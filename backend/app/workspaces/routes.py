"""
Workspace management API.

POST   /api/workspaces/                  — create workspace
GET    /api/workspaces/{workspace_id}    — get workspace info
POST   /api/workspaces/{workspace_id}/upload — upload file and kick off ingest
GET    /api/workspaces/{workspace_id}/status — ingest job status
"""

from __future__ import annotations

import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import settings
from app.workspaces.models import (
    Document,
    IngestJob,
    JobStatus,
    Workspace,
    create_db_and_tables,
    get_session,
)

router = APIRouter()

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB per workspace cap
ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".ipynb", ".html", ".htm"}


# DB tables are created in main.py lifespan; nothing needed here


class CreateWorkspaceRequest(BaseModel):
    name: str


@router.post("/")
def create_workspace(req: CreateWorkspaceRequest, session: Session = Depends(get_session)):
    workspace_id = re.sub(r"[^a-z0-9\-]", "-", req.name.lower().strip())[:40]
    existing = session.exec(select(Workspace).where(Workspace.workspace_id == workspace_id)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Workspace ID already exists")

    ws = Workspace(workspace_id=workspace_id, name=req.name)
    session.add(ws)
    session.commit()
    session.refresh(ws)
    return {"workspace_id": ws.workspace_id, "token": ws.token, "name": ws.name}


@router.get("/{workspace_id}")
def get_workspace(workspace_id: str, session: Session = Depends(get_session)):
    ws = session.exec(select(Workspace).where(Workspace.workspace_id == workspace_id)).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws


def _verify_token(workspace_id: str, token: str, session: Session) -> Workspace:
    ws = session.exec(select(Workspace).where(Workspace.workspace_id == workspace_id)).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if ws.token != token:
        raise HTTPException(status_code=403, detail="Invalid workspace token")
    return ws


@router.post("/{workspace_id}/upload")
async def upload_file(
    workspace_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_workspace_token: str = Header(...),
    session: Session = Depends(get_session),
):
    ws = _verify_token(workspace_id, x_workspace_token, session)

    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.filename}")

    # Save file to workspace directory
    ws_dir = settings.workspace_data_dir / workspace_id
    ws_dir.mkdir(parents=True, exist_ok=True)
    dest = ws_dir / file.filename

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 200 MB)")

    dest.write_bytes(content)

    # Create document record
    doc = Document(
        workspace_id=workspace_id,
        filename=file.filename,
        file_size_bytes=len(content),
        status=JobStatus.pending,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # Create ingest job
    job = IngestJob(workspace_id=workspace_id, document_id=doc.id)
    session.add(job)
    session.commit()
    session.refresh(job)

    # Kick off background ingest
    background_tasks.add_task(_run_ingest_job, job.id, workspace_id, dest, doc.id)

    return {"document_id": doc.id, "job_id": job.id, "status": "pending"}


def _run_ingest_job(job_id: int, workspace_id: str, file_path: Path, doc_id: int):
    """Background task: parse → chunk → enrich → index."""
    from sqlmodel import Session as S

    from app.indexing.pipeline import ingest_files, run_pipeline
    from app.workspaces.models import engine

    with S(engine) as session:
        job = session.get(IngestJob, job_id)
        doc = session.get(Document, doc_id)
        if not job or not doc:
            return

        def update_status(status: str):
            job.status = JobStatus(status)
            doc.status = JobStatus(status)
            session.add(job)
            session.add(doc)
            session.commit()

        try:
            update_status("parsing")
            chunks = ingest_files([file_path], workspace_id)

            stats = run_pipeline(workspace_id, chunks, update_status_fn=update_status)

            doc.chunk_count = stats.get("chunk_count", 0)
            doc.status = JobStatus.done
            job.status = JobStatus.done
            job.completed_at = datetime.utcnow()

            # Update workspace totals
            ws = session.exec(select(Workspace).where(Workspace.workspace_id == workspace_id)).first()
            if ws:
                ws.doc_count += 1
                ws.chunk_count += doc.chunk_count
                session.add(ws)

            session.add(doc)
            session.add(job)
            session.commit()

        except Exception as e:
            job.status = JobStatus.error
            job.error_message = str(e)[:500]
            doc.status = JobStatus.error
            doc.error_message = str(e)[:500]
            session.add(job)
            session.add(doc)
            session.commit()


@router.get("/{workspace_id}/status")
def get_status(workspace_id: str, session: Session = Depends(get_session)):
    jobs = session.exec(
        select(IngestJob).where(IngestJob.workspace_id == workspace_id)
    ).all()
    docs = session.exec(
        select(Document).where(Document.workspace_id == workspace_id)
    ).all()
    return {
        "workspace_id": workspace_id,
        "documents": [{"id": d.id, "filename": d.filename, "status": d.status,
                        "chunks": d.chunk_count, "error": d.error_message} for d in docs],
        "jobs": [{"id": j.id, "status": j.status, "error": j.error_message} for j in jobs],
    }
