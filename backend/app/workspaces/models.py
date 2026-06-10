"""SQLite data model via SQLModel: workspaces, documents, ingest jobs."""

from __future__ import annotations

import secrets
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from app.config import settings

DATABASE_URL = f"sqlite:///{settings.sqlite_db}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


class JobStatus(str, Enum):
    pending = "pending"
    parsing = "parsing"
    indexing_sparse = "indexing_sparse"
    indexing_dense = "indexing_dense"
    done = "done"
    error = "error"


class Workspace(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    workspace_id: str = Field(unique=True, index=True)
    name: str
    token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    doc_count: int = 0
    chunk_count: int = 0


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    workspace_id: str = Field(index=True)
    filename: str
    file_size_bytes: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    status: JobStatus = JobStatus.pending
    chunk_count: int = 0
    error_message: str = ""


class IngestJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    workspace_id: str = Field(index=True)
    document_id: int
    status: JobStatus = JobStatus.pending
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: str = ""


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
