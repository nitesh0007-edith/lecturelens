"""
POST /api/ask  — streaming SSE cited answer (with cache + trace)
POST /api/quiz — generate MCQs
POST /api/exam — generate exam questions
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.generation.answer import generate_answer, stream_answer
from app.generation.quiz import generate_mcqs, generate_short_answers
from app.generation.exam import generate_exam_questions
from app.retrieval.retriever import retrieve
from app.middleware.tracing import Trace, traced_stage

router = APIRouter()


class AskRequest(BaseModel):
    query: str
    workspace_id: str = "uofg-msds-demo"
    module: Optional[str] = None
    week: Optional[int] = None
    doc_type: Optional[str] = None
    stream: bool = False


class QuizRequest(BaseModel):
    topic: str
    workspace_id: str = "uofg-msds-demo"
    module: Optional[str] = None
    week: Optional[int] = None
    n: int = 5
    quiz_type: str = "mcq"


class ExamRequest(BaseModel):
    topic: str
    workspace_id: str = "uofg-msds-demo"
    module: Optional[str] = None
    difficulty: str = "medium"
    n: int = 3


@router.post("/ask")
async def ask(req: AskRequest):
    trace = Trace(query=req.query, workspace_id=req.workspace_id)

    # --- Semantic cache lookup ---
    try:
        from app.middleware.cache import lookup, store
        with traced_stage(trace, "cache_lookup"):
            cached = lookup(req.query, req.workspace_id)
        if cached:
            trace.emit(answer_length=len(cached.get("answer", "")), cached=True)
            return cached
    except Exception:
        cached = None

    filters: dict = {}
    if req.module:
        filters["module"] = req.module
    if req.week:
        filters["week"] = req.week
    if req.doc_type:
        filters["doc_type"] = req.doc_type

    # --- Retrieval ---
    with traced_stage(trace, "retrieval", filters=filters):
        chunks = retrieve(req.query, req.workspace_id, filters=filters or None)

    chunk_ids = [c.get("chunk_id", "") for c in chunks]

    if req.stream:
        async def event_generator():
            async for token_json in stream_answer(req.query, chunks):
                yield f"data: {token_json}\n\n"
            trace.end_stage("generation", chunk_ids=chunk_ids)
            trace.emit()

        trace.start_stage("generation")
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # --- Generation ---
    with traced_stage(trace, "generation", chunk_ids=chunk_ids):
        result = generate_answer(req.query, chunks)

    # Cache successful answers
    try:
        if result.get("answer") and "unavailable" not in result["answer"].lower():
            store(req.query, req.workspace_id, result)
    except Exception:
        pass

    trace.emit(answer_length=len(result.get("answer", "")))
    return result


@router.post("/quiz")
def quiz(req: QuizRequest):
    filters: dict = {}
    if req.module:
        filters["module"] = req.module
    if req.week:
        filters["week"] = req.week

    chunks = retrieve(
        req.topic, req.workspace_id,
        filters=filters or None,
        rerank_top_k=8,
    )

    if req.quiz_type == "short_answer":
        questions = generate_short_answers(chunks, n=req.n)
    else:
        questions = generate_mcqs(chunks, n=req.n)

    return {"questions": questions, "source_chunks": len(chunks)}


@router.post("/exam")
def exam(req: ExamRequest):
    questions = generate_exam_questions(
        topic=req.topic,
        workspace_id=req.workspace_id,
        module=req.module,
        difficulty=req.difficulty,
        n=req.n,
    )
    return {"questions": questions}
