"""
GET /api/recommend?q=...&workspace_id=...&top_n=5

Aggregate chunk similarity scores by (module, week) to recommend lectures.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Query

from app.retrieval.retriever import retrieve

router = APIRouter()


@router.get("/recommend")
def recommend(
    q: str = Query(..., description="Topic or concept to find lectures for"),
    workspace_id: str = Query("uofg-msds-demo"),
    top_n: int = Query(5, ge=1, le=20),
    module: Optional[str] = Query(None),
):
    """
    Return the top-N lectures most relevant to the query,
    grouped by (module, week) with aggregated relevance.
    """
    filters = {"module": module} if module else None
    chunks = retrieve(
        q, workspace_id,
        filters=filters,
        rerank_top_k=30,
        use_rerank=False,
    )

    # Aggregate by (module, week)
    lecture_scores: dict[tuple, list[float]] = defaultdict(list)
    lecture_meta: dict[tuple, dict] = {}

    for rank, chunk in enumerate(chunks):
        mod = chunk.get("module", "Unknown")
        week = chunk.get("week")
        key = (mod, week)
        # RRF-style score by rank position
        score = 1.0 / (60 + rank + 1)
        lecture_scores[key].append(score)
        if key not in lecture_meta:
            lecture_meta[key] = {
                "module": mod,
                "module_code": chunk.get("module_code", ""),
                "week": week,
                "sample_titles": [],
            }
        if chunk.get("title"):
            titles = lecture_meta[key]["sample_titles"]
            if chunk["title"] not in titles and len(titles) < 3:
                titles.append(chunk["title"])

    ranked = sorted(
        lecture_scores.keys(),
        key=lambda k: sum(lecture_scores[k]),
        reverse=True,
    )[:top_n]

    return {
        "query": q,
        "recommendations": [
            {
                **lecture_meta[k],
                "score": round(sum(lecture_scores[k]), 4),
                "chunk_hits": len(lecture_scores[k]),
            }
            for k in ranked
        ],
    }
