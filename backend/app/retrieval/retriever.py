"""
Orchestrates: query → BM25 → dense → RRF → rerank → top-k chunk dicts.

Designed to run headless (no FastAPI context) so the eval harness can import
and call retrieve() directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.indexing.sparse import get_bm25_index
from app.retrieval.hybrid import fuse_bm25_dense
from app.retrieval.rerank import rerank

# Fallback chunk lookup loaded from chunks.jsonl — used when Qdrant is unavailable
_chunk_lookup: dict[str, dict] | None = None


def _get_chunk_lookup() -> dict[str, dict]:
    global _chunk_lookup
    if _chunk_lookup is not None:
        return _chunk_lookup
    jsonl = settings.data_dir / "chunks.jsonl"
    if not jsonl.exists():
        return {}
    lookup: dict[str, dict] = {}
    with jsonl.open() as f:
        for line in f:
            c = json.loads(line)
            lookup[c["chunk_id"]] = c
    _chunk_lookup = lookup
    return lookup


def _fetch_payloads(chunk_ids: list[str]) -> list[dict]:
    """Try Qdrant first; fall back to chunks.jsonl lookup."""
    try:
        from app.indexing.dense import get_chunks_by_ids
        results = get_chunks_by_ids(chunk_ids)
        if results:
            return results
    except Exception:
        pass
    # Fallback: lookup from JSONL
    lookup = _get_chunk_lookup()
    return [lookup[cid] for cid in chunk_ids if cid in lookup]


def retrieve(
    query: str,
    workspace_id: str,
    filters: dict | None = None,
    bm25_top_k: int | None = None,
    dense_top_k: int | None = None,
    rrf_k: int | None = None,
    rerank_top_k: int | None = None,
    use_rerank: bool = True,
    use_bm25: bool = True,
    use_dense: bool = True,
    index_dir: Path | None = None,
) -> list[dict]:
    bm25_k = bm25_top_k or settings.bm25_top_k
    dense_k = dense_top_k or settings.dense_top_k
    rrf_constant = rrf_k or settings.rrf_k
    final_k = rerank_top_k or settings.rerank_top_k
    idx_dir = index_dir or settings.bm25_index_dir

    bm25_results: list[tuple[str, float]] = []
    dense_results: list[tuple[str, float]] = []

    if use_bm25:
        try:
            bm25_idx = get_bm25_index(workspace_id, idx_dir)
            bm25_results = bm25_idx.search(query, top_k=bm25_k)
        except FileNotFoundError:
            pass

    if use_dense:
        try:
            from app.indexing.dense import search_dense
            dense_results = search_dense(query, workspace_id, top_k=dense_k, filters=filters)
        except Exception:
            pass

    if not bm25_results and not dense_results:
        return []

    fused = fuse_bm25_dense(bm25_results, dense_results, k=rrf_constant)
    pool_size = max(bm25_k, dense_k)
    candidate_ids = [doc_id for doc_id, _ in fused[:pool_size]]

    candidates = _fetch_payloads(candidate_ids)

    # Filter by workspace_id if needed
    if workspace_id:
        candidates = [c for c in candidates if c.get("workspace_id") == workspace_id or not c.get("workspace_id")]

    id_to_payload = {c["chunk_id"]: c for c in candidates}
    ordered = [id_to_payload[cid] for cid in candidate_ids if cid in id_to_payload]

    if use_rerank and ordered:
        return rerank(query, ordered, top_k=final_k)

    return ordered[:final_k]
