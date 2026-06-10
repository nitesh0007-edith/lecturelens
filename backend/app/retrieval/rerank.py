"""
Cross-encoder re-ranking via FastEmbed (ONNX, CPU).

We only re-rank the top 30 RRF candidates, which takes ~1-2 s on CPU.
This is where most of the NDCG@10 gain comes from over hybrid-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    pass

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(settings.rerank_model, max_length=512)
        except (ImportError, Exception):
            _reranker = None
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 6,
    text_field: str = "text",
) -> list[dict]:
    """
    Re-rank candidate chunk dicts by cross-encoder score.

    Args:
        query: The search query string.
        candidates: List of chunk payload dicts (must have `text_field`).
        top_k: Number of chunks to return.
        text_field: Which field contains the passage text.

    Returns:
        Top-k candidates sorted by cross-encoder score desc.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    if reranker is None:
        # Graceful degradation: return top-k by insertion order (RRF score)
        return candidates[:top_k]

    texts = [c.get(text_field, c.get("contextualised_text", "")) for c in candidates]
    try:
        pairs = [[query, t] for t in texts]
        scores = reranker.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]
    except Exception:
        return candidates[:top_k]
