"""
Reciprocal Rank Fusion (RRF) over BM25 + dense ranked lists.

RRF(d) = sum_r 1 / (k + r(d))  where k=60 is the standard default.

Why RRF over weighted score fusion: BM25 and cosine scores live in incompatible
distributions (BM25 scores are unbounded; cosine is [-1,1]). RRF is rank-based
and parameter-light — this is the IR-module reasoning interviewers want.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists via RRF.

    Args:
        ranked_lists: Each list is [(doc_id, score), …] sorted desc.
        k: RRF constant (default 60 from literature).

    Returns:
        Merged [(doc_id, rrf_score), …] sorted desc, deduplicated.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_bm25_dense(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Convenience wrapper: fuse two ranked lists and deduplicate."""
    return reciprocal_rank_fusion([bm25_results, dense_results], k=k)
