"""
Tests for RRF fusion — pure function, no external dependencies.
"""

import pytest
from app.retrieval.hybrid import fuse_bm25_dense, reciprocal_rank_fusion


def test_rrf_basic():
    bm25 = [("doc_a", 3.5), ("doc_b", 2.1), ("doc_c", 1.0)]
    dense = [("doc_b", 0.95), ("doc_a", 0.90), ("doc_d", 0.80)]

    fused = fuse_bm25_dense(bm25, dense)
    assert len(fused) == 4

    ids = [doc_id for doc_id, _ in fused]
    # doc_a and doc_b appear in both lists → should rank higher than doc_c or doc_d
    assert ids.index("doc_a") < ids.index("doc_c")
    assert ids.index("doc_b") < ids.index("doc_d")


def test_rrf_deduplication():
    shared = [("doc_x", 10.0), ("doc_y", 5.0)]
    fused = reciprocal_rank_fusion([shared, shared])
    ids = [doc_id for doc_id, _ in fused]
    assert len(ids) == len(set(ids)), "RRF must deduplicate"


def test_rrf_empty_list():
    fused = fuse_bm25_dense([], [])
    assert fused == []


def test_rrf_single_list():
    bm25 = [("a", 2.0), ("b", 1.0)]
    fused = fuse_bm25_dense(bm25, [])
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["a", "b"]


def test_rrf_scores_positive():
    bm25 = [("a", 1.0), ("b", 0.5)]
    dense = [("a", 0.9), ("c", 0.7)]
    fused = fuse_bm25_dense(bm25, dense)
    for _, score in fused:
        assert score > 0


def test_rrf_k_parameter():
    """Lower k amplifies rank differences (top docs score higher)."""
    bm25 = [("a", 1.0), ("b", 0.5)]
    dense = [("a", 0.9)]

    fused_60 = dict(fuse_bm25_dense(bm25, dense, k=60))
    fused_1 = dict(fuse_bm25_dense(bm25, dense, k=1))

    # k=1 gives higher total scores than k=60 for top doc
    assert fused_1["a"] > fused_60["a"]


def test_rrf_multiple_lists():
    list1 = [("a", 1.0), ("b", 0.5), ("c", 0.3)]
    list2 = [("b", 0.9), ("a", 0.7), ("d", 0.4)]
    list3 = [("a", 0.8), ("c", 0.6), ("e", 0.2)]

    fused = reciprocal_rank_fusion([list1, list2, list3])
    ids = [doc_id for doc_id, _ in fused]
    # "a" appears in all 3 lists at rank 1 → should be top
    assert ids[0] == "a"
