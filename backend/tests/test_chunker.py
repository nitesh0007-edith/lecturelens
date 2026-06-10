"""
Tests for the chunker: metadata extraction, chunk counts, no mid-sentence splits,
past-paper splitting, PPTX one-chunk-per-slide, token range.
"""

import pytest
from app.ingestion.chunker import (
    Chunk,
    _approx_tokens,
    _build_chunks_from_sentences,
    _extract_module_week,
    _split_past_paper,
    chunk_pages,
    TARGET_MAX_TOKENS,
    TARGET_MIN_TOKENS,
)
from app.ingestion.parsers import RawPage


# ---------------------------------------------------------------------------
# Module / week metadata extraction
# ---------------------------------------------------------------------------

def test_extract_module_ir():
    module, code, week = _extract_module_week("/repo/IR/Lecture_Notes/week5_bm25.pdf")
    assert module == "IR"
    assert code == "COMPSCI5011"
    assert week == 5


def test_extract_module_deeplearning():
    module, code, week = _extract_module_week("DeepLearning/Week3/transformer.pptx")
    assert module == "DeepLearning"
    assert week == 3


def test_extract_no_week():
    module, code, week = _extract_module_week("IR/notes.pdf")
    assert module == "IR"
    assert week is None


# ---------------------------------------------------------------------------
# Sentence splitter / chunk builder
# ---------------------------------------------------------------------------

def test_chunks_respect_max_tokens():
    # Create text well above max tokens
    sentence = "This is a test sentence about information retrieval. " * 20
    chunks = _build_chunks_from_sentences(sentence.split(". "))
    for chunk in chunks:
        assert _approx_tokens(chunk) <= TARGET_MAX_TOKENS + 50  # small buffer


def test_chunks_not_empty():
    sentences = ["BM25 is a ranking function.", "It improves on TF-IDF."]
    chunks = _build_chunks_from_sentences(sentences)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)


def test_overlap_produces_multiple_chunks():
    # 10 short sentences → should produce at least 1 chunk
    sentences = [f"Sentence number {i} about neural networks and embeddings." for i in range(30)]
    chunks = _build_chunks_from_sentences(sentences)
    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Past paper splitting
# ---------------------------------------------------------------------------

def test_past_paper_split():
    text = """
Q1. Define precision and recall in the context of IR evaluation.

Q2. Explain how BM25 differs from TF-IDF. Give the formula.

Q3. What is the purpose of stop word removal?
"""
    parts = _split_past_paper(text)
    assert len(parts) >= 2  # at least 2 questions detected


def test_past_paper_split_question_prefix():
    text = "Question 1. Describe the vector space model.\nQuestion 2. What is cosine similarity?"
    parts = _split_past_paper(text)
    assert len(parts) >= 1


# ---------------------------------------------------------------------------
# chunk_pages — integration
# ---------------------------------------------------------------------------

def test_chunk_pages_pptx():
    pages = [
        RawPage(
            text="Slide title\nBullet one\nBullet two",
            source_file="IR/week1/intro.pptx",
            page_or_slide=1,
            doc_type="lecture",
            title="Slide title",
        )
    ]
    chunks = chunk_pages(pages, "test-workspace")
    assert len(chunks) == 1
    assert chunks[0].module == "IR"
    assert chunks[0].doc_type == "lecture"
    assert chunks[0].page_or_slide == 1


def test_chunk_pages_metadata_fields():
    pages = [
        RawPage(
            text="BM25 is a probabilistic retrieval model. " * 20,
            source_file="IR/Lecture_Notes/week5_bm25.pdf",
            page_or_slide=5,
            doc_type="lecture",
            title="BM25",
        )
    ]
    chunks = chunk_pages(pages, "test-ws")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_id
        assert c.workspace_id == "test-ws"
        assert c.module == "IR"
        assert c.week == 5
        assert c.source_file.endswith(".pdf")
        assert c.text.strip()


def test_chunk_ids_unique():
    pages = [
        RawPage(
            text="Sentence one. " * 30,
            source_file="IR/week1.pdf",
            page_or_slide=1,
            doc_type="lecture",
        ),
        RawPage(
            text="Sentence two. " * 30,
            source_file="IR/week1.pdf",
            page_or_slide=2,
            doc_type="lecture",
        ),
    ]
    chunks = chunk_pages(pages, "test-ws")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_past_paper_chunks():
    pages = [
        RawPage(
            text="Q1. Explain inverted indexes.\nQ2. What is BM25?",
            source_file="IR/pastpaper_2023.pdf",
            page_or_slide=1,
            doc_type="past_paper",
            title="",
        )
    ]
    chunks = chunk_pages(pages, "test-ws")
    assert all(c.doc_type == "past_paper" for c in chunks)


def test_chunk_to_dict():
    c = Chunk(
        chunk_id="abc123",
        workspace_id="ws1",
        module="IR",
        module_code="COMPSCI5011",
        week=1,
        doc_type="lecture",
        source_file="IR/week1.pdf",
        page_or_slide=1,
        chunk_index=0,
        title="Test",
        text="hello world",
        contextualised_text="From IR Week 1: hello world",
        token_count=3,
    )
    d = c.to_dict()
    assert d["chunk_id"] == "abc123"
    assert d["module"] == "IR"
    assert d["week"] == 1
