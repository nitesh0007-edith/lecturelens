"""
Parser tests with synthetic in-memory fixtures.
Tests the dispatch logic, doc_type inference, and HTML / notebook parsing
without requiring actual PDF/PPTX binary files.
"""

import io
import json
import textwrap
from pathlib import Path

import pytest

from app.ingestion.parsers import infer_doc_type, parse_html, parse_ipynb


# ---------------------------------------------------------------------------
# doc_type inference
# ---------------------------------------------------------------------------

def test_infer_past_paper():
    assert infer_doc_type(Path("IR/past_paper_2023.pdf")) == "past_paper"
    assert infer_doc_type(Path("exam_2021.pdf")) == "past_paper"


def test_infer_lab():
    assert infer_doc_type(Path("MLAI/lab_session3.ipynb")) == "lab"
    assert infer_doc_type(Path("tutorial_week2.html")) == "lab"


def test_infer_lecture():
    assert infer_doc_type(Path("IR/lecture_notes_week5.pdf")) == "lecture"


def test_infer_coursework():
    assert infer_doc_type(Path("IR/coursework_1.pdf")) == "coursework"


def test_infer_cheatsheet():
    assert infer_doc_type(Path("summary_sheet.pdf")) == "cheatsheet"


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def test_parse_html_basic(tmp_path):
    html = """
    <html><body>
    <h1>Introduction to IR</h1>
    <p>Information retrieval is the process of finding documents.</p>
    <h2>BM25</h2>
    <p>BM25 is a probabilistic ranking model.</p>
    </body></html>
    """
    p = tmp_path / "lecture.html"
    p.write_text(html)
    pages = parse_html(p)
    assert len(pages) >= 1
    full_text = " ".join(pg.text for pg in pages)
    assert "BM25" in full_text
    assert "Information retrieval" in full_text


def test_parse_html_strips_script(tmp_path):
    html = """
    <html><body>
    <script>alert('xss')</script>
    <h1>NLP</h1>
    <p>Natural language processing.</p>
    </body></html>
    """
    p = tmp_path / "test.html"
    p.write_text(html)
    pages = parse_html(p)
    full_text = " ".join(pg.text for pg in pages)
    assert "alert" not in full_text


def test_parse_html_source_file(tmp_path):
    p = tmp_path / "notes.html"
    p.write_text("<html><body><p>Hello world.</p></body></html>")
    pages = parse_html(p)
    assert all(pg.source_file == str(p) for pg in pages)


# ---------------------------------------------------------------------------
# Notebook parser (synthetic .ipynb)
# ---------------------------------------------------------------------------

def _make_notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3"}},
        "cells": cells,
    }


def test_parse_ipynb_basic(tmp_path):
    nb = _make_notebook([
        {"cell_type": "markdown", "source": "## Introduction\nThis notebook covers BM25.", "metadata": {}, "id": "1"},
        {"cell_type": "code", "source": "import numpy as np", "outputs": [], "metadata": {}, "id": "2", "execution_count": None},
        {"cell_type": "markdown", "source": "## Results\nThe results show improvement.", "metadata": {}, "id": "3"},
    ])
    p = tmp_path / "lecture.ipynb"
    p.write_text(json.dumps(nb))
    pages = parse_ipynb(p)
    assert len(pages) >= 1
    full_text = " ".join(pg.text for pg in pages)
    assert "BM25" in full_text


def test_parse_ipynb_heading_extraction(tmp_path):
    nb = _make_notebook([
        {"cell_type": "markdown", "source": "# Topic Modelling\nLDA is a generative model.", "metadata": {}, "id": "1"},
    ])
    p = tmp_path / "test.ipynb"
    p.write_text(json.dumps(nb))
    pages = parse_ipynb(p)
    assert any(pg.title == "Topic Modelling" for pg in pages)


def test_parse_ipynb_code_python_fence(tmp_path):
    nb = _make_notebook([
        {"cell_type": "markdown", "source": "## Code example", "metadata": {}, "id": "1"},
        {"cell_type": "code", "source": "x = 42\nprint(x)", "outputs": [], "metadata": {}, "id": "2", "execution_count": 1},
    ])
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    pages = parse_ipynb(p)
    full = " ".join(pg.text for pg in pages)
    assert "python" in full.lower() or "x = 42" in full


def test_parse_ipynb_strips_large_output(tmp_path):
    large_output = "x" * 1000
    nb = _make_notebook([
        {"cell_type": "markdown", "source": "## Results", "metadata": {}, "id": "1"},
        {
            "cell_type": "code",
            "source": "print('big output')",
            "outputs": [{"output_type": "stream", "text": large_output}],
            "metadata": {},
            "id": "2",
            "execution_count": 1,
        },
    ])
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    pages = parse_ipynb(p)
    full = " ".join(pg.text for pg in pages)
    # Large output should be stripped
    assert large_output not in full
