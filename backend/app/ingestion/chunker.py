"""
Structure-aware chunker.

Rules per §5 of PROJECT_PLAN.md:
- PPTX: one chunk per slide (already done by parse_pptx — just wrap).
- PDF: split on headings, target 250–400 tokens, 15% overlap, no mid-sentence split.
- Notebooks: markdown+code pairs are already logical units; just wrap.
- Past papers: one chunk per question (regex Q\\d+ / Question \\d+).
- HTML: section-per-heading already from parse_html.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.ingestion.parsers import RawPage


APPROX_TOKENS_PER_CHAR = 0.25   # rough chars→tokens for Latin text
TARGET_MIN_TOKENS = 250
TARGET_MAX_TOKENS = 400
OVERLAP_FRACTION = 0.15


@dataclass
class Chunk:
    """A single retrieval unit ready for indexing."""

    chunk_id: str              # sha256 of workspace_id + source_file + page + chunk_index
    workspace_id: str
    module: str
    module_code: str
    week: int | None
    doc_type: str
    source_file: str
    page_or_slide: int
    chunk_index: int           # 0-based within the page/slide
    title: str
    text: str                  # raw text
    contextualised_text: str   # prepended context (filled during enrich step)
    entities: list[str] = field(default_factory=list)
    keyphrases: list[str] = field(default_factory=list)
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "workspace_id": self.workspace_id,
            "module": self.module,
            "module_code": self.module_code,
            "week": self.week,
            "doc_type": self.doc_type,
            "source_file": self.source_file,
            "page_or_slide": self.page_or_slide,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "text": self.text,
            "contextualised_text": self.contextualised_text,
            "entities": self.entities,
            "keyphrases": self.keyphrases,
            "token_count": self.token_count,
        }


# ---------------------------------------------------------------------------
# Module / week metadata extraction from file paths
# ---------------------------------------------------------------------------

MODULE_DIRS = {
    "bigdata": ("BigData", "COMPSCI5074"),
    "cybersec": ("CyberSec", "COMPSCI5083"),
    "cybersecurityfundamentals": ("CyberSec", "COMPSCI5083"),
    "idss": ("IDSS", "COMPSCI5070"),
    "ds": ("IDSS", "COMPSCI5070"),
    "deeplearning": ("DeepLearning", "COMPSCI5079"),
    "ir": ("IR", "COMPSCI5011"),
    "iv": ("IV", "COMPSCI5060"),
    "mlai": ("MLAI", "COMPSCI5078"),
    "progsd": ("ProgSD", "COMPSCI5073"),
    "rps": ("RPS", "COMPSCI5098"),
    "textasdata": ("TextasData", "COMPSCI5069"),
}


def _extract_module_week(source_file: str) -> tuple[str, str, int | None]:
    """Derive (module_name, module_code, week) from the repo path."""
    parts = Path(source_file).parts
    module_name = "Unknown"
    module_code = ""
    week: int | None = None

    for part in parts:
        key = part.lower().replace("_", "").replace("-", "")
        if key in MODULE_DIRS:
            module_name, module_code = MODULE_DIRS[key]
            break
        # Partial match
        for mod_key, (name, code) in MODULE_DIRS.items():
            if mod_key in key:
                module_name, module_code = name, code
                break

    # Try to extract week number from path or filename
    week_match = re.search(r"week[_\-]?(\d{1,2})", source_file, re.IGNORECASE)
    if not week_match:
        week_match = re.search(r"w(\d{1,2})[_\-\.]", source_file, re.IGNORECASE)
    if week_match:
        week = int(week_match.group(1))

    return module_name, module_code, week


# ---------------------------------------------------------------------------
# Token estimation (cheap, no tokenizer dependency)
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    return max(1, int(len(text) * APPROX_TOKENS_PER_CHAR))


# ---------------------------------------------------------------------------
# Sentence-boundary-safe text splitting
# ---------------------------------------------------------------------------

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_END.split(text) if s.strip()]


def _build_chunks_from_sentences(
    sentences: list[str],
    min_tokens: int = TARGET_MIN_TOKENS,
    max_tokens: int = TARGET_MAX_TOKENS,
    overlap_frac: float = OVERLAP_FRACTION,
) -> list[str]:
    """Greedy sentence packing with overlap."""
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tok = 0

    for sent in sentences:
        sent_tok = _approx_tokens(sent)
        if current_tok + sent_tok > max_tokens and current_tok >= min_tokens:
            chunks.append(" ".join(current))
            # Overlap: keep last ~15% by token count
            overlap_target = int(current_tok * overlap_frac)
            carry: list[str] = []
            carry_tok = 0
            for s in reversed(current):
                if carry_tok + _approx_tokens(s) > overlap_target:
                    break
                carry.insert(0, s)
                carry_tok += _approx_tokens(s)
            current = carry
            current_tok = carry_tok

        current.append(sent)
        current_tok += sent_tok

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Past-paper question splitter
# ---------------------------------------------------------------------------

_QUESTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Q(?:uestion)?\s*\.?\s*(\d+)|(\d+)\s*\.)\s*",
    re.IGNORECASE,
)


def _split_past_paper(text: str) -> list[str]:
    parts = _QUESTION_PATTERN.split(text)
    # Recombine: every two "splits" form a question
    questions: list[str] = []
    # Flatten and filter None
    clean = [p for p in parts if p is not None]
    # Rebuild question texts (skip match group captures that are just numbers)
    rebuilt: list[str] = []
    current = ""
    for p in clean:
        if re.fullmatch(r"\d+", p.strip()):
            if current.strip():
                rebuilt.append(current.strip())
            current = ""
        else:
            current += p
    if current.strip():
        rebuilt.append(current.strip())

    return [q for q in rebuilt if len(q.strip()) > 30]


# ---------------------------------------------------------------------------
# Main chunking entry point
# ---------------------------------------------------------------------------

def _make_chunk_id(workspace_id: str, source_file: str, page: int, idx: int) -> str:
    import hashlib

    raw = f"{workspace_id}|{source_file}|{page}|{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_pages(
    pages: list[RawPage],
    workspace_id: str,
) -> list[Chunk]:
    """Convert RawPages → Chunks with full metadata."""
    chunks: list[Chunk] = []

    for page in pages:
        module_name, module_code, week = _extract_module_week(page.source_file)
        source_file = page.source_file
        doc_type = page.doc_type

        if doc_type == "past_paper":
            questions = _split_past_paper(page.text)
            if not questions:
                questions = [page.text]
            for q_idx, q_text in enumerate(questions):
                cid = _make_chunk_id(workspace_id, source_file, page.page_or_slide, q_idx)
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        workspace_id=workspace_id,
                        module=module_name,
                        module_code=module_code,
                        week=week,
                        doc_type=doc_type,
                        source_file=source_file,
                        page_or_slide=page.page_or_slide,
                        chunk_index=q_idx,
                        title=page.title or f"Question {q_idx + 1}",
                        text=q_text,
                        contextualised_text=q_text,
                        token_count=_approx_tokens(q_text),
                    )
                )

        elif page.source_file.lower().endswith((".pptx", ".ppt")):
            # One chunk per slide — already the right granularity
            cid = _make_chunk_id(workspace_id, source_file, page.page_or_slide, 0)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    workspace_id=workspace_id,
                    module=module_name,
                    module_code=module_code,
                    week=week,
                    doc_type=doc_type,
                    source_file=source_file,
                    page_or_slide=page.page_or_slide,
                    chunk_index=0,
                    title=page.title,
                    text=page.text,
                    contextualised_text=page.text,
                    token_count=_approx_tokens(page.text),
                )
            )

        elif page.source_file.lower().endswith(".ipynb"):
            # Notebook markdown+code pairs are already logical units
            cid = _make_chunk_id(workspace_id, source_file, page.page_or_slide, 0)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    workspace_id=workspace_id,
                    module=module_name,
                    module_code=module_code,
                    week=week,
                    doc_type=doc_type,
                    source_file=source_file,
                    page_or_slide=page.page_or_slide,
                    chunk_index=0,
                    title=page.title,
                    text=page.text,
                    contextualised_text=page.text,
                    token_count=_approx_tokens(page.text),
                )
            )

        else:
            # PDF / HTML: sentence-based chunking with overlap
            sentences = _split_sentences(page.text)
            text_chunks = _build_chunks_from_sentences(sentences)
            if not text_chunks:
                text_chunks = [page.text]

            for c_idx, chunk_text in enumerate(text_chunks):
                cid = _make_chunk_id(workspace_id, source_file, page.page_or_slide, c_idx)
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        workspace_id=workspace_id,
                        module=module_name,
                        module_code=module_code,
                        week=week,
                        doc_type=doc_type,
                        source_file=source_file,
                        page_or_slide=page.page_or_slide,
                        chunk_index=c_idx,
                        title=page.title,
                        text=chunk_text,
                        contextualised_text=chunk_text,
                        token_count=_approx_tokens(chunk_text),
                    )
                )

    return chunks
