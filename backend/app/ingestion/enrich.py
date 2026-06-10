"""
Chunk enrichment: NER (spaCy) + keyphrases (YAKE) + contextual prefix.

The contextual prefix ("From IR Week 5 lecture on probabilistic ranking: …")
is Anthropic's 'contextual retrieval' technique — cheap one-time offline cost,
meaningful retrieval lift on slide decks where individual slides lack context.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk


# ---------------------------------------------------------------------------
# NER — spaCy
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy

            _NLP = spacy.load("en_core_web_sm", disable=["parser", "senter"])
        except OSError:
            # Model not downloaded yet; fall back to blank model
            import spacy

            _NLP = spacy.blank("en")
    return _NLP


def extract_entities(text: str) -> list[str]:
    nlp = _get_nlp()
    doc = nlp(text[:5000])  # cap to avoid slow processing on very long chunks
    seen: set[str] = set()
    entities: list[str] = []
    for ent in doc.ents:
        label = ent.text.strip()
        if label and label not in seen and len(label) > 2:
            seen.add(label)
            entities.append(label)
    return entities[:20]


# ---------------------------------------------------------------------------
# Keyphrases — YAKE
# ---------------------------------------------------------------------------

_YAKE_EXTRACTOR = None


def _get_yake():
    global _YAKE_EXTRACTOR
    if _YAKE_EXTRACTOR is None:
        try:
            import yake

            _YAKE_EXTRACTOR = yake.KeywordExtractor(
                lan="en", n=3, dedupLim=0.7, top=10, features=None
            )
        except ImportError:
            pass
    return _YAKE_EXTRACTOR


def extract_keyphrases(text: str) -> list[str]:
    extractor = _get_yake()
    if extractor is None:
        return []
    try:
        kws = extractor.extract_keywords(text[:3000])
        return [kw for kw, _score in kws]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Contextual prefix
# ---------------------------------------------------------------------------

def build_context_prefix(chunk: "Chunk") -> str:
    """Build a one-line context prefix for embedding."""
    parts: list[str] = ["From"]
    if chunk.module and chunk.module != "Unknown":
        parts.append(chunk.module)
    if chunk.week:
        parts.append(f"Week {chunk.week}")
    if chunk.doc_type and chunk.doc_type != "lecture":
        parts.append(chunk.doc_type.replace("_", " "))
    if chunk.title:
        clean_title = re.sub(r"\s+", " ", chunk.title).strip()
        parts.append(f"on {clean_title}")
    return " ".join(parts) + ":"


def enrich_chunk(chunk: "Chunk") -> "Chunk":
    """Add NER entities, keyphrases, and contextual prefix in-place."""
    chunk.entities = extract_entities(chunk.text)
    chunk.keyphrases = extract_keyphrases(chunk.text)
    prefix = build_context_prefix(chunk)
    chunk.contextualised_text = f"{prefix} {chunk.text}"
    return chunk


def enrich_chunks(chunks: list["Chunk"], verbose: bool = False) -> list["Chunk"]:
    enriched = []
    for i, chunk in enumerate(chunks):
        enriched.append(enrich_chunk(chunk))
        if verbose and (i + 1) % 100 == 0:
            print(f"  enriched {i + 1}/{len(chunks)}")
    return enriched
