"""
Concept + relation extraction from chunks.
Extracts (concept, relation, concept) triples via LLM and spaCy co-occurrence.
"""

from __future__ import annotations

import json
import re

from app.config import settings

EXTRACT_SYSTEM_PROMPT = """Extract knowledge graph triples from lecture text.
Return ONLY valid JSON — no markdown.
Format: [{"subject": ..., "predicate": ..., "object": ...}, ...]
Focus on academic concepts and their relationships. Max 10 triples per chunk.
"""

EXTRACT_USER_PROMPT = """Extract concept triples from this lecture material:

{text}

Return JSON array of triples."""


def extract_triples_llm(text: str) -> list[dict]:
    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACT_USER_PROMPT.format(text=text[:2000])},
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    except Exception:
        return []


def extract_triples_from_chunks(chunks: list[dict], max_chunks: int = 50) -> list[dict]:
    """Extract triples from a sample of chunks (LLM cost control)."""
    all_triples: list[dict] = []
    for chunk in chunks[:max_chunks]:
        triples = extract_triples_llm(chunk.get("text", ""))
        for triple in triples:
            triple["source_module"] = chunk.get("module", "")
            triple["source_week"] = chunk.get("week")
        all_triples.extend(triples)
    return all_triples
