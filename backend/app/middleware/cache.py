"""
Semantic answer cache — Phase 7.

Caches answers per workspace keyed by query-embedding similarity.
Threshold: cosine > 0.95 → cache hit. TTL: 7 days.
SQLite-backed so it survives restarts without Qdrant.

Why: students ask "explain BM25" repeatedly; sub-second cached responses
     also act as Groq rate-limit defence for the public demo.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from app.config import settings

_CACHE_DB = settings.sqlite_db.parent / "answer_cache.db"


def _conn() -> sqlite3.Connection:
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_CACHE_DB))
    c.execute(
        """CREATE TABLE IF NOT EXISTS answer_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            query TEXT NOT NULL,
            query_embedding BLOB NOT NULL,
            answer_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )"""
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_ws ON answer_cache(workspace_id)")
    c.commit()
    return c


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embed(text: str) -> list[float]:
    from app.indexing.dense import embed_texts
    return embed_texts([text])[0]


def lookup(query: str, workspace_id: str) -> dict | None:
    """Return cached answer dict if a similar query exists, else None."""
    ttl_cutoff = time.time() - settings.cache_ttl_days * 86400
    threshold = settings.cache_similarity_threshold

    c = _conn()
    rows = c.execute(
        "SELECT query_embedding, answer_json FROM answer_cache "
        "WHERE workspace_id=? AND created_at>?",
        (workspace_id, ttl_cutoff),
    ).fetchall()
    c.close()

    if not rows:
        return None

    try:
        q_emb = _embed(query)
    except Exception:
        return None

    for emb_blob, answer_json in rows:
        cached_emb = json.loads(emb_blob)
        if _cosine(q_emb, cached_emb) >= threshold:
            result = json.loads(answer_json)
            result["cached"] = True
            return result

    return None


def store(query: str, workspace_id: str, answer: dict) -> None:
    """Store a new answer in the cache."""
    try:
        q_emb = _embed(query)
    except Exception:
        return

    c = _conn()
    c.execute(
        "INSERT INTO answer_cache (workspace_id, query, query_embedding, answer_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (workspace_id, query, json.dumps(q_emb), json.dumps(answer), time.time()),
    )
    c.commit()
    c.close()


def evict_expired() -> int:
    ttl_cutoff = time.time() - settings.cache_ttl_days * 86400
    c = _conn()
    cur = c.execute("DELETE FROM answer_cache WHERE created_at<?", (ttl_cutoff,))
    deleted = cur.rowcount
    c.commit()
    c.close()
    return deleted
