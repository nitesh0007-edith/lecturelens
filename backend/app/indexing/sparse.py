"""
BM25 sparse index using bm25s (pure Python, Lucene-comparable scores).
Serialised per workspace to disk; loaded in-process for retrieval.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk


_LOADED_INDEXES: dict[str, "BM25Index"] = {}


class BM25Index:
    def __init__(self, workspace_id: str, index_dir: Path):
        self.workspace_id = workspace_id
        self.index_dir = index_dir
        self._index = None
        self._chunk_ids: list[str] = []

    # ------------------------------------------------------------------
    # Tokenisation (must match what was used to build the index)
    # ------------------------------------------------------------------

    @staticmethod
    def tokenise(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list["Chunk"], use_contextualised: bool = True) -> None:
        try:
            import bm25s
        except ImportError as e:
            raise ImportError("bm25s required: pip install bm25s") from e

        corpus_text = [
            c.contextualised_text if use_contextualised else c.text for c in chunks
        ]
        self._chunk_ids = [c.chunk_id for c in chunks]

        tokenised = bm25s.tokenize(corpus_text, stopwords="en")
        self._index = bm25s.BM25()
        self._index.index(tokenised)

    # ------------------------------------------------------------------
    # Persist / load
    # ------------------------------------------------------------------

    def _workspace_dir(self) -> Path:
        d = self.index_dir / self.workspace_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self) -> None:
        d = self._workspace_dir()
        self._index.save(str(d / "bm25_index"), corpus=None)
        (d / "chunk_ids.json").write_text(json.dumps(self._chunk_ids))

    def load(self) -> bool:
        """Return True if loaded successfully, False if not found."""
        try:
            import bm25s
        except ImportError:
            return False

        d = self._workspace_dir()
        index_path = d / "bm25_index"
        ids_path = d / "chunk_ids.json"
        if not ids_path.exists():
            return False

        self._index = bm25s.BM25.load(str(index_path), load_corpus=False)
        self._chunk_ids = json.loads(ids_path.read_text())
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        """Return [(chunk_id, score), …] sorted descending."""
        if self._index is None:
            return []
        try:
            import bm25s

            tokenised_query = bm25s.tokenize([query], stopwords="en")
            results, scores = self._index.retrieve(tokenised_query, k=min(top_k, len(self._chunk_ids)))
            out: list[tuple[str, float]] = []
            for doc_idx, score in zip(results[0], scores[0]):
                if int(doc_idx) < len(self._chunk_ids):
                    out.append((self._chunk_ids[int(doc_idx)], float(score)))
            return out
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def get_bm25_index(workspace_id: str, index_dir: Path) -> BM25Index:
    if workspace_id not in _LOADED_INDEXES:
        idx = BM25Index(workspace_id, index_dir)
        if not idx.load():
            raise FileNotFoundError(f"BM25 index not found for workspace {workspace_id}")
        _LOADED_INDEXES[workspace_id] = idx
    return _LOADED_INDEXES[workspace_id]


def invalidate_cache(workspace_id: str) -> None:
    _LOADED_INDEXES.pop(workspace_id, None)
