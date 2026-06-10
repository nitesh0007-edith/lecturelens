"""
Rebuild BM25 index from Qdrant Cloud at container startup.
Idempotent: skips if chunk_ids.json already exists for the workspace.

Usage:
    PYTHONPATH=/app/backend python scripts/build_bm25_from_qdrant.py
    PYTHONPATH=/app/backend python scripts/build_bm25_from_qdrant.py --workspace uofg-msds-demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly with PYTHONPATH set
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings
from app.indexing.sparse import BM25Index


def build_for_workspace(workspace_id: str) -> None:
    index_dir = settings.bm25_index_dir
    workspace_dir = index_dir / workspace_id
    ids_path = workspace_dir / "chunk_ids.json"

    if ids_path.exists():
        print(f"[bm25-build] Index already exists at {workspace_dir}, skipping.")
        return

    print(f"[bm25-build] Fetching chunks for workspace '{workspace_id}' from Qdrant...")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    url = settings.qdrant_url
    if url in ("local", ""):
        local_path = str(settings.data_dir / "qdrant_local")
        client = QdrantClient(path=local_path)
    else:
        kwargs: dict = {"url": url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        client = QdrantClient(**kwargs)

    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id))]
            ),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        print(f"[bm25-build]   fetched {len(points)} chunks so far...", end="\r")
        if offset is None:
            break

    print(f"\n[bm25-build] Total: {len(points)} chunks fetched.")

    if not points:
        print(f"[bm25-build] WARNING: no chunks found for workspace '{workspace_id}'. Skipping.")
        return

    from app.ingestion.chunker import Chunk

    chunks = []
    for point in points:
        p = point.payload
        chunk = Chunk(
            chunk_id=p["chunk_id"],
            workspace_id=p.get("workspace_id", workspace_id),
            module=p.get("module", ""),
            module_code=p.get("module_code", ""),
            week=p.get("week"),
            doc_type=p.get("doc_type", ""),
            source_file=p.get("source_file", ""),
            page_or_slide=p.get("page_or_slide") or 0,
            chunk_index=p.get("chunk_index", 0),
            title=p.get("title", ""),
            text=p.get("text", ""),
            contextualised_text=p.get("text", ""),  # contextualised_text not stored in Qdrant
        )
        chunks.append(chunk)

    print(f"[bm25-build] Building BM25 index...")
    bm25_idx = BM25Index(workspace_id, index_dir)
    bm25_idx.build(chunks, use_contextualised=False)
    bm25_idx.save()
    print(f"[bm25-build] Index saved to {workspace_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index from Qdrant")
    parser.add_argument(
        "--workspace",
        default="uofg-msds-demo",
        help="Workspace ID to rebuild (default: uofg-msds-demo)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rebuild BM25 for all workspaces found in Qdrant",
    )
    args = parser.parse_args()

    if args.all:
        # Discover distinct workspace_ids via scroll (no aggregate API in free tier)
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter

        url = settings.qdrant_url
        kwargs: dict = {"url": url}
        if url not in ("local", "") and settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        client = QdrantClient(**kwargs)

        seen: set[str] = set()
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=500,
                offset=offset,
                with_payload=["workspace_id"],
                with_vectors=False,
            )
            for point in batch:
                wid = point.payload.get("workspace_id")
                if wid:
                    seen.add(wid)
            if offset is None:
                break

        print(f"[bm25-build] Found workspaces: {sorted(seen)}")
        for wid in sorted(seen):
            build_for_workspace(wid)
    else:
        build_for_workspace(args.workspace)


if __name__ == "__main__":
    main()
