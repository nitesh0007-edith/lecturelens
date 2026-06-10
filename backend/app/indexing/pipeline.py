"""
Full ingest pipeline for a workspace.

Usage:
    python -m app.indexing.pipeline --workspace uofg-msds-demo --chunks-file data/chunks.jsonl
    python -m app.indexing.pipeline --workspace uofg-msds-demo --ingest-dir /path/to/files
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from app.config import settings
from app.indexing.sparse import BM25Index, invalidate_cache


def load_chunks_from_jsonl(path: Path) -> list:
    """Load Chunk dicts from a .jsonl file."""
    from app.ingestion.chunker import Chunk

    chunks = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            c = Chunk(
                chunk_id=d["chunk_id"],
                workspace_id=d["workspace_id"],
                module=d["module"],
                module_code=d.get("module_code", ""),
                week=d.get("week"),
                doc_type=d["doc_type"],
                source_file=d["source_file"],
                page_or_slide=d["page_or_slide"],
                chunk_index=d["chunk_index"],
                title=d.get("title", ""),
                text=d["text"],
                contextualised_text=d.get("contextualised_text", d["text"]),
                entities=d.get("entities", []),
                keyphrases=d.get("keyphrases", []),
                token_count=d.get("token_count", 0),
            )
            chunks.append(c)
    return chunks


def ingest_files(file_paths: list[Path], workspace_id: str) -> list:
    """Parse and chunk a list of files."""
    from app.ingestion.parsers import parse_file
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.enrich import enrich_chunks

    all_chunks = []
    for fp in file_paths:
        try:
            pages = parse_file(fp)
            chunks = chunk_pages(pages, workspace_id)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [WARN] Failed to parse {fp}: {e}", file=sys.stderr)

    print(f"  enriching {len(all_chunks)} chunks...")
    return enrich_chunks(all_chunks, verbose=True)


def run_pipeline(
    workspace_id: str,
    chunks: list,
    update_status_fn=None,
) -> dict:
    """Build BM25 + Qdrant indexes for a list of Chunk objects."""
    from app.indexing.dense import upsert_chunks

    stats: dict = {"workspace_id": workspace_id, "chunk_count": len(chunks)}

    if update_status_fn:
        update_status_fn("indexing_sparse")

    print(f"  building BM25 index for {len(chunks)} chunks...")
    t0 = time.time()
    bm25_idx = BM25Index(workspace_id, settings.bm25_index_dir)
    bm25_idx.build(chunks)
    bm25_idx.save()
    invalidate_cache(workspace_id)
    stats["bm25_seconds"] = round(time.time() - t0, 2)

    if update_status_fn:
        update_status_fn("indexing_dense")

    print(f"  upserting {len(chunks)} chunks to Qdrant...")
    t0 = time.time()
    upserted = upsert_chunks(chunks)
    stats["qdrant_upserted"] = upserted
    stats["dense_seconds"] = round(time.time() - t0, 2)

    if update_status_fn:
        update_status_fn("done")

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LectureLens ingest pipeline")
    parser.add_argument("--workspace", required=True, help="Workspace ID")
    parser.add_argument("--chunks-file", help="Path to chunks.jsonl")
    parser.add_argument("--ingest-dir", help="Directory of files to parse and ingest")
    parser.add_argument("--glob", default="**/*.pdf,**/*.pptx,**/*.ipynb,**/*.html",
                        help="Glob patterns (comma-separated)")
    args = parser.parse_args()

    if args.chunks_file:
        path = Path(args.chunks_file)
        print(f"Loading chunks from {path}")
        chunks = load_chunks_from_jsonl(path)
    elif args.ingest_dir:
        base = Path(args.ingest_dir)
        patterns = args.glob.split(",")
        file_paths = []
        for pattern in patterns:
            file_paths.extend(base.glob(pattern.strip()))
        print(f"Found {len(file_paths)} files in {base}")
        chunks = ingest_files(file_paths, args.workspace)
    else:
        print("Either --chunks-file or --ingest-dir required", file=sys.stderr)
        sys.exit(1)

    print(f"Running pipeline for workspace '{args.workspace}' with {len(chunks)} chunks")
    stats = run_pipeline(args.workspace, chunks)
    print("Done:", stats)
