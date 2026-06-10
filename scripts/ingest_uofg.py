"""
Clone the UofGMScDS repo and build the demo workspace.
Emits data/chunks.jsonl with full metadata.
Then indexes into BM25 + Qdrant.

Usage:
    PYTHONPATH=backend python scripts/ingest_uofg.py [--skip-index] [--repo-dir /path]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/nitesh0007-edith/UofGMScDS"
DEFAULT_REPO_DIR = Path("data/UofGMScDS")
CHUNKS_FILE = Path("data/chunks.jsonl")
WORKSPACE_ID = "uofg-msds-demo"

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".ipynb", ".html", ".htm"}


def clone_or_pull(repo_dir: Path):
    if repo_dir.exists():
        print(f"Repo already exists at {repo_dir}, pulling latest...")
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=False)
    else:
        print(f"Cloning {REPO_URL} → {repo_dir}")
        subprocess.run(["git", "clone", "--depth=1", REPO_URL, str(repo_dir)], check=True)


def collect_files(repo_dir: Path) -> list[Path]:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(repo_dir.rglob(f"*{ext}"))
    # Exclude .git directory
    files = [f for f in files if ".git" not in f.parts]
    return sorted(files)


def emit_chunks_jsonl(chunks: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict()) + "\n")


def print_stats(chunks: list):
    from collections import Counter

    module_counts: Counter = Counter()
    type_counts: Counter = Counter()
    total_tokens = 0

    for c in chunks:
        module_counts[c.module] += 1
        type_counts[c.doc_type] += 1
        total_tokens += c.token_count

    print(f"\n{'='*50}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total tokens (approx): {total_tokens:,}")
    print(f"\nChunks per module:")
    for module, count in sorted(module_counts.items()):
        print(f"  {module:20s}: {count:5d}")
    print(f"\nChunks per doc type:")
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype:20s}: {count:5d}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Ingest UofGMScDS corpus")
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-index", action="store_true",
                        help="Only produce chunks.jsonl, skip BM25+Qdrant indexing")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of files (for testing)")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)

    if not args.skip_clone:
        clone_or_pull(repo_dir)

    files = collect_files(repo_dir)
    if args.max_files:
        files = files[:args.max_files]

    print(f"\nFound {len(files)} files across all modules")

    # Add backend to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

    from app.ingestion.parsers import parse_file
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.enrich import enrich_chunks

    all_chunks = []
    failed = 0

    for i, fp in enumerate(files):
        try:
            pages = parse_file(fp)
            chunks = chunk_pages(pages, WORKSPACE_ID)
            all_chunks.extend(chunks)
            if (i + 1) % 20 == 0:
                print(f"  Parsed {i+1}/{len(files)} files, {len(all_chunks)} chunks so far")
        except Exception as e:
            print(f"  [WARN] {fp.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"\nParsed {len(files) - failed} files successfully ({failed} failed)")
    print(f"Enriching {len(all_chunks)} chunks (NER + keyphrases)...")
    all_chunks = enrich_chunks(all_chunks, verbose=True)

    emit_chunks_jsonl(all_chunks, CHUNKS_FILE)
    print(f"\nWrote {len(all_chunks)} chunks to {CHUNKS_FILE}")
    print_stats(all_chunks)

    if not args.skip_index:
        print("\nBuilding BM25 + Qdrant indexes...")
        from app.indexing.pipeline import run_pipeline

        stats = run_pipeline(WORKSPACE_ID, all_chunks)
        print("Indexing complete:", stats)

    print("\nDemo workspace ready. Try:")
    print(f"  PYTHONPATH=backend python -c \"from app.retrieval.retriever import retrieve; "
          f"print(retrieve('explain BM25', '{WORKSPACE_ID}'))\"")


if __name__ == "__main__":
    main()
