"""
Index all chunks from data/chunks.jsonl to Qdrant Cloud.
Idempotent — safe to re-run; existing points are overwritten.
Usage: PYTHONPATH=backend python scripts/index_dense.py [--start N] [--end N]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.indexing.pipeline import load_chunks_from_jsonl
from app.indexing.dense import ensure_collection, upsert_chunks, _get_qdrant, COLLECTION_NAME

CHUNKS_FILE = Path("data/chunks.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2000)
    args = parser.parse_args()

    print(f"Loading {CHUNKS_FILE} ...", flush=True)
    chunks = load_chunks_from_jsonl(CHUNKS_FILE)
    total = len(chunks)
    print(f"Total chunks: {total}", flush=True)

    ensure_collection()
    before = _get_qdrant().get_collection(COLLECTION_NAME).points_count
    print(f"Cloud points before: {before}", flush=True)

    subset = chunks[args.start : args.end]
    print(f"Upserting slice [{args.start}:{args.end}] = {len(subset)} chunks ...", flush=True)

    t0 = time.time()
    for i in range(0, len(subset), args.batch):
        batch = subset[i : i + args.batch]
        upsert_chunks(batch)
        cloud_n = _get_qdrant().get_collection(COLLECTION_NAME).points_count
        elapsed = round(time.time() - t0, 1)
        print(f"  [{args.start + i}:{args.start + i + len(batch)}] done | cloud total={cloud_n} | {elapsed}s", flush=True)

    after = _get_qdrant().get_collection(COLLECTION_NAME).points_count
    print(f"Done. Cloud points after: {after} (added {after - before})", flush=True)


if __name__ == "__main__":
    main()
