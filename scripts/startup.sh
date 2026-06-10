#!/bin/bash
set -e

echo "[startup] Building BM25 index from Qdrant (skipped if already built)..."
python /app/scripts/build_bm25_from_qdrant.py --workspace uofg-msds-demo

echo "[startup] Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
