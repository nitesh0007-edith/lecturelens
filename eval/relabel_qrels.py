"""
Re-label qrels using Gemini Flash as judge.

For each query, retrieves top-10 chunks via hybrid retrieval, then asks
Gemini to score each (query, chunk) pair 0-2:
  0 = not relevant
  1 = relevant (partially answers the question)
  2 = highly relevant (directly and completely answers)

Writes eval/qrels.txt (overwrites) in TREC format: qid 0 chunk_id rel

Usage:
    PYTHONPATH=backend python eval/relabel_qrels.py
    PYTHONPATH=backend python eval/relabel_qrels.py --dry-run   # first 5 queries only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

WORKSPACE_ID = "uofg-msds-demo"
QUERIES_FILE = Path(__file__).parent / "queries.jsonl"
QRELS_OUT = Path(__file__).parent / "qrels.txt"
CACHE_FILE = Path(__file__).parent / "relabel_cache.json"

JUDGE_PROMPT = """You are an expert relevance assessor for an information retrieval evaluation.

Query: {query}

Below are {n} text chunks from university lecture materials. For each chunk, assign a relevance score:
  0 = Not relevant (does not help answer the query at all)
  1 = Partially relevant (touches on the topic but doesn't directly answer)
  2 = Highly relevant (directly and completely answers the query)

Return ONLY a JSON array of integers, one per chunk, in order.
Example for 3 chunks: [2, 0, 1]

Chunks:
{chunks}"""


def load_queries(path: Path) -> list[dict]:
    queries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def retrieve_candidates(query: str, workspace_id: str, top_k: int = 10) -> list[dict]:
    from app.retrieval.retriever import retrieve
    return retrieve(query, workspace_id, use_rerank=False, rerank_top_k=top_k)


def judge_batch(query: str, chunks: list[dict], gemini_key: str, model: str) -> list[int]:
    from google import genai

    client = genai.Client(api_key=gemini_key)
    judge_model = "gemini-2.5-flash"

    chunk_texts = []
    for i, c in enumerate(chunks, 1):
        text = c.get("text", "")[:400]
        title = c.get("title", "")
        module = c.get("module", "")
        week = c.get("week", "")
        label = f"[{i}] {module}{' W'+str(week) if week else ''} — {title}\n{text}"
        chunk_texts.append(label)

    prompt = JUDGE_PROMPT.format(
        query=query,
        n=len(chunks),
        chunks="\n\n".join(chunk_texts),
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(model=judge_model, contents=prompt)
            text = response.text.strip()
            # Extract JSON array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError(f"No JSON array in response: {text[:200]}")
            scores = json.loads(text[start:end])
            if len(scores) != len(chunks):
                raise ValueError(f"Expected {len(chunks)} scores, got {len(scores)}")
            return [max(0, min(2, int(s))) for s in scores]
        except Exception as e:
            if attempt == 2:
                print(f"    [WARN] Judge failed after 3 attempts: {e}")
                return [1] * len(chunks)  # fallback: mark all as relevant
            time.sleep(2 ** attempt)

    return [1] * len(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only process first 5 queries")
    parser.add_argument("--top-k", type=int, default=10, help="Chunks per query to judge")
    args = parser.parse_args()

    from app.config import settings

    gemini_key = settings.gemini_api_key
    model = settings.gemini_model
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)

    queries = load_queries(QUERIES_FILE)
    if args.dry_run:
        queries = queries[:5]
        print(f"[DRY RUN] Processing {len(queries)} queries")
    else:
        print(f"Processing {len(queries)} queries, {args.top_k} chunks each")

    # Load cache (so we can resume if interrupted)
    cache: dict[str, dict[str, int]] = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        print(f"Loaded {len(cache)} cached queries")

    qrels: dict[str, dict[str, int]] = dict(cache)

    for i, q in enumerate(queries):
        qid = q["qid"]
        query_text = q["query"]

        if qid in qrels:
            print(f"[{i+1}/{len(queries)}] {qid} — cached, skipping")
            continue

        print(f"[{i+1}/{len(queries)}] {qid}: {query_text[:60]}…")

        try:
            chunks = retrieve_candidates(query_text, WORKSPACE_ID, top_k=args.top_k)
        except Exception as e:
            print(f"  [WARN] Retrieval failed: {e}")
            continue

        if not chunks:
            print(f"  [WARN] No chunks retrieved")
            continue

        scores = judge_batch(query_text, chunks, gemini_key, model)

        qrels[qid] = {}
        for chunk, score in zip(chunks, scores):
            cid = chunk.get("chunk_id", "")
            if cid:
                qrels[qid][cid] = score

        # Always include the gold source_chunk_id with score 2 if present
        gold = q.get("source_chunk_id")
        if gold and gold not in qrels[qid]:
            qrels[qid][gold] = 2

        print(f"  → {len(qrels[qid])} judgments, {sum(1 for s in qrels[qid].values() if s > 0)} relevant")

        # Save cache after each query
        CACHE_FILE.write_text(json.dumps(qrels, indent=2))

        # Rate limit: ~1 req/s to stay within free tier
        time.sleep(1.2)

    # Write TREC-format qrels
    lines = []
    for qid in sorted(qrels.keys()):
        for chunk_id, rel in sorted(qrels[qid].items()):
            lines.append(f"{qid} 0 {chunk_id} {rel}")

    QRELS_OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {len(lines)} judgments to {QRELS_OUT}")

    # Summary stats
    total_rel = sum(1 for q in qrels.values() for s in q.values() if s > 0)
    total_highly = sum(1 for q in qrels.values() for s in q.values() if s == 2)
    print(f"Relevant (≥1): {total_rel}, Highly relevant (2): {total_highly}")


if __name__ == "__main__":
    main()
