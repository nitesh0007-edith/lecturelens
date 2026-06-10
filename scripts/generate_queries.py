"""
Generate eval/queries.jsonl + eval/qrels.txt from the actual corpus.

For each module, samples lecture chunks, uses Groq to generate realistic
student questions, then uses BM25 search to find relevant chunk_ids (bootstrap
relevance — you should review and correct qrels manually for best eval quality).

Usage:
    PYTHONPATH=backend python scripts/generate_queries.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.config import settings
from app.indexing.sparse import BM25Index

QUERIES_OUT = Path("eval/queries.jsonl")
QRELS_OUT = Path("eval/qrels.txt")
CHUNKS_FILE = Path("data/chunks.jsonl")
WORKSPACE_ID = "uofg-msds-demo"
QUESTIONS_PER_MODULE = 8
UNANSWERABLE_QUESTIONS = 10

UNANSWERABLE = [
    {"qid": "u01", "query": "What was Napoleon's favourite theorem?", "tag": "unanswerable"},
    {"qid": "u02", "query": "How many FIFA World Cups did Scotland win?", "tag": "unanswerable"},
    {"qid": "u03", "query": "What is the boiling point of dark matter?", "tag": "unanswerable"},
    {"qid": "u04", "query": "Explain how to bake a soufflé using gradient descent", "tag": "unanswerable"},
    {"qid": "u05", "query": "What did Plato say about convolutional neural networks?", "tag": "unanswerable"},
    {"qid": "u06", "query": "Describe the Hadoop setup used at NASA in 1969", "tag": "unanswerable"},
    {"qid": "u07", "query": "What is the stock price of BM25 Corporation?", "tag": "unanswerable"},
    {"qid": "u08", "query": "How many calories are in a support vector machine?", "tag": "unanswerable"},
    {"qid": "u09", "query": "Explain why Python was invented in the 1800s", "tag": "unanswerable"},
    {"qid": "u10", "query": "What are the IR module's views on cryptocurrency trading?", "tag": "unanswerable"},
]

QUESTION_PROMPT = """You are a university student studying MSc Data Science.
Given the lecture excerpt below, write {n} realistic exam-style questions a student would ask.

Rules:
- Questions must be answerable from the excerpt
- Vary difficulty: 2 factual, 2 conceptual, 2 application, 2 comparison
- Each question on its own line, no numbering

Lecture excerpt ({module}, Week {week}):
{text}

Questions:"""


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_FILE.open() as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def sample_chunks_per_module(chunks: list[dict]) -> dict[str, list[dict]]:
    """Sample representative lecture chunks per module."""
    from collections import defaultdict
    by_module: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c.get("doc_type") == "lecture" and c.get("token_count", 0) > 100:
            by_module[c["module"]].append(c)

    sampled = {}
    for module, mod_chunks in by_module.items():
        # Pick diverse weeks
        by_week: dict = defaultdict(list)
        for c in mod_chunks:
            by_week[c.get("week", 0)].append(c)
        selected = []
        for week_chunks in sorted(by_week.values(), key=len, reverse=True):
            if week_chunks:
                selected.append(random.choice(week_chunks))
            if len(selected) >= 4:
                break
        sampled[module] = selected
    return sampled


def generate_questions_for_chunk(chunk: dict, n: int = 4) -> list[str]:
    """Use Groq to generate n questions from a chunk."""
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    prompt = QUESTION_PROMPT.format(
        n=n,
        module=chunk.get("module", ""),
        week=chunk.get("week", "?"),
        text=chunk.get("text", "")[:1500],
    )
    try:
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        lines = [l.strip() for l in raw.splitlines() if l.strip() and len(l.strip()) > 15]
        return lines[:n]
    except Exception as e:
        print(f"    [WARN] Groq error: {e}")
        return []


def bm25_find_relevant(query: str, bm25: BM25Index, top_k: int = 10) -> list[tuple[str, int]]:
    """Return [(chunk_id, relevance), ...] using BM25 scores for bootstrap qrels."""
    results = bm25.search(query, top_k=top_k)
    if not results:
        return []
    max_score = results[0][1] if results else 1.0
    qrels = []
    for chunk_id, score in results:
        if score <= 0:
            continue
        # Map to 0-2 relevance scale
        rel = 2 if score >= max_score * 0.7 else 1
        qrels.append((chunk_id, rel))
    return qrels


def main():
    random.seed(42)
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    bm25 = BM25Index(WORKSPACE_ID, settings.bm25_index_dir)
    if not bm25.load():
        print("ERROR: BM25 index not found. Run `make ingest-demo` first.")
        sys.exit(1)

    sampled = sample_chunks_per_module(chunks)
    print(f"Sampled chunks from {len(sampled)} modules")

    all_queries = []
    all_qrels: list[tuple[str, str, int]] = []  # (qid, chunk_id, rel)
    qid_counter = 1

    for module, mod_chunks in sorted(sampled.items()):
        print(f"\nGenerating questions for {module} ({len(mod_chunks)} source chunks)...")
        for source_chunk in mod_chunks:
            questions = generate_questions_for_chunk(source_chunk, n=4)
            for q_text in questions:
                qid = f"q{qid_counter:03d}"
                qid_counter += 1
                entry = {
                    "qid": qid,
                    "query": q_text,
                    "module": module,
                    "source_chunk_id": source_chunk.get("chunk_id", ""),
                }
                all_queries.append(entry)

                # Bootstrap qrels via BM25
                relevant = bm25_find_relevant(q_text, bm25, top_k=10)
                # Always mark the source chunk as highly relevant
                source_id = source_chunk.get("chunk_id", "")
                source_in_results = any(cid == source_id for cid, _ in relevant)
                if not source_in_results and source_id:
                    all_qrels.append((qid, source_id, 2))
                for chunk_id, rel in relevant:
                    all_qrels.append((qid, chunk_id, rel))

            time.sleep(0.3)  # Rate limit

    # Add unanswerable questions
    for u in UNANSWERABLE:
        all_queries.append(u)

    # Write queries.jsonl
    QUERIES_OUT.parent.mkdir(exist_ok=True)
    with QUERIES_OUT.open("w") as f:
        for q in all_queries:
            f.write(json.dumps(q) + "\n")
    print(f"\nWrote {len(all_queries)} queries to {QUERIES_OUT}")

    # Write qrels.txt (TREC format)
    with QRELS_OUT.open("w") as f:
        for qid, chunk_id, rel in all_qrels:
            f.write(f"{qid} 0 {chunk_id} {rel}\n")
    print(f"Wrote {len(all_qrels)} qrel entries to {QRELS_OUT}")
    print("\nNOTE: These are BM25-bootstrapped qrels. Review and correct manually")
    print("      for highest eval quality (focus on the IR module questions).")


if __name__ == "__main__":
    main()
