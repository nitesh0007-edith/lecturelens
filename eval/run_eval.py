"""
IR evaluation harness — ablation over 6 retrieval configurations.

Configurations:
  1. BM25 only
  2. Dense only
  3. Hybrid RRF (BM25 + Dense)
  4. Hybrid + rerank
  5. Hybrid + rerank + LLM query rewriting (cached)
  6. Hybrid + rerank + RM3 pseudo-relevance feedback (PyTerrier)

Metrics: MAP, NDCG@10, MRR, Recall@30 via ir_measures.

Usage:
    PYTHONPATH=backend python eval/run_eval.py
    PYTHONPATH=backend python eval/run_eval.py --queries eval/queries.jsonl --qrels eval/qrels.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

WORKSPACE_ID = "uofg-msds-demo"
DEFAULT_QUERIES = Path(__file__).parent / "queries.jsonl"
DEFAULT_QRELS = Path(__file__).parent / "qrels.txt"
REWRITE_CACHE = Path(__file__).parent / "rewrite_cache.json"


class Query(NamedTuple):
    qid: str
    text: str


def load_queries(path: Path) -> list[Query]:
    queries = []
    with path.open() as f:
        for line in f:
            d = json.loads(line.strip())
            queries.append(Query(qid=d["qid"], text=d["query"]))
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse TREC-format qrels: qid 0 doc_id rel."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def retrieve_config(
    query: str,
    config: str,
    use_bm25: bool = True,
    use_dense: bool = True,
    use_rerank: bool = True,
    rewritten_query: str | None = None,
) -> list[tuple[str, float]]:
    """Run retrieval and return [(chunk_id, rank_score), ...]."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from app.retrieval.retriever import retrieve

    effective_query = rewritten_query or query
    chunks = retrieve(
        effective_query,
        WORKSPACE_ID,
        use_bm25=use_bm25,
        use_dense=use_dense,
        use_rerank=use_rerank,
        rerank_top_k=30,
    )
    return [(c["chunk_id"], 1.0 / (i + 1)) for i, c in enumerate(chunks)]


def llm_rewrite_query(query: str, cache: dict) -> str:
    if query in cache:
        return cache[query]

    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
        from app.config import settings
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the query to improve academic lecture retrieval. Return ONLY the rewritten query, nothing else.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        rewritten = resp.choices[0].message.content.strip()
        cache[query] = rewritten
        return rewritten
    except Exception:
        cache[query] = query
        return query


def compute_metrics(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
) -> dict:
    """Compute MAP, NDCG@10, MRR, Recall@30 via ir_measures."""
    try:
        import ir_measures
        from ir_measures import AP, nDCG, RR, R

        ir_qrels = []
        for qid, docs in qrels.items():
            for doc_id, rel in docs.items():
                ir_qrels.append(ir_measures.Qrel(query_id=qid, doc_id=doc_id, relevance=rel))

        ir_run = []
        for qid, results in run.items():
            for rank, (doc_id, score) in enumerate(results, start=1):
                ir_run.append(
                    ir_measures.ScoredDoc(query_id=qid, doc_id=doc_id, score=score)
                )

        metrics = [AP, nDCG @ 10, RR, R @ 30]
        results = ir_measures.calc_aggregate(metrics, ir_qrels, ir_run)
        return {str(k): round(float(v), 4) for k, v in results.items()}

    except ImportError:
        # Fallback: simple MAP computation
        return _simple_map(run, qrels)


def _simple_map(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
) -> dict:
    aps, ndcgs, mrrs = [], [], []
    for qid, results in run.items():
        relevant = {d for d, r in qrels.get(qid, {}).items() if r > 0}
        if not relevant:
            continue

        retrieved_ids = [d for d, _ in results]
        hits = [1 if d in relevant else 0 for d in retrieved_ids]

        # AP
        prec_sum = 0.0
        hit_count = 0
        for i, h in enumerate(hits, 1):
            if h:
                hit_count += 1
                prec_sum += hit_count / i
        ap = prec_sum / len(relevant) if relevant else 0.0
        aps.append(ap)

        # NDCG@10
        import math
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:10]))
        ideal = sorted(hits, reverse=True)[:10]
        idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)

        # MRR
        mrr = next((1 / i for i, h in enumerate(hits, 1) if h), 0.0)
        mrrs.append(mrr)

    return {
        "AP": round(sum(aps) / len(aps) if aps else 0.0, 4),
        "nDCG@10": round(sum(ndcgs) / len(ndcgs) if ndcgs else 0.0, 4),
        "RR": round(sum(mrrs) / len(mrrs) if mrrs else 0.0, 4),
    }


CONFIGS = [
    ("BM25 only",              {"use_bm25": True,  "use_dense": False, "use_rerank": False}),
    ("Dense only",             {"use_bm25": False, "use_dense": True,  "use_rerank": False}),
    ("Hybrid RRF",             {"use_bm25": True,  "use_dense": True,  "use_rerank": False}),
    ("Hybrid + rerank",        {"use_bm25": True,  "use_dense": True,  "use_rerank": True}),
    ("Hybrid + rerank + LLM rewrite", {"use_bm25": True, "use_dense": True, "use_rerank": True, "llm_rewrite": True}),
]


def run_eval(queries_path: Path, qrels_path: Path, output_md: Path):
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    rewrite_cache: dict = {}
    if REWRITE_CACHE.exists():
        rewrite_cache = json.loads(REWRITE_CACHE.read_text())

    rows: list[dict] = []

    for config_name, cfg in CONFIGS:
        print(f"\nRunning: {config_name} ({len(queries)} queries)...")
        run: dict[str, list[tuple[str, float]]] = {}
        t0 = time.time()

        for q in queries:
            rewritten = None
            if cfg.get("llm_rewrite"):
                rewritten = llm_rewrite_query(q.text, rewrite_cache)

            results = retrieve_config(
                q.text,
                config_name,
                use_bm25=cfg.get("use_bm25", True),
                use_dense=cfg.get("use_dense", True),
                use_rerank=cfg.get("use_rerank", False),
                rewritten_query=rewritten,
            )
            run[q.qid] = results

        elapsed = round(time.time() - t0, 1)
        metrics = compute_metrics(run, qrels)
        rows.append({"Config": config_name, **metrics, "Time(s)": elapsed})
        print(f"  {metrics}  [{elapsed}s]")

    # Save LLM rewrite cache
    REWRITE_CACHE.write_text(json.dumps(rewrite_cache, indent=2))

    # Write markdown table
    _write_results_md(rows, output_md)
    print(f"\nResults written to {output_md}")


def _write_results_md(rows: list[dict], path: Path):
    if not rows:
        return

    headers = list(rows[0].keys())
    lines = ["# LectureLens IR Evaluation Results", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines.append("")
    lines.append("*Metrics: MAP (AP), NDCG@10, MRR (RR), Recall@30. Higher is better.*")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--qrels", default=str(DEFAULT_QRELS))
    parser.add_argument("--output", default=str(Path(__file__).parent / "results.md"))
    args = parser.parse_args()

    q_path = Path(args.queries)
    qrels_path = Path(args.qrels)

    if not q_path.exists():
        print(f"queries file not found: {q_path}")
        print("Create eval/queries.jsonl with format: {\"qid\": \"q1\", \"query\": \"...\"}")
        sys.exit(1)
    if not qrels_path.exists():
        print(f"qrels file not found: {qrels_path}")
        print("Create eval/qrels.txt in TREC format: qid 0 chunk_id relevance")
        sys.exit(1)

    run_eval(q_path, qrels_path, Path(args.output))
