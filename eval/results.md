# LectureLens IR Evaluation Results

| Config | RR (MRR) | R@30 | nDCG@10 | MAP (AP) |
|--------|----------|------|---------|---------|
| BM25 only | 1.0 | 0.9904 | 0.9965 | 0.9895 |
| Dense only | — | — | — | — |
| Hybrid RRF | 1.0 | 0.9904 | 0.9965 | 0.9895 |
| Hybrid+rerank | 1.0 | 0.9904 | 0.9965 | 0.9895 |
| Hybrid+rerank+LLM rewrite | — | — | — | — |

*85 answerable queries across 10 MSc DS modules. Higher is better.*

**Notes:**
- Qrels are BM25-bootstrapped (Groq-generated questions + BM25 relevance labels). BM25 scores near-perfectly on its own labels — expected. For interview-ready numbers, manually review 30–40 IR-module qrels and re-run.
- Dense-only and hybrid rows will populate once the local Qdrant index finishes building (`data/qdrant_local/`). Re-run `make eval` after indexing completes.
- The BM25 → Hybrid+rerank delta will appear once dense is available; expect NDCG@10 lift of ~5–15% from re-ranking based on the literature.
- LLM-rewrite row: run `PYTHONPATH=backend python eval/run_eval.py` with Groq key set.

**How to improve qrel quality (highest ROI):**
1. Pick 30 IR-module queries from `eval/queries.jsonl`
2. Search `data/chunks.jsonl` for their chunk_ids
3. Correct relevance scores in `eval/qrels.txt` (0=not relevant, 1=relevant, 2=highly relevant)
4. Re-run `make eval` — the BM25/hybrid delta will be honest and CV-ready
