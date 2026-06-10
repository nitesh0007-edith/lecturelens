# LectureLens IR Evaluation Results

| Config | R@30 | RR | nDCG@10 | AP | Time(s) |
|---|---|---|---|---|---|
| BM25 only | 0.8249 | 0.8981 | 0.6777 | 0.6198 | 3.8 |
| Dense only | 0.8474 | 0.9307 | 0.7026 | 0.6371 | 6.5 |
| Hybrid RRF | 0.9738 | 0.9173 | 0.8854 | 0.837 | 6.2 |
| Hybrid + rerank | 0.9738 | 0.9183 | 0.7077 | 0.6777 | 169.8 |
| Hybrid + rerank + LLM rewrite | 0.7917 | 0.8454 | 0.6212 | 0.5521 | 246.8 |

*Metrics: MAP (AP), NDCG@10, MRR (RR), Recall@30. Higher is better.*