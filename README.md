---
title: LectureLens
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# LectureLens

**Hybrid RAG learning copilot for university course materials.**

A live web app where you ask *"Explain BM25 and how it improves on TF-IDF"* and get a cited answer grounded in your lecture notes — plus recommended lectures to revisit, an auto-generated quiz, and a concept graph.

Ships with the full UofG MSc Data Science corpus as the public demo workspace. Any student can create their own workspace by uploading PDFs/slides.

---

## Architecture

```
PDFs / PPTX / ipynb / HTML
        │
   INGESTION (PyMuPDF · python-pptx · nbformat · BS4)
   structure-aware chunks + metadata {module, week, source, page}
        │
   DUAL INDEX
   ├── bm25s  (sparse, in-process)
   └── BGE-small-en-v1.5 → Qdrant  (dense, FastEmbed/ONNX)
        │
   HYBRID RETRIEVAL
   BM25 top-30 ∪ dense top-30 → Reciprocal Rank Fusion
        │
   RE-RANKING  (bge-reranker-base ONNX CPU → top-6)
        │
   GENERATION  (Groq Llama 3.3 70B, cite-or-abstain prompt)
        │
   Cited answer · lecture recs · quiz · exam mode · concept graph
```

**Stack:** FastAPI · Qdrant Cloud · bm25s · FastEmbed · Groq · Next.js · SQLite · Docker → HF Spaces

---

## Quickstart

```bash
cp .env.example .env   # fill in GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
make dev               # starts api + qdrant via docker-compose
make ingest-demo       # clones UofGMScDS repo, builds demo workspace
```

Then visit `http://localhost:8000/docs` for the API.

## Commands

| Command | Description |
|---------|-------------|
| `make dev` | uvicorn + local Qdrant via docker-compose |
| `make test` | pytest -q (43 tests) |
| `make lint` | ruff check |
| `make ingest-demo` | Clone UofGMScDS → build demo workspace |
| `make eval` | Print MAP/NDCG ablation table |

## IR Evaluation

> Retrieval ablation results will appear here after `make eval`.
> See [eval/results.md](eval/results.md) once populated.

| Config | MAP | NDCG@10 | MRR | Recall@30 |
|--------|-----|---------|-----|-----------|
| BM25 only | — | — | — | — |
| Dense only | — | — | — | — |
| Hybrid RRF | — | — | — | — |
| Hybrid + rerank | — | — | — | — |
| Hybrid + LLM rewrite | — | — | — | — |

> Generation quality results: [eval/generation_results.md](eval/generation_results.md)

## Key design decisions

- **RRF over weighted score fusion** — BM25 and cosine scores live in incompatible distributions; RRF is rank-based and parameter-light.
- **bm25s not ElasticSearch** — real BM25 (Lucene-comparable) in pure Python, serialisable, CPU-only, free.
- **Single Qdrant collection, multi-tenancy via payload filter** — Qdrant's documented pattern; avoids blowing the 1 GB free tier.
- **CPU-only ONNX inference** — no GPU needed; cross-encoder re-ranks 30 candidates in ~1-2 s.
- **Contextual chunk enrichment** — LLM-generated one-line context prepended before embedding (Anthropic's contextual retrieval technique), especially valuable for slide decks where individual slides lack standalone context.
- **Indirect prompt injection defence** — retrieved chunks wrapped in `<retrieved_data>` delimiters; role-marker strings sanitised at ingest.

## Evaluation harness

The eval harness in `eval/` runs 5 retrieval configurations headlessly and produces an ablation table — this is the CV differentiator. See `eval/run_eval.py`.

## Cost: £0

| Component | Service | Free tier |
|-----------|---------|-----------|
| Vectors | Qdrant Cloud | 1 GB cluster |
| LLM | Groq Llama 3.3 70B | free tier |
| Fallback | Gemini 2.0 Flash | free tier |
| Embeddings + reranker | FastEmbed ONNX | CPU only |
| Hosting | HF Spaces Docker | free CPU Space |
