# LectureLens

Hybrid RAG learning copilot for university course materials.
Stack: FastAPI, Qdrant Cloud, bm25s, FastEmbed (BGE-small + bge-reranker, ONNX CPU),
Groq Llama 3.3, Next.js, SQLite (SQLModel), Docker → Hugging Face Spaces.

## Hard constraints
- CPU-only everywhere. No GPU dependencies. No LangChain/LlamaIndex — build retrieval by hand.
- Free tiers only: Qdrant Cloud 1GB, Groq free, HF Spaces. Single Qdrant collection,
  multi-tenancy via workspace_id payload filter.
- Every answer must carry citations resolvable to {module, week, source_file, page}.
- All config via env vars (pydantic-settings). Never hardcode keys.
- Tests required for: parsers, chunker, RRF fusion, citation resolution. pytest, run before commit.
- Type hints everywhere; ruff for lint/format.

## Commands
- `make dev` → uvicorn + local qdrant via docker-compose
- `make test` → pytest -q
- `make ingest-demo` → scripts/ingest_uofg.py (clones UofGMScDS repo)
- `make eval` → eval/run_eval.py (prints MAP/NDCG ablation table)

## Style
- Small modules, pure functions for ranking logic (easy to unit test).
- Retrieval pipeline must be runnable headless (no API) for the eval harness.

## Source corpus
https://github.com/nitesh0007-edith/UofGMScDS
Modules: BigData, CyberSec, IDSS, DeepLearning, IR, IV, MLAI, ProgSD, RPS, TextasData
