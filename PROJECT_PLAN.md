# LectureLens — Course Knowledge Copilot

**A hybrid-retrieval RAG learning assistant, pre-loaded with the UofG MSc Data Science curriculum, that any student can use with their own course materials.**

Source corpus: https://github.com/nitesh0007-edith/UofGMScDS (10 modules: BigData, CyberSec, IDSS, DeepLearning, IR, IV, MLAI, ProgSD, RPS, TextasData — PDFs, PPTX slides, Jupyter notebooks, HTML notes, past papers 2015–2025)

---

## 1. What you are building (one paragraph)

A live web app where a student asks "Explain BM25 and how it improves on TF-IDF" and gets a cited answer ("Per IR Week 5, slide 12…") grounded only in their lecture notes, plus recommended lectures to revisit, an auto-generated quiz on the topic, and a concept graph showing how BM25 connects to inverted indexing, query expansion, and dense retrieval. It ships with your full UofG corpus as the public demo workspace, and any student can create their own workspace by uploading their own PDFs/slides. Retrieval is hybrid (BM25 + dense embeddings) with cross-encoder re-ranking, and — the differentiator — you evaluate it like an IR researcher: a labelled test set, MAP/NDCG@10 via PyTerrier/ir_measures, ablations showing exactly what re-ranking buys you.

## 2. Why this beats the generic "PDF chatbot"

| Generic RAG project | LectureLens |
|---|---|
| LangChain wrapper, one vector DB | Hand-built hybrid retrieval: BM25 + dense + RRF fusion + cross-encoder re-rank |
| "It works" (no numbers) | Offline eval harness: MAP, NDCG@10, MRR with ablation table (BM25-only vs dense-only vs hybrid vs hybrid+rerank) |
| Single hardcoded corpus | Multi-tenant workspaces — any student uploads their own notes |
| Localhost demo | Live, free-tier deployment with CI/CD |
| No domain grounding | Directly applies your Sem-2 modules: IR (ranking, evaluation), Text as Data (preprocessing, NER, topic metadata), Deep Learning (embeddings, cross-encoders, LLMs) |

## 3. Architecture

```
                        ┌─────────────────────────────┐
  PDFs / PPTX / ipynb   │        INGESTION            │
  / HTML  ─────────────▶│ PyMuPDF · python-pptx ·     │
  (upload or GitHub)    │ nbformat · BeautifulSoup    │
                        └──────────┬──────────────────┘
                                   │ structure-aware chunks
                                   │ + metadata {workspace, module,
                                   │   week, source, page, type}
                        ┌──────────▼──────────────────┐
                        │      INDEXING (dual)        │
                        │  bm25s (sparse, in-process) │
                        │  BGE-small-en-v1.5 → Qdrant │
                        │  (dense, FastEmbed/ONNX)    │
                        └──────────┬──────────────────┘
                                   │
              query ──────▶ HYBRID RETRIEVAL
                            BM25 top-30 ∪ dense top-30
                            → Reciprocal Rank Fusion
                                   │
                        ┌──────────▼──────────────────┐
                        │  RE-RANKING                 │
                        │  bge-reranker-base (ONNX,   │
                        │  CPU) → top 6 chunks        │
                        └──────────┬──────────────────┘
                                   │
                        ┌──────────▼──────────────────┐
                        │  GENERATION                 │
                        │  Llama 3.3 70B via Groq     │
                        │  (free tier) — strict       │
                        │  cite-or-abstain prompt     │
                        └──────────┬──────────────────┘
                                   │
                 Cited answer · lecture recs · quiz ·
                 exam mode · concept graph (frontend)
```

**Backend:** FastAPI (Python 3.11) · **Frontend:** Next.js (or a React SPA served by FastAPI) · **Vector DB:** Qdrant Cloud free tier (1 GB) · **Sparse index:** `bm25s` (serialised to disk, loaded in-process) · **Embeddings/reranker:** FastEmbed (ONNX, CPU-only — no GPU needed) · **LLM:** Groq free tier (Llama 3.3 70B), Gemini 2.0 Flash free tier as fallback · **Graph:** NetworkX + SQLite, rendered with Cytoscape.js (Neo4j Aura free tier optional later) · **Eval:** PyTerrier + ir_measures · **Hosting:** Hugging Face Spaces (Docker, free) or Render free tier; frontend on Vercel if split.

**Why these choices (be ready to defend in interviews):**
- ElasticSearch/OpenSearch need a paid managed cluster or a beefy VM. `bm25s` gives you real BM25 (Lucene-comparable scores) in pure Python, serialisable, fast on 10k–50k chunks. You still discuss inverted indexes and term weighting — you implemented the IR concepts, not just configured a service.
- Qdrant Cloud free tier = 1 GB ≈ plenty for ~50k chunks of 384-dim BGE-small vectors. One collection, multi-tenancy via a `workspace_id` payload filter (Qdrant's documented multi-tenant pattern) — not one collection per user, which would blow the free tier.
- Cross-encoder on CPU is fine because you only re-rank 30 candidates per query (~1–2 s with ONNX).
- Groq is free and fast; the LLM is deliberately the *least* interesting part of your system. Your value-add is retrieval quality, and you can prove it with numbers.

## 4. Repo layout (new repo, e.g. `lecturelens`)

```
lecturelens/
├── CLAUDE.md                  # Claude Code project memory (see §7)
├── README.md                  # Architecture diagram, demo GIF, eval table
├── docker-compose.yml         # local dev: api + qdrant
├── Dockerfile                 # HF Spaces deployment
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, routers
│   │   ├── config.py          # pydantic-settings, all env vars
│   │   ├── ingestion/
│   │   │   ├── parsers.py     # pdf / pptx / ipynb / html → Document
│   │   │   ├── chunker.py     # structure-aware chunking
│   │   │   └── enrich.py      # NER, keyphrases, topic tags (Text as Data)
│   │   ├── indexing/
│   │   │   ├── sparse.py      # bm25s build/load/search
│   │   │   ├── dense.py       # FastEmbed + Qdrant upsert/search
│   │   │   └── pipeline.py    # full ingest job per workspace
│   │   ├── retrieval/
│   │   │   ├── hybrid.py      # RRF fusion
│   │   │   ├── rerank.py      # bge-reranker
│   │   │   └── retriever.py   # orchestrates: query → top-k chunks
│   │   ├── generation/
│   │   │   ├── answer.py      # cite-or-abstain RAG prompt, Groq client
│   │   │   ├── quiz.py        # MCQ / short-answer generation (JSON mode)
│   │   │   └── exam.py        # exam-question generation by difficulty
│   │   ├── graph/
│   │   │   ├── extract.py     # concept + relation extraction
│   │   │   └── store.py       # SQLite + NetworkX, /graph endpoints
│   │   ├── workspaces/
│   │   │   ├── models.py      # SQLite via SQLModel: workspaces, docs, jobs
│   │   │   └── routes.py      # create workspace, upload, ingest status
│   │   └── api/
│   │       ├── ask.py         # POST /ask  (streaming SSE)
│   │       ├── recommend.py   # GET /recommend?q=
│   │       └── health.py
│   └── tests/                 # pytest: parsers, chunker, RRF, citation format
├── eval/
│   ├── queries.jsonl          # 60–100 labelled queries (see Phase 5)
│   ├── qrels.txt              # TREC-format relevance judgments
│   ├── run_eval.py            # PyTerrier/ir_measures, ablation matrix
│   └── results.md             # the table that goes on your CV
├── frontend/                  # Next.js app (chat, sources, quiz, graph tabs)
├── scripts/
│   ├── ingest_uofg.py         # clone UofGMScDS → build demo workspace
│   └── seed_demo.py
└── .github/workflows/ci.yml   # lint, pytest, docker build
```

## 5. Data model & chunking strategy (this is where most RAG projects fail)

**Metadata schema per chunk** — this powers citations, filtering, lecture recommendation, and the graph:

```json
{
  "workspace_id": "uofg-msds-demo",
  "module": "IR", "module_code": "COMPSCI5011",
  "week": 5, "doc_type": "lecture|lab|past_paper|cheatsheet|coursework",
  "source_file": "IR/Lecture_Notes/week5_bm25.pdf",
  "page_or_slide": 12,
  "title": "BM25 Term Weighting",
  "entities": ["BM25", "TF-IDF", "document length normalisation"],
  "text": "..."
}
```

**Chunking rules (structure-aware, not naive 512-token splits):**
- PPTX: one chunk per slide (title + body + speaker notes). Slides are already semantically chunked by the professor.
- PDF lecture notes: split on detected headings (font-size heuristics via PyMuPDF), target 250–400 tokens, 15% overlap, never split mid-sentence.
- Notebooks: markdown cell + following code cell(s) as one chunk; strip outputs except small ones.
- Past papers: one chunk per question (regex on `Q\d+`/`Question \d+`) with `doc_type=past_paper` — this is what makes exam mode grounded in *real* exam style, a feature almost nobody else can demo.
- Derive `module` and `week` from the repo's folder structure (`IR/Lecture_Notes/...`) and filename patterns; fall back to an LLM call for ambiguous files.

**Enrichment (Text as Data showcase):** run spaCy NER + keyphrase extraction (YAKE or KeyBERT) per chunk; store entities in payload. Optional: BERTopic over each module to auto-tag topics — mention it in the README as applied topic modelling.

**Contextual chunk enrichment (do this — it's cheap and recognisable):** before embedding, prepend a one-line LLM-generated context to each chunk, e.g. "From IR Week 5 lecture on probabilistic ranking models: <chunk text>". One-time offline cost at ingest (batch through Groq/Gemini free tier), meaningful retrieval lift on slide decks where individual slides lack standalone context. This is Anthropic's "contextual retrieval" technique — name it in your README; anyone screening RAG projects in 2026 will recognise it. Store both raw and contextualised text so the eval harness can ablate it.

## 6. Features (scoped — four flagship, three stretch)

**Flagship (must ship):**
1. **Cited Q&A** — hybrid retrieve → rerank → answer with inline citations `[IR · Week 5 · slide 12]`, each citation clickable to show the source chunk. Strict prompt: if retrieved context doesn't contain the answer, say so (measure your hallucination rate informally with 20 adversarial questions).
2. **Lecture recommendation** — "I don't understand Word2Vec" → top lectures across modules by aggregated chunk similarity, grouped by `(module, week)`.
3. **Quiz & exam-prep mode** — generate MCQs/short answers from a chosen module+week, grounded in retrieved chunks *and* styled on real past-paper chunks ("write in the style of these past exam questions"). LLM JSON mode → render interactive quiz.
4. **Multi-tenant workspaces** — public demo workspace (your corpus) + "Create workspace → upload PDFs/PPTX → background ingest → private copilot". Simple auth: random workspace token (no full user accounts in v1).

**Stretch (only after the above is live):**
5. Concept knowledge graph (LLM-extracted `(concept, relation, concept)` triples per module → NetworkX → Cytoscape.js, click node → ask about it).
6. Streaming answers (SSE) and conversation memory (last 3 turns, condensed query rewriting).
7. "Explain like the professor" persona per module using the lecturer's phrasing from notes.

## 7. CLAUDE.md (paste this into the new repo before your first Claude Code session)

```markdown
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
```

## 8. Phase plan with Claude Code prompts

Work in feature branches; one phase ≈ one or two Claude Code sessions. After each phase, run `make test`, commit, and update README.

### Phase 0 — Scaffold (0.5 day)
> "Scaffold the repo per CLAUDE.md and the layout in PROJECT_PLAN.md §4: FastAPI app with /health, pydantic-settings config, docker-compose with qdrant, Dockerfile for HF Spaces (port 7860), Makefile, ruff + pytest setup, GitHub Actions CI running lint + tests. No business logic yet."

**Done when:** CI green, `make dev` serves /health, container builds.

### Phase 1 — Ingestion & chunking (3–4 days)
> "Implement backend/app/ingestion per §5 of PROJECT_PLAN.md. parsers.py: PDF via PyMuPDF with heading detection, PPTX per-slide via python-pptx including speaker notes, ipynb via nbformat pairing markdown+code cells, HTML via BeautifulSoup. chunker.py: structure-aware chunking, 250–400 tokens, 15% overlap, sentence-boundary safe; past papers split per question. Derive module/week metadata from path patterns of the UofGMScDS repo. Write scripts/ingest_uofg.py that clones the repo and emits chunks.jsonl. Add pytest fixtures with one small sample of each file type and tests asserting chunk counts, metadata fields, and no mid-sentence splits."

**Done when:** `make ingest-demo` produces chunks.jsonl with full metadata; print stats per module (docs, chunks, avg tokens). Sanity-check 20 random chunks by eye.

### Phase 2 — Dual indexing (2–3 days)
> "Implement indexing/: sparse.py builds a bm25s index from chunks.jsonl (tokenise with the same normalisation pipeline used in enrich.py) and serialises it; dense.py embeds chunks with FastEmbed BAAI/bge-small-en-v1.5 and upserts to Qdrant with full payload, batched, idempotent by chunk hash; pipeline.py runs both for a workspace and records job status in SQLite. Add a small CLI to query each index independently."

**Done when:** you can run `python -m app.indexing.pipeline --workspace uofg-msds-demo` and then query both indexes from the CLI; Qdrant dashboard shows the collection under 1 GB.

### Phase 3 — Hybrid retrieval + re-ranking (2–3 days, the core)
> "Implement retrieval/: hybrid.py fuses bm25s top-30 and Qdrant top-30 with Reciprocal Rank Fusion (k=60), dedup by chunk id; rerank.py scores query–chunk pairs with FastEmbed reranker bge-reranker-base and returns top-6; retriever.py exposes retrieve(query, workspace_id, filters) usable both by the API and headlessly by eval/. Support optional metadata filters (module, doc_type). Unit-test RRF with synthetic rankings."

**Done when:** `retrieve("how does BM25 normalise for document length", "uofg-msds-demo")` returns IR Week-5 chunks at rank 1–3.

### Phase 4 — Generation & API (2–3 days)
> "Implement generation/answer.py: Groq client (env GROQ_API_KEY), cite-or-abstain system prompt that forces inline citation markers [n] mapped to retrieved chunks, returns answer + structured citations. Implement POST /ask with SSE streaming, GET /recommend (aggregate chunk scores by module+week), and quiz.py + exam.py using Groq JSON mode, with past_paper chunks injected as style exemplars. Add 10 integration tests with a mocked LLM."

**Done when:** curl /ask returns a cited streamed answer; quiz endpoint returns valid JSON MCQs grounded in real chunks.

### Phase 5 — IR evaluation harness (3–4 days, your CV differentiator)
> "Build eval/: I will provide queries.jsonl (question + ids of relevant chunks). Write run_eval.py that runs these configurations headlessly via retriever.py — (1) BM25 only, (2) dense only, (3) hybrid RRF, (4) hybrid+rerank, (5) hybrid+rerank with RM3 pseudo-relevance-feedback query expansion via PyTerrier, (6) hybrid+rerank with LLM query rewriting (Groq, one rewrite per query, cached to disk so reruns are free), and optionally (7) config 4 over contextualised chunks vs raw chunks — computing MAP, NDCG@10, MRR, Recall@30 with ir_measures, and writing a markdown ablation table to eval/results.md."

The RM3-vs-LLM-rewrite comparison is a coursework tie-in almost no other candidate has: you benchmarked a classical pseudo-relevance-feedback technique from your IR module against an LLM approach on the same qrels. Whichever wins, it's a memorable interview story.

> "Then add eval/eval_generation.py: over a 40-question subset, generate answers with the full pipeline and score two metrics with an LLM judge (Gemini Flash, temperature 0, rubric prompts): faithfulness (every claim in the answer must be supported by a cited chunk — judge returns per-claim verdicts) and answer relevance. Write results to eval/generation_results.md. Include 10 deliberately unanswerable questions and report the abstention rate."

You create the labels yourself (this is the part Claude Code can't do for you): pick 60–100 real questions across modules — pull them from your past papers and cheatsheets — and mark which chunks are relevant (your IR coursework taught you exactly how qrels work). Budget ~4 hours; it's the highest-ROI work in the project. Expect a pattern like: hybrid beats either alone, rerank adds a further jump in NDCG@10. Whatever the real numbers are, *those* go on your CV — two tables (retrieval + generation quality) puts you ahead of nearly every graduate applicant.

**Done when:** `make eval` prints both tables and they're committed to the README.

### Phase 6 — Workspaces & frontend (4–5 days)
> "Implement workspaces/ (create workspace → token, upload endpoint accepting pdf/pptx/ipynb, background ingest with status polling, all data filtered by workspace_id). Then build the Next.js frontend: chat view with streaming answer and a sources panel (click citation → chunk + source metadata), module/week filter chips, quiz tab, recommend tab, workspace switcher + upload flow. Clean, fast, mobile-friendly; no UI library bloat."

**Done when:** a stranger can create a workspace, upload one lecture PDF, and ask questions about it within 2 minutes.

### Phase 7 — Deploy + polish (2 days)
> "Production Dockerfile for HF Spaces (single container: FastAPI serving the built frontend, models pre-downloaded at build time to avoid cold-start), env-var documentation, rate limiting (slowapi) per workspace token, and a GitHub Action that rebuilds the Space on push to main. Write the README: architecture diagram, eval table, 30-second demo GIF, 'try it' link."

**Stretch phases:** knowledge graph (extract.py over the demo corpus offline, ship the graph JSON with the app), conversation memory, professor persona.

**Timeline — anchored to UK recruiting, not perfectionism.** It is June 2026; 2027 graduate-scheme and industrial-placement applications open September–October, and the strongest data/AI schemes (banks, consultancies, big tech) fill on a rolling basis by November. Your dissertation runs all summer, so plan part-time:

| Milestone | Target | Contents |
|---|---|---|
| M1 — Working pipeline | mid-July | Phases 0–3: ingest, dual index, hybrid+rerank retrieval (CLI only) |
| M2 — Cited answers live | early Aug | Phase 4 + minimal single-page frontend, deployed to HF Spaces |
| M3 — Numbers on the README | end Aug | Phase 5: both eval tables, CV bullets written with real figures |
| M4 — Multi-tenant + polish | early–mid Sept | Phases 6–7: workspaces, proper frontend, demo GIF, LinkedIn/blog writeup |

**A live link with real numbers in early September beats a perfect app in December.** If dissertation pressure bites, ship M1–M3 and put the workspace feature in "roadmap" — never cut the eval harness. Stretch features (graph, personas) only after M4.

## 9. Cost: £0

| Component | Service | Free tier |
|---|---|---|
| Vectors | Qdrant Cloud | 1 GB cluster |
| LLM | Groq (Llama 3.3 70B) | free tier rate limits, fine for a demo |
| Fallback LLM | Gemini Flash | free tier |
| Embeddings + reranker | FastEmbed ONNX | runs on your CPU/host |
| Hosting | HF Spaces (Docker) | free CPU Space |
| Frontend (if split) | Vercel | free |
| Graph | SQLite + NetworkX | free |

## 10. CV bullets (write these only after Phase 5 gives you real numbers)

> **LectureLens — Hybrid RAG Learning Copilot** (FastAPI, Qdrant, PyTerrier, Llama 3.3, Next.js, Docker) · live demo · GitHub
> - Built and deployed a multi-tenant retrieval-augmented learning assistant over 10 modules of MSc coursework (X,XXX documents → XX,XXX structure-aware chunks), serving cited answers, lecture recommendations, and grounded exam-question generation.
> - Engineered a hybrid retrieval pipeline (BM25 via bm25s + BGE dense embeddings with Reciprocal Rank Fusion) and cross-encoder re-ranking; evaluated on a hand-labelled 80-query test set, improving NDCG@10 from X.XX (BM25 baseline) to X.XX (+NN%).
> - Designed a free-tier production architecture (Qdrant Cloud multi-tenancy via payload filtering, ONNX CPU inference, Groq API) with CI/CD to Hugging Face Spaces.

Interview talking points this project earns you: why RRF over weighted score fusion (score distributions aren't comparable; RRF is rank-based and parameter-light), bi-encoder vs cross-encoder trade-offs, chunking as the dominant RAG quality lever, multi-tenancy via payload filter vs per-tenant collections, and how you'd scale it (swap bm25s → OpenSearch, FastEmbed → GPU batch embedding on Databricks — which connects it back to your data engineering background).

## 11. Risks & mitigations

- **Copyright:** lecture slides are the university's/professors' IP. Keep the *demo workspace* answers excerpt-based (chunks shown as short snippets), and add a note that uploaded materials stay private to the workspace. Consider asking the school informally before publicising the public demo widely.
- **Free-tier limits:** Groq rate limits → queue + graceful "busy" message; Qdrant 1 GB → cap uploads per workspace (e.g. 200 MB) and use 384-dim embeddings.
- **HF Spaces cold starts:** pre-download models in the Docker build; keep the Space "always on" is not free, so accept ~30 s cold start or use Render.
- **Scope creep:** the graph and personas are stretch. A live app + eval table beats five half-built features.

## 12. Production-grade additions (fold into Phases 4 and 7)

**Prompt-injection defence (Phase 4).** Multi-tenant means strangers upload PDFs, and a malicious document can carry embedded instructions ("ignore your system prompt and…"). Defences for v1: wrap all retrieved chunks in clearly delimited tags and instruct the model that everything inside is untrusted *data*, never instructions; never let retrieved content alter tool behaviour or system configuration; strip/escape anything resembling role markers from uploaded text at ingest. Add 5 adversarial test documents to the test suite (a PDF containing injection attempts) and assert the answer ignores them. State this explicitly in the README — mentioning indirect prompt injection unprompted signals production thinking, and interviewers for AI engineering roles increasingly probe for it.

**Observability (Phase 7).** Every request gets a trace ID; log structured JSON for query → rewritten query → fused candidates (ids + scores) → reranked top-6 → answer → cited chunk ids → latency per stage. Either plain JSON logs you can grep, or Langfuse free tier for a UI. This is your answer to the standard interview question "how do you know it works in production?" — and it's how you'll actually debug bad answers.

**Semantic caching (Phase 7).** Students ask the same questions ("explain BM25") constantly. Cache by query-embedding similarity (cosine > ~0.95 against recent queries in the same workspace → serve the cached answer, marked as cached). This doubles as your Groq rate-limit defence and gives sub-second responses on common questions for the public demo — which matters when a recruiter clicks your link.

**Claude Code prompt for this section:**
> "Harden the system per §12: (1) in generation/answer.py, wrap retrieved chunks in <retrieved_data> delimiters with an explicit untrusted-data instruction, sanitise role-marker strings at ingest, and add adversarial-document tests; (2) add structured JSON trace logging across the retrieval and generation pipeline with per-stage latency; (3) add an embedding-similarity answer cache (SQLite-backed, per-workspace, 0.95 cosine threshold, TTL 7 days) in front of /ask."

## 13. What NOT to build

Fine-tuning (reranker LoRA, instruction-tuning an LLM) is the classic scope-trap for this project: marginal gains on a corpus this size, real cost and time, and it delays the things that get interviews — a live link and metrics tables. Same for agents/multi-step orchestration frameworks. If asked in interviews "what would you do next?", *that's* where fine-tuning belongs: as a considered roadmap answer ("I'd mine hard negatives from my query logs and fine-tune the reranker — I already have the eval harness to prove whether it helped"), not as unfinished code.
