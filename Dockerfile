FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# System deps for pymupdf, lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgl1 libglib2.0-0 curl git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time — eliminates cold-start latency
RUN python -m spacy download en_core_web_sm
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" || true

COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY eval/ ./eval/
# BM25 indexes pre-built by `make ingest-demo` — committed to repo
COPY data/bm25_indexes/ ./data/bm25_indexes/

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app/backend
# QDRANT_URL is injected via HF Spaces secret — do not hardcode here

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
