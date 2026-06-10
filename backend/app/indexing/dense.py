"""
Dense indexing: FastEmbed (ONNX CPU) → Qdrant upsert.
Single Qdrant collection, multi-tenancy via workspace_id payload filter.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk

COLLECTION_NAME = settings.qdrant_collection
VECTOR_SIZE = 384  # BGE-small-en-v1.5

_embed_model = None
_qdrant_client = None


def _get_embedder():
    global _embed_model
    if _embed_model is None:
        try:
            from fastembed import TextEmbedding

            _embed_model = TextEmbedding(model_name=settings.embed_model)
        except ImportError as e:
            raise ImportError("fastembed required: pip install fastembed") from e
    return _embed_model


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient

            url = settings.qdrant_url
            if url in ("local", ""):
                # Local file-based Qdrant — no Docker needed
                local_path = str(settings.data_dir / "qdrant_local")
                import os; os.makedirs(local_path, exist_ok=True)
                _qdrant_client = QdrantClient(path=local_path)
            else:
                kwargs: dict = {"url": url}
                if settings.qdrant_api_key:
                    kwargs["api_key"] = settings.qdrant_api_key
                _qdrant_client = QdrantClient(**kwargs)
        except ImportError as e:
            raise ImportError("qdrant-client required: pip install qdrant-client") from e
    return _qdrant_client


def ensure_collection() -> None:
    from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

    client = _get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    # Payload indexes required for filtered queries on Qdrant Cloud
    for field in ("workspace_id", "module", "doc_type", "week"):
        try:
            schema = PayloadSchemaType.INTEGER if field == "week" else PayloadSchemaType.KEYWORD
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # index already exists — safe to ignore


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = _get_embedder()
    return [list(v) for v in embedder.embed(texts)]


def upsert_chunks(chunks: list["Chunk"], batch_size: int = 64, use_contextualised: bool = True) -> int:
    """Upsert chunks to Qdrant; idempotent by chunk_id. Returns upserted count."""
    from qdrant_client.models import PointStruct

    ensure_collection()
    client = _get_qdrant()
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.contextualised_text if use_contextualised else c.text for c in batch]
        vectors = embed_texts(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c.chunk_id)),
                vector=vec,
                payload={
                    "chunk_id": c.chunk_id,
                    "workspace_id": c.workspace_id,
                    "module": c.module,
                    "module_code": c.module_code,
                    "week": c.week,
                    "doc_type": c.doc_type,
                    "source_file": c.source_file,
                    "page_or_slide": c.page_or_slide,
                    "chunk_index": c.chunk_index,
                    "title": c.title,
                    "text": c.text,
                    "entities": c.entities,
                    "keyphrases": c.keyphrases,
                },
            )
            for c, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)

    return total


def search_dense(
    query: str,
    workspace_id: str,
    top_k: int = 30,
    filters: dict | None = None,
) -> list[tuple[str, float]]:
    """Return [(chunk_id, score), …] for a workspace."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = _get_qdrant()
    query_vec = embed_texts([query])[0]

    must_conditions = [
        FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id))
    ]
    if filters:
        if "module" in filters:
            must_conditions.append(
                FieldCondition(key="module", match=MatchValue(value=filters["module"]))
            )
        if "doc_type" in filters:
            must_conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=filters["doc_type"]))
            )
        if "week" in filters:
            must_conditions.append(
                FieldCondition(key="week", match=MatchValue(value=filters["week"]))
            )

    qdrant_filter = Filter(must=must_conditions)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )
    return [(r.payload["chunk_id"], r.score) for r in response.points]


def get_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    """Fetch chunk payloads from Qdrant by chunk_id."""
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    client = _get_qdrant()
    if not chunk_ids:
        return []

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids))]
        ),
        limit=len(chunk_ids),
        with_payload=True,
    )
    return [r.payload for r in results[0]]
