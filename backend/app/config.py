from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "lecturelens"

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Retrieval
    bm25_top_k: int = 30
    dense_top_k: int = 30
    rrf_k: int = 60
    rerank_top_k: int = 6

    # Embedding model (FastEmbed ONNX)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Storage
    data_dir: Path = Path("data")
    workspace_data_dir: Path = Path("data/workspaces")
    bm25_index_dir: Path = Path("data/bm25_indexes")
    sqlite_db: Path = Path("data/lecturelens.db")

    # Rate limiting
    rate_limit_per_minute: int = 20

    # Semantic cache
    cache_similarity_threshold: float = 0.95
    cache_ttl_days: int = 7

    @property
    def workspace_data_path(self) -> Path:
        return self.workspace_data_dir

    @property
    def bm25_index_path(self) -> Path:
        return self.bm25_index_dir


settings = Settings()
