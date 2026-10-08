from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Free tiers: Groq (LLM) + Pinecone Starter (vector DB). Embeddings run locally.
    groq_api_key: str = ""
    pinecone_api_key: str = ""
    # New default name: an index created earlier with 1536 dims cannot be reused.
    pinecone_index: str = "rag-orchestrator-free"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    namespace: str = "default"

    embedding_model: str = "BAAI/bge-small-en-v1.5"   # runs locally via fastembed
    embedding_dim: int = 384                          # must match the model above
    llm_model: str = "openai/gpt-oss-120b"            # any chat model your Groq key can access (see README)

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    max_rewrites: int = 2

    docs_dir: str = "data/docs"
    manifest_path: str = "data/.ingest_manifest.json"  # remembers which files are already indexed

    # Keep ingestion gentle on laptops
    embed_threads: int = 2        # CPU threads used by the embedding model
    embed_batch_size: int = 16    # texts embedded at once
    upsert_batch_size: int = 64   # chunks embedded + sent to Pinecone per batch


@lru_cache
def get_settings() -> Settings:
    return Settings()