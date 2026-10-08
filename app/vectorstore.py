from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings


class LocalEmbeddings(Embeddings):
    """Free, local embeddings (ONNX via fastembed). No API key, no cost."""

    def __init__(self, model_name: str, threads: int, batch_size: int):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, threads=threads)
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()


@lru_cache
def get_embeddings() -> LocalEmbeddings:
    s = get_settings()
    return LocalEmbeddings(s.embedding_model, s.embed_threads, s.embed_batch_size)


def ensure_index() -> None:
    """Create the Pinecone serverless index if missing; fail clearly on a dimension mismatch."""
    s = get_settings()
    pc = Pinecone(api_key=s.pinecone_api_key)
    existing = {i.name: i for i in pc.list_indexes()}
    if s.pinecone_index not in existing:
        pc.create_index(
            name=s.pinecone_index,
            dimension=s.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
        )
        return
    dim = existing[s.pinecone_index].dimension
    if dim != s.embedding_dim:
        raise RuntimeError(
            f"Pinecone index '{s.pinecone_index}' has dimension {dim}, but the embedding model "
            f"needs {s.embedding_dim}. Delete that index in the Pinecone console or set "
            f"PINECONE_INDEX to a new name in .env."
        )


@lru_cache
def get_vectorstore() -> PineconeVectorStore:
    s = get_settings()
    ensure_index()
    return PineconeVectorStore(
        index_name=s.pinecone_index,
        embedding=get_embeddings(),
        namespace=s.namespace,
        pinecone_api_key=s.pinecone_api_key,
    )
