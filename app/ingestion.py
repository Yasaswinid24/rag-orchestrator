"""Document ingestion: load -> chunk -> embed -> upsert into Pinecone (batched and resumable)."""
import gc
import hashlib
import json
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": Docx2txtLoader,
}


def load_file(path: Path) -> list[Document]:
    loader_cls = LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    docs = loader_cls(str(path)).load()
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


def chunk_id(chunk: Document) -> str:
    """Deterministic ID so re-ingesting the same file overwrites, not duplicates."""
    key = f"{chunk.metadata.get('source')}|{chunk.metadata.get('chunk_index')}|{chunk.page_content}"
    return hashlib.sha1(key.encode()).hexdigest()


def ingest_file(path: Path) -> int:
    """Embed and upsert one file in small batches so memory and CPU stay low."""
    from app.vectorstore import get_vectorstore

    s = get_settings()
    chunks = chunk_documents(load_file(path))
    if not chunks:
        return 0
    store = get_vectorstore()
    for start in range(0, len(chunks), s.upsert_batch_size):
        batch = chunks[start : start + s.upsert_batch_size]
        store.add_documents(batch, ids=[chunk_id(c) for c in batch])
    return len(chunks)


def _load_manifest() -> dict:
    try:
        return json.loads(Path(get_settings().manifest_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_manifest(manifest: dict) -> None:
    p = Path(get_settings().manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(p)


def ingest_directory(directory: str | None = None) -> dict[str, int]:
    """Ingest every supported file under a directory.

    Resumable: files already indexed (same name and size) are skipped, and progress is saved
    after every file, so a crash or a closed laptop loses at most one file.
    A bad file is reported and skipped.
    """
    root = Path(directory or get_settings().docs_dir)
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in LOADERS]
    manifest = _load_manifest()
    results: dict[str, int] = {}
    for i, path in enumerate(files, 1):
        tag = f"[{i}/{len(files)}] {path.name}"
        size = path.stat().st_size
        done = manifest.get(path.name)
        if done and done.get("size") == size:
            print(f"{tag}: already indexed ({done['chunks']} chunks), skipped", flush=True)
            results[path.name] = done["chunks"]
            continue
        try:
            n = ingest_file(path)
        except Exception as e:  # keep going; one bad file must not stop the batch
            print(f"{tag}: FAILED ({type(e).__name__}: {e})", flush=True)
            results[path.name] = -1
            continue
        if n == 0:
            print(f"{tag}: no text extracted (scanned PDF?), skipped", flush=True)
        else:
            print(f"{tag}: {n} chunks", flush=True)
            manifest[path.name] = {"size": size, "chunks": n}
            _save_manifest(manifest)
        results[path.name] = n
        gc.collect()
    return results
