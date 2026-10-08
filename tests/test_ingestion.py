from langchain_core.documents import Document

from app.ingestion import chunk_documents, chunk_id


def test_chunking_splits_and_tags_metadata():
    doc = Document(page_content="Sentence about RAG. " * 200, metadata={"source": "a.txt"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "a.txt" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic_and_unique():
    doc = Document(page_content="Different words here. " * 200, metadata={"source": "b.txt"})
    chunks = chunk_documents([doc])
    ids = [chunk_id(c) for c in chunks]
    assert ids == [chunk_id(c) for c in chunks]
    assert len(set(ids)) == len(ids)
