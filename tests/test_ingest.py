"""Unit tests for ingestion pipeline."""

import pytest
from langchain_core.documents import Document
from pathlib import Path

from exceptions import DocumentIngestionError
from ingest import (
    ResilientOpenAIEmbeddings,
    build_faiss_index,
    load_documents,
    split_documents,
)


def test_load_documents_success(tmp_path):
    """Verify loading text documents from a valid directory."""
    doc_file = tmp_path / "sample.txt"
    doc_file.write_text("This is test document content.", encoding="utf-8")

    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert "This is test document content." in docs[0].page_content
    assert docs[0].metadata["filename"] == "sample.txt"


def test_load_documents_empty_directory_raises(tmp_path):
    """Verify empty directory raises DocumentIngestionError."""
    with pytest.raises(DocumentIngestionError):
        load_documents(tmp_path)


def test_load_documents_nonexistent_directory_raises():
    """Verify nonexistent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_documents(Path("nonexistent_directory_path_12345"))


def test_split_documents_chunks_and_metadata():
    """Verify splitting generates chunks with correct metadata."""
    long_text = "Word " * 200
    doc = Document(page_content=long_text, metadata={"source": "test.txt"})

    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert "chunk_id" in chunks[0].metadata
    assert "char_length" in chunks[0].metadata


def test_resilient_embeddings_fallback():
    """Verify fallback vector generation is deterministic 1536-dim."""
    emb = ResilientOpenAIEmbeddings(model="text-embedding-3-small", openai_api_key="test-key")
    v1 = emb._fallback_vector("hello world")
    v2 = emb._fallback_vector("hello world")
    v3 = emb._fallback_vector("different text")

    assert len(v1) == 1536
    assert v1 == v2
    assert v1 != v3


def test_build_faiss_index_clean_overwrite(tmp_path):
    """Verify index creation cleanly overwrites existing index path."""
    chunks = [Document(page_content=f"Fact number {i}", metadata={}) for i in range(5)]
    index_dir = tmp_path / "test_faiss"

    # Build first time
    vs1 = build_faiss_index(chunks, index_path=index_dir, api_key="test-key")
    assert vs1 is not None
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()

    # Re-build second time on same path to ensure clean overwrite
    vs2 = build_faiss_index(chunks[:3], index_path=index_dir, api_key="test-key")
    assert vs2 is not None
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()
