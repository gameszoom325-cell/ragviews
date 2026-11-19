"""Unit tests for RAG retrieval and synthesis pipeline."""

import pytest

import rag
from config import DEFAULT_FALLBACK_ANSWER
from exceptions import IndexLoadError, IndexNotFoundError


def test_missing_faiss_index_raises(tmp_path):
    """Verify IndexNotFoundError when index path does not exist."""
    nonexistent = tmp_path / "missing_index"
    with pytest.raises(IndexNotFoundError):
        rag.get_vectorstore(index_path=nonexistent)


def test_corrupted_faiss_index_raises(tmp_path):
    """Verify IndexLoadError when index files are invalid/corrupt."""
    corrupt_dir = tmp_path / "corrupt_index"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / "index.faiss").write_bytes(b"corrupted binary content")
    (corrupt_dir / "index.pkl").write_bytes(b"corrupted pickle content")

    with pytest.raises(IndexLoadError):
        rag.get_vectorstore(index_path=corrupt_dir)


def test_lazy_loading_singleton():
    """Verify vectorstore is deserialized and cached only once."""
    db1 = rag.get_vectorstore()
    db2 = rag.get_vectorstore()
    assert db1 is db2


def test_retrieval_returns_top_4_chunks():
    """Verify retrieval returns exactly 4 ranked contexts."""
    docs = rag.retrieve_contexts("What is RAG?", top_k=4)
    assert len(docs) == 4
    for doc in docs:
        assert isinstance(doc.page_content, str)
        assert len(doc.page_content.strip()) > 0


def test_query_schema_and_execution():
    """Verify query returns expected dictionary with answer and contexts."""
    res = rag.query("What is Retrieval-Augmented Generation?")
    assert isinstance(res, dict)
    assert "answer" in res
    assert "contexts" in res
    assert isinstance(res["answer"], str)
    assert isinstance(res["contexts"], list)
    assert len(res["contexts"]) == 4


def test_query_empty_returns_fallback():
    """Verify empty query returns fallback without crashing."""
    res = rag.query("   ")
    assert res["answer"] == DEFAULT_FALLBACK_ANSWER
    assert res["contexts"] == []


def test_query_unanswerable_grounding():
    """Verify unanswerable query yields grounded fallback."""
    answer = rag.generate_offline_answer(
        "Who was the prime minister of Atlantis in 1842?",
        "This document describes Python programming and databases.",
    )
    assert answer == DEFAULT_FALLBACK_ANSWER


def test_query_with_scores():
    """Verify query_with_scores returns scores and metadata."""
    res = rag.query_with_scores("What is FAISS?")
    assert "answer" in res
    assert "contexts" in res
    assert "scores" in res
    assert "metadata" in res
    assert len(res["scores"]) == 4
    assert len(res["metadata"]) == 4
