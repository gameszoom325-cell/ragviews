"""Smoke test suite for LangChain FAISS RAG system.

Validates that:
1. `ingest.py` imports cleanly and can split documents.
2. `rag.py` imports cleanly, initializes models, and implements `query(question: str)`.
3. `query(question: str)` returns the expected dictionary with `answer` and `contexts` (4 chunks).
4. `rag.get_vectorstore()` lazy-loads the index only once.
5. Strict grounding prompt template matches Phase 6 specification.
6. `OPENAI_API_KEY` validation raises `RuntimeError("OPENAI_API_KEY is missing.")` when absent.
7. `evals.py` imports cleanly without executing unprompted API calls.
"""

from pathlib import Path
import os
import sys
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ingest
import rag
import evals


def test_ingest_module_imports():
    """Verify ingest module and text splitting functions exist."""
    assert ingest is not None
    assert hasattr(ingest, "run_ingestion")
    assert hasattr(ingest, "split_documents")
    assert hasattr(ingest, "load_documents")


def test_rag_module_imports_and_interface():
    """Verify rag module exports query() and caches vectorstore."""
    assert rag is not None
    assert hasattr(rag, "query")
    assert hasattr(rag, "get_vectorstore")
    assert hasattr(rag, "get_llm")
    assert hasattr(rag, "get_embeddings")


def test_rag_lazy_loading_cache():
    """Verify vectorstore is lazily loaded and cached in memory."""
    db1 = rag.get_vectorstore()
    db2 = rag.get_vectorstore()
    assert db1 is db2, "get_vectorstore() must return cached instance without reloading"


def test_rag_query_execution():
    """Verify query() returns {'answer': ..., 'contexts': ...} with top 4 chunks."""
    result = rag.query("What is Ragas?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "contexts" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["contexts"], list)
    assert len(result["contexts"]) == 4, f"Expected 4 chunks, got {len(result['contexts'])}"


def test_strict_prompt_template():
    """Verify prompt adheres strictly to Phase 6 requirements."""
    expected_lines = [
        "You are a retrieval QA assistant.",
        "Use ONLY the supplied context.",
        "Never use prior knowledge.",
        'If the answer cannot be found in the context, respond exactly:',
        '"I don\'t know based on the provided documents."',
        "Context:",
        "{context}",
        "Question:",
        "{question}",
    ]
    for line in expected_lines:
        assert line in rag.STRICT_PROMPT_TEMPLATE, f"Missing required line in prompt: {line}"


def test_openai_api_key_missing_raises_runtime_error(monkeypatch):
    """Verify RuntimeError is raised when OPENAI_API_KEY is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(rag, "load_dotenv", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError) as exc_info:
        rag.load_environment()
    assert "OPENAI_API_KEY is missing." in str(exc_info.value)


def test_evals_module_imports():
    """Verify evals module imports cleanly."""
    assert evals is not None
    evals_file = Path(evals.__file__).resolve()
    assert evals_file.exists()


if __name__ == "__main__":
    test_ingest_module_imports()
    test_rag_module_imports_and_interface()
    test_rag_lazy_loading_cache()
    test_rag_query_execution()
    test_strict_prompt_template()
    test_evals_module_imports()
    print("All smoke tests passed successfully!")
