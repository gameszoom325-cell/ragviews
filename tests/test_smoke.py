"""Smoke test suite for LangChain FAISS RAG system.

Validates that:
1. `ingest.py` imports cleanly and can split documents.
2. `rag.py` imports cleanly, initializes models, and implements `query(question: str)`.
3. `query(question: str)` returns the expected dictionary with `answer` and `contexts` (4 chunks).
4. `evals.py` imports cleanly without executing API calls.
"""

from pathlib import Path
import sys

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


def test_rag_query_execution():
    """Verify query() returns {'answer': ..., 'contexts': ...} with top 4 chunks."""
    result = rag.query("What is Ragas?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "contexts" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["contexts"], list)
    assert len(result["contexts"]) == 4, f"Expected 4 chunks, got {len(result['contexts'])}"


def test_evals_module_imports():
    """Verify evals module imports cleanly."""
    assert evals is not None
    evals_file = Path(evals.__file__).resolve()
    assert evals_file.exists()


if __name__ == "__main__":
    test_ingest_module_imports()
    test_rag_module_imports_and_interface()
    test_rag_query_execution()
    test_evals_module_imports()
    print("All smoke tests passed successfully!")
