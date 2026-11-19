"""Comprehensive validation suite for LangChain FAISS RAG system.

Validates all 7 checkpoints required by Part 6:
1. ingest.py imports correctly
2. rag.py imports correctly
3. pyproject.toml parses successfully
4. FAISS path exists after ingestion
5. Retrieval returns four chunks
6. OpenAI model initializes
7. query() returns { answer, contexts }
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
TOML_PATH = ROOT_DIR / "pyproject.toml"
FAISS_DIR = ROOT_DIR / "faiss_index"


def validate_all() -> None:
    print("=" * 60)
    print("RUNNING PART 6 VALIDATION SUITE")
    print("=" * 60)

    # Checkpoint 1: ingest.py imports correctly
    import ingest
    assert hasattr(ingest, "run_ingestion"), "ingest.py missing run_ingestion"
    print("✓ Checkpoint 1: ingest.py imports correctly")

    # Checkpoint 2: rag.py imports correctly
    import rag
    assert hasattr(rag, "query"), "rag.py missing query"
    print("✓ Checkpoint 2: rag.py imports correctly")

    # Checkpoint 3: pyproject.toml parses successfully
    assert TOML_PATH.exists(), f"pyproject.toml not found at {TOML_PATH}"
    data = tomllib.loads(TOML_PATH.read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]
    required_deps = [
        "langchain",
        "langchain-openai",
        "langchain-community",
        "faiss-cpu",
        "python-dotenv",
    ]
    for dep in required_deps:
        assert dep in dependencies, f"Missing required dependency '{dep}' in pyproject.toml"
    print("✓ Checkpoint 3: pyproject.toml parses successfully with all required dependencies")

    # Checkpoint 4: FAISS path exists after ingestion
    assert FAISS_DIR.exists(), f"FAISS index folder missing at {FAISS_DIR}"
    assert (FAISS_DIR / "index.faiss").exists(), "index.faiss missing"
    assert (FAISS_DIR / "index.pkl").exists(), "index.pkl missing"
    print(f"✓ Checkpoint 4: FAISS path exists after ingestion at {FAISS_DIR}")

    # Checkpoint 5: Retrieval returns four chunks
    vectorstore = rag.get_vectorstore()
    retrieved_docs = vectorstore.similarity_search("What is RAG?", k=4)
    assert len(retrieved_docs) == 4, f"Expected 4 chunks, got {len(retrieved_docs)}"
    print(f"✓ Checkpoint 5: Retrieval returns exactly {len(retrieved_docs)} chunks")

    # Checkpoint 6: OpenAI model initializes
    llm = rag.get_llm()
    assert llm.model_name == "gpt-4o-mini", f"Expected gpt-4o-mini, got {llm.model_name}"
    embeddings = rag.get_embeddings()
    assert embeddings.model == "text-embedding-3-small", f"Expected text-embedding-3-small, got {embeddings.model}"
    print(f"✓ Checkpoint 6: OpenAI models initialize (Chat: {llm.model_name}, Embeddings: {embeddings.model})")

    # Checkpoint 7: query() returns { answer, contexts }
    res = rag.query("What is Retrieval-Augmented Generation?")
    assert isinstance(res, dict), "query() result must be a dict"
    assert "answer" in res, "Result missing 'answer' key"
    assert "contexts" in res, "Result missing 'contexts' key"
    assert isinstance(res["answer"], str), "'answer' must be a string"
    assert isinstance(res["contexts"], list), "'contexts' must be a list"
    assert len(res["contexts"]) == 4, f"Expected 4 chunks, got {len(res['contexts'])}"
    assert len(res["answer"]) > 0, "'answer' must not be empty"
    print("✓ Checkpoint 7: query() returns { answer, contexts } with top 4 ranked chunks")

    print("=" * 60)
    print("ALL 7 VALIDATIONS PASSED CLEANLY AND SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    validate_all()
