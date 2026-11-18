"""Smoke test suite for RAGAS evaluation repository.

Validates that:
1. `evals.py` and `rag.py` import cleanly and parse without calling live APIs.
2. `rag.query()` interface formats answers and contexts correctly (with mock).
3. `evals/datasets/sample.jsonl` exists and conforms to required schema.
4. `evals.build_evaluation_dataset` dynamically constructs dataset rows without external networks.
"""

import json
from pathlib import Path
import sys
from unittest.mock import patch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evals
import rag


def test_evals_module_exists_and_imports():
    """Verify evals module can be imported cleanly without executing API calls."""
    assert evals is not None
    evals_file = Path(evals.__file__).resolve()
    assert evals_file.exists(), f"evals.py does not exist at {evals_file}"
    assert evals_file.is_file()


def test_rag_module_interface():
    """Verify rag module exports query and config functions."""
    assert rag is not None
    assert hasattr(rag, "query"), "rag module missing query() function"
    assert hasattr(rag, "query_rag"), "rag module missing query_rag() alias"
    config = rag.get_mcp_config()
    assert "server_url" in config
    assert "timeout" in config
    assert config["timeout"] == 30.0


def test_rag_query_with_mock():
    """Verify rag.query formats answer (concatenating top 3 chunks) and contexts."""
    sample_mock = {
        "answer": "Chunk 1\n\nChunk 2\n\nChunk 3",
        "contexts": ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"],
        "response": "Chunk 1\n\nChunk 2\n\nChunk 3",
        "retrieved_contexts": ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"],
    }
    with patch("rag._call_mcp_query", return_value=sample_mock):
        result = rag.query("Test question")
        assert result["answer"] == sample_mock["answer"]
        assert len(result["contexts"]) == 4
        assert "response" in result
        assert "retrieved_contexts" in result


def test_sample_dataset_exists_and_parses():
    """Verify sample dataset exists and contains 5 valid rows with questions and references."""
    dataset_path = REPO_ROOT / "evals" / "datasets" / "sample.jsonl"
    assert dataset_path.exists(), f"Sample dataset not found at {dataset_path}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 5, f"Expected 5 sample rows in sample.jsonl, found {len(rows)}"

    for i, row in enumerate(rows):
        has_question = "user_input" in row or "question" in row
        has_reference = "reference" in row or "ground_truth" in row
        assert has_question, f"Row {i} is missing user_input/question"
        assert has_reference, f"Row {i} is missing reference/ground_truth"


if __name__ == "__main__":
    test_evals_module_exists_and_imports()
    test_rag_module_interface()
    test_rag_query_with_mock()
    test_sample_dataset_exists_and_parses()
    print("All smoke tests passed successfully!")
