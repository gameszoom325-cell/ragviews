"""Smoke test suite for RAGAS evaluation repository.

Validates that:
1. `evals.py` imports cleanly and parses without calling OpenAI API.
2. `rag.py` placeholder function executes simulated retrieval.
3. `evals/datasets/sample.jsonl` exists and conforms to required schema.
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
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


def test_rag_module_placeholder():
    """Verify rag module placeholder runs simulated query without network calls."""
    assert rag is not None
    rag_file = Path(rag.__file__).resolve()
    assert rag_file.exists(), f"rag.py does not exist at {rag_file}"

    test_query = "What is Ragas?"
    result = rag.query_rag(test_query)

    assert isinstance(result, dict)
    assert "response" in result
    assert "retrieved_contexts" in result
    assert isinstance(result["response"], str)
    assert isinstance(result["retrieved_contexts"], list)
    assert len(result["retrieved_contexts"]) > 0


def test_sample_dataset_exists_and_parses():
    """Verify sample dataset exists and contains 5 valid rows with required fields."""
    dataset_path = REPO_ROOT / "evals" / "datasets" / "sample.jsonl"
    assert dataset_path.exists(), f"Sample dataset not found at {dataset_path}"

    required_fields = ["user_input", "response", "retrieved_contexts", "reference"]

    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 5, f"Expected 5 sample rows in sample.jsonl, found {len(rows)}"

    for i, row in enumerate(rows):
        for field in required_fields:
            assert field in row, f"Row {i} is missing required field '{field}'"
        assert isinstance(row["user_input"], str) and len(row["user_input"].strip()) > 0
        assert isinstance(row["response"], str) and len(row["response"].strip()) > 0
        assert isinstance(row["retrieved_contexts"], list) and len(row["retrieved_contexts"]) > 0
        assert isinstance(row["reference"], str) and len(row["reference"].strip()) > 0


if __name__ == "__main__":
    test_evals_module_exists_and_imports()
    test_rag_module_placeholder()
    test_sample_dataset_exists_and_parses()
    print("All smoke tests passed successfully!")
