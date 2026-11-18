"""RAGAS Evaluation Runner with Live MCP Retrieval.

Loads questions and reference ground truth from sample datasets,
queries the real MCP RAG server dynamically for retrieved contexts
and dummy answers, and benchmarks quality using 4 core RAGAS metrics:
- faithfulness
- answer_relevancy
- context_precision
- context_recall
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Lazy / conditional imports so that smoke tests and CI can import evals
# without requiring runtime API calls or pre-installed heavy dependencies.
try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
except ImportError:
    evaluate = None  # type: ignore
    faithfulness = None  # type: ignore
    answer_relevancy = None  # type: ignore
    context_precision = None  # type: ignore
    context_recall = None  # type: ignore

import rag

DATASET_PATH = Path(__file__).parent / "evals" / "datasets" / "sample.jsonl"
EXPERIMENTS_DIR = Path(__file__).parent / "evals" / "experiments"


def check_environment() -> str:
    """Validate that OPENAI_API_KEY is configured in the environment.

    Exits with an informative error message if the key is missing.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "Error: OPENAI_API_KEY environment variable is missing.\n\n"
            "Please configure OPENAI_API_KEY in your .env file or environment:\n"
            "  Example in .env:     OPENAI_API_KEY=sk-...\n"
            "  Windows PowerShell: $env:OPENAI_API_KEY=\"sk-...\"\n"
            "  Linux / macOS:      export OPENAI_API_KEY=\"sk-...\""
        )
    return api_key


def build_evaluation_dataset(sample_file: Path) -> Any:
    """Read questions and ground truth from sample.jsonl and query MCP server dynamically.

    Iterates over each question in sample.jsonl, calls rag.query(question), and
    dynamically constructs the evaluation dataset:
    - `user_input`: Question from JSONL
    - `response`: Dummy concatenated top-3 chunks from rag.query()
    - `retrieved_contexts`: Retrieved context chunks from MCP
    - `reference`: Ground truth reference from JSONL

    Args:
        sample_file: Path to sample.jsonl file.

    Returns:
        datasets.Dataset: Evaluation dataset ready for Ragas.
    """
    if Dataset is None:
        raise ImportError(
            "The 'datasets' package is required to load datasets. "
            "Install it via: pip install datasets"
        )

    if not sample_file.exists():
        sys.exit(f"Error: Dataset file not found at {sample_file}")

    samples_raw: List[Dict[str, Any]] = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples_raw.append(json.loads(line))
            except json.JSONDecodeError as err:
                sys.exit(f"Error: Invalid JSON on line {line_num} in {sample_file}: {err}")

    if not samples_raw:
        sys.exit(f"Error: Dataset at {sample_file} is empty.")

    print(f"Loaded {len(samples_raw)} question-reference pairs from {sample_file}")
    print("Querying MCP RAG server dynamically for real contexts...\n")

    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(samples_raw, start=1):
        question = item.get("user_input", item.get("question", "")).strip()
        reference = item.get("reference", item.get("ground_truth", "")).strip()

        if not question:
            continue

        print(f"[{idx}/{len(samples_raw)}] Querying: '{question}'")

        # Calls rag.query(question) which queries the live MCP server
        rag_output = rag.query(question)

        dummy_answer = rag_output.get("answer", rag_output.get("response", ""))
        contexts = rag_output.get("contexts", rag_output.get("retrieved_contexts", []))

        print(f"       -> Retrieved {len(contexts)} contexts from MCP.")

        # Construct evaluation row with both modern and legacy Ragas keys
        record = {
            "user_input": question,
            "response": dummy_answer,
            "retrieved_contexts": contexts,
            "reference": reference,
            # Legacy aliases
            "question": question,
            "answer": dummy_answer,
            "contexts": contexts,
            "ground_truth": reference,
        }
        records.append(record)

    return Dataset.from_list(records)


def run_eval(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = EXPERIMENTS_DIR,
) -> Any:
    """Execute RAGAS evaluation against real MCP retrieved data.

    Args:
        dataset_path: Path to the seed JSONL dataset containing queries and references.
        output_dir: Output directory for benchmark run CSV results.

    Returns:
        Evaluation results object from ragas.evaluate().
    """
    # 1. Enforce API key presence
    check_environment()

    # 2. Check dependencies
    if evaluate is None:
        sys.exit(
            "Error: 'ragas' is not installed.\n"
            "Install all dependencies via: pip install -e . or pip install ragas datasets openai"
        )

    # 3. Dynamically build dataset by querying MCP server
    dataset = build_evaluation_dataset(dataset_path)

    # 4. Define the 4 required RAGAS metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print("\nRunning RAGAS evaluation with 4 core metrics:")
    print("  1. faithfulness")
    print("  2. answer_relevancy")
    print("  3. context_precision")
    print("  4. context_recall")
    print("\nExecuting evaluation with LLM backend...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    df = result.to_pandas()

    # 5. Print Summary Table (Metric Name, Mean Score, Min Score)
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    print("\n" + "=" * 58)
    print(f"{'Metric Name':<22} | {'Mean Score':<14} | {'Min Score':<14}")
    print("-" * 58)
    for m in metric_names:
        if m in df.columns:
            mean_score = float(df[m].mean())
            min_score = float(df[m].min())
            print(f"{m:<22} | {mean_score:<14.4f} | {min_score:<14.4f}")
        else:
            print(f"{m:<22} | {'N/A':<14} | {'N/A':<14}")
    print("=" * 58)

    # 6. Save full per-row results to evals/experiments/run_<timestamp>.csv
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = output_dir / f"run_{timestamp}.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Full per-row results saved to: {output_csv}")

    # 7. Flag any row where faithfulness < 0.7 or context_recall < 0.8
    if "faithfulness" in df.columns and "context_recall" in df.columns:
        flagged_mask = (df["faithfulness"] < 0.7) | (df["context_recall"] < 0.8)
        flagged_df = df[flagged_mask]

        if not flagged_df.empty:
            print("\n" + "!" * 70)
            print("FLAGGED ROWS (faithfulness < 0.7 OR context_recall < 0.8):")
            print("!" * 70)
            for idx, row in flagged_df.iterrows():
                f_score = row.get("faithfulness", float("nan"))
                cr_score = row.get("context_recall", float("nan"))
                u_input = row.get("user_input", row.get("question", "N/A"))
                resp = row.get("response", row.get("answer", "N/A"))
                ctxs = row.get("retrieved_contexts", row.get("contexts", []))
                ref = row.get("reference", row.get("ground_truth", "N/A"))

                print(f"\n[Row {idx}] Faithfulness: {f_score:.4f} | Context Recall: {cr_score:.4f}")
                print(f"  User Input : {u_input}")
                print(f"  Response   : {resp}")
                print(f"  Reference  : {ref}")
                print(f"  Contexts   : {ctxs}")
            print("!" * 70)
        else:
            print("\n[OK] All rows passed thresholds (faithfulness >= 0.7 and context_recall >= 0.8).")

    return result


if __name__ == "__main__":
    run_eval()
