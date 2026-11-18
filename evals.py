"""RAGAS Evaluation Runner.

Loads evaluation datasets and executes RAGAS metric evaluations for:
- faithfulness
- answer_relevancy
- context_precision
- context_recall

Outputs a metric summary table, writes per-row results to CSV,
and flags low-performing samples below target thresholds.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            "Please set OPENAI_API_KEY before running evaluations.\n"
            "Example:\n"
            "  Windows PowerShell: $env:OPENAI_API_KEY=\"sk-...\"\n"
            "  Windows CMD:        set OPENAI_API_KEY=sk-...\n"
            "  Linux / macOS:      export OPENAI_API_KEY=\"sk-...\""
        )
    return api_key


def load_dataset_from_jsonl(file_path: Path) -> Any:
    """Load sample JSONL dataset into a Hugging Face Dataset.

    Populates both modern Ragas fields (user_input, response, retrieved_contexts, reference)
    and legacy Ragas fields (question, answer, contexts, ground_truth) to maintain
    seamless cross-version compatibility.

    Args:
        file_path: Path to the .jsonl dataset file.

    Returns:
        datasets.Dataset: The loaded dataset.
    """
    if Dataset is None:
        raise ImportError(
            "The 'datasets' package is required to load datasets. "
            "Install it via: pip install datasets"
        )

    if not file_path.exists():
        sys.exit(f"Error: Dataset file not found at {file_path}")

    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as err:
                sys.exit(f"Error: Invalid JSON on line {line_num} in {file_path}: {err}")

            user_input = item.get("user_input", item.get("question", ""))
            response = item.get("response", item.get("answer", ""))
            retrieved_contexts = item.get("retrieved_contexts", item.get("contexts", []))
            reference = item.get("reference", item.get("ground_truth", ""))

            # Support both current Ragas and legacy schemas
            record = {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
                "question": user_input,
                "answer": response,
                "contexts": retrieved_contexts,
                "ground_truth": reference,
            }
            records.append(record)

    if not records:
        sys.exit(f"Error: Dataset at {file_path} is empty.")

    return Dataset.from_list(records)


def run_eval(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = EXPERIMENTS_DIR,
) -> Any:
    """Execute Ragas evaluation on the dataset and print summary statistics.

    Args:
        dataset_path: Path to the JSONL dataset.
        output_dir: Directory where per-row CSV results will be saved.

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

    print(f"Loading evaluation dataset from: {dataset_path}")
    dataset = load_dataset_from_jsonl(dataset_path)
    print(f"Loaded {len(dataset)} evaluation samples.")

    # 3. Define the four required metrics
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

    # 4. Print Summary Table (Metric Name, Mean Score, Min Score)
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

    # 5. Save full per-row results to evals/experiments/run_<timestamp>.csv
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = output_dir / f"run_{timestamp}.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Full per-row results saved to: {output_csv}")

    # 6. Flag any row where faithfulness < 0.7 or context_recall < 0.8
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
