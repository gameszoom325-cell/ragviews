"""RAGAS Evaluation Runner with Live MCP Retrieval and Quality Checks.

Loads evaluation questions and ground truth from datasets,
queries the RAG pipeline dynamically for contexts and answers,
and evaluates quality across 4 core RAGAS metrics:
- faithfulness
- answer_relevancy
- context_precision
- context_recall

Supports `--check` mode to enforce thresholds defined in evals/thresholds.json,
writes failure diagnostics to evals/last_failure.json, and exits with code 0 or 1.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

import rag

DATASET_PATH = Path(__file__).parent / "evals" / "datasets" / "sample.jsonl"
EXPERIMENTS_DIR = Path(__file__).parent / "evals" / "experiments"
THRESHOLDS_PATH = Path(__file__).parent / "evals" / "thresholds.json"
LAST_FAILURE_PATH = Path(__file__).parent / "evals" / "last_failure.json"

STOP_WORDS = {
    "what",
    "is",
    "the",
    "of",
    "and",
    "a",
    "an",
    "in",
    "to",
    "for",
    "how",
    "does",
    "can",
    "are",
    "with",
    "without",
    "from",
    "on",
    "it",
    "by",
    "or",
    "as",
    "at",
    "be",
    "this",
    "that",
    "which",
    "there",
    "have",
    "has",
    "do",
    "you",
    "your",
    "they",
    "their",
    "we",
    "our",
}


def _tokenize(text: str) -> List[str]:
    """Extract lowercase non-stopword tokens from text."""
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def compute_row_metrics(
    user_input: str,
    response: str,
    retrieved_contexts: List[str],
    reference: str,
) -> Dict[str, float]:
    """Calculate RAGAS metric scores for a single evaluation row."""
    q_tokens = set(_tokenize(user_input))
    ref_tokens = set(_tokenize(reference))

    # 1. Context Recall: fraction of reference facts/tokens present in contexts
    all_context_tokens = set()
    for ctx in retrieved_contexts:
        all_context_tokens.update(_tokenize(ctx))

    if ref_tokens:
        recall = len(ref_tokens & all_context_tokens) / len(ref_tokens)
    else:
        recall = 1.0
    context_recall_score = min(1.0, max(0.0, recall))

    # 2. Context Precision: Average precision of retrieved chunks
    relevance_flags = []
    for ctx in retrieved_contexts:
        ctx_tokens = set(_tokenize(ctx))
        # Relevant if it shares concepts with query or reference
        overlap_q = len(ctx_tokens & q_tokens) / max(1, len(q_tokens))
        overlap_ref = len(ctx_tokens & ref_tokens) / max(1, len(ref_tokens))
        is_relevant = (overlap_q >= 0.25) or (overlap_ref >= 0.35)
        relevance_flags.append(1 if is_relevant else 0)

    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        context_precision_score = 0.0
    else:
        cumulative_hits = 0
        precision_sum = 0.0
        for rank, flag in enumerate(relevance_flags, start=1):
            if flag == 1:
                cumulative_hits += 1
                precision_sum += cumulative_hits / rank
        context_precision_score = min(1.0, max(0.0, precision_sum / total_relevant))

    # 3. Faithfulness: proportion of statements in response grounded in context
    # and focused on the topic (penalizes inclusion of noisy distractor chunks)
    resp_paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if not resp_paragraphs:
        resp_paragraphs = [response]

    supported_count: float = 0.0
    for para in resp_paragraphs:
        para_tokens = set(_tokenize(para))
        overlap_with_q = len(para_tokens & q_tokens) / max(1, len(q_tokens))
        overlap_with_ctx = len(para_tokens & all_context_tokens) / max(1, len(para_tokens))
        # Supported and topic-grounded
        if overlap_with_ctx >= 0.70 and overlap_with_q >= 0.20:
            supported_count += 1
        elif overlap_with_ctx >= 0.85:
            supported_count += 0.5

    faithfulness_score = min(1.0, max(0.0, supported_count / max(1, len(resp_paragraphs))))

    # 4. Answer Relevancy: semantic alignment between response and user input
    resp_tokens = set(_tokenize(response))
    if q_tokens:
        overlap = len(q_tokens & resp_tokens) / len(q_tokens)
        relevancy_score = min(1.0, max(0.0, 0.5 + 0.5 * overlap))
    else:
        relevancy_score = 0.85

    return {
        "faithfulness": round(faithfulness_score, 4),
        "answer_relevancy": round(relevancy_score, 4),
        "context_precision": round(context_precision_score, 4),
        "context_recall": round(context_recall_score, 4),
    }


def load_sample_queries(sample_file: Path) -> List[Dict[str, Any]]:
    """Load question-reference pairs from sample JSONL file."""
    if not sample_file.exists():
        sys.exit(f"Error: Dataset file not found at {sample_file}")

    records: List[Dict[str, Any]] = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                sys.exit(f"Error: Invalid JSON on line {line_num} in {sample_file}: {err}")

    if not records:
        sys.exit(f"Error: Dataset at {sample_file} is empty.")

    return records


def run_evaluation(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = EXPERIMENTS_DIR,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Execute dynamic evaluation against RAG pipeline and compute metric scores."""
    samples = load_sample_queries(dataset_path)
    print(f"Loaded {len(samples)} query-reference pairs from {dataset_path}")
    print("Executing dynamic RAG retrieval for evaluation...\n")

    eval_rows: List[Dict[str, Any]] = []
    metric_accum: Dict[str, List[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for idx, item in enumerate(samples, start=1):
        user_input = item.get("user_input", item.get("question", "")).strip()
        reference = item.get("reference", item.get("ground_truth", "")).strip()

        print(f"[{idx}/{len(samples)}] Querying: '{user_input}'")
        rag_output = rag.query(user_input)

        response = rag_output.get("answer", rag_output.get("response", ""))
        contexts = rag_output.get("contexts", rag_output.get("retrieved_contexts", []))

        # Compute metric scores for this row
        scores = compute_row_metrics(user_input, response, contexts, reference)
        for m, s in scores.items():
            metric_accum[m].append(s)

        row_data = {
            "row_index": idx - 1,
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": contexts,
            "reference": reference,
            **scores,
        }
        eval_rows.append(row_data)

    mean_scores = {
        m: round(sum(scores) / max(1, len(scores)), 4) for m, scores in metric_accum.items()
    }

    # Print summary table
    print("\n" + "=" * 58)
    print(f"{'Metric Name':<22} | {'Mean Score':<14} | {'Min Score':<14}")
    print("-" * 58)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        mean_val = mean_scores[m]
        min_val = min(metric_accum[m]) if metric_accum[m] else 0.0
        print(f"{m:<22} | {mean_val:<14.4f} | {min_val:<14.4f}")
    print("=" * 58)

    # Save per-row CSV results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = output_dir / f"run_{timestamp}.csv"

    import csv

    if eval_rows:
        keys = list(eval_rows[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in eval_rows:
                writer.writerow(r)
        print(f"\n[OK] Results saved to: {output_csv}")

    return mean_scores, eval_rows


def load_thresholds(thresholds_path: Path = THRESHOLDS_PATH) -> Dict[str, float]:
    """Load evaluation pass/fail thresholds from thresholds.json."""
    if not thresholds_path.exists():
        sys.exit(f"Error: Thresholds file not found at {thresholds_path}")
    with open(thresholds_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_thresholds(
    mean_scores: Dict[str, float],
    eval_rows: List[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> bool:
    """Validate mean scores against target thresholds and record failure state."""
    failed_metrics = []
    worst_rows: Dict[str, List[Dict[str, Any]]] = {}

    for metric, threshold in thresholds.items():
        score = mean_scores.get(metric, 0.0)
        if score < threshold:
            failed_metrics.append(metric)
            # Find the lowest performing rows for this failed metric
            sorted_rows = sorted(eval_rows, key=lambda r: r.get(metric, 1.0))
            worst_rows[metric] = [
                {
                    "row_index": r["row_index"],
                    "score": r.get(metric, 0.0),
                    "user_input": r["user_input"],
                    "response": r["response"][:200] if len(r["response"]) > 200 else r["response"],
                    "retrieved_contexts": r["retrieved_contexts"],
                    "reference": r["reference"],
                }
                for r in sorted_rows[:2]  # Top 2 worst rows
            ]

    if failed_metrics:
        # Write failure record to evals/last_failure.json
        failure_data = {
            "failed_metrics": failed_metrics,
            "scores": mean_scores,
            "thresholds": thresholds,
            "worst_rows": worst_rows,
        }
        with open(LAST_FAILURE_PATH, "w", encoding="utf-8") as f:
            json.dump(failure_data, f, indent=2)

        print("\n" + "!" * 64)
        print("QUALITY CHECK FAILED: Metrics below threshold!")
        for m in failed_metrics:
            actual = mean_scores.get(m, 0.0)
            target = thresholds.get(m, 0.0)
            print(f"  - {m}: {actual:.4f} (Required: >= {target:.4f})")
        print(f"\nDiagnostics recorded to: {LAST_FAILURE_PATH}")
        print("!" * 64)
        return False

    # Check passed: clear or report success
    print("\n" + "=" * 64)
    print("SUCCESS: All RAGAS metrics meet or exceed target thresholds!")
    for m, target in thresholds.items():
        actual = mean_scores.get(m, 0.0)
        print(f"  [PASS] {m:<20}: {actual:.4f} >= {target:.4f}")
    print("=" * 64)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Runner")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Enforce thresholds defined in evals/thresholds.json and exit with code 0/1",
    )
    args = parser.parse_args()

    mean_scores, eval_rows = run_evaluation()

    if args.check:
        thresholds = load_thresholds()
        passed = check_thresholds(mean_scores, eval_rows, thresholds)
        if not passed:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
