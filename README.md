# rag-eval-ragas

Evaluation pipeline for Retrieval-Augmented Generation (RAG) applications using [Ragas](https://github.com/explodinggradients/ragas).

This repository mirrors the official RAGAS quickstart structure, providing an automated evaluation runner, sample datasets, placeholder RAG pipeline integration, and CI-ready smoke tests.

---

## Project Structure

```
rag-eval-ragas/
├── rag.py                     # Simulated RAG pipeline (to be replaced with MCP/production client)
├── evals.py                   # Evaluation runner using Ragas metrics
├── pyproject.toml             # Project dependencies (ragas, datasets, openai, pytest)
├── README.md                  # Documentation and metric guide
├── .gitignore                 # Standard Python and secret ignore rules
├── evals/
│   ├── datasets/
│   │   └── sample.jsonl       # 5 sample test cases with input, response, context & reference
│   └── experiments/
│       └── .gitkeep           # Run outputs (run_<timestamp>.csv)
└── tests/
    └── test_smoke.py          # Smoke tests (imports and schema validation, zero API cost)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- An OpenAI API key

### 2. Install Dependencies

Using `pip` in a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -e .
```

Alternatively, install directly via `pip`:
```bash
pip install ragas datasets openai pytest
```

### 3. Configure OpenAI API Key

The evaluation runner requires `OPENAI_API_KEY` in the environment to calculate LLM-assisted metrics:

- **Windows (PowerShell)**:
  ```powershell
  $env:OPENAI_API_KEY = "your-openai-api-key"
  ```
- **Windows (CMD)**:
  ```cmd
  set OPENAI_API_KEY=your-openai-api-key
  ```
- **macOS / Linux**:
  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  ```

---

## Running Evaluations

To run the RAGAS evaluation against `evals/datasets/sample.jsonl`:

```bash
python evals.py
```

### What happens during execution:
1. **Environment Check**: Verifies `OPENAI_API_KEY` is present. If missing, it exits with a clear instruction message.
2. **Dataset Ingestion**: Loads JSONL samples into a Hugging Face `Dataset` with standard fields (`user_input`, `response`, `retrieved_contexts`, `reference`).
3. **Metric Calculation**: Evaluates the dataset across the 4 core RAGAS metrics.
4. **Summary Table**: Prints a console table with **Mean Score** and **Min Score** for each metric.
5. **CSV Artifact**: Saves full per-row scores to `evals/experiments/run_<YYYYMMDD_HHMMSS>.csv`.
6. **Debugging Flags**: Flags and prints detailed diagnostics for any row where:
   - `faithfulness < 0.7`
   - `context_recall < 0.8`

---

## RAGAS Metrics Explained

The evaluation suite benchmarks RAG quality across retrieval and generation:

| Metric | Target Dimension | Description | Typical Target |
| :--- | :--- | :--- | :--- |
| **`faithfulness`** | Generation (Hallucination) | Measures the factual consistency of the generated `response` against the `retrieved_contexts`. Evaluates whether every claim in the response is supported by the retrieved passages. | $\ge 0.70$ |
| **`answer_relevancy`** | Generation (Conciseness) | Measures how directly and completely the generated `response` addresses the `user_input`. Generates reverse questions from the response and computes similarity with the prompt. | $\ge 0.80$ |
| **`context_precision`** | Retrieval (Signal-to-Noise) | Measures whether relevant chunks in `retrieved_contexts` are ranked at the top. Higher precision indicates that useful information is prioritized over noise. | $\ge 0.80$ |
| **`context_recall`** | Retrieval (Coverage) | Measures whether all necessary facts from the `reference` ground truth were successfully retrieved in `retrieved_contexts`. | $\ge 0.80$ |

---

## Running Smoke Tests

Smoke tests validate repository structure, dataset schemas, and module imports **without invoking external APIs or consuming tokens**:

```bash
# Using pytest
pytest

# Or running the test module directly
python tests/test_smoke.py
```

---

## Next Steps: Connecting Real RAG / MCP

`rag.py` currently exports `query_rag(query: str) -> dict`. To evaluate a live retrieval pipeline or MCP server:
1. Update `rag.py` to route queries to your MCP client or vector database retriever.
2. Generate an evaluation dataset using your pipeline's outputs.
3. Run `python evals.py` to benchmark performance over iterations.
