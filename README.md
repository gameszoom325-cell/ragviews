# rag-eval-ragas

Evaluation pipeline for Retrieval-Augmented Generation (RAG) applications using [Ragas](https://github.com/explodinggradients/ragas) integrated with a live **Model Context Protocol (MCP)** RAG server.

This project queries your real MCP retrieval server dynamically for context chunks, builds dummy concatenated answers (ready for LLM generation), and benchmarks retrieval/generation quality across 4 core RAGAS metrics.

---

## Project Structure

```
rag-eval-ragas/
├── rag.py                     # MCP RAG client (connects to MCP server, calls retrieval tool)
├── evals.py                   # Dynamic RAGAS evaluation runner
├── pyproject.toml             # Project dependencies (ragas, datasets, openai, pytest, mcp, python-dotenv)
├── .env.example               # Environment variable configuration template
├── README.md                  # Setup, server launch, and evaluation guide
├── .gitignore                 # Standard Python, virtualenv, and .env ignore rules
├── evals/
│   ├── datasets/
│   │   └── sample.jsonl       # Seed evaluation questions and ground truth references
│   └── experiments/
│       └── .gitkeep           # Run benchmark CSV artifacts (run_<timestamp>.csv)
└── tests/
    └── test_smoke.py          # CI smoke tests (imports, schemas & mocked MCP calls)
```

---

## Configuration & Environment Variables

Create a `.env` file in the project root (you can copy `.env.example`):

```bash
cp .env.example .env
```

Set the required environment variables in `.env`:

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key used by Ragas LLM evaluators. |
| `MCP_SERVER_URL` | **Yes** | `http://localhost:8000/mcp` | SSE endpoint URL or stdio command for your MCP server. |
| `MCP_TOOL_NAME` | No | Auto-detected | Specific retrieval tool name (auto-discovers `retrieve`, `search`, etc.). |
| `MCP_TIMEOUT` | No | `30` | Per-query timeout in seconds. |

### Example `.env`

```ini
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
MCP_SERVER_URL=http://localhost:8000/mcp
MCP_TIMEOUT=30
```

> **Note**: If your MCP server runs via stdio rather than HTTP SSE, you can set `MCP_SERVER_URL` to the launch command, e.g.:
> ```ini
> MCP_SERVER_URL=python server.py
> ```

---

## Starting Your MCP RAG Server

Before running evaluations, ensure your MCP server is running and accessible:

```bash
# In your MCP server directory / environment:
python server.py
```

*By default, the client expects the server to listen on `http://localhost:8000/mcp` or as configured in `MCP_SERVER_URL`.*

If the server is unreachable or offline, `rag.py` and `evals.py` will catch the error and display an actionable troubleshooting message rather than crashing with an unhandled exception.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- An OpenAI API key
- A running MCP RAG server

### 2. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install all dependencies (including mcp and python-dotenv)
pip install -e .
```

Or install dependencies directly:
```bash
pip install ragas datasets openai pytest mcp python-dotenv
```

---

## Running Evaluations

Once your MCP server is up and `.env` is configured:

```bash
python evals.py
```

### Execution Flow:
1. **Validation**: Confirms `OPENAI_API_KEY` is set.
2. **Dynamic Ingestion**: Reads question-reference pairs from `evals/datasets/sample.jsonl`.
3. **Live MCP Querying**: For every question, calls `rag.query(question)` with a 30s timeout to fetch live retrieved chunks.
4. **Answer Synthesis**: Formulates dummy answers by concatenating the top 3 retrieved chunks.
5. **Ragas Evaluation**: Computes the 4 core metrics using the OpenAI backend.
6. **Summary Table**: Prints a summary with Mean Score and Min Score per metric.
7. **CSV Export**: Writes detailed per-sample scores to `evals/experiments/run_<YYYYMMDD_HHMMSS>.csv`.
8. **Threshold Alerts**: Flags any sample where `faithfulness < 0.7` or `context_recall < 0.8` for debugging.

---

## RAGAS Metrics Reference

| Metric | Target | Description |
| :--- | :---: | :--- |
| **`faithfulness`** | $\ge 0.70$ | Factual consistency: checks whether claims in the generated response are grounded in the retrieved context chunks. |
| **`answer_relevancy`** | $\ge 0.80$ | Relevance: evaluates whether the response directly addresses the user question without superfluous noise. |
| **`context_precision`** | $\ge 0.80$ | Ranking signal-to-noise: measures if relevant chunks are ranked higher in the retrieved list. |
| **`context_recall`** | $\ge 0.80$ | Retrieval coverage: checks if all key facts from the reference ground truth were captured in retrieved contexts. |

---

## Running Smoke Tests

Smoke tests validate file schemas, imports, and interface formatting **offline without API token costs or live MCP servers**:

```bash
pytest
```
*Or execute standalone:*
```bash
python tests/test_smoke.py
```
