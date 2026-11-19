# rag-eval-ragas (ragviews)

Production-ready Retrieval-Augmented Generation (RAG) system built with **LangChain**, **OpenAI**, **FAISS**, and benchmarked using **Ragas**.

---

## Project Structure

```
rag-eval-ragas/
├── ingest.py                  # Ingests data/*.txt, splits into chunks, builds & saves FAISS index
├── rag.py                     # Production RAG pipeline (loads cached FAISS, retrieves top 4, strict prompt, GPT-4o-mini)
├── evals.py                   # Dynamic Ragas evaluation runner with threshold checks
├── validate_pipeline.py       # Comprehensive Part 6 validation test suite
├── pyproject.toml             # Project dependencies (LangChain, FAISS, OpenAI, Ragas, Pytest)
├── .env.example               # Environment variable configuration template
├── README.md                  # System documentation and usage guide
├── .gitignore                 # Standard Python, virtualenv, and .env ignore rules
├── data/                      # Source knowledge base documents (.txt)
│   ├── rag_overview.txt
│   ├── ragas_metrics.txt
│   └── vector_search.txt
├── faiss_index/               # Local FAISS vector database
│   ├── index.faiss
│   └── index.pkl
├── evals/
│   ├── datasets/
│   │   └── sample.jsonl       # Seed evaluation questions and ground truth references
│   ├── thresholds.json        # Pass/fail threshold definitions for Ragas metrics
│   └── experiments/
│       └── .gitkeep           # Run benchmark CSV artifacts
└── tests/
    └── test_smoke.py          # Smoke tests (imports, schemas & retrieval validation)
```

---

## Architecture & Workflow

### 1. Ingestion Pipeline (`ingest.py`)
- Reads all `.txt` documents from `data/` using LangChain's `DirectoryLoader` and `TextLoader`.
- Splits text using `RecursiveCharacterTextSplitter` with `chunk_size = 500` and `chunk_overlap = 50`.
- Generates 1536-dimensional embeddings with `OpenAIEmbeddings(model="text-embedding-3-small")`.
- Builds a FAISS vector index, cleanly overwriting any existing index to eliminate duplicates.
- Serializes the vector database locally to `./faiss_index`.

### 2. RAG Query Pipeline (`rag.py`)
- **One-time Loading & Memory Caching**: Loads the FAISS vector database from `./faiss_index` only once and keeps it in memory across queries.
- **Top-4 Chunk Retrieval**: Retrieves the top 4 most relevant context chunks for any incoming user question.
- **Strict Grounded Prompt**:
  ```
  You are a question-answering assistant.

  Answer ONLY using the supplied context.

  Do not use outside knowledge.

  If the answer cannot be found in the provided context, respond exactly with:

  "I don't know based on the provided documents."

  Context:

  {context}

  Question:

  {question}
  ```
- **Language Model**: Uses `ChatOpenAI(model="gpt-4o-mini")` with zero temperature for grounded synthesis.
- **Return Contract**:
  ```python
  {
      "answer": "...",
      "contexts": [
          "...",
          "...",
          "...",
          "..."
      ]
  }
  ```

---

## Configuration & Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure your OpenAI API key in `.env`:

```ini
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
```

---

## Installation & Execution

### 1. Install Dependencies

```bash
# In virtual environment:
pip install -e .
```

Or install with pip:

```bash
pip install langchain langchain-openai langchain-community faiss-cpu python-dotenv ragas datasets openai pytest
```

### 2. Build the FAISS Vector Database

```bash
python ingest.py
```

Output:
```
Loading documents...
Loaded 3 document(s) from C:\MY PROJECTS\rag-eval-ragas\data.
Splitting...
Created 16 text chunk(s).
Creating embeddings...
Saving FAISS...
Done.
```

### 3. Run a Query

```bash
python rag.py
```

### 4. Run Ragas Evaluations

```bash
python evals.py --check
```

### 5. Run Verification & Tests

```bash
# Part 6 comprehensive validation
python validate_pipeline.py

# Pytest smoke test suite
pytest
```
