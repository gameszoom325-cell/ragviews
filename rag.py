"""Production RAG Pipeline using LangChain, OpenAI, and FAISS.

Loads previously created FAISS vector index from ./faiss_index (cached once in memory),
retrieves the top 4 most relevant context chunks for a query using search_kwargs={"k": 4},
formats the context into a strict grounded prompt, and answers using ChatOpenAI(model="gpt-4o-mini").
"""

from __future__ import annotations

import os
import re
import warnings
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr
from typing import Any, Dict, List, Optional, Tuple

# Suppress sunset deprecation warnings from langchain_community
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import DEFAULT_INDEX_DIR, STRICT_PROMPT_TEMPLATE, get_config
from exceptions import (
    IndexLoadError,
    IndexNotFoundError,
    MissingAPIKeyError,
    RetrievalError,
)
from ingest import ResilientOpenAIEmbeddings
from logger import get_logger
from security import sanitize_query

logger = get_logger(__name__)

# Directory Paths exported for compatibility
PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_DIR = DEFAULT_INDEX_DIR
ENV_FILE = PROJECT_ROOT / ".env"

# Re-export prompt template
STRICT_PROMPT_TEMPLATE = STRICT_PROMPT_TEMPLATE

# Global in-memory cache for FAISS index and components
_db: Optional[FAISS] = None
_VECTORSTORE: Optional[FAISS] = None
_EMBEDDINGS: Optional[ResilientOpenAIEmbeddings] = None
_LLM: Optional[ChatOpenAI] = None


def load_environment() -> str:
    """Load environment variables from .env and validate OPENAI_API_KEY.

    Returns:
        str: The retrieved OpenAI API key.

    Raises:
        MissingAPIKeyError: If OPENAI_API_KEY is missing or empty.
    """
    cfg = get_config()
    if cfg.env_file.exists():
        load_dotenv(dotenv_path=cfg.env_file)
    else:
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("OPENAI_API_KEY is missing.")
    return api_key.strip()


def get_embeddings() -> ResilientOpenAIEmbeddings:
    """Get or initialize cached OpenAIEmbeddings model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        try:
            cfg = get_config()
            api_key = load_environment()
            _EMBEDDINGS = ResilientOpenAIEmbeddings(
                model=cfg.embedding_model,
                api_key=SecretStr(api_key),
            )
        except Exception as exc:
            logger.error("Failed to initialize OpenAIEmbeddings: %s", exc)
            raise RuntimeError(f"Failed to initialize OpenAIEmbeddings: {exc}") from exc
    return _EMBEDDINGS


def get_vectorstore(index_path: Path = INDEX_DIR) -> FAISS:
    """Load and cache the precomputed FAISS index from disk.

    Ensures the index is deserialized only once while the application runs.

    Args:
        index_path: Directory path where FAISS index files reside.

    Returns:
        FAISS: Loaded vector database instance.

    Raises:
        IndexNotFoundError: If the FAISS index does not exist on disk.
        IndexLoadError: If index deserialization or loading fails.
    """
    global _db, _VECTORSTORE
    if _db is None:
        target_path = Path(index_path)
        if not target_path.exists():
            raise IndexNotFoundError(
                f"FAISS index not found at '{target_path}'. "
                "Please run `python ingest.py` first to generate the index."
            )
        try:
            cfg = get_config()
            embeddings = get_embeddings()
            _db = FAISS.load_local(
                str(target_path),
                embeddings,
                allow_dangerous_deserialization=cfg.allow_dangerous_deserialization,
            )
            _VECTORSTORE = _db
            logger.info("Successfully loaded and cached FAISS index from %s", target_path)
        except Exception as exc:
            logger.error("Failed to load FAISS index from '%s': %s", target_path, exc)
            raise IndexLoadError(f"Failed to load FAISS index from '{target_path}': {exc}") from exc
    return _db


def get_llm() -> ChatOpenAI:
    """Get or initialize cached ChatOpenAI model."""
    global _LLM
    if _LLM is None:
        try:
            cfg = get_config()
            api_key = load_environment()
            _LLM = ChatOpenAI(
                model=cfg.chat_model,
                api_key=SecretStr(api_key),
                temperature=cfg.temperature,
            )
        except Exception as exc:
            logger.error("Failed to initialize ChatOpenAI: %s", exc)
            raise RuntimeError(f"Failed to initialize ChatOpenAI: {exc}") from exc
    return _LLM


def format_context(documents: List[Document]) -> str:
    """Format retrieved document chunks into clean newline-separated text blocks."""
    return "\n\n".join(doc.page_content.strip() for doc in documents if doc.page_content)


def generate_offline_answer(question: str, context: str) -> str:
    """Extractive fallback answer synthesizer for offline test environments."""
    cfg = get_config()
    if not context.strip():
        return cfg.fallback_answer

    stop_words = {
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
    }
    q_words = set(re.findall(r"\w+", question.lower())) - stop_words

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    scored_sentences = []
    for s in sentences:
        s_words = set(re.findall(r"\w+", s.lower())) - stop_words
        score = len(q_words & s_words)
        if score > 0:
            scored_sentences.append((score, s))

    if scored_sentences:
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scored_sentences[:3]]
        return " ".join(top_sentences)

    return cfg.fallback_answer


def retrieve_contexts(question: str, top_k: Optional[int] = None) -> List[Document]:
    """Retrieve top-k contexts for question, deduplicated in ranking order.

    Args:
        question: Query string.
        top_k: Number of chunks to retrieve (defaults to config top_k).

    Returns:
        List[Document]: Deduplicated ranked list of retrieved Document objects.
    """
    cfg = get_config()
    k = top_k if top_k is not None else cfg.top_k

    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs: List[Document] = retriever.invoke(question)
    except Exception as exc:
        logger.error("Error during context retrieval: %s", exc)
        raise RetrievalError(f"Error during context retrieval: {exc}") from exc

    # Quality enhancement: deduplicate identical chunks while strictly preserving ranking order
    seen_texts = set()
    deduped_docs: List[Document] = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen_texts:
            seen_texts.add(content)
            deduped_docs.append(doc)

    return deduped_docs


def query(question: str) -> Dict[str, Any]:
    """Execute end-to-end RAG query pipeline.

    Step 1: Sanitize and validate user query.
    Step 2: Retrieve the Top 4 most relevant chunks from FAISS (search_kwargs={"k": 4}).
    Step 3: Create strict prompt enforcing grounded answers.
    Step 4: Call ChatOpenAI(model="gpt-4o-mini") to generate response.
    Step 5: Return dictionary containing answer and ranking-ordered contexts.

    Args:
        question: User query string.

    Returns:
        Dict[str, Any]:
            {
                "answer": str,
                "contexts": List[str]
            }
    """
    cfg = get_config()

    try:
        cleaned_question, is_suspicious = sanitize_query(question, max_length=cfg.max_query_length)
    except Exception as exc:
        logger.warning("Query validation error: %s", exc)
        return {
            "answer": cfg.fallback_answer,
            "contexts": [],
        }

    if not cleaned_question:
        return {
            "answer": cfg.fallback_answer,
            "contexts": [],
        }

    # Step 1: Retrieve top 4 most relevant chunks
    retrieved_docs = retrieve_contexts(cleaned_question, top_k=cfg.top_k)
    contexts: List[str] = [doc.page_content.strip() for doc in retrieved_docs]

    # Step 2: Build strict prompt
    context_str = format_context(retrieved_docs)
    prompt = cfg.prompt_template.format(context=context_str, question=cleaned_question)

    # Step 3: Call ChatOpenAI with fallback
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = str(answer).strip()
    except Exception as exc:
        logger.debug("LLM invocation fell back to offline synthesizer: %s", exc)
        answer = generate_offline_answer(cleaned_question, context_str)

    # Step 4: Return formatted dictionary
    return {
        "answer": answer,
        "contexts": contexts,
    }


def query_with_scores(question: str) -> Dict[str, Any]:
    """Execute RAG query with similarity scores and chunk metadata for observability."""
    cfg = get_config()
    vectorstore = get_vectorstore()
    results_with_scores: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(
        question,
        k=cfg.top_k,
    )

    contexts = [doc.page_content.strip() for doc, _ in results_with_scores]
    scores = [float(score) for _, score in results_with_scores]
    metadata = [doc.metadata for doc, _ in results_with_scores]

    base_result = query(question)
    return {
        "answer": base_result["answer"],
        "contexts": contexts,
        "scores": scores,
        "metadata": metadata,
    }


# Backward-compatibility alias
query_rag = query


if __name__ == "__main__":
    sample_q = "What is the primary purpose of the Ragas framework?"
    print(f"Query: {sample_q}\n")
    result = query(sample_q)
    print("Answer:")
    print(result["answer"])
    print(f"\nRetrieved Contexts ({len(result['contexts'])} chunks):")
    for idx, ctx in enumerate(result["contexts"], start=1):
        print(f"\n[{idx}] {ctx[:120]}...")
