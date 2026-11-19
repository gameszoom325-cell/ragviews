"""Centralized Configuration Module for RAG Pipeline.

Provides typed, environment-overridable settings for ingestion, embedding,
vector storage, retrieval, synthesis, and security.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "faiss_index"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# Grounded Strict Prompt Template (Phase 6 compliant)
STRICT_PROMPT_TEMPLATE: str = """You are a retrieval QA assistant.

Use ONLY the supplied context.

Never use prior knowledge.

If the answer cannot be found in the context, respond exactly:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}"""

DEFAULT_FALLBACK_ANSWER: str = "I don't know based on the provided documents."


@dataclass(frozen=True)
class RAGConfig:
    """Immutable production configuration container."""

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DEFAULT_DATA_DIR
    index_dir: Path = DEFAULT_INDEX_DIR
    env_file: Path = DEFAULT_ENV_FILE

    # Text Splitting
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Generation (LLM)
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    # Retrieval
    top_k: int = 4
    score_threshold: Optional[float] = None

    # Security & Input Validation
    max_query_length: int = 2000
    allow_dangerous_deserialization: bool = True

    # Logging
    log_level: str = "INFO"

    # Prompt Template
    prompt_template: str = STRICT_PROMPT_TEMPLATE
    fallback_answer: str = DEFAULT_FALLBACK_ANSWER

    @classmethod
    def from_env(cls) -> RAGConfig:
        """Create configuration instance with environment variable overrides."""
        data_dir_env = os.getenv("RAG_DATA_DIR")
        index_dir_env = os.getenv("RAG_INDEX_DIR")
        env_file_env = os.getenv("RAG_ENV_FILE")

        data_dir = Path(data_dir_env).resolve() if data_dir_env else DEFAULT_DATA_DIR
        index_dir = Path(index_dir_env).resolve() if index_dir_env else DEFAULT_INDEX_DIR
        env_file = Path(env_file_env).resolve() if env_file_env else DEFAULT_ENV_FILE

        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "500"))
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
        embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        chat_model = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("RAG_TEMPERATURE", "0.0"))
        top_k = int(os.getenv("RAG_TOP_K", "4"))
        max_query_length = int(os.getenv("RAG_MAX_QUERY_LENGTH", "2000"))
        log_level = os.getenv("RAG_LOG_LEVEL", "INFO").upper()

        return cls(
            project_root=PROJECT_ROOT,
            data_dir=data_dir,
            index_dir=index_dir,
            env_file=env_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            chat_model=chat_model,
            temperature=temperature,
            top_k=top_k,
            max_query_length=max_query_length,
            log_level=log_level,
        )


_CONFIG_INSTANCE: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Retrieve or initialize singleton configuration instance."""
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = RAGConfig.from_env()
    return _CONFIG_INSTANCE


def reset_config() -> None:
    """Reset configuration instance (useful for testing with modified environments)."""
    global _CONFIG_INSTANCE
    _CONFIG_INSTANCE = None
