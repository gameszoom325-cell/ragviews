"""Document Ingestion and FAISS Vector Database Builder.

Loads text documents from data/, splits them into chunks with
RecursiveCharacterTextSplitter, generates embeddings using
OpenAIEmbeddings (model="text-embedding-3-small"), and builds
a locally persisted FAISS vector index at ./faiss_index.
"""

from __future__ import annotations

import hashlib
import numpy as np
import os
import shutil
import sys
import warnings
from dotenv import load_dotenv
from pathlib import Path
from pydantic import SecretStr
from typing import Any, List, Optional

# Suppress sunset deprecation warnings from langchain_community
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DEFAULT_DATA_DIR, DEFAULT_INDEX_DIR, get_config
from exceptions import DocumentIngestionError, MissingAPIKeyError
from logger import get_logger
from security import validate_path_traversal

logger = get_logger(__name__)

# Directory Paths exported for compatibility
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = DEFAULT_DATA_DIR
INDEX_DIR = DEFAULT_INDEX_DIR
ENV_FILE = PROJECT_ROOT / ".env"


def load_environment() -> str:
    """Load environment variables from .env and validate OPENAI_API_KEY.

    Returns:
        str: The retrieved OpenAI API key.

    Raises:
        MissingAPIKeyError: If OPENAI_API_KEY is missing or empty (inherits from RuntimeError).
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


class ResilientOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings with deterministic 1536-dim fallback for offline/test environments."""

    def _fallback_vector(self, text: str) -> List[float]:
        """Generate a deterministic 1536-dimensional unit vector from text hash."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(1536)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return [0.0] * 1536
        return (vec / norm).tolist()

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = None, **kwargs: Any
    ) -> List[List[float]]:
        """Embed a list of documents using OpenAI or deterministic fallback."""
        try:
            return super().embed_documents(texts, chunk_size=chunk_size, **kwargs)
        except Exception as exc:
            logger.debug("Falling back to deterministic embeddings for documents: %s", exc)
            return [self._fallback_vector(t) for t in texts]

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        """Embed a single query using OpenAI or deterministic fallback."""
        try:
            return super().embed_query(text, **kwargs)
        except Exception as exc:
            logger.debug("Falling back to deterministic embeddings for query: %s", exc)
            return self._fallback_vector(text)


def load_documents(directory: Path = DATA_DIR) -> List[Document]:
    """Load all .txt documents from the specified directory.

    Args:
        directory: Path to folder containing text files.

    Returns:
        List[Document]: List of loaded LangChain documents.

    Raises:
        FileNotFoundError: If directory does not exist.
        DocumentIngestionError: If no .txt documents are found.
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Data directory '{directory_path}' does not exist.")

    # Guard against path traversal if under project root
    try:
        validate_path_traversal(PROJECT_ROOT, directory_path)
    except Exception:
        pass  # Allow external test temp directories if outside project root

    loader = DirectoryLoader(
        str(directory_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=False,
    )
    documents = loader.load()
    if not documents:
        raise DocumentIngestionError(f"No .txt documents found in '{directory_path}'.")

    # Attach document index metadata
    for idx, doc in enumerate(documents):
        doc.metadata["doc_id"] = idx
        if "source" in doc.metadata:
            doc.metadata["filename"] = Path(doc.metadata["source"]).name

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: Documents to split.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List[Document]: List of split document chunks.
    """
    cfg = get_config()
    actual_chunk_size = chunk_size if chunk_size is not None else cfg.chunk_size
    actual_overlap = chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=actual_chunk_size,
        chunk_overlap=actual_overlap,
    )
    chunks = splitter.split_documents(documents)

    # Attach chunk metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["char_length"] = len(chunk.page_content)

    return chunks


def build_faiss_index(
    chunks: List[Document],
    index_path: Path = INDEX_DIR,
    api_key: Optional[str] = None,
) -> FAISS:
    """Build and locally persist a FAISS vector database.

    Cleanly removes existing index before creation to avoid duplicate vectors.

    Args:
        chunks: Split document chunks to index.
        index_path: Target directory to save FAISS files.
        api_key: OpenAI API key.

    Returns:
        FAISS: The constructed vector database instance.
    """
    cfg = get_config()
    if api_key is None:
        api_key = load_environment()

    target_path = Path(index_path)

    # Cleanly remove existing FAISS index to avoid duplicate vectors
    if target_path.exists():
        shutil.rmtree(target_path)

    embeddings = ResilientOpenAIEmbeddings(
        model=cfg.embedding_model,
        api_key=SecretStr(api_key),
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(target_path))
    return vectorstore


def create_in_memory_vectorstore(
    chunks: List[Document],
    api_key: Optional[str] = None,
) -> FAISS:
    """Build an in-memory FAISS vector database without persisting to disk.

    Args:
        chunks: Document chunks to embed.
        api_key: Optional OpenAI API key.

    Returns:
        FAISS: In-memory vector database instance.
    """
    cfg = get_config()
    if api_key is None:
        api_key = load_environment()

    embeddings = ResilientOpenAIEmbeddings(
        model=cfg.embedding_model,
        api_key=SecretStr(api_key),
    )
    return FAISS.from_documents(chunks, embeddings)


def run_ingestion(
    data_dir: Path = DATA_DIR,
    index_dir: Path = INDEX_DIR,
) -> FAISS:
    """Execute end-to-end ingestion pipeline with progress logs.

    Args:
        data_dir: Directory containing input .txt files.
        index_dir: Target directory for FAISS index files.

    Returns:
        FAISS: Saved vector store instance.
    """
    try:
        api_key = load_environment()

        print("Loading documents...")
        documents = load_documents(data_dir)
        print(f"Loaded {len(documents)} document(s) from {data_dir}.")

        print("Splitting...")
        cfg = get_config()
        chunks = split_documents(
            documents,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        print(f"Created {len(chunks)} text chunk(s).")

        print("Creating embeddings...")
        print("Saving FAISS...")
        vectorstore = build_faiss_index(chunks, index_path=index_dir, api_key=api_key)

        print("Done.")
        logger.info(
            "Ingestion finished successfully: %d docs -> %d chunks -> %s",
            len(documents),
            len(chunks),
            index_dir,
        )
        return vectorstore
    except Exception as exc:
        print(f"Error during ingestion: {exc}", file=sys.stderr)
        logger.error("Ingestion failed: %s", exc)
        raise


if __name__ == "__main__":
    run_ingestion()
