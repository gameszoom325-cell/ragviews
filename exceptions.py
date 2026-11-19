"""Domain-Specific Exceptions for RAG System.

Provides structured, hierarchical exception classes for vector indexing,
retrieval, embedding, language modeling, and environment configuration.
"""

from __future__ import annotations


class RAGException(Exception):
    """Base exception for all RAG pipeline errors."""


class MissingAPIKeyError(RAGException, RuntimeError):
    """Raised when OPENAI_API_KEY is not defined or empty.

    Inherits from RuntimeError to maintain strict backward compatibility.
    """


class IndexNotFoundError(RAGException, FileNotFoundError):
    """Raised when FAISS index files do not exist at expected path.

    Inherits from FileNotFoundError to maintain backward compatibility.
    """


class IndexLoadError(RAGException, RuntimeError):
    """Raised when FAISS index loading or deserialization fails."""


class DocumentIngestionError(RAGException, ValueError):
    """Raised when source documents cannot be loaded or split."""


class RetrievalError(RAGException, RuntimeError):
    """Raised when nearest-neighbor context retrieval fails."""


class LLMGenerationError(RAGException, RuntimeError):
    """Raised when the language model fails during answer generation."""


class InputValidationError(RAGException, ValueError):
    """Raised when query input fails validation or security checks."""
