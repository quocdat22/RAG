"""
Custom exceptions for RAG system.

This module defines a hierarchy of custom exceptions for better error handling
and debugging throughout the application.
"""


class RAGException(Exception):
    """Base exception for all RAG system errors."""

    def __init__(self, message: str, error_code: str = "RAG_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigurationError(RAGException):
    """Raised when there's a configuration issue."""

    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")


class MissingAPIKeyError(ConfigurationError):
    """Raised when required API keys are missing."""

    def __init__(self, key_name: str):
        super().__init__(
            f"Missing required API key: {key_name}. Please check your .env file.",
            "MISSING_API_KEY",
        )


# ============================================================================
# Document Processing Errors
# ============================================================================


class DocumentError(RAGException):
    """Base exception for document processing errors."""

    pass


class DocumentLoadError(DocumentError):
    """Raised when a document cannot be loaded."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"Failed to load document '{filename}': {reason}", "DOCUMENT_LOAD_ERROR"
        )


class UnsupportedFileTypeError(DocumentError):
    """Raised when file type is not supported."""

    def __init__(self, filename: str, file_type: str):
        super().__init__(
            f"Unsupported file type '{file_type}' for file '{filename}'",
            "UNSUPPORTED_FILE_TYPE",
        )


class FileTooLargeError(DocumentError):
    """Raised when file size exceeds maximum allowed."""

    def __init__(self, filename: str, size_mb: float, max_size_mb: int):
        super().__init__(
            f"File '{filename}' ({size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)",
            "FILE_TOO_LARGE",
        )


class ChunkingError(DocumentError):
    """Raised when document chunking fails."""

    def __init__(self, reason: str):
        super().__init__(f"Failed to chunk document: {reason}", "CHUNKING_ERROR")


# ============================================================================
# Embedding Errors
# ============================================================================


class EmbeddingError(RAGException):
    """Base exception for embedding errors."""

    pass


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""

    def __init__(self, text_preview: str, reason: str):
        preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        super().__init__(
            f"Failed to generate embedding for text '{preview}': {reason}",
            "EMBEDDING_GENERATION_ERROR",
        )


class EmbeddingAPIError(EmbeddingError):
    """Raised when embedding API call fails."""

    def __init__(self, reason: str):
        super().__init__(
            f"Embedding API request failed: {reason}", "EMBEDDING_API_ERROR"
        )


# ============================================================================
# Vector Store Errors
# ============================================================================


class VectorStoreError(RAGException):
    """Base exception for vector store errors."""

    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when cannot connect to vector store."""

    def __init__(self, reason: str):
        super().__init__(
            f"Failed to connect to vector store: {reason}",
            "VECTOR_STORE_CONNECTION_ERROR",
        )


class VectorStoreIndexError(VectorStoreError):
    """Raised when indexing fails."""

    def __init__(self, doc_id: str, reason: str):
        super().__init__(
            f"Failed to index document '{doc_id}': {reason}",
            "VECTOR_STORE_INDEX_ERROR",
        )


class VectorStoreQueryError(VectorStoreError):
    """Raised when query fails."""

    def __init__(self, reason: str):
        super().__init__(
            f"Vector store query failed: {reason}", "VECTOR_STORE_QUERY_ERROR"
        )


# ============================================================================
# Retrieval Errors
# ============================================================================


class RetrievalError(RAGException):
    """Base exception for retrieval errors."""

    pass


class QueryProcessingError(RetrievalError):
    """Raised when query processing fails."""

    def __init__(self, query: str, reason: str):
        super().__init__(
            f"Failed to process query '{query}': {reason}", "QUERY_PROCESSING_ERROR"
        )


class NoResultsFoundError(RetrievalError):
    """Raised when no relevant results are found."""

    def __init__(self, query: str):
        super().__init__(
            f"No relevant results found for query: '{query}'", "NO_RESULTS_FOUND"
        )


class RerankingError(RetrievalError):
    """Raised when reranking fails."""

    def __init__(self, reason: str):
        super().__init__(f"Reranking failed: {reason}", "RERANKING_ERROR")


# ============================================================================
# LLM Errors
# ============================================================================


class LLMError(RAGException):
    """Base exception for LLM errors."""

    pass


class LLMAPIError(LLMError):
    """Raised when LLM API call fails."""

    def __init__(self, reason: str):
        super().__init__(f"LLM API request failed: {reason}", "LLM_API_ERROR")


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or cannot be parsed."""

    def __init__(self, reason: str):
        super().__init__(f"Invalid LLM response: {reason}", "LLM_RESPONSE_ERROR")


class TokenLimitExceededError(LLMError):
    """Raised when token limit is exceeded."""

    def __init__(self, used_tokens: int, max_tokens: int):
        super().__init__(
            f"Token limit exceeded: {used_tokens} > {max_tokens}",
            "TOKEN_LIMIT_EXCEEDED",
        )


# ============================================================================
# Cache Errors
# ============================================================================


class CacheError(RAGException):
    """Base exception for cache errors."""

    pass


class CacheConnectionError(CacheError):
    """Raised when cannot connect to cache."""

    def __init__(self, reason: str):
        super().__init__(
            f"Failed to connect to cache: {reason}", "CACHE_CONNECTION_ERROR"
        )


class CacheOperationError(CacheError):
    """Raised when cache operation fails."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            f"Cache {operation} operation failed: {reason}", "CACHE_OPERATION_ERROR"
        )


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(RAGException):
    """Base exception for validation errors."""

    pass


class InvalidInputError(ValidationError):
    """Raised when input validation fails."""

    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Invalid input for field '{field}': {reason}", "INVALID_INPUT"
        )


__all__ = [
    "RAGException",
    "ConfigurationError",
    "MissingAPIKeyError",
    "DocumentError",
    "DocumentLoadError",
    "UnsupportedFileTypeError",
    "FileTooLargeError",
    "ChunkingError",
    "EmbeddingError",
    "EmbeddingGenerationError",
    "EmbeddingAPIError",
    "VectorStoreError",
    "VectorStoreConnectionError",
    "VectorStoreIndexError",
    "VectorStoreQueryError",
    "RetrievalError",
    "QueryProcessingError",
    "NoResultsFoundError",
    "RerankingError",
    "LLMError",
    "LLMAPIError",
    "LLMResponseError",
    "TokenLimitExceededError",
    "CacheError",
    "CacheConnectionError",
    "CacheOperationError",
    "ValidationError",
    "InvalidInputError",
]
