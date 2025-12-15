"""
Utility functions for RAG system.

This module contains helper functions used throughout the application.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import tiktoken

from config.settings import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Text Processing
# ============================================================================


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove multiple newlines
    text = re.sub(r"\n+", "\n", text)

    # Strip whitespace
    text = text.strip()

    return text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


# ============================================================================
# Token Counting
# ============================================================================


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text for a specific model.

    Args:
        text: Text to count tokens for
        model: Model name (for tokenizer selection)

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def estimate_tokens(text: str) -> int:
    """
    Quick token estimation (4 chars ≈ 1 token).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def split_text_by_tokens(
    text: str, max_tokens: int, model: str = "gpt-4", overlap: int = 0
) -> list[str]:
    """
    Split text into chunks by token count.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        model: Model for tokenization
        overlap: Overlap between chunks in tokens

    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move start position considering overlap
        start = end - overlap if overlap > 0 else end

    return chunks


# ============================================================================
# File Operations
# ============================================================================


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.

    Args:
        filename: Filename or path

    Returns:
        Lowercase file extension (without dot)
    """
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def validate_file(file_path: Path) -> tuple[bool, str]:
    """
    Validate if file can be processed.

    Args:
        file_path: Path to file

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    # Check file size
    size_mb = get_file_size_mb(file_path)
    if size_mb > settings.max_file_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max: {settings.max_file_size_mb}MB)"

    # Check file extension
    extension = get_file_extension(file_path.name)
    if extension not in settings.allowed_extensions_list:
        return (
            False,
            f"Unsupported file type: {extension} (allowed: {', '.join(settings.allowed_extensions_list)})",
        )

    return True, ""


# ============================================================================
# Hashing
# ============================================================================


def generate_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of text.

    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        Hash hexdigest
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode("utf-8"))
    return hash_obj.hexdigest()


def generate_doc_id(filename: str, content: str | None = None) -> str:
    """
    Generate unique document ID.

    Args:
        filename: Document filename
        content: Optional content to include in hash

    Returns:
        Document ID
    """
    if content:
        # Hash filename + content
        hash_input = f"{filename}:{content[:1000]}"  # Use first 1000 chars
    else:
        # Hash only filename
        hash_input = filename

    return f"doc_{generate_hash(hash_input, 'md5')[:16]}"


# ============================================================================
# Data Formatting
# ============================================================================


def format_metadata(metadata: dict[str, Any]) -> str:
    """
    Format metadata dictionary as readable string.

    Args:
        metadata: Metadata dictionary

    Returns:
        Formatted metadata string
    """
    lines = []
    for key, value in metadata.items():
        # Convert key to title case
        display_key = key.replace("_", " ").title()
        lines.append(f"{display_key}: {value}")

    return "\n".join(lines)


def format_sources(source_ids: list[str]) -> str:
    """
    Format source IDs as citation string.

    Args:
        source_ids: List of source document IDs

    Returns:
        Formatted citation string
    """
    if not source_ids:
        return ""

    unique_ids = list(dict.fromkeys(source_ids))  # Remove duplicates, preserve order
    return "[" + ", ".join(unique_ids) + "]"


# ============================================================================
# Validation
# ============================================================================


def is_valid_query(query: str, min_length: int = 3, max_length: int = 1000) -> bool:
    """
    Validate user query.

    Args:
        query: User query
        min_length: Minimum query length
        max_length: Maximum query length

    Returns:
        True if valid
    """
    if not query or not query.strip():
        return False

    query_length = len(query.strip())
    return min_length <= query_length <= max_length


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be safe for filesystem.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip(". ")

    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"

    return filename


# ============================================================================
# Retry Logic
# ============================================================================


def should_retry_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.

    Args:
        error: Exception that occurred

    Returns:
        True if should retry
    """
    # Retry on network errors, timeouts, rate limits
    retry_keywords = ["timeout", "connection", "rate limit", "429", "503", "502"]

    error_str = str(error).lower()
    return any(keyword in error_str for keyword in retry_keywords)


__all__ = [
    "clean_text",
    "truncate_text",
    "count_tokens",
    "estimate_tokens",
    "split_text_by_tokens",
    "get_file_size_mb",
    "get_file_extension",
    "validate_file",
    "generate_hash",
    "generate_doc_id",
    "format_metadata",
    "format_sources",
    "is_valid_query",
    "sanitize_filename",
    "should_retry_error",
]
