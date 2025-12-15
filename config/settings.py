"""
Configuration Management for RAG System

This module provides centralized configuration using Pydantic Settings.
All configuration is loaded from environment variables with validation.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============================================================================
    # LLM Configuration
    # ============================================================================
    openai_api_key: str = Field(..., description="OpenAI/GitHub Models API key")
    openai_base_url: str = Field(
        default="https://models.inference.ai.azure.com",
        description="OpenAI API base URL",
    )
    model_name: str = Field(default="gpt-4o", description="Primary LLM model name")
    fallback_model_name: str = Field(
        default="gpt-4o-mini", description="Fallback model for cost optimization"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )

    # ============================================================================
    # Cohere Configuration
    # ============================================================================
    cohere_api_key: str = Field(..., description="Cohere API key for reranking")
    rerank_model: str = Field(default="rerank-v3.5", description="Reranking model")

    # ============================================================================
    # Vector Database Configuration
    # ============================================================================
    chroma_persist_dir: Path = Field(
        default=Path("./data/vector_db"), description="Chroma persistence directory"
    )
    chroma_collection_name: str = Field(
        default="documents", description="Chroma collection name"
    )

    # ============================================================================
    # Chunking Configuration
    # ============================================================================
    chunk_size: int = Field(
        default=512, ge=100, le=2048, description="Chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=500, description="Overlap between chunks in tokens"
    )
    min_chunk_size: int = Field(
        default=100, ge=50, le=500, description="Minimum chunk size"
    )

    # ============================================================================
    # Retrieval Configuration
    # ============================================================================
    retrieval_top_k: int = Field(
        default=5, ge=1, le=50, description="Number of documents to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    enable_reranking: bool = Field(
        default=True, description="Enable Cohere reranking"
    )
    rerank_top_n: int = Field(
        default=3, ge=1, le=20, description="Top N after reranking"
    )

    # ============================================================================
    # Cache Configuration
    # ============================================================================
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )
    enable_cache: bool = Field(default=False, description="Enable caching layer")
    cache_ttl_seconds: int = Field(
        default=3600, ge=60, description="Cache TTL in seconds"
    )

    # ============================================================================
    # Application Configuration
    # ============================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    data_dir: Path = Field(
        default=Path("./data/documents"), description="Document storage directory"
    )
    max_file_size_mb: int = Field(
        default=50, ge=1, le=500, description="Maximum file size in MB"
    )
    allowed_extensions: str = Field(
        default="pdf,txt,csv,docx,xlsx", description="Allowed file extensions"
    )

    # ============================================================================
    # Performance & Limits
    # ============================================================================
    max_tokens: int = Field(
        default=4096, ge=256, le=128000, description="Maximum tokens for LLM"
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="LLM temperature"
    )
    request_timeout: int = Field(
        default=30, ge=5, le=300, description="API request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, ge=1, le=10, description="Maximum API retry attempts"
    )

    # ============================================================================
    # UI Configuration
    # ============================================================================
    streamlit_server_port: int = Field(
        default=8501, ge=1024, le=65535, description="Streamlit server port"
    )
    streamlit_server_address: str = Field(
        default="localhost", description="Streamlit server address"
    )

    # ============================================================================
    # Hybrid Search Configuration (Phase 2)
    # ============================================================================
    enable_hybrid_search: bool = Field(
        default=True, description="Enable hybrid BM25+Vector search"
    )
    hybrid_alpha: float = Field(
        default=0.5, ge=0.0, le=1.0, 
        description="Weight for vector vs BM25 (1.0=pure vector, 0.0=pure BM25)"
    )

    # ============================================================================
    # Conversation Memory Configuration (Phase 2)
    # ============================================================================
    enable_memory: bool = Field(
        default=True, description="Enable conversation memory for multi-turn chat"
    )
    memory_max_turns: int = Field(
        default=10, ge=1, le=50, description="Maximum conversation turns to keep"
    )
    memory_max_tokens: int = Field(
        default=4000, ge=500, le=16000, description="Maximum tokens for conversation context"
    )

    # ============================================================================
    # API Configuration (Phase 2)
    # ============================================================================
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API server port")

    @field_validator("chroma_persist_dir", "data_dir")
    @classmethod
    def ensure_directory_exists(cls, v: Path) -> Path:
        """Ensure directories exist, create if they don't."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def allowed_extensions_list(self) -> list[str]:
        """Get allowed extensions as a list."""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    def is_file_allowed(self, filename: str) -> bool:
        """Check if a file extension is allowed."""
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return extension in self.allowed_extensions_list


# Global settings instance
settings = Settings()


# Export for easy import
__all__ = ["settings", "Settings"]
