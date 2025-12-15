"""
Caching layer for RAG system using Redis.

This module provides multi-level caching to improve performance and reduce API costs.
"""

import json
from typing import Any, Optional

from config.settings import settings
from src.core.exceptions import CacheConnectionError, CacheOperationError
from src.core.logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class CacheManager(LoggerMixin):
    """
    Manages caching operations with Redis.

    Supports:
    - Query cache (full responses)
    - Embedding cache (vectors)
    - Retrieval cache (document IDs)
    """

    def __init__(self, enabled: bool | None = None):
        """
        Initialize cache manager.

        Args:
            enabled: Override settings.enable_cache if provided
        """
        self.enabled = enabled if enabled is not None else settings.enable_cache
        self.redis_client = None

        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis

            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            self.redis_client.ping()
            self.logger.info("Connected to Redis cache")

        except ImportError:
            self.logger.warning("Redis package not installed. Caching disabled.")
            self.enabled = False

        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
            self.enabled = False
            self.redis_client = None

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                self.logger.debug(f"Cache hit: {key}")
                return json.loads(value)

            self.logger.debug(f"Cache miss: {key}")
            return None

        except Exception as e:
            self.logger.error(f"Cache get error for key '{key}': {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (uses settings.cache_ttl_seconds if None)

        Returns:
            True if successful
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            ttl = ttl or settings.cache_ttl_seconds
            serialized_value = json.dumps(value)

            self.redis_client.setex(key, ttl, serialized_value)
            self.logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            self.logger.error(f"Cache set error for key '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            self.redis_client.delete(key)
            self.logger.debug(f"Cache delete: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Cache delete error for key '{key}': {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "query:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0

        try:
            keys = list(self.redis_client.scan_iter(match=pattern))
            if keys:
                deleted = self.redis_client.delete(*keys)
                self.logger.info(f"Cleared {deleted} cache keys matching '{pattern}'")
                return deleted

            return 0

        except Exception as e:
            self.logger.error(f"Cache clear pattern error for '{pattern}': {e}")
            return 0

    def clear_all(self) -> bool:
        """
        Clear all cache keys.

        Returns:
            True if successful
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            self.redis_client.flushdb()
            self.logger.warning("Cleared all cache keys")
            return True

        except Exception as e:
            self.logger.error(f"Cache clear all error: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.enabled or not self.redis_client:
            return {"enabled": False}

        try:
            info = self.redis_client.info("stats")
            return {
                "enabled": True,
                "keys": self.redis_client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0), info.get("keyspace_misses", 0)
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}

    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


# Global cache instance
cache = CacheManager()


# ============================================================================
# Cache Key Generators
# ============================================================================


def generate_query_cache_key(query: str, **kwargs) -> str:
    """
    Generate cache key for query results.

    Args:
        query: User query
        **kwargs: Additional parameters (top_k, reranking, etc.)

    Returns:
        Cache key
    """
    from src.core.utils import generate_hash

    # Include query and relevant parameters in hash
    params_str = json.dumps(kwargs, sort_keys=True)
    hash_input = f"{query}:{params_str}"
    key_hash = generate_hash(hash_input, "md5")[:16]

    return f"query:{key_hash}"


def generate_embedding_cache_key(text: str, model: str) -> str:
    """
    Generate cache key for embeddings.

    Args:
        text: Text that was embedded
        model: Embedding model name

    Returns:
        Cache key
    """
    from src.core.utils import generate_hash

    text_hash = generate_hash(text, "md5")[:16]
    return f"embedding:{model}:{text_hash}"


def generate_retrieval_cache_key(query_embedding: list[float], top_k: int) -> str:
    """
    Generate cache key for retrieval results.

    Args:
        query_embedding: Query embedding vector
        top_k: Number of results

    Returns:
        Cache key
    """
    from src.core.utils import generate_hash

    # Hash first/last few values of embedding (for performance)
    embedding_sample = str(query_embedding[:5] + query_embedding[-5:])
    embedding_hash = generate_hash(embedding_sample, "md5")[:16]

    return f"retrieval:{embedding_hash}:{top_k}"


__all__ = [
    "CacheManager",
    "cache",
    "generate_query_cache_key",
    "generate_embedding_cache_key",
    "generate_retrieval_cache_key",
]
