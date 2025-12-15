"""
Embedding generation using OpenAI models.

This module handles vector embedding generation with caching and batch processing.
"""

from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.core.cache import cache, generate_embedding_cache_key
from src.core.exceptions import EmbeddingAPIError, EmbeddingGenerationError
from src.core.logging import LoggerMixin, log_execution_time
from src.core.utils import should_retry_error


class EmbeddingGenerator(LoggerMixin):
    """
    Generates embeddings using OpenAI models.

    Supports:
    - Single and batch embedding generation
    - Automatic caching
    - Retry logic for API failures
    - Multiple embedding models
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize embedding generator.

        Args:
            model: Embedding model name (uses settings if None)
            api_key: OpenAI API key (uses settings if None)
            base_url: OpenAI API base URL (uses settings if None)
        """
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=settings.request_timeout,
        )

        self.logger.info(f"Initialized EmbeddingGenerator with model: {self.model}")

    @log_execution_time
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=lambda retry_state: should_retry_error(retry_state.outcome.exception())
        if retry_state.outcome.failed
        else False,
    )
    def generate_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding for single text.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            Embedding vector

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingGenerationError(text, "Text is empty")

        # Check cache first
        if use_cache and cache.enabled:
            cache_key = generate_embedding_cache_key(text, self.model)
            cached_embedding = cache.get(cache_key)

            if cached_embedding is not None:
                self.logger.debug(f"Retrieved embedding from cache")
                return cached_embedding

        try:
            # Generate embedding
            self.logger.debug(f"Generating embedding for text (length: {len(text)})")

            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )

            embedding = response.data[0].embedding

            # Cache the embedding (with infinite TTL since embeddings are immutable)
            if use_cache and cache.enabled:
                cache_key = generate_embedding_cache_key(text, self.model)
                cache.set(cache_key, embedding, ttl=None)  # Infinite TTL

            self.logger.debug(
                f"Generated embedding with dimension: {len(embedding)}"
            )

            return embedding

        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            raise EmbeddingGenerationError(text[:100], str(e))

    @log_execution_time
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=lambda retry_state: should_retry_error(retry_state.outcome.exception())
        if retry_state.outcome.failed
        else False,
    )
    def generate_embeddings_batch(
        self, texts: list[str], use_cache: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        if not texts:
            return []

        self.logger.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = []
        texts_to_embed = []
        cache_keys = []

        # Check cache for each text
        for text in texts:
            if not text or not text.strip():
                embeddings.append(None)
                continue

            if use_cache and cache.enabled:
                cache_key = generate_embedding_cache_key(text, self.model)
                cached_embedding = cache.get(cache_key)

                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    continue

            # Need to generate embedding for this text
            texts_to_embed.append(text)
            cache_keys.append(generate_embedding_cache_key(text, self.model))
            embeddings.append(None)  # Placeholder

        # Generate embeddings for uncached texts
        if texts_to_embed:
            try:
                self.logger.debug(
                    f"Generating {len(texts_to_embed)} embeddings via API"
                )

                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts_to_embed,
                )

                generated_embeddings = [item.embedding for item in response.data]

                # Fill in the embeddings and cache them
                j = 0
                for i, embedding in enumerate(embeddings):
                    if embedding is None and texts[i] and texts[i].strip():
                        embeddings[i] = generated_embeddings[j]

                        # Cache the embedding
                        if use_cache and cache.enabled:
                            cache.set(cache_keys[j], generated_embeddings[j], ttl=None)

                        j += 1

                self.logger.info(
                    f"Generated {len(generated_embeddings)} embeddings successfully"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to generate batch embeddings: {e}", exc_info=True
                )
                raise EmbeddingAPIError(str(e))

        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for this model.

        Returns:
            Embedding dimension

        Raises:
            EmbeddingGenerationError: If unable to determine dimension
        """
        # Known dimensions for OpenAI models
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        if self.model in model_dimensions:
            return model_dimensions[self.model]

        # Generate a test embedding to determine dimension
        try:
            test_embedding = self.generate_embedding("test", use_cache=False)
            return len(test_embedding)
        except Exception as e:
            raise EmbeddingGenerationError(
                "test", f"Unable to determine embedding dimension: {e}"
            )


# Default embedder instance
default_embedder = EmbeddingGenerator()


def generate_embedding(text: str, use_cache: bool = True) -> list[float]:
    """
    Generate embedding for text using default embedder.

    Args:
        text: Text to embed
        use_cache: Whether to use cache

    Returns:
        Embedding vector
    """
    return default_embedder.generate_embedding(text, use_cache)


def generate_embeddings_batch(
    texts: list[str], use_cache: bool = True
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts using default embedder.

    Args:
        texts: List of texts to embed
        use_cache: Whether to use cache

    Returns:
        List of embedding vectors
    """
    return default_embedder.generate_embeddings_batch(texts, use_cache)


__all__ = [
    "EmbeddingGenerator",
    "default_embedder",
    "generate_embedding",
    "generate_embeddings_batch",
]
