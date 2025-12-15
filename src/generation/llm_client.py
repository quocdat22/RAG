"""
LLM Client for text generation using OpenAI models.

This module provides a client for interacting with OpenAI LLMs
with features like streaming, token tracking, retry logic, and fallback models.
"""

from typing import Any, Generator, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.core.exceptions import (
    LLMAPIError,
    LLMResponseError,
    TokenLimitExceededError,
)
from src.core.logging import LoggerMixin, log_execution_time
from src.core.utils import count_tokens, should_retry_error


class LLMClient(LoggerMixin):
    """
    Client for interacting with OpenAI LLMs.

    Supports:
    - Streaming and non-streaming responses
    - Token counting and cost tracking
    - Automatic retry with exponential backoff
    - Model fallback for cost optimization
    - Temperature and parameter control
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Initialize LLM client.

        Args:
            model: Model name (uses settings if None)
            api_key: OpenAI API key (uses settings if None)
            base_url: OpenAI API base URL (uses settings if None)
            temperature: Generation temperature (uses settings if None)
            max_tokens: Maximum tokens to generate (uses settings if None)
        """
        self.model = model or settings.model_name
        self.fallback_model = settings.fallback_model_name
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=settings.request_timeout,
        )

        # Track usage statistics
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

        self.logger.info(
            f"Initialized LLMClient: model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    @log_execution_time
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=lambda retry_state: should_retry_error(retry_state.outcome.exception())
        if retry_state.outcome.failed
        else False,
    )
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_fallback_on_error: bool = True,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            use_fallback_on_error: Use fallback model on error

        Returns:
            Generated text

        Raises:
            LLMAPIError: If API call fails
            TokenLimitExceededError: If token limit exceeded
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Check token limit
        prompt_tokens = count_tokens(prompt, self.model)
        if system_prompt:
            prompt_tokens += count_tokens(system_prompt, self.model)

        if prompt_tokens > max_tokens:
            raise TokenLimitExceededError(prompt_tokens, max_tokens)

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            self.logger.debug(
                f"Generating completion: model={self.model}, "
                f"prompt_tokens={prompt_tokens}, temperature={temperature}"
            )

            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens - prompt_tokens,  # Leave room for response
            )

            # Extract response
            if not response.choices:
                raise LLMResponseError("No choices in response")

            content = response.choices[0].message.content

            if not content:
                raise LLMResponseError("Empty response content")

            # Track usage
            if response.usage:
                self._track_usage(response.usage)

            self.logger.info(
                f"Generated {len(content)} chars "
                f"(tokens: {response.usage.completion_tokens if response.usage else 'unknown'})"
            )

            return content

        except TokenLimitExceededError:
            raise

        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)

            # Try fallback model if enabled
            if use_fallback_on_error and self.fallback_model != self.model:
                self.logger.warning(f"Retrying with fallback model: {self.fallback_model}")
                return self._generate_with_fallback(
                    messages, temperature, max_tokens - prompt_tokens
                )

            raise LLMAPIError(str(e))

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """
        Generate text completion with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Yields:
            Text chunks as they're generated

        Raises:
            LLMAPIError: If API call fails
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            self.logger.debug(f"Starting streaming generation: model={self.model}")

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}", exc_info=True)
            raise LLMAPIError(str(e))

    def _generate_with_fallback(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
    ) -> str:
        """Generate with fallback model."""
        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.fallback_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_completion_tokens,
            )

            if not response.choices or not response.choices[0].message.content:
                raise LLMResponseError("Empty fallback response")

            # Track usage
            if response.usage:
                self._track_usage(response.usage)

            self.logger.info(f"Fallback generation successful with {self.fallback_model}")

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Fallback generation also failed: {e}")
            raise LLMAPIError(f"Both primary and fallback models failed: {str(e)}")

    def _track_usage(self, usage: Any) -> None:
        """Track token usage and estimate cost."""
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens

        # Rough cost estimation (adjust based on actual pricing)
        # GPT-4o: $2.50/1M input, $10/1M output
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        if "mini" in self.model.lower():
            input_cost = usage.prompt_tokens * 0.15 / 1_000_000
            output_cost = usage.completion_tokens * 0.60 / 1_000_000
        else:
            input_cost = usage.prompt_tokens * 2.50 / 1_000_000
            output_cost = usage.completion_tokens * 10.0 / 1_000_000

        self.total_cost += input_cost + output_cost

        self.logger.debug(
            f"Usage: prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"cost=${input_cost + output_cost:.6f}"
        )

    def get_usage_stats(self) -> dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with usage stats
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost": self.total_cost,
            "model": self.model,
            "fallback_model": self.fallback_model,
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.logger.info("Reset usage statistics")


# Default LLM client instance
default_llm_client = LLMClient()


def generate_text(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Generate text using default LLM client.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        temperature: Generation temperature

    Returns:
        Generated text
    """
    return default_llm_client.generate(prompt, system_prompt, temperature)


__all__ = [
    "LLMClient",
    "default_llm_client",
    "generate_text",
]
