"""
Response synthesis for RAG system.

This module combines retrieved documents with LLM generation
to produce coherent, well-cited answers.
"""

from typing import Any

from config import prompts
from src.core.logging import LoggerMixin, log_execution_time
from src.core.utils import format_metadata, format_sources
from src.generation.llm_client import LLMClient


class ResponseSynthesizer(LoggerMixin):
    """
    Synthesizes responses from retrieved documents and LLM.

    Handles:
    - Context building from retrieved chunks
    - Prompt formatting
    - Response generation
    - Citation extraction
    """

    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize response synthesizer.

        Args:
            llm_client: LLM client (creates default if None)
        """
        self.llm_client = llm_client or LLMClient()
        self.logger.info("Initialized ResponseSynthesizer")

    @log_execution_time
    def synthesize(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        query_type: str = "SIMPLE",
        include_metadata: bool = True,
        conversation_context: str = "",
    ) -> dict[str, Any]:
        """
        Synthesize response from query and retrieved documents.

        Args:
            query: User query
            retrieved_docs: List of retrieved documents from vector store
            query_type: Type of query (SIMPLE, COMPLEX, ANALYTICAL)
            include_metadata: Include metadata in context
            conversation_context: Previous conversation for multi-turn chat

        Returns:
            Dictionary with answer, sources, and metadata
        """
        self.logger.info(
            f"Synthesizing response for query type '{query_type}' "
            f"with {len(retrieved_docs)} documents"
            f"{' (with conversation context)' if conversation_context else ''}"
        )

        # Handle no results case
        if not retrieved_docs:
            return self._handle_no_results(query)

        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)

        # Build metadata summary
        metadata_str = ""
        if include_metadata and retrieved_docs:
            metadata_str = self._build_metadata_summary(retrieved_docs)

        # Add conversation context if provided
        if conversation_context:
            context = f"{conversation_context}\n\n---\n\nCurrent documents:\n{context}"

        # Get appropriate prompt template
        prompt_template = prompts.get_prompt_for_query_type(query_type)

        # Format prompt
        formatted_prompt = prompts.format_prompt(
            prompt_template,
            context=context,
            metadata=metadata_str,
            query=query,
        )

        # Generate response
        try:
            answer = self.llm_client.generate(
                prompt=formatted_prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
            )

            # Extract source IDs from retrieved docs
            source_ids = [doc["id"] for doc in retrieved_docs]

            # Extract cited sources from answer (look for [doc_id] patterns)
            cited_sources = self._extract_citations(answer, source_ids)

            self.logger.info(
                f"Generated response ({len(answer)} chars) "
                f"with {len(cited_sources)} citations"
            )

            return {
                "answer": answer,
                "sources": cited_sources,
                "source_documents": retrieved_docs,
                "query_type": query_type,
                "token_usage": self.llm_client.get_usage_stats(),
            }

        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}", exc_info=True)
            return {
                "answer": f"Sorry, I encountered an error generating the response: {str(e)}",
                "sources": [],
                "source_documents": retrieved_docs,
                "query_type": query_type,
                "error": str(e),
            }

    def synthesize_stream(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        query_type: str = "SIMPLE",
    ):
        """
        Synthesize response with streaming.

        Args:
            query: User query
            retrieved_docs: Retrieved documents
            query_type: Query type

        Yields:
            Text chunks
        """
        if not retrieved_docs:
            yield "I don't have any relevant documents to answer your question."
            return

        # Build context
        context = self._build_context(retrieved_docs)
        metadata_str = self._build_metadata_summary(retrieved_docs)

        # Get prompt
        prompt_template = prompts.get_prompt_for_query_type(query_type)
        formatted_prompt = prompts.format_prompt(
            prompt_template,
            context=context,
            metadata=metadata_str,
            query=query,
        )

        # Stream response
        try:
            for chunk in self.llm_client.generate_stream(
                prompt=formatted_prompt,
                system_prompt=prompts.SYSTEM_PROMPT,
            ):
                yield chunk

        except Exception as e:
            self.logger.error(f"Streaming failed: {e}", exc_info=True)
            yield f"\n\n[Error: {str(e)}]"

    def _build_context(self, retrieved_docs: list[dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents.

        Args:
            retrieved_docs: Retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get("id", f"doc_{i}")
            content = doc.get("document", "")
            similarity = doc.get("similarity", 0.0)

            # Format each document
            doc_context = f"[{doc_id}] (Relevance: {similarity:.2f})\n{content}"
            context_parts.append(doc_context)

        # Join with separators
        context = "\n\n" + "-" * 50 + "\n\n"
        context += "\n\n".join(context_parts)

        return context

    def _build_metadata_summary(self, retrieved_docs: list[dict[str, Any]]) -> str:
        """
        Build metadata summary from retrieved documents.

        Args:
            retrieved_docs: Retrieved documents

        Returns:
            Formatted metadata string
        """
        metadata_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get("id", f"doc_{i}")
            metadata = doc.get("metadata", {})

            # Include only useful metadata
            useful_metadata = {
                "source": metadata.get("filename", "Unknown"),
                "category": metadata.get("category", "N/A"),
                "chunk": f"{metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}",
            }

            metadata_str = f"[{doc_id}]: {format_metadata(useful_metadata)}"
            metadata_parts.append(metadata_str)

        return "\n".join(metadata_parts)

    def _extract_citations(
        self, answer: str, available_sources: list[str]
    ) -> list[str]:
        """
        Extract citations from answer.

        Args:
            answer: Generated answer
            available_sources: List of available source IDs

        Returns:
            List of cited source IDs
        """
        import re

        # Find all citation patterns like [doc_abc123]
        citation_pattern = r"\[([^\]]+)\]"
        matches = re.findall(citation_pattern, answer)

        # Filter to only include actual source IDs
        cited_sources = []
        for match in matches:
            if match in available_sources and match not in cited_sources:
                cited_sources.append(match)

        return cited_sources

    def _handle_no_results(self, query: str) -> dict[str, Any]:
        """
        Handle case when no results are found.

        Args:
            query: User query

        Returns:
            Response dictionary
        """
        self.logger.warning(f"No results found for query: '{query}'")

        no_context_msg = prompts.format_prompt(
            prompts.NO_CONTEXT_PROMPT,
            query=query,
        )

        return {
            "answer": no_context_msg,
            "sources": [],
            "source_documents": [],
            "query_type": "NO_RESULTS",
        }


# Default synthesizer instance
default_synthesizer = ResponseSynthesizer()


def synthesize_response(
    query: str,
    retrieved_docs: list[dict[str, Any]],
    query_type: str = "SIMPLE",
) -> dict[str, Any]:
    """
    Synthesize response using default synthesizer.

    Args:
        query: User query
        retrieved_docs: Retrieved documents
        query_type: Query type

    Returns:
        Response dictionary
    """
    return default_synthesizer.synthesize(query, retrieved_docs, query_type)


__all__ = [
    "ResponseSynthesizer",
    "default_synthesizer",
    "synthesize_response",
]
