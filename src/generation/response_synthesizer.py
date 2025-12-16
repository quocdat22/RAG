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
from src.generation.research_analyzer import ResearchAnalyzer


class ResponseSynthesizer(LoggerMixin):
    """
    Synthesizes responses from retrieved documents and LLM.

    Handles:
    - Context building from retrieved chunks
    - Prompt formatting
    - Response generation
    - Citation extraction
    - Research analysis routing
    """

    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize response synthesizer.

        Args:
            llm_client: LLM client (creates default if None)
        """
        self.llm_client = llm_client or LLMClient()
        self.research_analyzer = ResearchAnalyzer(self.llm_client)
        self.logger.info("Initialized ResponseSynthesizer with ResearchAnalyzer")

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
        
        # Route research analysis queries to ResearchAnalyzer
        if query_type in ["COMPARISON", "TREND_ANALYSIS", "GAP_IDENTIFICATION", "CONSENSUS_DETECTION"]:
            self.logger.info(f"Routing to ResearchAnalyzer for {query_type}")
            analysis_result = self.research_analyzer.analyze(
                query=query,
                documents=retrieved_docs,
                analysis_type=query_type,
            )
            
            # Handle errors from research analyzer
            if "error" in analysis_result:
                return {
                    "answer": f"Analysis failed: {analysis_result['error']}",
                    "sources": [],
                    "source_documents": retrieved_docs,
                    "query_type": query_type,
                    "error": analysis_result["error"],
                }
            
            # Format the analysis result for display
            # Extract the main analysis text from the result
            analysis_key_map = {
                "COMPARISON": "comparison_matrix",
                "TREND_ANALYSIS": "trend_analysis",
                "GAP_IDENTIFICATION": "gap_analysis",
                "CONSENSUS_DETECTION": "consensus_analysis",
            }
            
            analysis_text_key = analysis_key_map.get(query_type, "analysis")
            answer = analysis_result.get(analysis_text_key, "Analysis completed but no output generated.")
            
            return {
                "answer": answer,
                "sources": analysis_result.get("sources", []),
                "source_documents": retrieved_docs,
                "query_type": query_type,
                "analysis_metadata": analysis_result,  # Include full result for reference
                "token_usage": self.llm_client.get_usage_stats(),
            }

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
        
        # Select appropriate system prompt based on query type
        analytical_types = ["ANALYTICAL", "COMPARISON", "TREND_ANALYSIS", "GAP_IDENTIFICATION", "CONSENSUS_DETECTION"]
        if query_type in analytical_types:
            system_prompt = prompts.ANALYTICAL_SYSTEM_PROMPT
            self.logger.debug(f"Using analytical system prompt for {query_type}")
        else:
            system_prompt = prompts.SYSTEM_PROMPT
            self.logger.debug(f"Using standard system prompt for {query_type}")

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
                system_prompt=system_prompt,
                query_type=query_type,  # Pass query_type for model selection
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
            Formatted metadata string with academic citations
        """
        metadata_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get("id", f"doc_{i}")
            metadata = doc.get("metadata", {})

            # Extract citation components
            # Author (formatted with et al.)
            author = metadata.get("author_formatted", "").strip()
            if not author:
                author = metadata.get("author", "").strip()
            if not author:
                author = "Unknown Author"
            
            # Year
            year = metadata.get("year", "n.d.")
            
            # Title (prioritize PDF title metadata, fallback to filename)
            title = metadata.get("title", "").strip()
            if not title:
                title = metadata.get("filename", "Unknown Paper")
            
            # Section
            section = metadata.get("section_title", "")
            if not section or section == "Unknown Section":
                # Fallback to chunk position
                chunk_idx = metadata.get("chunk_index", 0)
                section = f"Section {chunk_idx + 1}"
            
            # Page (estimated)
            page = metadata.get("estimated_page", metadata.get("chunk_index", 0) + 1)
            
            # Build citation: [1] Author et al., Year, Title, Section, p. X
            citation = f"[{doc_id}]: {author}, {year}, {title}, {section}, p. {page}"
            
            metadata_parts.append(citation)

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
