"""
Research Analyzer for advanced academic research analysis.

This module provides specialized analyzers for:
- Comparison matrices between research approaches
- Trend analysis over time
- Research gap identification
- Consensus vs controversy detection
"""

import json
import re
from typing import Any

from src.core.logging import LoggerMixin, log_execution_time
from src.generation.llm_client import LLMClient


class StructuredDataExtractor(LoggerMixin):
    """
    Extracts structured data from research papers.
    
    Extracts:
    - Method/approach name
    - Results (accuracy, F1, speed, etc.)
    - Dataset used
    - Publication year
    - Authors
    """
    
    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize structured data extractor.
        
        Args:
            llm_client: LLM client for extraction
        """
        self.llm_client = llm_client or LLMClient()
        self.logger.info("StructuredDataExtractor initialized")
    
    @log_execution_time
    def extract(
        self,
        documents: list[dict[str, Any]],
        extraction_schema: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract structured data from documents.
        
        Args:
            documents: Retrieved documents
            extraction_schema: Schema defining what to extract
                Example: {"metrics": ["accuracy", "f1_score"], "info": ["method", "dataset"]}
        
        Returns:
            List of structured data dictionaries
        """
        if extraction_schema is None:
            extraction_schema = {
                "method": ["method_name", "approach", "technique"],
                "results": ["accuracy", "f1_score", "speed", "performance"],
                "dataset": ["dataset", "benchmark", "corpus"],
                "metadata": ["year", "authors"],
            }
        
        self.logger.info(f"Extracting structured data from {len(documents)} documents")
        
        structured_data = []
        
        for doc in documents:
            try:
                extracted = self._extract_from_document(doc, extraction_schema)
                structured_data.append(extracted)
            except Exception as e:
                self.logger.error(f"Failed to extract from document: {e}")
                # Add placeholder data
                structured_data.append({
                    "source": doc.get("id", "unknown"),
                    "error": str(e),
                })
        
        return structured_data
    
    def _extract_from_document(
        self,
        document: dict[str, Any],
        schema: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Extract structured data from a single document."""
        content = document.get("document", "")
        metadata = document.get("metadata", {})
        
        # Build extraction prompt
        fields_description = []
        for category, fields in schema.items():
            fields_description.append(f"{category}: {', '.join(fields)}")
        
        prompt = f"""Extract the following information from this research paper excerpt:

Paper Content:
{content}

Extract these fields as a JSON object:
{chr(10).join(fields_description)}

Additional metadata available:
- Filename: {metadata.get('filename', 'N/A')}
- Section: {metadata.get('section_title', 'N/A')}

Return ONLY a valid JSON object with the extracted information. Use "N/A" for missing fields.
Example format:
{{
    "method_name": "Transformer",
    "accuracy": "92.5%",
    "dataset": "WMT14",
    "year": "2017",
    "authors": "Vaswani et al."
}}

JSON:"""
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are a precise data extraction assistant. Extract only factual information from the text and return valid JSON.",
                temperature=0.1,  # Low temperature for more deterministic extraction
            )
            
            # Parse JSON from response
            extracted_data = self._parse_json_response(response)
            
            # Add source information
            extracted_data["source"] = document.get("id", "unknown")
            extracted_data["source_metadata"] = {
                "filename": metadata.get("filename", "Unknown"),
                "title": metadata.get("title", metadata.get("filename", "Unknown")),
                "section": metadata.get("section_title", "Unknown"),
            }
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON: {e}. Response: {response[:200]}")
            # Return empty dict as fallback
            return {}


class ComparisonAnalyzer(LoggerMixin):
    """
    Generates comparison matrices for research approaches.
    
    Example: Compare BERT vs GPT-2 vs T5 on accuracy, speed, dataset
    """
    
    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize comparison analyzer."""
        self.llm_client = llm_client or LLMClient()
        self.extractor = StructuredDataExtractor(llm_client)
        self.logger.info("ComparisonAnalyzer initialized")
    
    @log_execution_time
    def analyze(
        self,
        query: str,
        documents: list[dict[str, Any]],
        criteria: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate comparison matrix.
        
        Args:
            query: Comparison query
            documents: Retrieved documents
            criteria: Comparison criteria (e.g., ["accuracy", "speed", "dataset"])
        
        Returns:
            Comparison result with matrix and analysis
        """
        if criteria is None:
            criteria = ["accuracy", "speed", "dataset", "year", "method_type"]
        
        self.logger.info(f"Generating comparison matrix with criteria: {criteria}")
        
        # Step 1: Extract structured data from all documents
        extraction_schema = {
            "method": ["method_name", "approach"],
            "metrics": criteria,
            "metadata": ["year", "authors", "paper_title"],
        }
        
        structured_data = self.extractor.extract(documents, extraction_schema)
        
        # Step 2: Build comparison prompt
        from config import prompts
        
        # Format structured data for prompt
        data_summary = self._format_structured_data(structured_data)
        
        comparison_prompt = prompts.format_prompt(
            prompts.COMPARISON_MATRIX_PROMPT,
            query=query,
            structured_data=data_summary,
            criteria=", ".join(criteria),
        )
        
        # Step 3: Generate comparison analysis
        analysis = self.llm_client.generate(
            prompt=comparison_prompt,
            system_prompt=prompts.SYSTEM_PROMPT,
        )
        
        return {
            "query": query,
            "comparison_matrix": analysis,
            "structured_data": structured_data,
            "criteria": criteria,
            "sources": [doc.get("id") for doc in documents],
        }
    
    def _format_structured_data(self, data: list[dict[str, Any]]) -> str:
        """Format structured data for prompt."""
        formatted_parts = []
        for i, item in enumerate(data, 1):
            source = item.get("source_metadata", {}).get("title", f"Paper {i}")
            formatted_parts.append(f"**Paper {i}: {source}**")
            for key, value in item.items():
                if key not in ["source", "source_metadata", "error"]:
                    formatted_parts.append(f"  - {key}: {value}")
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)


class TrendAnalyzer(LoggerMixin):
    """
    Analyzes temporal trends in research.
    
    Example: How did attention mechanisms evolve from 2017-2024?
    """
    
    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize trend analyzer."""
        self.llm_client = llm_client or LLMClient()
        self.extractor = StructuredDataExtractor(llm_client)
        self.logger.info("TrendAnalyzer initialized")
    
    @log_execution_time
    def analyze(
        self,
        query: str,
        documents: list[dict[str, Any]],
        time_range: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze research trends over time.
        
        Args:
            query: Trend analysis query
            documents: Retrieved documents
            time_range: Optional (start_year, end_year) tuple
        
        Returns:
            Trend analysis result
        """
        self.logger.info(f"Analyzing trends for query: {query}")
        
        # Extract temporal data
        extraction_schema = {
            "method": ["method_name", "technique", "innovation"],
            "metadata": ["year", "authors", "paper_title"],
            "impact": ["key_contribution", "improvement_over_previous"],
        }
        
        structured_data = self.extractor.extract(documents, extraction_schema)
        
        # Sort by year
        structured_data_sorted = sorted(
            structured_data,
            key=lambda x: int(x.get("year", "0").replace("N/A", "0").strip()) if isinstance(x.get("year"), str) else x.get("year", 0)
        )
        
        # Build temporal context
        temporal_context = self._build_temporal_context(structured_data_sorted)
        
        # Generate trend analysis
        from config import prompts
        
        trend_prompt = prompts.format_prompt(
            prompts.TREND_ANALYSIS_PROMPT,
            query=query,
            temporal_data=temporal_context,
            time_range=f"{time_range[0]}-{time_range[1]}" if time_range else "all available years",
        )
        
        analysis = self.llm_client.generate(
            prompt=trend_prompt,
            system_prompt=prompts.SYSTEM_PROMPT,
        )
        
        return {
            "query": query,
            "trend_analysis": analysis,
            "temporal_data": structured_data_sorted,
            "time_range": time_range,
            "sources": [doc.get("id") for doc in documents],
        }
    
    def _build_temporal_context(self, sorted_data: list[dict[str, Any]]) -> str:
        """Build temporal context from sorted data."""
        timeline_parts = []
        for item in sorted_data:
            year = item.get("year", "Unknown")
            method = item.get("method_name", item.get("approach", "Unknown method"))
            paper = item.get("source_metadata", {}).get("title", "Unknown paper")
            contribution = item.get("key_contribution", "N/A")
            
            timeline_parts.append(
                f"**{year}** - {method} ({paper})\n"
                f"  Key contribution: {contribution}"
            )
        
        return "\n\n".join(timeline_parts)


class GapAnalyzer(LoggerMixin):
    """
    Identifies research gaps in a corpus.
    
    Example: What problems in NLP haven't been thoroughly researched?
    """
    
    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize gap analyzer."""
        self.llm_client = llm_client or LLMClient()
        self.logger.info("GapAnalyzer initialized")
    
    @log_execution_time
    def analyze(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Identify research gaps.
        
        Args:
            query: Gap identification query
            documents: Retrieved documents
        
        Returns:
            Gap analysis result
        """
        self.logger.info(f"Identifying research gaps for: {query}")
        
        # Build corpus coverage summary
        coverage_summary = self._build_coverage_summary(documents)
        
        # Generate gap analysis
        from config import prompts
        
        gap_prompt = prompts.format_prompt(
            prompts.GAP_IDENTIFICATION_PROMPT,
            query=query,
            corpus_coverage=coverage_summary,
        )
        
        analysis = self.llm_client.generate(
            prompt=gap_prompt,
            system_prompt=prompts.SYSTEM_PROMPT,
        )
        
        return {
            "query": query,
            "gap_analysis": analysis,
            "corpus_size": len(documents),
            "sources": [doc.get("id") for doc in documents],
        }
    
    def _build_coverage_summary(self, documents: list[dict[str, Any]]) -> str:
        """Build summary of what the corpus covers."""
        # Extract topics and methods mentioned
        topics = set()
        methods = set()
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            # Simple extraction - could be enhanced with NER
            content = doc.get("document", "").lower()
            
            # Look for common ML/NLP topics
            topic_keywords = ["classification", "generation", "translation", "summarization", 
                            "question answering", "sentiment analysis", "named entity"]
            for topic in topic_keywords:
                if topic in content:
                    topics.add(topic)
            
            # Add filename as topic indicator
            filename = metadata.get("filename", "")
            if filename:
                topics.add(filename)
        
        summary_parts = [
            f"**Corpus size**: {len(documents)} documents",
            f"**Topics covered**: {', '.join(list(topics)[:10]) if topics else 'Various topics'}",
            "",
            "**Document summaries**:"
        ]
        
        for i, doc in enumerate(documents[:10], 1):  # Limit to first 10
            title = doc.get("metadata", {}).get("title", 
                   doc.get("metadata", {}).get("filename", f"Document {i}"))
            content_preview = doc.get("document", "")[:200]
            summary_parts.append(f"{i}. {title}: {content_preview}...")
        
        return "\n".join(summary_parts)


class ConsensusAnalyzer(LoggerMixin):
    """
    Detects consensus vs controversy in research findings.
    
    Example: Is pre-training always beneficial? What do papers agree/disagree on?
    """
    
    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize consensus analyzer."""
        self.llm_client = llm_client or LLMClient()
        self.logger.info("ConsensusAnalyzer initialized")
    
    @log_execution_time
    def analyze(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Detect consensus and controversy.
        
        Args:
            query: Consensus detection query
            documents: Retrieved documents
        
        Returns:
            Consensus analysis result
        """
        self.logger.info(f"Analyzing consensus for: {query}")
        
        # Extract claims and findings from each paper
        claims = self._extract_claims(documents)
        
        # Generate consensus analysis
        from config import prompts
        
        consensus_prompt = prompts.format_prompt(
            prompts.CONSENSUS_DETECTION_PROMPT,
            query=query,
            claims=claims,
        )
        
        analysis = self.llm_client.generate(
            prompt=consensus_prompt,
            system_prompt=prompts.SYSTEM_PROMPT,
        )
        
        return {
            "query": query,
            "consensus_analysis": analysis,
            "paper_count": len(documents),
            "sources": [doc.get("id") for doc in documents],
        }
    
    def _extract_claims(self, documents: list[dict[str, Any]]) -> str:
        """Extract key claims from documents."""
        claim_parts = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.get("metadata", {}).get("title",
                   doc.get("metadata", {}).get("filename", f"Paper {i}"))
            content = doc.get("document", "")
            
            # Extract conclusion or key findings section if available
            # For now, use full content (could be enhanced with section detection)
            claim_parts.append(
                f"**Paper {i}: {title}**\n"
                f"Content: {content[:500]}...\n"
            )
        
        return "\n".join(claim_parts)


class ResearchAnalyzer(LoggerMixin):
    """
    Main research analysis coordinator.
    
    Routes queries to appropriate specialized analyzers.
    """
    
    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize research analyzer."""
        self.llm_client = llm_client or LLMClient()
        
        # Initialize specialized analyzers
        self.comparison = ComparisonAnalyzer(llm_client)
        self.trend = TrendAnalyzer(llm_client)
        self.gap = GapAnalyzer(llm_client)
        self.consensus = ConsensusAnalyzer(llm_client)
        
        self.logger.info("ResearchAnalyzer initialized with all sub-analyzers")
    
    @log_execution_time
    def analyze(
        self,
        query: str,
        documents: list[dict[str, Any]],
        analysis_type: str,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Perform research analysis.
        
        Args:
            query: Analysis query
            documents: Retrieved documents
            analysis_type: Type of analysis (COMPARISON, TREND_ANALYSIS, etc.)
            **kwargs: Additional parameters for specific analyzers
        
        Returns:
            Analysis result
        """
        self.logger.info(f"Performing {analysis_type} analysis")
        
        if not documents:
            return {
                "error": "No documents provided for analysis",
                "query": query,
                "analysis_type": analysis_type,
            }
        
        try:
            if analysis_type == "COMPARISON":
                return self.comparison.analyze(query, documents, **kwargs)
            elif analysis_type == "TREND_ANALYSIS":
                return self.trend.analyze(query, documents, **kwargs)
            elif analysis_type == "GAP_IDENTIFICATION":
                return self.gap.analyze(query, documents, **kwargs)
            elif analysis_type == "CONSENSUS_DETECTION":
                return self.consensus.analyze(query, documents, **kwargs)
            else:
                return {
                    "error": f"Unknown analysis type: {analysis_type}",
                    "query": query,
                    "analysis_type": analysis_type,
                }
        
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "query": query,
                "analysis_type": analysis_type,
            }


# Default instance
default_research_analyzer = ResearchAnalyzer()


__all__ = [
    "ResearchAnalyzer",
    "ComparisonAnalyzer",
    "TrendAnalyzer",
    "GapAnalyzer",
    "ConsensusAnalyzer",
    "StructuredDataExtractor",
    "default_research_analyzer",
]
