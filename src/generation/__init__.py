"""
Generation layer for RAG system.

This module handles LLM-based text generation, response synthesis,
and specialized research analysis.
"""

from src.generation.llm_client import LLMClient, default_llm_client, generate_text
from src.generation.research_analyzer import (
    ComparisonAnalyzer,
    ConsensusAnalyzer,
    GapAnalyzer,
    ResearchAnalyzer,
    StructuredDataExtractor,
    TrendAnalyzer,
    default_research_analyzer,
)
from src.generation.response_synthesizer import (
    ResponseSynthesizer,
    default_synthesizer,
    synthesize_response,
)

__all__ = [
    # LLM Client
    "LLMClient",
    "default_llm_client",
    "generate_text",
    # Response Synthesizer
    "ResponseSynthesizer",
    "default_synthesizer",
    "synthesize_response",
    # Research Analyzer
    "ResearchAnalyzer",
    "ComparisonAnalyzer",
    "TrendAnalyzer",
    "GapAnalyzer",
    "ConsensusAnalyzer",
    "StructuredDataExtractor",
    "default_research_analyzer",
]
