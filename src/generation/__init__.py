"""
Generation layer for RAG system.

This module handles LLM-based text generation, response synthesis,
multi-step analysis, chart generation, and research analysis.
"""

from src.generation.chart_generator import ChartGenerator, default_chart_generator
from src.generation.llm_client import LLMClient, default_llm_client, generate_text
from src.generation.multi_step_analyzer import (
    MultiStepAnalyzer,
    default_multi_step_analyzer,
)
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
    # Multi-step Analyzer
    "MultiStepAnalyzer",
    "default_multi_step_analyzer",
    # Chart Generator
    "ChartGenerator",
    "default_chart_generator",
    # Research Analyzer
    "ResearchAnalyzer",
    "ComparisonAnalyzer",
    "TrendAnalyzer",
    "GapAnalyzer",
    "ConsensusAnalyzer",
    "StructuredDataExtractor",
    "default_research_analyzer",
]
