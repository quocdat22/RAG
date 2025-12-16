"""
Unit tests for Research Analyzer module.

Tests structured data extraction, comparison, trend analysis,
gap identification, and consensus detection.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.generation.research_analyzer import (
    StructuredDataExtractor,
    ComparisonAnalyzer,
    TrendAnalyzer,
    GapAnalyzer,
    ConsensusAnalyzer,
    ResearchAnalyzer,
)


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client for testing."""
    client = Mock()
    client.generate = Mock(return_value='{"method_name": "BERT", "accuracy": "88.5%", "year": "2019"}')
    return client


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc_1",
            "document": "BERT achieves 88.5% accuracy on GLUE benchmark. Published in 2019.",
            "metadata": {
                "filename": "bert_paper.pdf",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "author": "Devlin et al.",
                "year": "2019",
            },
        },
        {
            "id": "doc_2",
            "document": "GPT-2 shows strong performance with 1.5B parameters. Released in 2019.",
            "metadata": {
                "filename": "gpt2_paper.pdf",
                "title": "Language Models are Unsupervised Multitask Learners",
                "author": "Radford et al.",
                "year": "2019",
            },
        },
        {
            "id": "doc_3",
            "document": "T5 unifies NLP tasks with 92% accuracy. Published in 2020.",
            "metadata": {
                "filename": "t5_paper.pdf",
                "title": "Exploring the Limits of Transfer Learning",
                "author": "Raffel et al.",
                "year": "2020",
            },
        },
    ]


class TestStructuredDataExtractor:
    """Test structured data extraction."""
    
    def test_extraction_basic(self, mock_llm_client, sample_documents):
        """Test basic data extraction."""
        extractor = StructuredDataExtractor(mock_llm_client)
        
        # Mock JSON response
        mock_llm_client.generate.return_value = '''
        {
            "method_name": "BERT",
            "accuracy": "88.5%",
            "dataset": "GLUE",
            "year": "2019"
        }
        '''
        
        results = extractor.extract([sample_documents[0]])
        
        assert len(results) == 1
        assert results[0]["method_name"] == "BERT"
        assert results[0]["accuracy"] == "88.5%"
        assert results[0]["source"] == "doc_1"
    
    def test_parse_json_with_code_blocks(self, mock_llm_client):
        """Test JSON parsing with markdown code blocks."""
        extractor = StructuredDataExtractor(mock_llm_client)
        
        response = '''
        Here's the data:
        ```json
        {"method": "BERT", "year": "2019"}
        ```
        '''
        
        parsed = extractor._parse_json_response(response)
        assert parsed["method"] == "BERT"
        assert parsed["year"] == "2019"
    
    def test_parse_json_without_code_blocks(self, mock_llm_client):
        """Test JSON parsing without code blocks."""
        extractor = StructuredDataExtractor(mock_llm_client)
        
        response = '{"method": "GPT-2", "year": "2019"}'
        
        parsed = extractor._parse_json_response(response)
        assert parsed["method"] == "GPT-2"


class TestComparisonAnalyzer:
    """Test comparison matrix generation."""
    
    @patch('src.generation.research_analyzer.StructuredDataExtractor')
    def test_comparison_analysis(self, mock_extractor_class, mock_llm_client, sample_documents):
        """Test comparison matrix generation."""
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = [
            {"method_name": "BERT", "accuracy": "88.5%", "year": "2019"},
            {"method_name": "GPT-2", "accuracy": "N/A", "year": "2019"},
        ]
        mock_extractor_class.return_value = mock_extractor
        
        # Mock LLM response with comparison table
        mock_llm_client.generate.return_value = """
        ## Comparison Matrix
        
        | Method | Accuracy | Year |
        |--------|----------|------|
        | BERT   | 88.5%    | 2019 |
        | GPT-2  | N/A      | 2019 |
        """
        
        analyzer = ComparisonAnalyzer(mock_llm_client)
        result = analyzer.analyze(
            query="Compare BERT and GPT-2",
            documents=sample_documents[:2],
            criteria=["accuracy", "year"],
        )
        
        assert "comparison_matrix" in result
        assert result["query"] == "Compare BERT and GPT-2"
        assert len(result["sources"]) == 2


class TestTrendAnalyzer:
    """Test trend analysis."""
    
    @patch('src.generation.research_analyzer.StructuredDataExtractor')
    def test_trend_analysis(self, mock_extractor_class, mock_llm_client, sample_documents):
        """Test temporal trend analysis."""
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = [
            {"method_name": "BERT", "year": "2019", "key_contribution": "Bidirectional pretraining"},
            {"method_name": "GPT-2", "year": "2019", "key_contribution": "Large-scale LM"},
            {"method_name": "T5", "year": "2020", "key_contribution": "Unified text-to-text"},
        ]
        mock_extractor_class.return_value = mock_extractor
        
        mock_llm_client.generate.return_value = """
        ## Timeline of Key Developments
        
        2019: BERT and GPT-2 introduced transformer-based pretraining
        2020: T5 unified NLP tasks
        """
        
        analyzer = TrendAnalyzer(mock_llm_client)
        result = analyzer.analyze(
            query="How did transformers evolve?",
            documents=sample_documents,
            time_range=(2019, 2020),
        )
        
        assert "trend_analysis" in result
        assert result["time_range"] == (2019, 2020)
        assert len(result["temporal_data"]) == 3


class TestGapAnalyzer:
    """Test research gap identification."""
    
    def test_gap_analysis(self, mock_llm_client, sample_documents):
        """Test gap identification."""
        mock_llm_client.generate.return_value = """
        ## Identified Research Gaps
        
        ### High Priority
        1. **Multilingual models**: Limited coverage
        2. **Efficiency**: Few studies on compression
        """
        
        analyzer = GapAnalyzer(mock_llm_client)
        result = analyzer.analyze(
            query="What gaps exist in NLP?",
            documents=sample_documents,
        )
        
        assert "gap_analysis" in result
        assert result["corpus_size"] == 3


class TestConsensusAnalyzer:
    """Test consensus detection."""
    
    def test_consensus_analysis(self, mock_llm_client, sample_documents):
        """Test consensus vs controversy detection."""
        mock_llm_client.generate.return_value = """
        ## Strong Consensus
        
        1. **Pretraining is beneficial**: All papers agree
        
        ## Active Controversies
        
        1. **Model size vs quality**: Debated
        """
        
        analyzer = ConsensusAnalyzer(mock_llm_client)
        result = analyzer.analyze(
            query="Is pretraining beneficial?",
            documents=sample_documents,
        )
        
        assert "consensus_analysis" in result
        assert result["paper_count"] == 3


class TestResearchAnalyzer:
    """Test main research analyzer coordinator."""
    
    def test_routing_comparison(self, mock_llm_client, sample_documents):
        """Test routing to comparison analyzer."""
        with patch.object(ComparisonAnalyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {
                "comparison_matrix": "Test result",
                "sources": ["doc_1", "doc_2"],
            }
            
            analyzer = ResearchAnalyzer(mock_llm_client)
            result = analyzer.analyze(
                query="Compare methods",
                documents=sample_documents,
                analysis_type="COMPARISON",
            )
            
            assert "comparison_matrix" in result
            mock_analyze.assert_called_once()
    
    def test_routing_trend(self, mock_llm_client, sample_documents):
        """Test routing to trend analyzer."""
        with patch.object(TrendAnalyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = {
                "trend_analysis": "Test result",
                "sources": ["doc_1"],
            }
            
            analyzer = ResearchAnalyzer(mock_llm_client)
            result = analyzer.analyze(
                query="Show trends",
                documents=sample_documents,
                analysis_type="TREND_ANALYSIS",
            )
            
            assert "trend_analysis" in result
            mock_analyze.assert_called_once()
    
    def test_unknown_analysis_type(self, mock_llm_client, sample_documents):
        """Test handling of unknown analysis type."""
        analyzer = ResearchAnalyzer(mock_llm_client)
        result = analyzer.analyze(
            query="Test",
            documents=sample_documents,
            analysis_type="UNKNOWN_TYPE",
        )
        
        assert "error" in result
        assert "Unknown analysis type" in result["error"]
    
    def test_no_documents(self, mock_llm_client):
        """Test handling of empty document list."""
        analyzer = ResearchAnalyzer(mock_llm_client)
        result = analyzer.analyze(
            query="Test",
            documents=[],
            analysis_type="COMPARISON",
        )
        
        assert "error" in result
        assert "No documents" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
