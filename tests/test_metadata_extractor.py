"""
Unit tests for MetadataExtractor.

Tests for academic paper metadata extraction including abstract, DOI,
arXiv ID, venue detection, institution extraction, and LLM-based extraction.
"""

import pytest
from unittest.mock import Mock, patch

from src.ingestion.metadata import MetadataExtractor
from src.ingestion.loaders import Document


@pytest.fixture
def extractor():
    """Create MetadataExtractor without LLM client."""
    return MetadataExtractor(llm_client=None)


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock()
    client.generate = Mock(return_value='{"contribution": "Test contribution", "methodology": "neural network"}')
    return client


@pytest.fixture
def sample_academic_paper():
    """Sample academic paper content."""
    return """
arXiv:2301.12345v2

Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar
Google Brain, Google Research
University of Toronto

Abstract
We propose a new simple network architecture, the Transformer, based solely on 
attention mechanisms, dispensing with recurrence and convolutions entirely.
Experiments on two machine translation tasks show these models to be superior
in quality while being more parallelizable and requiring significantly less time to train.

DOI: 10.1234/nips.2017.12345

Published at NeurIPS 2017

1. Introduction
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder.
"""


@pytest.fixture
def sample_document(sample_academic_paper):
    """Create sample Document."""
    return Document(
        content=sample_academic_paper,
        metadata={"filename": "attention.pdf", "title": "Attention Is All You Need"}
    )


class TestAbstractExtraction:
    """Test abstract extraction."""
    
    def test_extract_abstract_basic(self, extractor, sample_academic_paper):
        """Test basic abstract extraction."""
        abstract = extractor.extract_abstract(sample_academic_paper)
        assert "Transformer" in abstract
        assert "attention mechanisms" in abstract
    
    def test_extract_abstract_no_abstract(self, extractor):
        """Test when no abstract present."""
        text = "This is a document without an abstract section."
        abstract = extractor.extract_abstract(text)
        assert abstract == ""
    
    def test_extract_abstract_short_text(self, extractor):
        """Test with very short abstract (should be filtered)."""
        text = "Abstract\nToo short.\n\nIntroduction\nSome text."
        abstract = extractor.extract_abstract(text)
        assert abstract == ""


class TestDOIExtraction:
    """Test DOI extraction."""
    
    def test_extract_doi_basic(self, extractor, sample_academic_paper):
        """Test basic DOI extraction."""
        doi = extractor.extract_doi(sample_academic_paper)
        assert doi == "10.1234/nips.2017.12345"
    
    def test_extract_doi_from_url(self, extractor):
        """Test DOI extraction from URL."""
        text = "Available at https://doi.org/10.5555/paper.2023.001"
        doi = extractor.extract_doi(text)
        assert "10.5555/paper.2023.001" in doi
    
    def test_extract_doi_none(self, extractor):
        """Test when no DOI present."""
        text = "This paper has no DOI."
        doi = extractor.extract_doi(text)
        assert doi == ""


class TestArxivExtraction:
    """Test arXiv ID extraction."""
    
    def test_extract_arxiv_basic(self, extractor, sample_academic_paper):
        """Test basic arXiv extraction."""
        arxiv_id = extractor.extract_arxiv_id(sample_academic_paper)
        assert arxiv_id == "2301.12345v2"
    
    def test_extract_arxiv_from_url(self, extractor):
        """Test arXiv extraction from URL."""
        text = "Paper available at arxiv.org/abs/2312.01234"
        arxiv_id = extractor.extract_arxiv_id(text)
        assert arxiv_id == "2312.01234"
    
    def test_extract_arxiv_none(self, extractor):
        """Test when no arXiv ID present."""
        text = "This paper is not on arXiv."
        arxiv_id = extractor.extract_arxiv_id(text)
        assert arxiv_id == ""


class TestVenueExtraction:
    """Test venue (conference/journal) detection."""
    
    def test_extract_venue_conference(self, extractor, sample_academic_paper):
        """Test conference detection."""
        venue = extractor.extract_venue(sample_academic_paper)
        assert venue["type"] == "conference"
        assert "NeurIPS" in venue["name"] or "NIPS" in venue["name"]
    
    def test_extract_venue_journal(self, extractor):
        """Test journal detection."""
        text = "Published in Nature Machine Intelligence, 2023"
        venue = extractor.extract_venue(text)
        assert venue["type"] == "journal"
        assert "Nature" in venue["name"]
    
    def test_extract_venue_none(self, extractor):
        """Test when no venue detected."""
        text = "This is just some random text without venue."
        venue = extractor.extract_venue(text)
        assert venue["name"] == ""


class TestInstitutionExtraction:
    """Test institution/affiliation extraction."""
    
    def test_extract_institutions_basic(self, extractor, sample_academic_paper):
        """Test basic institution extraction."""
        institutions = extractor.extract_institutions(sample_academic_paper)
        # Should find Google, University of Toronto
        assert len(institutions) > 0
        institution_names = " ".join(institutions).lower()
        assert "google" in institution_names or "toronto" in institution_names
    
    def test_extract_institutions_multiple(self, extractor):
        """Test multiple institution extraction."""
        text = """
        Authors from Stanford University, MIT, and Google Research
        collaborated on this project.
        """
        institutions = extractor.extract_institutions(text)
        assert len(institutions) >= 2


class TestLLMExtraction:
    """Test LLM-based extraction."""
    
    def test_extract_with_llm(self, mock_llm_client, sample_academic_paper):
        """Test LLM extraction."""
        extractor = MetadataExtractor(llm_client=mock_llm_client)
        result = extractor.extract_with_llm(sample_academic_paper)
        
        assert result["contribution"] == "Test contribution"
        assert result["methodology"] == "neural network"
    
    def test_parse_json_with_code_blocks(self, extractor):
        """Test JSON parsing with markdown code blocks."""
        response = '''
        Here's the data:
        ```json
        {"contribution": "Test", "methodology": "survey"}
        ```
        '''
        parsed = extractor._parse_json_response(response)
        assert parsed["contribution"] == "Test"
    
    def test_parse_json_without_code_blocks(self, extractor):
        """Test JSON parsing without code blocks."""
        response = '{"contribution": "Test", "methodology": "survey"}'
        parsed = extractor._parse_json_response(response)
        assert parsed["contribution"] == "Test"


class TestEnrichMetadata:
    """Test full metadata enrichment."""
    
    def test_enrich_metadata_no_llm(self, extractor, sample_document):
        """Test metadata enrichment without LLM."""
        enriched = extractor.enrich_metadata(sample_document, use_llm=False)
        
        # Basic metadata
        assert enriched.metadata.get("language") == "en"
        assert len(enriched.metadata.get("keywords", [])) > 0
        assert enriched.metadata.get("category") == "computer_science"
        
        # Academic metadata
        assert "abstract" in enriched.metadata
        assert "doi" in enriched.metadata
        assert "arxiv_id" in enriched.metadata
        assert "venue" in enriched.metadata
    
    def test_enrich_metadata_with_llm(self, mock_llm_client, sample_document):
        """Test metadata enrichment with LLM."""
        extractor = MetadataExtractor(llm_client=mock_llm_client)
        enriched = extractor.enrich_metadata(sample_document, use_llm=True)
        
        # Should have LLM extracted data
        assert "llm_extracted" in enriched.metadata or "contribution" in enriched.metadata


class TestCategorization:
    """Test document categorization."""
    
    def test_categorize_cs(self, extractor):
        """Test computer science categorization."""
        text = "This paper presents a new neural network architecture using transformers."
        category = extractor.categorize_document(text, {})
        assert category == "computer_science"
    
    def test_categorize_medicine(self, extractor):
        """Test medicine categorization."""
        text = "Clinical trial results show improved patient outcomes with new treatment."
        category = extractor.categorize_document(text, {})
        assert category == "medicine_biology"
    
    def test_categorize_physics(self, extractor):
        """Test physics categorization."""
        text = "Quantum entanglement experiments with photon particles."
        category = extractor.categorize_document(text, {})
        assert category == "physics_chemistry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
