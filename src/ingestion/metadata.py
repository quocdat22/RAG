"""
Metadata extraction from documents.

This module provides utilities for extracting and enriching document metadata.
"""

from typing import Any

from langdetect import detect, LangDetectException

from src.core.logging import LoggerMixin
from src.ingestion.loaders import Document


class MetadataExtractor(LoggerMixin):
    """Extracts and enriches document metadata."""

    def extract_language(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            ISO 639-1 language code (e.g., 'en', 'vi')
        """
        try:
            # Use first 1000 chars for detection
            sample = text[:1000]
            lang = detect(sample)
            return lang
        except LangDetectException:
            self.logger.warning("Failed to detect language, defaulting to 'en'")
            return "en"

    def extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """
        Extract keywords from text.

        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords

        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be improved with TF-IDF or NLP)
        import re
        from collections import Counter

        # Convert to lowercase and split into words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Common stop words to filter
        stop_words = {
            "the",
            "is",
            "at",
            "which",
            "on",
            "and",
            "or",
            "but",
            "in",
            "with",
            "to",
            "for",
            "of",
            "as",
            "by",
            "that",
            "this",
            "from",
            "are",
            "was",
            "were",
            "been",
            "have",
            "has",
            "had",
            "will",
            "would",
            "can",
            "could",
        }

        # Filter stop words
        words = [w for w in words if w not in stop_words]

        # Count frequency
        word_counts = Counter(words)

        # Get top keywords
        keywords = [word for word, count in word_counts.most_common(max_keywords)]

        return keywords

    def categorize_document(self, text: str, metadata: dict[str, Any]) -> str:
        """
        Categorize document based on content and metadata.

        Args:
            text: Document text
            metadata: Document metadata

        Returns:
            Category name
        """
        # Simple rule-based categorization
        text_lower = text.lower()

        # Check for financial terms
        financial_terms = [
            "revenue",
            "profit",
            "financial",
            "balance sheet",
            "income statement",
            "quarterly",
            "fiscal",
        ]
        if any(term in text_lower for term in financial_terms):
            return "financial_report"

        # Check for technical terms
        technical_terms = ["api", "function", "class", "code", "software", "system"]
        if any(term in text_lower for term in technical_terms):
            return "technical_document"

        # Check for meeting-related terms
        meeting_terms = ["meeting", "agenda", "minutes", "discussion", "action item"]
        if any(term in text_lower for term in meeting_terms):
            return "meeting_notes"

        # Check file type for data files
        if metadata.get("file_type") in ["csv", "xlsx", "xls"]:
            return "data_file"

        # Default category
        return "general_document"

    def enrich_metadata(self, document: Document) -> Document:
        """
        Enrich document metadata with extracted information.

        Args:
            document: Document to enrich

        Returns:
            Document with enriched metadata
        """
        try:
            self.logger.info(f"Enriching metadata for document: {document.doc_id}")

            # Extract language
            language = self.extract_language(document.content)
            document.metadata["language"] = language

            # Extract keywords
            keywords = self.extract_keywords(document.content)
            document.metadata["keywords"] = keywords

            # Categorize document
            category = self.categorize_document(document.content, document.metadata)
            document.metadata["category"] = category

            # Add content stats
            document.metadata["content_length"] = len(document.content)
            document.metadata["word_count"] = len(document.content.split())

            self.logger.info(
                f"Enriched metadata: lang={language}, category={category}, keywords={len(keywords)}"
            )

            return document

        except Exception as e:
            self.logger.error(f"Failed to enrich metadata: {e}")
            # Return document unchanged if enrichment fails
            return document


# Default extractor instance
default_extractor = MetadataExtractor()


def enrich_document_metadata(document: Document) -> Document:
    """
    Enrich document metadata using default extractor.

    Args:
        document: Document to enrich

    Returns:
        Document with enriched metadata
    """
    return default_extractor.enrich_metadata(document)


__all__ = [
    "MetadataExtractor",
    "enrich_document_metadata",
    "default_extractor",
]
