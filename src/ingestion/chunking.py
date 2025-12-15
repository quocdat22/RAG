"""
Text chunking strategies for document processing.

This module provides intelligent text splitting strategies that preserve
semantic meaning and document structure.
"""

import re
from typing import Any

from config.settings import settings
from src.core.exceptions import ChunkingError
from src.core.logging import LoggerMixin
from src.core.utils import clean_text, count_tokens
from src.ingestion.loaders import Document


class DocumentChunk:
    """Represents a chunk of a document."""

    def __init__(
        self,
        content: str,
        metadata: dict[str, Any],
        chunk_id: str,
        chunk_index: int,
    ):
        """
        Initialize document chunk.

        Args:
            content: Chunk text content
            metadata: Chunk metadata (inherited from parent document)
            chunk_id: Unique chunk identifier
            chunk_index: Index of chunk in document
        """
        self.content = content
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.chunk_index = chunk_index

    def __repr__(self) -> str:
        return f"DocumentChunk(id={self.chunk_id}, index={self.chunk_index}, length={len(self.content)})"


class SemanticChunker(LoggerMixin):
    """
    Semantic-aware text chunker that preserves meaning.

    Uses intelligent splitting based on:
    - Paragraph boundaries
    - Sentence boundaries
    - Token limits
    - Configurable overlap
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
    ):
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size in tokens
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.min_chunk_size

        self.logger.info(
            f"Initialized SemanticChunker: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, min={self.min_chunk_size}"
        )

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document to chunk

        Returns:
            List of document chunks

        Raises:
            ChunkingError: If chunking fails
        """
        try:
            self.logger.info(
                f"Chunking document: {document.doc_id} (length: {len(document.content)})"
            )

            # Clean text
            text = clean_text(document.content)

            if not text.strip():
                raise ChunkingError("Document content is empty after cleaning")

            # Split into chunks
            chunks = self._split_text(text)

            # Create DocumentChunk objects
            doc_chunks = []
            for i, chunk_text in enumerate(chunks):
                # Skip chunks that are too small (except the last one)
                if i < len(chunks) - 1 and count_tokens(chunk_text) < self.min_chunk_size:
                    continue

                # Create chunk metadata (inherit from document)
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "doc_id": document.doc_id,
                    }
                )

                chunk_id = f"{document.doc_id}_chunk_{i}"

                doc_chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    chunk_index=i,
                )

                doc_chunks.append(doc_chunk)

            self.logger.info(
                f"Created {len(doc_chunks)} chunks from document {document.doc_id}"
            )

            return doc_chunks

        except Exception as e:
            raise ChunkingError(f"Failed to chunk document {document.doc_id}: {str(e)}")

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks using semantic-aware strategy.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # First, split by paragraphs
        paragraphs = self._split_paragraphs(text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = count_tokens(paragraph)

            # If single paragraph exceeds chunk size, split by sentences
            if paragraph_tokens > self.chunk_size:
                # Flush current chunk if not empty
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentence_chunks = self._split_large_paragraph(paragraph)
                chunks.extend(sentence_chunks)

            # If adding paragraph would exceed chunk size
            elif current_tokens + paragraph_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))

                # Start new chunk with overlap from previous
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    overlap_tokens = count_tokens(overlap_text)

                    if overlap_tokens <= self.chunk_overlap:
                        current_chunk = [overlap_text, paragraph]
                        current_tokens = overlap_tokens + paragraph_tokens
                    else:
                        current_chunk = [paragraph]
                        current_tokens = paragraph_tokens
                else:
                    current_chunk = [paragraph]
                    current_tokens = paragraph_tokens

            # Add paragraph to current chunk
            else:
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        # Split on double newlines or multiple newlines
        paragraphs = re.split(r"\n\s*\n", text)

        # Clean and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def _split_large_paragraph(self, paragraph: str) -> list[str]:
        """
        Split a large paragraph into smaller chunks by sentences.

        Args:
            paragraph: Paragraph to split

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = self._split_sentences(paragraph)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            # If single sentence exceeds chunk size, split by characters
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split very long sentence by characters
                char_chunks = self._split_by_characters(sentence)
                chunks.extend(char_chunks)

            elif current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_sentence = current_chunk[-1]
                    overlap_tokens = count_tokens(overlap_sentence)

                    if overlap_tokens <= self.chunk_overlap:
                        current_chunk = [overlap_sentence, sentence]
                        current_tokens = overlap_tokens + sentence_tokens
                    else:
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens

            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with nltk/spacy)
        # Handle common abbreviations
        text = re.sub(r"(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|e\.g|i\.e)\.", r"\1<PERIOD>", text)

        # Split on sentence endings
        sentences = re.split(r"[.!?]+\s+", text)

        # Restore periods in abbreviations
        sentences = [s.replace("<PERIOD>", ".").strip() for s in sentences if s.strip()]

        return sentences

    def _split_by_characters(self, text: str, max_chars: int | None = None) -> list[str]:
        """
        Split text by character count (fallback for very long text).

        Args:
            text: Text to split
            max_chars: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if max_chars is None:
            # Estimate characters from token limit (1 token ≈ 4 chars)
            max_chars = self.chunk_size * 4

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_chars

            # Try to break at word boundary
            if end < len(text):
                # Look back for space
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos

            chunks.append(text[start:end].strip())
            start = end - (self.chunk_overlap * 4)  # Character-based overlap

        return chunks


# Default chunker instance
default_chunker = SemanticChunker()


def chunk_document(
    document: Document,
    chunker: SemanticChunker | None = None,
) -> list[DocumentChunk]:
    """
    Chunk a document using specified or default chunker.

    Args:
        document: Document to chunk
        chunker: Optional custom chunker (uses default if None)

    Returns:
        List of document chunks
    """
    chunker = chunker or default_chunker
    return chunker.chunk_document(document)


__all__ = [
    "DocumentChunk",
    "SemanticChunker",
    "chunk_document",
    "default_chunker",
]
