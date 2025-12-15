"""
Vector store implementation using Chroma.

This module provides integration with Chroma vector database for
document storage, indexing, and similarity search.
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.core.exceptions import (
    VectorStoreConnectionError,
    VectorStoreIndexError,
    VectorStoreQueryError,
)
from src.core.logging import LoggerMixin, log_execution_time
from src.embedding.embedder import EmbeddingGenerator
from src.ingestion.chunking import DocumentChunk


class VectorStore(LoggerMixin):
    """
    Manages vector storage and retrieval using Chroma.

    Supports:
    - Document indexing with metadata
    - Similarity search
    - Metadata filtering
    - Incremental updates
    - Persistence to disk
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: Path | None = None,
        embedder: EmbeddingGenerator | None = None,
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection (uses settings if None)
            persist_dir: Directory for persistence (uses settings if None)
            embedder: Embedding generator (creates default if None)
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.embedder = embedder or EmbeddingGenerator()

        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = None
        self.collection = None

        self._connect()

    def _connect(self) -> None:
        """Connect to Chroma and get/create collection."""
        try:
            self.logger.info(f"Connecting to Chroma at: {self.persist_dir}")

            # Create Chroma client with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

            doc_count = self.collection.count()
            self.logger.info(
                f"Connected to collection '{self.collection_name}' with {doc_count} documents"
            )

        except Exception as e:
            raise VectorStoreConnectionError(str(e))

    @log_execution_time
    def index_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Index a single document chunk.

        Args:
            chunk: Document chunk to index

        Returns:
            True if successful

        Raises:
            VectorStoreIndexError: If indexing fails
        """
        try:
            self.logger.debug(f"Indexing chunk: {chunk.chunk_id}")

            # Generate embedding
            embedding = self.embedder.generate_embedding(chunk.content)

            # Prepare metadata (Chroma requires all values to be strings, ints, floats, or bools)
            metadata = self._prepare_metadata(chunk.metadata)

            # Add to collection
            self.collection.add(
                ids=[chunk.chunk_id],
                embeddings=[embedding],
                documents=[chunk.content],
                metadatas=[metadata],
            )

            self.logger.debug(f"Successfully indexed chunk: {chunk.chunk_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to index chunk {chunk.chunk_id}: {e}")
            raise VectorStoreIndexError(chunk.chunk_id, str(e))

    @log_execution_time
    def index_chunks_batch(self, chunks: list[DocumentChunk]) -> int:
        """
        Index multiple document chunks in batch.

        Args:
            chunks: List of document chunks

        Returns:
            Number of successfully indexed chunks

        Raises:
            VectorStoreIndexError: If batch indexing fails
        """
        if not chunks:
            return 0

        try:
            self.logger.info(f"Batch indexing {len(chunks)} chunks")

            # Generate embeddings in batch
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedder.generate_embeddings_batch(texts)

            # Prepare data for Chroma
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [self._prepare_metadata(chunk.metadata) for chunk in chunks]

            # Add to collection in batch
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            self.logger.info(f"Successfully indexed {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            self.logger.error(f"Failed to batch index chunks: {e}", exc_info=True)
            raise VectorStoreIndexError("batch", str(e))

    @log_execution_time
    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            top_k: Number of results to return (uses settings if None)
            where: Metadata filter conditions
            where_document: Document content filter conditions

        Returns:
            List of search results with documents, metadata, and distances

        Raises:
            VectorStoreQueryError: If search fails
        """
        top_k = top_k or settings.retrieval_top_k

        try:
            self.logger.debug(f"Searching for: '{query}' (top_k={top_k})")

            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query)

            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                            "similarity": 1
                            - results["distances"][0][i],  # Convert distance to similarity
                        }
                    )

            self.logger.info(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}", exc_info=True)
            raise VectorStoreQueryError(str(e))

    def get_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """
        Get document chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Document data or None if not found
        """
        try:
            results = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])

            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0],
                }

            return None

        except Exception as e:
            self.logger.error(f"Failed to get document {chunk_id}: {e}")
            return None

    def delete_by_id(self, chunk_id: str) -> bool:
        """
        Delete document chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[chunk_id])
            self.logger.info(f"Deleted chunk: {chunk_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            doc_id: Document ID

        Returns:
            Number of deleted chunks
        """
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )

            if results["ids"]:
                # Delete all chunks
                self.collection.delete(ids=results["ids"])
                count = len(results["ids"])
                self.logger.info(f"Deleted {count} chunks for document: {doc_id}")
                return count

            return 0

        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {e}")
            return 0

    def count(self) -> int:
        """
        Get total number of indexed chunks.

        Returns:
            Chunk count
        """
        try:
            return self.collection.count()
        except Exception as e:
            self.logger.error(f"Failed to get count: {e}")
            return 0

    def list_documents(self) -> list[str]:
        """
        List all unique document IDs in the store.

        Returns:
            List of document IDs
        """
        try:
            results = self.collection.get(include=["metadatas"])

            doc_ids = set()
            for metadata in results["metadatas"]:
                if "doc_id" in metadata:
                    doc_ids.add(metadata["doc_id"])

            return sorted(list(doc_ids))

        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            return []

    def reset(self) -> bool:
        """
        Delete all documents from the collection.

        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.logger.warning(f"Reset collection: {self.collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
            return False

    def _prepare_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare metadata for Chroma (convert complex types to strings).

        Args:
            metadata: Original metadata

        Returns:
            Prepared metadata
        """
        prepared = {}

        for key, value in metadata.items():
            # Chroma only supports: str, int, float, bool
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                prepared[key] = ", ".join(str(v) for v in value)
            elif value is not None:
                # Convert other types to string
                prepared[key] = str(value)

        return prepared


# Default vector store instance
default_vector_store = VectorStore()


__all__ = [
    "VectorStore",
    "default_vector_store",
]
