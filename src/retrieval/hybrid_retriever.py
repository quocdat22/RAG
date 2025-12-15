"""
Hybrid Retriever combining BM25 and Vector search.

This module implements hybrid search using:
- BM25 (keyword-based) for exact term matching
- Vector search (semantic) for conceptual similarity
- Reciprocal Rank Fusion (RRF) for combining results
"""

from typing import Any

from rank_bm25 import BM25Okapi

from config.settings import settings
from src.core.logging import LoggerMixin, log_execution_time
from src.embedding.vector_store import VectorStore, default_vector_store


class HybridRetriever(LoggerMixin):
    """
    Hybrid retriever combining BM25 and vector search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both
    retrieval methods for improved recall and precision.
    """
    
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for semantic search
            bm25_k1: BM25 k1 parameter (term frequency saturation)
            bm25_b: BM25 b parameter (document length normalization)
        """
        self.vector_store = vector_store or default_vector_store
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        
        # BM25 index state
        self._bm25: BM25Okapi | None = None
        self._corpus: list[str] = []
        self._doc_ids: list[str] = []
        self._doc_metadata: list[dict] = []
        
        self.logger.info("HybridRetriever initialized")
    
    def build_bm25_index(self) -> int:
        """
        Build BM25 index from all documents in vector store.
        
        Returns:
            Number of documents indexed
        """
        try:
            # Get all documents from vector store
            all_docs = self._get_all_documents()
            
            if not all_docs:
                self.logger.warning("No documents found for BM25 indexing")
                return 0
            
            # Extract texts and build corpus
            self._corpus = []
            self._doc_ids = []
            self._doc_metadata = []
            
            for doc in all_docs:
                text = doc.get("document", "")
                if text:
                    # Tokenize for BM25
                    tokens = self._tokenize(text)
                    self._corpus.append(tokens)
                    self._doc_ids.append(doc.get("id", ""))
                    self._doc_metadata.append(doc.get("metadata", {}))
            
            # Build BM25 index
            if self._corpus:
                self._bm25 = BM25Okapi(
                    self._corpus,
                    k1=self.bm25_k1,
                    b=self.bm25_b
                )
                self.logger.info(f"BM25 index built with {len(self._corpus)} documents")
            
            return len(self._corpus)
            
        except Exception as e:
            self.logger.error(f"Failed to build BM25 index: {e}")
            raise
    
    def _get_all_documents(self) -> list[dict]:
        """Get all documents from vector store."""
        try:
            # Get all document IDs
            collection = self.vector_store.collection
            result = collection.get()
            
            documents = []
            if result and result.get("ids"):
                for i, doc_id in enumerate(result["ids"]):
                    documents.append({
                        "id": doc_id,
                        "document": result["documents"][i] if result.get("documents") else "",
                        "metadata": result["metadatas"][i] if result.get("metadatas") else {},
                    })
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to get documents: {e}")
            return []
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on whitespace/punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    @log_execution_time
    def search(
        self,
        query: str,
        top_k: int | None = None,
        alpha: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for vector search (1.0=pure vector, 0.0=pure BM25)
            where: Metadata filter for vector search
            
        Returns:
            List of search results with combined scores
        """
        top_k = top_k or settings.retrieval_top_k
        alpha = alpha if alpha is not None else getattr(settings, 'hybrid_alpha', 0.5)
        
        # Ensure BM25 index exists
        if self._bm25 is None:
            self.build_bm25_index()
        
        # Get results from both methods
        vector_results = self._vector_search(query, top_k * 2, where)
        bm25_results = self._bm25_search(query, top_k * 2) if self._bm25 else []
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            alpha=alpha,
            k=60  # RRF constant
        )
        
        # Return top_k results
        return combined[:top_k]
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Perform vector similarity search."""
        try:
            results = self.vector_store.search(query, top_k=top_k, where=where)
            # Add rank for RRF
            for i, result in enumerate(results):
                result["vector_rank"] = i + 1
            return results
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Perform BM25 keyword search."""
        if not self._bm25 or not self._corpus:
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            
            # Get BM25 scores
            scores = self._bm25.get_scores(query_tokens)
            
            # Get top results
            scored_docs = list(zip(range(len(scores)), scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, (idx, score) in enumerate(scored_docs[:top_k]):
                if score > 0:  # Only include docs with positive score
                    # Reconstruct document text from tokens
                    doc_text = " ".join(self._corpus[idx])
                    results.append({
                        "id": self._doc_ids[idx],
                        "document": doc_text,
                        "metadata": self._doc_metadata[idx],
                        "bm25_score": float(score),
                        "bm25_rank": i + 1,
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"BM25 search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        alpha: float = 0.5,
        k: int = 60,
    ) -> list[dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(1 / (k + rank))
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search  
            alpha: Weight for vector results (0-1)
            k: RRF constant (default 60)
            
        Returns:
            Combined and re-ranked results
        """
        # Build score map by document ID
        scores: dict[str, dict] = {}
        
        # Add vector scores
        for result in vector_results:
            doc_id = result["id"]
            rank = result.get("vector_rank", 1)
            rrf_score = alpha * (1 / (k + rank))
            
            if doc_id not in scores:
                scores[doc_id] = {
                    "id": doc_id,
                    "document": result.get("document", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity": result.get("similarity", 0),
                    "rrf_score": 0,
                    "vector_rank": rank,
                    "bm25_rank": None,
                }
            scores[doc_id]["rrf_score"] += rrf_score
        
        # Add BM25 scores
        for result in bm25_results:
            doc_id = result["id"]
            rank = result.get("bm25_rank", 1)
            rrf_score = (1 - alpha) * (1 / (k + rank))
            
            if doc_id not in scores:
                scores[doc_id] = {
                    "id": doc_id,
                    "document": result.get("document", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity": 0,
                    "rrf_score": 0,
                    "vector_rank": None,
                    "bm25_rank": rank,
                }
            else:
                scores[doc_id]["bm25_rank"] = rank
                
            scores[doc_id]["rrf_score"] += rrf_score
            scores[doc_id]["bm25_score"] = result.get("bm25_score", 0)
        
        # Sort by RRF score
        combined = list(scores.values())
        combined.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        self.logger.debug(
            f"RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 "
            f"= {len(combined)} unique results"
        )
        
        return combined
    
    def refresh_index(self) -> int:
        """Refresh BM25 index with current documents."""
        self._bm25 = None
        self._corpus = []
        self._doc_ids = []
        self._doc_metadata = []
        return self.build_bm25_index()


# Default hybrid retriever instance
default_hybrid_retriever = HybridRetriever()


__all__ = [
    "HybridRetriever",
    "default_hybrid_retriever",
]
