"""
Cohere Reranker for post-retrieval reranking.

This module integrates Cohere's rerank-v3.5 model to improve
retrieval precision by reranking candidate documents.
"""

from typing import Any

import cohere

from config.settings import settings
from src.core.logging import LoggerMixin, log_execution_time


class CohereReranker(LoggerMixin):
    """
    Reranker using Cohere's rerank API.
    
    Improves retrieval precision by reranking candidate documents
    based on their relevance to the query.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key (uses settings if None)
            model: Reranking model name (uses settings if None)
        """
        self.api_key = api_key or settings.cohere_api_key
        self.model = model or settings.rerank_model
        
        # Initialize Cohere client
        self.client = cohere.Client(self.api_key)
        
        self.logger.info(f"CohereReranker initialized with model: {self.model}")
    
    @log_execution_time
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int | None = None,
        return_documents: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of documents with 'document' field
            top_n: Number of results to return (uses settings if None)
            return_documents: Whether to return document content
            
        Returns:
            Reranked documents with relevance scores
        """
        if not documents:
            return []
        
        top_n = top_n or settings.rerank_top_n
        
        # Extract document texts
        doc_texts = []
        for doc in documents:
            text = doc.get("document", "")
            if not text and "content" in doc:
                text = doc["content"]
            doc_texts.append(text)
        
        # Filter empty documents
        valid_indices = [i for i, text in enumerate(doc_texts) if text.strip()]
        valid_texts = [doc_texts[i] for i in valid_indices]
        
        if not valid_texts:
            self.logger.warning("No valid document texts for reranking")
            return documents[:top_n]
        
        try:
            # Call Cohere rerank API
            response = self.client.rerank(
                query=query,
                documents=valid_texts,
                model=self.model,
                top_n=min(top_n, len(valid_texts)),
                return_documents=return_documents,
            )
            
            # Build reranked results
            reranked = []
            for result in response.results:
                original_idx = valid_indices[result.index]
                original_doc = documents[original_idx].copy()
                
                # Add rerank metadata
                original_doc["rerank_score"] = result.relevance_score
                original_doc["rerank_index"] = result.index
                
                # Update document text if returned
                if return_documents and hasattr(result, "document") and result.document:
                    original_doc["document"] = result.document.text
                
                reranked.append(original_doc)
            
            self.logger.info(
                f"Reranked {len(documents)} docs → top {len(reranked)} "
                f"(best score: {reranked[0]['rerank_score']:.3f})"
            )
            
            return reranked
            
        except cohere.errors.BadRequestError as e:
            self.logger.error(f"Cohere API error: {e}")
            # Fallback to original order
            return documents[:top_n]
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if reranker is available (API key configured)."""
        return bool(self.api_key and self.api_key.strip())


# Default reranker instance (may not be available if API key not set)
try:
    default_reranker = CohereReranker()
except Exception:
    default_reranker = None


__all__ = [
    "CohereReranker",
    "default_reranker",
]
