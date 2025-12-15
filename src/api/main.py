"""
FastAPI Application for RAG System.

Provides REST API endpoints for:
- Document querying
- Document management
- Health checks
- System status
"""

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG System API",
    description="AI-powered document analysis and Q&A system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Request/Response Models
# ==============================================================================

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, max_length=5000, description="User query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    query_type: str = Field(default="SIMPLE", description="Query type: SIMPLE, ANALYTICAL")
    use_hybrid: bool = Field(default=True, description="Use hybrid search")
    use_reranking: bool = Field(default=True, description="Use Cohere reranking")


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: list[str]
    source_documents: list[dict[str, Any]]
    query_type: str
    token_usage: dict[str, Any] | None = None


class DocumentInfo(BaseModel):
    """Document information model."""
    doc_id: str
    filename: str | None = None
    chunk_count: int
    category: str | None = None
    file_type: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: dict[str, str]


class StatsResponse(BaseModel):
    """System statistics response."""
    total_documents: int
    total_chunks: int
    total_cost: float
    model_name: str


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to RAG System API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    components = {}
    
    # Check vector store
    try:
        from src.embedding import default_vector_store
        default_vector_store.count()
        components["vector_store"] = "healthy"
    except Exception as e:
        components["vector_store"] = f"unhealthy: {str(e)}"
    
    # Check LLM
    try:
        from src.generation import default_llm_client
        components["llm"] = "healthy"
    except Exception as e:
        components["llm"] = f"unhealthy: {str(e)}"
    
    # Check reranker
    try:
        from src.retrieval.reranker import default_reranker
        components["reranker"] = "healthy" if default_reranker else "not configured"
    except Exception:
        components["reranker"] = "not configured"
    
    status = "healthy" if all(v == "healthy" for v in components.values() if v != "not configured") else "degraded"
    
    return HealthResponse(
        status=status,
        version="2.0.0",
        components=components,
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get system statistics."""
    try:
        from src.embedding import default_vector_store
        from src.generation import default_llm_client
        
        doc_count = len(default_vector_store.list_documents())
        chunk_count = default_vector_store.count()
        usage = default_llm_client.get_usage_stats()
        
        return StatsResponse(
            total_documents=doc_count,
            total_chunks=chunk_count,
            total_cost=usage.get("total_cost", 0),
            model_name=settings.model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query documents and get AI-generated answer.
    
    Supports:
    - Hybrid search (BM25 + Vector)
    - Cohere reranking
    - Multiple query types
    """
    try:
        from src.embedding import default_vector_store
        from src.generation import default_synthesizer
        
        # Search
        if request.use_hybrid:
            try:
                from src.retrieval.hybrid_retriever import default_hybrid_retriever
                results = default_hybrid_retriever.search(
                    query=request.query,
                    top_k=request.top_k * 2 if request.use_reranking else request.top_k,
                )
            except Exception:
                results = default_vector_store.search(
                    request.query,
                    top_k=request.top_k * 2 if request.use_reranking else request.top_k,
                )
        else:
            results = default_vector_store.search(
                request.query,
                top_k=request.top_k * 2 if request.use_reranking else request.top_k,
            )
        
        if not results:
            return QueryResponse(
                answer="No relevant documents found for your query.",
                sources=[],
                source_documents=[],
                query_type=request.query_type,
            )
        
        # Rerank
        if request.use_reranking:
            try:
                from src.retrieval.reranker import default_reranker
                if default_reranker:
                    results = default_reranker.rerank(
                        query=request.query,
                        documents=results,
                        top_n=request.top_k,
                    )
                else:
                    results = results[:request.top_k]
            except Exception:
                results = results[:request.top_k]
        else:
            results = results[:request.top_k]
        
        # Synthesize
        response = default_synthesizer.synthesize(
            query=request.query,
            retrieved_docs=results,
            query_type=request.query_type,
        )
        
        return QueryResponse(
            answer=response["answer"],
            sources=response.get("sources", []),
            source_documents=response.get("source_documents", []),
            query_type=response.get("query_type", request.query_type),
            token_usage=response.get("token_usage"),
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    try:
        from src.embedding import default_vector_store
        
        documents = default_vector_store.list_documents()
        
        result = []
        for doc_id in documents:
            # Get document info
            doc_info = _get_document_info(doc_id)
            result.append(doc_info)
        
        return {
            "total": len(result),
            "documents": result,
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: str):
    """Get document details by ID."""
    try:
        from src.embedding import default_vector_store
        
        collection = default_vector_store.collection
        result = collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        
        if not result or not result.get("ids"):
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        chunks = []
        for i, chunk_id in enumerate(result["ids"]):
            chunks.append({
                "id": chunk_id,
                "content": result["documents"][i] if result.get("documents") else "",
                "metadata": result["metadatas"][i] if result.get("metadatas") else {},
            })
        
        metadata = chunks[0]["metadata"] if chunks else {}
        
        return {
            "doc_id": doc_id,
            "filename": metadata.get("filename"),
            "chunk_count": len(chunks),
            "metadata": metadata,
            "chunks": chunks,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document by ID."""
    try:
        from src.embedding import default_vector_store
        
        deleted_count = default_vector_store.delete_by_doc_id(doc_id)
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        
        return {
            "message": f"Document '{doc_id}' deleted successfully",
            "deleted_chunks": deleted_count,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents", tags=["Documents"])
async def delete_all_documents():
    """Delete all documents (use with caution!)."""
    try:
        from src.embedding import default_vector_store
        
        default_vector_store.reset()
        
        return {
            "message": "All documents deleted successfully",
        }
        
    except Exception as e:
        logger.error(f"Failed to delete all documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Helper Functions
# ==============================================================================

def _get_document_info(doc_id: str) -> dict:
    """Get basic document information."""
    try:
        from src.embedding import default_vector_store
        
        collection = default_vector_store.collection
        result = collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        
        if result and result.get("ids"):
            metadata = result["metadatas"][0] if result.get("metadatas") else {}
            return {
                "doc_id": doc_id,
                "filename": metadata.get("filename"),
                "chunk_count": len(result["ids"]),
                "category": metadata.get("category"),
                "file_type": metadata.get("file_type"),
            }
        
        return {
            "doc_id": doc_id,
            "filename": None,
            "chunk_count": 0,
            "category": None,
            "file_type": None,
        }
        
    except Exception:
        return {
            "doc_id": doc_id,
            "filename": None,
            "chunk_count": 0,
            "category": None,
            "file_type": None,
        }


# ==============================================================================
# Run Server
# ==============================================================================

def run_server(host: str = None, port: int = None):
    """Run the FastAPI server."""
    import uvicorn
    
    host = host or settings.api_host
    port = port or settings.api_port
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
