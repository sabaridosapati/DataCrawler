# gpu_services/embedding_service/main.py

"""
High-performance Embedding Service using Gemini API.
Provides async endpoints for document and query embedding.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from embedding import embedding_processor, GeminiEmbeddingProcessor
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

processor_instance: Optional[GeminiEmbeddingProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the Embedding Processor when the app starts"""
    global processor_instance
    logger.info("Gemini Embedding service is starting up...")
    
    try:
        processor_instance = embedding_processor
        logger.info("GeminiEmbeddingProcessor initialized successfully")
        
        # Log model information
        model_info = processor_instance.get_model_info()
        logger.info(f"Model Info: {model_info}")
        
        # Perform health check
        if await processor_instance.health_check():
            logger.info("Initial health check passed ✓")
        else:
            logger.warning("Initial health check failed!")
            
    except Exception as e:
        logger.critical(f"Failed to initialize GeminiEmbeddingProcessor: {e}")
        raise
    
    yield
    
    logger.info("Gemini Embedding service is shutting down.")


app = FastAPI(
    title="Gemini Embedding Service",
    description="High-performance embedding service using Gemini API for document and query embeddings",
    version="3.0.0",
    lifespan=lifespan
)


# --- Pydantic Models for API ---
class EmbedDocumentsRequest(BaseModel):
    texts: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "This is a document chunk from docling processing.",
                    "Another chunk with technical content and data."
                ]
            }
        }


class EmbedQueryRequest(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "What are the main topics in the document?"
            }
        }


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    processed_count: int
    embedding_dimension: int


class EmbedQueryResponse(BaseModel):
    embedding: List[float]
    embedding_dimension: int


class ModelInfoResponse(BaseModel):
    model_name: str
    embedding_dimension: int
    backend: str
    document_task_type: str
    query_task_type: str
    max_batch_size: int
    max_concurrent_requests: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    health_check_passed: bool
    backend: str


# --- API Endpoints ---
@app.post("/embed-documents", response_model=EmbedResponse)
async def embed_documents_endpoint(request: EmbedDocumentsRequest):
    """
    Embed a batch of document chunks using Gemini API.
    Optimized for semantic search with RETRIEVAL_DOCUMENT task type.
    """
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding processor not initialized"
        )
    
    if not request.texts:
        return EmbedResponse(
            embeddings=[],
            processed_count=0,
            embedding_dimension=settings.EMBEDDING_DIMENSION
        )
    
    try:
        logger.info(f"Processing embedding request for {len(request.texts)} texts")
        embeddings = await processor_instance.embed_documents_async(request.texts)
        
        # Count non-zero embeddings (successfully processed)
        processed_count = sum(
            1 for emb in embeddings 
            if any(val != 0.0 for val in emb)
        )
        
        return EmbedResponse(
            embeddings=embeddings,
            processed_count=processed_count,
            embedding_dimension=len(embeddings[0]) if embeddings else settings.EMBEDDING_DIMENSION
        )
        
    except Exception as e:
        logger.error(f"Error during document embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate document embeddings: {str(e)}"
        )


@app.post("/embed-query", response_model=EmbedQueryResponse)
async def embed_query_endpoint(request: EmbedQueryRequest):
    """
    Embed a single query using Gemini API.
    Optimized for search with RETRIEVAL_QUERY task type.
    """
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding processor not initialized"
        )
    
    try:
        logger.info("Processing query embedding request")
        embedding = await processor_instance.embed_query_async(request.text)
        
        return EmbedQueryResponse(
            embedding=embedding,
            embedding_dimension=len(embedding)
        )
        
    except Exception as e:
        logger.error(f"Error during query embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embedding: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check including API connectivity"""
    health_check_passed = False
    
    if processor_instance:
        try:
            health_check_passed = await processor_instance.health_check()
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    status_value = "healthy" if (processor_instance and health_check_passed) else "unhealthy"
    
    return HealthResponse(
        status=status_value,
        model_loaded=processor_instance is not None,
        health_check_passed=health_check_passed,
        backend="gemini-api"
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get detailed information about the embedding configuration"""
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding processor not initialized"
        )
    
    info = processor_instance.get_model_info()
    return ModelInfoResponse(**info)


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Gemini Embedding Service",
        "version": "3.0.0",
        "description": "High-performance embedding service using Gemini API",
        "model": settings.EMBEDDING_MODEL,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "backend": "gemini-api",
        "features": [
            "Async batch processing",
            "RETRIEVAL_DOCUMENT task type for documents",
            "RETRIEVAL_QUERY task type for queries",
            "L2 normalized embeddings",
            "Docling artifact cleaning",
            "Retry logic with exponential backoff"
        ],
        "endpoints": {
            "embed_documents": "/embed-documents",
            "embed_query": "/embed-query",
            "health": "/health",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )