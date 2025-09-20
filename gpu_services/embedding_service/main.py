# gpu_services/embedding_service/main.py

import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import torch

# Import the enhanced processor instance
from embedding import embedding_processor, EnhancedEmbeddingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

processor_instance: EnhancedEmbeddingProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the Enhanced EmbeddingProcessor when the app starts"""
    global processor_instance
    logger.info("Enhanced Embedding service is starting up...")
    try:
        processor_instance = embedding_processor
        logger.info("Enhanced EmbeddingProcessor initialized successfully")
        
        # Log model information
        model_info = processor_instance.get_model_info()
        logger.info(f"Model Info: {model_info}")
        
        # Perform health check
        if processor_instance.health_check():
            logger.info("Initial health check passed ✓")
        else:
            logger.warning("Initial health check failed!")
            
    except Exception as e:
        logger.critical(f"Failed to initialize Enhanced EmbeddingProcessor: {e}")
        raise
    
    yield
    
    logger.info("Enhanced Embedding service is shutting down.")

app = FastAPI(
    title="Enhanced Embedding Service",
    description="GPU-powered embedding service optimized for docling data processing with EmbeddingGemma-300M",
    version="2.0.0",
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
    device: str
    torch_dtype: str
    max_seq_length: str
    gpu_memory_info: dict = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    health_check_passed: bool
    gpu_available: bool
    gpu_memory_info: dict = None

# --- API Endpoints ---
@app.post("/embed-documents", response_model=EmbedResponse)
async def embed_documents_endpoint(request: EmbedDocumentsRequest):
    """
    Receives a batch of text chunks (typically from docling service) 
    and returns their vector embeddings optimized for semantic search.
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
            embedding_dimension=processor_instance.model.get_sentence_embedding_dimension()
        )
        
    try:
        logger.info(f"Processing embedding request for {len(request.texts)} texts")
        embeddings = processor_instance.embed_documents(request.texts)
        
        # Count non-zero embeddings (successfully processed)
        processed_count = sum(1 for emb in embeddings if any(val != 0.0 for val in emb))
        
        return EmbedResponse(
            embeddings=embeddings,
            processed_count=processed_count,
            embedding_dimension=len(embeddings[0]) if embeddings else 0
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
    Receives a single query text and returns its vector embedding
    optimized for semantic search against document embeddings.
    """
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding processor not initialized"
        )
    
    try:
        logger.info(f"Processing query embedding request")
        embedding = processor_instance.embed_query(request.text)
        
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
    """Comprehensive health check including GPU status and model verification"""
    
    gpu_available = torch.cuda.is_available()
    gpu_memory_info = None
    
    if gpu_available:
        try:
            gpu_memory_info = {
                "device_name": torch.cuda.get_device_name(0),
                "total_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                "allocated_memory": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB",
                "cached_memory": f"{torch.cuda.memory_reserved(0) / 1024**3:.1f}GB"
            }
        except Exception as e:
            gpu_memory_info = {"error": str(e)}
    
    health_check_passed = False
    if processor_instance:
        try:
            health_check_passed = processor_instance.health_check()
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    status_value = "healthy" if (processor_instance and health_check_passed) else "unhealthy"
    
    return HealthResponse(
        status=status_value,
        model_loaded=processor_instance is not None,
        health_check_passed=health_check_passed,
        gpu_available=gpu_available,
        gpu_memory_info=gpu_memory_info
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get detailed information about the loaded embedding model"""
    
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding processor not initialized"
        )
    
    model_info = processor_instance.get_model_info()
    
    gpu_memory_info = None
    if torch.cuda.is_available():
        try:
            gpu_memory_info = {
                "device_name": torch.cuda.get_device_name(0),
                "total_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                "allocated_memory": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB"
            }
        except Exception:
            pass
    
    return ModelInfoResponse(
        model_name=model_info["model_name"],
        embedding_dimension=model_info["embedding_dimension"],
        device=model_info["device"],
        torch_dtype=model_info["torch_dtype"],
        max_seq_length=str(model_info["max_seq_length"]),
        gpu_memory_info=gpu_memory_info
    )

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Enhanced Embedding Service",
        "version": "2.0.0",
        "description": "GPU-powered embedding service optimized for docling data processing",
        "model": "google/embeddinggemma-300m",
        "optimizations": [
            "RTX 4050 (6GB VRAM) optimized batch sizes",
            "Docling text cleaning and preprocessing", 
            "L2 normalized embeddings for better similarity",
            "Robust error handling and fallbacks"
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
        host="0.0.0.0", 
        port=8002,
        log_level="info"
    )