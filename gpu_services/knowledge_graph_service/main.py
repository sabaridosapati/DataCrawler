# gpu_services/knowledge_graph_service/main.py

"""
Advanced Knowledge Graph Service for building Neo4j graphs from document chunks.
Uses local LLM and Embedding services (Gemini-based).
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from graph_builder import AdvancedGraphBuilderProcessor, get_graph_builder
from config import settings

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

processor_instance: Optional[AdvancedGraphBuilderProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the GraphBuilderProcessor"""
    global processor_instance
    logger.info("Knowledge Graph service is starting up...")
    
    try:
        processor_instance = get_graph_builder()
        logger.info("Successfully initialized GraphBuilderProcessor")
    except Exception as e:
        logger.critical(f"Failed to initialize processor: {e}")
        raise
    
    yield
    
    logger.info("Knowledge Graph service is shutting down...")
    if processor_instance:
        await processor_instance.close()


app = FastAPI(
    title="Advanced Knowledge Graph Service",
    description="Builds Neo4j knowledge graphs using Gemini-based LLM and embedding services",
    version="3.0.0",
    lifespan=lifespan
)


# --- Pydantic Models ---
class Chunk(BaseModel):
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = {}


class BuildGraphRequest(BaseModel):
    doc_id: str
    user_id: str
    chunks: List[Chunk]


class BuildGraphResponse(BaseModel):
    message: str
    doc_id: str
    chunks_processed: int
    status: str


class HealthResponse(BaseModel):
    service: str
    status: str
    version: str
    connected_services: Dict[str, str]
    configuration: Dict[str, Any]


# --- API Endpoints ---
@app.post("/build-graph", response_model=BuildGraphResponse)
async def build_graph_endpoint(
    request: BuildGraphRequest,
    background_tasks: BackgroundTasks
):
    """
    Build knowledge graph from text chunks using background processing.
    Returns immediately, processing happens asynchronously.
    """
    if not request.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No chunks provided"
        )
    
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph builder not initialized"
        )
    
    logger.info(f"Received graph build request: doc_id={request.doc_id}, chunks={len(request.chunks)}")
    
    # Add task to background
    background_tasks.add_task(
        processor_instance.build_graph_from_chunks,
        doc_id=request.doc_id,
        user_id=request.user_id,
        chunks=[chunk.model_dump() for chunk in request.chunks]
    )
    
    return BuildGraphResponse(
        message="Graph construction initiated",
        doc_id=request.doc_id,
        chunks_processed=len(request.chunks),
        status="processing"
    )


@app.post("/build-graph/sync", response_model=BuildGraphResponse)
async def build_graph_sync_endpoint(request: BuildGraphRequest):
    """
    Synchronous graph building - waits for completion.
    Use for testing or when you need confirmation.
    """
    if not request.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks provided"
        )
    
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph builder not initialized"
        )
    
    logger.info(f"Sync graph build: doc_id={request.doc_id}, chunks={len(request.chunks)}")
    
    try:
        await processor_instance.build_graph_from_chunks(
            doc_id=request.doc_id,
            user_id=request.user_id,
            chunks=[chunk.model_dump() for chunk in request.chunks]
        )
        
        return BuildGraphResponse(
            message="Graph construction completed",
            doc_id=request.doc_id,
            chunks_processed=len(request.chunks),
            status="completed"
        )
        
    except Exception as e:
        logger.error(f"Graph construction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph construction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check including all connected services"""
    try:
        service_health = {}
        overall_status = "ok"
        
        if processor_instance:
            service_health = await processor_instance.health_check()
            
            if any(s == "error" for s in service_health.values()):
                overall_status = "degraded"
            if service_health.get("neo4j") == "error":
                overall_status = "error"
        else:
            overall_status = "error"
            service_health = {"processor": "not_initialized"}
        
        return HealthResponse(
            service="knowledge_graph_service",
            status=overall_status,
            version="3.0.0",
            connected_services=service_health,
            configuration={
                "neo4j_uri": settings.NEO4J_URI,
                "gemini_model": settings.GEMINI_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthResponse(
            service="knowledge_graph_service",
            status="error",
            version="3.0.0",
            connected_services={"error": str(e)},
            configuration={}
        )


class SimilaritySearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = 20


class SimilaritySearchResponse(BaseModel):
    entities: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]


@app.post("/similarity-search", response_model=SimilaritySearchResponse)
async def similarity_search_endpoint(request: SimilaritySearchRequest):
    """
    Vector similarity search on the knowledge graph.
    Returns similar entities with their relationships and relevant chunks.
    """
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph builder not initialized"
        )
    
    try:
        results = await processor_instance.similarity_search(
            query=request.query,
            user_id=request.user_id,
            top_k=request.top_k
        )
        return SimilaritySearchResponse(**results)
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/config")
async def get_configuration():
    """Get current service configuration"""
    return {
        "service": "knowledge_graph_service",
        "version": "3.0.0",
        "neo4j": {
            "uri": settings.NEO4J_URI,
            "user": settings.NEO4J_USER
        },
        "services": {
            "llm_service": settings.LOCAL_LLM_URL,
            "embedding_service": settings.EMBEDDING_SERVICE_URL
        },
        "parameters": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "batch_size": settings.BATCH_SIZE,
            "extraction_temperature": settings.EXTRACTION_TEMPERATURE,
            "max_extraction_tokens": settings.MAX_EXTRACTION_TOKENS
        }
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Advanced Knowledge Graph Service",
        "version": "3.0.0",
        "description": "Builds knowledge graphs using Gemini-based services",
        "backend": "gemini-api",
        "endpoints": {
            "build_graph": "/build-graph (async)",
            "build_graph_sync": "/build-graph/sync",
            "health": "/health",
            "config": "/config",
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