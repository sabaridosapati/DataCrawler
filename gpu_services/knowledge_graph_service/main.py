# gpu_services/knowledge_graph_service/main.py

import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from graph_builder import graph_builder_processor, AdvancedGraphBuilderProcessor
from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

processor_instance: AdvancedGraphBuilderProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the AdvancedGraphBuilderProcessor"""
    global processor_instance
    logger.info("Knowledge Graph service is starting up...")
    try:
        processor_instance = graph_builder_processor
        logger.info("Successfully initialized AdvancedGraphBuilderProcessor")
    except Exception as e:
        logger.critical(f"Failed to initialize processor: {e}")
        raise
    
    yield
    
    logger.info("Knowledge Graph service is shutting down...")
    if processor_instance:
        processor_instance.close()

app = FastAPI(
    title="Advanced Knowledge Graph Service",
    description="A distributed service for building Neo4j knowledge graphs using remote LLM and embedding services",
    version="2.0.0",
    lifespan=lifespan
)

# --- Pydantic Models for API ---
class Chunk(BaseModel):
    chunk_index: int
    text: str
    metadata: Dict[str, Any]

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
    Build knowledge graph from text chunks using distributed services.
    Processing happens in background for better responsiveness.
    """
    if not request.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No chunks provided for graph building"
        )
    
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph builder processor not initialized"
        )
        
    logger.info(f"Received graph build request for doc_id: {request.doc_id} with {len(request.chunks)} chunks")
    
    try:
        # Start the graph building process in background
        background_tasks.add_task(
            processor_instance.build_graph_from_chunks,
            doc_id=request.doc_id,
            user_id=request.user_id,
            chunks=[chunk.model_dump() for chunk in request.chunks]
        )
        
        return BuildGraphResponse(
            message="Knowledge graph construction initiated successfully",
            doc_id=request.doc_id,
            chunks_processed=len(request.chunks),
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Failed to initiate graph construction for doc_id {request.doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate knowledge graph construction: {str(e)}"
        )

@app.post("/build-graph/sync", response_model=BuildGraphResponse)
async def build_graph_sync_endpoint(request: BuildGraphRequest):
    """
    Synchronous version of graph building (waits for completion).
    Use for testing or when you need to know when processing is complete.
    """
    if not request.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks provided for graph building"
        )
    
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph builder processor not initialized"
        )
        
    logger.info(f"Received sync graph build request for doc_id: {request.doc_id}")
    
    try:
        # Process synchronously
        await processor_instance.build_graph_from_chunks(
            doc_id=request.doc_id,
            user_id=request.user_id,
            chunks=[chunk.model_dump() for chunk in request.chunks]
        )
        
        return BuildGraphResponse(
            message="Knowledge graph construction completed successfully",
            doc_id=request.doc_id,
            chunks_processed=len(request.chunks),
            status="completed"
        )
        
    except Exception as e:
        logger.error(f"Sync graph construction failed for doc_id {request.doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge graph construction failed: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def comprehensive_health_check():
    """
    Comprehensive health check including all connected services
    """
    try:
        service_health = await processor_instance.health_check() if processor_instance else {}
        
        overall_status = "ok"
        if any(status != "ok" for status in service_health.values()):
            overall_status = "degraded"
            
        if service_health.get("neo4j") == "error":
            overall_status = "error"
            
        return HealthResponse(
            service="knowledge_graph_service",
            status=overall_status,
            version="2.0.0",
            connected_services=service_health,
            configuration={
                "neo4j_uri": settings.NEO4J_URI,
                "llm_service_url": settings.LOCAL_LLM_URL,
                "embedding_service_url": settings.EMBEDDING_SERVICE_URL,
                "extraction_temperature": settings.EXTRACTION_TEMPERATURE,
                "max_extraction_tokens": settings.MAX_EXTRACTION_TOKENS
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthResponse(
            service="knowledge_graph_service",
            status="error",
            version="2.0.0",
            connected_services={"error": str(e)},
            configuration={}
        )

@app.get("/config")
async def get_configuration():
    """
    Get current service configuration for debugging
    """
    return {
        "service": "knowledge_graph_service",
        "version": "2.0.0",
        "neo4j": {
            "uri": settings.NEO4J_URI,
            "user": settings.NEO4J_USER
        },
        "remote_services": {
            "llm_service": settings.LOCAL_LLM_URL,
            "embedding_service": settings.EMBEDDING_SERVICE_URL
        },
        "parameters": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "extraction_temperature": settings.EXTRACTION_TEMPERATURE,
            "max_extraction_tokens": settings.MAX_EXTRACTION_TOKENS
        }
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Advanced Knowledge Graph Service",
        "version": "2.0.0",
        "description": "Builds knowledge graphs using distributed LLM and embedding services",
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