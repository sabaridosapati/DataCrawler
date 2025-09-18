# gpu_services/knowledge_graph_service/main.py

import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from graph_builder import graph_builder_processor, GraphBuilderProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

processor_instance: GraphBuilderProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and closes the GraphBuilderProcessor."""
    global processor_instance
    logger.info("Knowledge Graph service is starting up...")
    processor_instance = graph_builder_processor
    yield
    logger.info("Knowledge Graph service is shutting down.")
    processor_instance.close()

app = FastAPI(
    title="Knowledge Graph Service",
    description="A worker for building a Neo4j graph from text using a local LLM.",
    version="1.0.0",
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

# --- API Endpoint ---
@app.post("/build-graph", response_model=BuildGraphResponse)
async def build_graph_endpoint(request: BuildGraphRequest):
    """
    Receives text chunks and orchestrates the knowledge graph construction.
    """
    if not request.chunks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No chunks provided.")
        
    try:
        await processor_instance.build_graph_from_chunks(
            doc_id=request.doc_id,
            user_id=request.user_id,
            chunks=request.chunks
        )
        return BuildGraphResponse(
            message="Knowledge graph construction initiated successfully.",
            doc_id=request.doc_id
        )
    except Exception as e:
        logger.error(f"Error during graph construction for doc_id {request.doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to build knowledge graph.")

@app.get("/health")
def health_check():
    return {"status": "ok", "processor_initialized": processor_instance is not None}```

