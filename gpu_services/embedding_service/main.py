# gpu_services/embedding_service/main.py

import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

# Import the processor instance
from embedding import embedding_processor, EmbeddingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

processor_instance: EmbeddingProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the heavy EmbeddingProcessor when the app starts."""
    global processor_instance
    logger.info("Embedding service is starting up...")
    processor_instance = embedding_processor
    yield
    logger.info("Embedding service is shutting down.")

app = FastAPI(
    title="Embedding Service",
    description="A GPU-powered worker for generating text embeddings with EmbeddingGemma.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models for API ---
class EmbedDocumentsRequest(BaseModel):
    texts: List[str]

class EmbedQueryRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class EmbedQueryResponse(BaseModel):
    embedding: List[float]

# --- API Endpoints ---
@app.post("/embed-documents", response_model=EmbedResponse)
async def embed_documents_endpoint(request: EmbedDocumentsRequest):
    """
    Receives a batch of text chunks and returns their vector embeddings.
    """
    if not request.texts:
        return EmbedResponse(embeddings=[])
        
    try:
        embeddings = processor_instance.embed_documents(request.texts)
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error during document embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate document embeddings.")

@app.post("/embed-query", response_model=EmbedQueryResponse)
async def embed_query_endpoint(request: EmbedQueryRequest):
    """
    Receives a single query text and returns its vector embedding.
    """
    try:
        embedding = processor_instance.embed_query(request.text)
        return EmbedQueryResponse(embedding=embedding)
    except Exception as e:
        logger.error(f"Error during query embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

@app.get("/health")
def health_check():
    """A simple health check endpoint."""
    return {"status": "ok", "model_initialized": processor_instance is not None}