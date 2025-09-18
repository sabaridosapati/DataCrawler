# orchestrator_api/app/main.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import auth, documents, query
from app.db import neo4j_handler, milvus_handler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    - On startup: Connects to Neo4j and Milvus databases.
    - On shutdown: Gracefully closes the database connections.
    """
    logger.info("Application startup: Connecting to databases...")
    # Connect to Neo4j
    await neo4j_handler.db_handler.connect()
    
    # Connect to Milvus (this is a synchronous call)
    milvus_handler.milvus_db_handler.connect()
    
    yield
    
    logger.info("Application shutdown: Closing database connections...")
    # Close Neo4j connection
    await neo4j_handler.db_handler.close()
    
    # Milvus connection is managed by the library's connection manager,
    # but explicit cleanup is good practice if available.
    # connections.disconnect(alias="default")

# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="The central orchestrator for a distributed document intelligence system.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# --- Middleware Configuration ---
# Configure CORS to allow requests from any origin (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routers ---
# Include all the API endpoints from the /api directory
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(documents.router, prefix=f"{settings.API_V1_STR}/documents", tags=["Documents"])
app.include_router(query.router, prefix=f"{settings.API_V1_STR}/query", tags=["Query & Chat"])

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """A simple health check endpoint."""
    return {"message": f"Welcome to the {settings.PROJECT_NAME}. API is running."}