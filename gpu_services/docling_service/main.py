# gpu_services/docling_service/main.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

# Import the processor instance we created
from processing import docling_processor, DoclingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This global variable will hold our processor instance
processor_instance: DoclingProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager. This is the modern way to handle startup/shutdown.
    It initializes the heavy DoclingProcessor when the app starts.
    """
    global processor_instance
    logger.info("Docling service is starting up...")
    # Initialize the processor and store it in the global variable
    processor_instance = docling_processor
    yield
    logger.info("Docling service is shutting down.")
    # No explicit cleanup needed for docling, but this is where it would go.

app = FastAPI(
    title="Docling Processing Service",
    description="A GPU-powered worker for document and audio extraction.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models for API ---
class ProcessRequest(BaseModel):
    input_file_path: str
    output_directory_path: str

class ProcessResponse(BaseModel):
    message: str
    extracted_markdown_path: str
    extracted_chunks_path: str

# --- API Endpoint ---
@app.post("/process", response_model=ProcessResponse)
async def process_document_endpoint(request: ProcessRequest):
    """
    Receives a file path, processes it using the appropriate engine (Docling/AssemblyAI),
    saves the output, and returns the paths to the extracted content.
    """
    logger.info(f"Received processing request for: {request.input_file_path}")
    try:
        if processor_instance is None:
            # This should not happen with the lifespan manager, but it's a good safeguard
            raise HTTPException(status_code=503, detail="Processor not initialized")

        result_paths = await processor_instance.process_file(
            request.input_file_path,
            request.output_directory_path
        )
        
        return ProcessResponse(
            message="File processed successfully.",
            extracted_markdown_path=result_paths["markdown_path"],
            extracted_chunks_path=result_paths["chunks_path"]
        )
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        logger.error(f"Unsupported file type error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal processing error occurred: {e}")

@app.get("/health")
def health_check():
    """A simple health check endpoint."""
    return {"status": "ok", "processor_initialized": processor_instance is not None}