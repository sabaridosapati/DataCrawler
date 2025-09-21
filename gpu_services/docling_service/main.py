# gpu_services/docling_service/main.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any

# Import the processor instance we created
from processing import granite_processor, GraniteDoclingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This global variable will hold our processor instance
processor_instance: GraniteDoclingProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager. This is the modern way to handle startup/shutdown.
    It initializes the heavy GraniteDoclingProcessor when the app starts.
    """
    global processor_instance
    logger.info("Granite Docling service is starting up...")
    try:
        # Initialize the processor and store it in the global variable
        processor_instance = granite_processor
        logger.info("GraniteDoclingProcessor initialized successfully")
        
        # Log model information
        model_info = processor_instance.get_model_info()
        logger.info(f"Model Info: {model_info}")
        
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize GraniteDoclingProcessor: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("Granite Docling service is shutting down.")

app = FastAPI(
    title="Granite Docling Processing Service",
    description="A GPU-powered document processing service using granite-docling-258M-mlx for Apple Silicon",
    version="2.0.0",
    lifespan=lifespan
)

# --- Pydantic Models for API ---
class ProcessRequest(BaseModel):
    input_file_path: str
    output_directory_path: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "input_file_path": "/path/to/document.pdf",
                "output_directory_path": "/path/to/output"
            }
        }

class ProcessResponse(BaseModel):
    message: str
    extracted_markdown_path: str
    extracted_chunks_path: str
    
class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    platform: str
    supported_formats: Dict[str, list]
    features: list

class HealthResponse(BaseModel):
    status: str
    processor_initialized: bool
    model_info: Dict[str, Any] = None

# --- API Endpoints ---
@app.post("/process", response_model=ProcessResponse)
async def process_document_endpoint(request: ProcessRequest):
    """
    Receives a file path, processes it using the Granite-Docling MLX model,
    saves the output, and returns the paths to the extracted content.
    """
    logger.info(f"Received processing request for: {request.input_file_path}")
    
    if processor_instance is None:
        # This should not happen with the lifespan manager, but it's a good safeguard
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Granite Docling processor not initialized"
        )
    
    try:
        result_paths = await processor_instance.process_file(
            input_path=request.input_file_path,
            output_dir=request.output_directory_path
        )
        
        logger.info(f"Processing completed successfully for: {request.input_file_path}")
        
        return ProcessResponse(
            message="File processed successfully with Granite-Docling MLX",
            extracted_markdown_path=result_paths["markdown_path"],
            extracted_chunks_path=result_paths["chunks_path"]
        )
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Input file not found: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Unsupported file type error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint with model information"""
    model_info = None
    
    if processor_instance:
        try:
            model_info = processor_instance.get_model_info()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
    
    return HealthResponse(
        status="healthy" if processor_instance else "unhealthy",
        processor_initialized=processor_instance is not None,
        model_info=model_info
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed information about the loaded Granite-Docling model"""
    if processor_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Granite Docling processor not initialized"
        )
    
    try:
        model_info = processor_instance.get_model_info()
        return ModelInfoResponse(
            model_name=model_info["model_name"],
            model_type=model_info["model_type"],
            platform=model_info["platform"],
            supported_formats=model_info["supported_formats"],
            features=model_info["features"]
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Granite Docling Processing Service",
        "version": "2.0.0",
        "description": "Document processing service using granite-docling-258M-mlx for Apple Silicon",
        "model": "granite-docling-258M-mlx",
        "platform": "Apple Silicon (MLX)",
        "endpoints": {
            "process": "/process",
            "health": "/health",
            "model_info": "/model-info",
            "docs": "/docs"
        },
        "supported_formats": [
            "PDF", "DOCX", "PPTX", "HTML", "MD", "TXT",
            "PNG", "JPG", "JPEG", "BMP", "TIFF", "WEBP",
            "MP3", "WAV", "M4A", "FLAC", "OGG", "MP4"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8004,
        log_level="info"
    )