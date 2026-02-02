# gpu_services/docling_service/main.py

"""
Document Processing Service using Docling.
Simplified for Docling 2.x API.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional
import tempfile
import shutil

from processing import docling_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    logger.info("Docling service is starting up...")
    
    try:
        # Log processor info
        info = docling_processor.get_info()
        logger.info(f"Processor Info: {info}")
    except Exception as e:
        logger.warning(f"Could not get processor info: {e}")
    
    yield
    
    logger.info("Docling service is shutting down.")


app = FastAPI(
    title="Docling Document Processing Service",
    description="Document processing service using Docling",
    version="2.0.0",
    lifespan=lifespan
)


# --- Pydantic Models ---
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
    success: bool
    message: str
    extracted_markdown_path: Optional[str] = None
    extracted_chunks_path: Optional[str] = None
    num_chunks: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    processor: str
    device: str
    supported_formats: list


# --- API Endpoints ---
@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """
    Process a document file from disk.
    """
    logger.info(f"Processing request for: {request.input_file_path}")
    
    input_path = Path(request.input_file_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Input file not found: {request.input_file_path}"
        )
    
    try:
        result = docling_processor.process_document(
            input_file=str(input_path),
            output_dir=request.output_directory_path
        )
        
        return ProcessResponse(
            success=True,
            message="Document processed successfully",
            extracted_markdown_path=result.get("extracted_markdown_path"),
            extracted_chunks_path=result.get("extracted_chunks_path"),
            num_chunks=result.get("num_chunks")
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@app.post("/process-upload", response_model=ProcessResponse)
async def process_uploaded_file(file: UploadFile = File(...)):
    """
    Process an uploaded file.
    """
    logger.info(f"Processing uploaded file: {file.filename}")
    
    # Create temp directory for this request
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save uploaded file
        input_file = temp_path / file.filename
        with open(input_file, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        output_dir = temp_path / "output"
        output_dir.mkdir()
        
        try:
            result = docling_processor.process_document(
                input_file=str(input_file),
                output_dir=str(output_dir)
            )
            
            # Read the chunks file content
            chunks_path = result.get("extracted_chunks_path")
            if chunks_path and Path(chunks_path).exists():
                import json
                with open(chunks_path) as f:
                    chunks = json.load(f)
            else:
                chunks = []
            
            return ProcessResponse(
                success=True,
                message="Document processed successfully",
                num_chunks=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {str(e)}"
            )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    info = docling_processor.get_info()
    return HealthResponse(
        status="healthy",
        processor=info.get("processor", "DoclingProcessor"),
        device=info.get("device", "cpu"),
        supported_formats=info.get("supported_formats", [])
    )


@app.get("/info")
async def get_info():
    """Get processor information."""
    return docling_processor.get_info()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Docling Document Processing Service",
        "version": "2.0.0",
        "endpoints": {
            "process": "/process",
            "process_upload": "/process-upload",
            "health": "/health",
            "info": "/info",
            "docs": "/docs"
        },
        "supported_formats": [
            "PDF", "DOCX", "PPTX", "XLSX", "HTML", "MD", "TXT"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")