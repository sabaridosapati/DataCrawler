# orchestrator_api/app/api/documents.py

"""
Document Management API with proper file handling and user isolation.
"""

import uuid
import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status

from app.core.security import get_current_user
from app.models.user import UserInDB
from app.models.document import DocumentInDB, DocumentPublic, DocumentStatus
from app.db.mongo_handler import create_document_record, get_document_by_id, document_collection
from app.services.processing_pipeline import start_document_processing

router = APIRouter()

# Use absolute path for data directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # orchestrator_api root
DATA_DIR = BASE_DIR / "data" / "user_files"


@router.post("/upload", response_model=DocumentPublic, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Uploads a document for processing.
    1. Saves the raw file to a user-specific directory.
    2. Creates a tracking record in MongoDB.
    3. Triggers background processing pipeline.
    """
    user_id = str(current_user.id)
    
    # Create user directories with absolute paths
    user_dir = DATA_DIR / user_id
    raw_dir = user_dir / "raw"
    extracted_dir = user_dir / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    saved_file_path = raw_dir / f"{file_id}{file_extension}"
    
    # Save file
    try:
        content = await file.read()
        with open(saved_file_path, "wb") as buffer:
            buffer.write(content)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Create document record with ABSOLUTE paths
    doc_in_db = DocumentInDB(
        _id=file_id,
        user_id=user_id,
        filename=file.filename,
        original_file_path=str(saved_file_path.absolute()),
        extracted_content_path=str((extracted_dir / f"{file_id}.md").absolute()),
    )
    await create_document_record(doc_in_db)

    # Start background processing
    background_tasks.add_task(start_document_processing, doc_id=file_id)

    return doc_in_db


@router.get("/", response_model=List[DocumentPublic])
async def list_documents(current_user: UserInDB = Depends(get_current_user)):
    """
    List all documents for the current user.
    """
    user_id = str(current_user.id)
    
    # Query MongoDB for user's documents
    cursor = document_collection.find({"user_id": user_id})
    documents = await cursor.to_list(length=100)
    
    # Convert to Pydantic models
    result = []
    for doc in documents:
        # Convert _id to id
        doc["id"] = doc.pop("_id")
        result.append(DocumentPublic(**doc))
    
    return result


@router.get("/{doc_id}", response_model=DocumentPublic)
async def get_document(doc_id: str, current_user: UserInDB = Depends(get_current_user)):
    """
    Get a specific document by ID.
    """
    doc = await get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    if doc.user_id != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    
    return doc


@router.get("/{doc_id}/status", response_model=DocumentPublic)
async def get_document_status(doc_id: str, current_user: UserInDB = Depends(get_current_user)):
    """
    Get the processing status of a document.
    """
    return await get_document(doc_id, current_user)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, current_user: UserInDB = Depends(get_current_user)):
    """
    Delete a document and its associated data.
    """
    doc = await get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    if doc.user_id != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    
    # Delete file from disk
    try:
        if doc.original_file_path and Path(doc.original_file_path).exists():
            Path(doc.original_file_path).unlink()
        if doc.extracted_content_path and Path(doc.extracted_content_path).exists():
            Path(doc.extracted_content_path).unlink()
    except Exception as e:
        pass  # File deletion errors are non-fatal
    
    # Delete from MongoDB
    await document_collection.delete_one({"_id": doc_id})
    
    # TODO: Delete from Milvus vectors
    # TODO: Delete from Neo4j graph
    
    return None