import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status

from app.core.security import get_current_user
from app.models.user import UserInDB
from app.models.document import DocumentInDB, DocumentPublic
from app.db.mongo_handler import create_document_record, get_document_by_id
from app.services.processing_pipeline import start_document_processing

router = APIRouter()
DATA_DIR = Path("./data/user_files")

@router.post("/upload", response_model=DocumentPublic, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Uploads a document for processing. The process runs in the background.
    1. Saves the raw file to a user-specific directory.
    2. Creates a tracking record in MongoDB linked to the user's _id.
    3. Triggers the background processing pipeline.
    """
    user_id = str(current_user.id)
    
    user_dir = DATA_DIR / user_id
    raw_dir = user_dir / "raw"
    extracted_dir = user_dir / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(exist_ok=True)

    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    saved_file_path = raw_dir / f"{file_id}{file_extension}"
    
    try:
        with open(saved_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file on server: {e}")

    doc_in_db = DocumentInDB(
        _id=file_id,
        user_id=user_id,
        filename=file.filename,
        original_file_path=str(saved_file_path),
        extracted_content_path=str(extracted_dir / f"{file_id}.md"),
    )
    await create_document_record(doc_in_db)

    background_tasks.add_task(start_document_processing, doc_id=file_id)

    return doc_in_db

@router.get("/{doc_id}/status", response_model=DocumentPublic)
async def get_document_status(doc_id: str, current_user: UserInDB = Depends(get_current_user)):
    """
    Retrieves the current processing status of a document.
    Ensures the user requesting the status is the owner of the document.
    """
    doc = await get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    if doc.user_id != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this document's status")
    
    return doc