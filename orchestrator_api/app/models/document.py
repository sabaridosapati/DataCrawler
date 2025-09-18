from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum
from app.models.object_id import PyObjectId

class DocumentStatus(str, Enum):
    """Enumeration for the processing status of a document."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    
class DocumentBase(BaseModel):
    """Base model for a document, containing common fields."""
    filename: str
    user_id: str

class DocumentInDB(DocumentBase):
    """Model representing a document as stored in the database."""
    id: str = Field(..., alias="_id") # Using simple string for the file-based UUID
    original_file_path: str
    extracted_content_path: str
    status: DocumentStatus = DocumentStatus.QUEUED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}

class DocumentPublic(BaseModel):
    """Model for a document's public information, safe to send to clients."""
    id: str
    filename: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None