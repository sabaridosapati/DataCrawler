from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone

from app.models.object_id import PyObjectId

class ChatMessage(BaseModel):
    """Represents a single message within a chat session."""
    role: str # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_number: int

class ChatSession(BaseModel):
    """Represents an entire conversation session as stored in MongoDB."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    chat_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    history: List[ChatMessage] = []

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {PyObjectId: str, datetime: lambda dt: dt.isoformat()}

class ChatRequest(BaseModel):
    """Request model for the /query endpoint."""
    prompt: str
    chat_id: Optional[str] = None

class ChatTurnResponse(BaseModel):
    """Response model for a single turn in a conversation."""
    chat_id: str
    ai_response: str
    turn_number: int