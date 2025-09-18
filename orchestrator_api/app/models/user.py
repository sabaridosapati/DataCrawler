from pydantic import BaseModel, EmailStr, Field
from typing import Optional

from app.models.object_id import PyObjectId

class UserBase(BaseModel):
    """Base model for a user, containing common fields."""
    username: EmailStr = Field(..., description="User's email address, used for login.")
    
    class Config:
        # Configuration for Pydantic model
        populate_by_name = True
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "username": "user@example.com"
            }
        }

class UserCreate(UserBase):
    """Model used when creating a new user."""
    password: str = Field(..., min_length=8, description="User's password (min 8 characters).")

class UserInDB(UserBase):
    """Model representing a user as stored in the database."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    hashed_password: str

class UserPublic(UserBase):
    """Model representing a user's public information, safe to send to clients."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")