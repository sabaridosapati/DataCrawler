from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class UserBase(BaseModel):
    """Base model for a user, containing common fields."""
    username: EmailStr = Field(..., description="User's email address, used for login.")
    
    class Config:
        populate_by_name = True
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
    id: Optional[str] = Field(default=None, alias="_id")
    hashed_password: str


class UserPublic(UserBase):
    """Model representing a user's public information, safe to send to clients."""
    id: Optional[str] = Field(default=None)