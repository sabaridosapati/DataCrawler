from pydantic import BaseModel
from typing import Optional

class Token(BaseModel):
    """Response model for a successful login, providing the access token."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Data model for the content encoded within a JWT."""
    username: Optional[str] = None