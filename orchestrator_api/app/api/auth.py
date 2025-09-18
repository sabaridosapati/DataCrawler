# orchestrator_api/app/api/auth.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.security import create_access_token, get_password_hash, verify_password, get_current_user
from app.db.mongo_handler import get_user_by_username, create_user
from app.db.neo4j_handler import create_user_node_in_graph # We will create this file next
from app.models.token import Token
from app.models.user import UserCreate, UserPublic, UserInDB

router = APIRouter()

@router.post("/signup", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def signup(user_in: UserCreate):
    """
    Handles new user registration.
    1. Checks if user already exists in MongoDB.
    2. Hashes the password.
    3. Creates the user in MongoDB.
    4. Creates a corresponding user node in Neo4j.
    """
    existing_user = await get_user_by_username(user_in.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )
    
    hashed_password = get_password_hash(user_in.password)
    user_to_create = UserInDB(username=user_in.username, hashed_password=hashed_password)
    
    # Create user in MongoDB
    created_user = await create_user(user=user_to_create)

    # Create user node in Neo4j
    try:
        await create_user_node_in_graph(email=created_user.username)
    except Exception as e:
        # Log this critical error but don't fail the signup
        print(f"CRITICAL: Failed to create Neo4j node for user {created_user.username}. Error: {e}")

    return created_user

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Handles user login and returns a JWT access token.
    """
    user = await get_user_by_username(username=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserPublic)
async def read_current_user(current_user: UserInDB = Depends(get_current_user)):
    """
    Returns the profile of the currently authenticated user.
    """
    return current_user