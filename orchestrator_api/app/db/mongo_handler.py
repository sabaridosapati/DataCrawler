# orchestrator_api/app/db/mongo_handler.py

from datetime import datetime, timezone
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from app.core.config import settings
from app.models.user import UserInDB
from app.models.document import DocumentInDB, DocumentStatus
from app.models.chat import ChatSession, ChatMessage

# --- Connection Setup ---
# A single client instance is created when the module is imported.
# Motor manages an internal connection pool.
client = AsyncIOMotorClient(settings.MONGO_URL)
db = client[settings.MONGO_DB_NAME]

# Get collection objects
user_collection = db.get_collection("users")
document_collection = db.get_collection("documents")
chat_collection = db.get_collection("chats")


# --- User Functions ---
async def get_user_by_username(username: str) -> Optional[UserInDB]:
    """Retrieves a single user from the database by their username (email)."""
    user_data = await user_collection.find_one({"username": username})
    if user_data:
        # Convert _id to id for the model
        user_data["id"] = str(user_data.pop("_id"))
        return UserInDB(**user_data)
    return None

async def create_user(user: UserInDB) -> UserInDB:
    """Inserts a new user record into the database."""
    user_dict = user.model_dump(exclude={'id'})
    result = await user_collection.insert_one(user_dict)
    user.id = str(result.inserted_id)
    return user


# --- Document Functions ---
async def create_document_record(doc: DocumentInDB) -> DocumentInDB:
    """Inserts a new document tracking record into the database."""
    await document_collection.insert_one(doc.model_dump(by_alias=True))
    return doc

async def get_document_by_id(doc_id: str) -> Optional[DocumentInDB]:
    """Retrieves a document record by its unique ID."""
    doc_data = await document_collection.find_one({"_id": doc_id})
    return DocumentInDB(**doc_data) if doc_data else None

async def update_document_status(doc_id: str, status: DocumentStatus, error_message: Optional[str] = None):
    """Updates the processing status and timestamp of a document."""
    update_fields = {"status": status.value, "updated_at": datetime.now(timezone.utc)}
    if error_message:
        update_fields["error_message"] = error_message
    
    await document_collection.update_one({"_id": doc_id}, {"$set": update_fields})


# --- Chat Functions ---
async def get_chat_session(chat_id: str) -> Optional[ChatSession]:
    """Retrieves a full chat session by its ID."""
    if not ObjectId.is_valid(chat_id):
        return None
    chat_data = await chat_collection.find_one({"_id": ObjectId(chat_id)})
    return ChatSession(**chat_data) if chat_data else None

async def create_or_update_chat_session(user_id: str, prompt: str, response: str, chat_id: Optional[str] = None) -> ChatSession:
    """
    Handles the core logic for continuous conversations.
    - If chat_id exists, it appends the new turn.
    - If chat_id is None, it creates a new chat session.
    """
    current_time = datetime.now(timezone.utc)
    
    if chat_id and ObjectId.is_valid(chat_id):
        # --- Update existing chat ---
        existing_chat = await get_chat_session(chat_id)
        last_turn_number = existing_chat.history[-1].turn_number if existing_chat and existing_chat.history else 0
        current_turn_number = last_turn_number + 1

        user_message = ChatMessage(role="user", content=prompt, turn_number=current_turn_number)
        assistant_message = ChatMessage(role="assistant", content=response, turn_number=current_turn_number)

        result = await chat_collection.find_one_and_update(
            {"_id": ObjectId(chat_id), "user_id": user_id},
            {
                "$push": {"history": {"$each": [user_message.model_dump(), assistant_message.model_dump()]}},
                "$set": {"updated_at": current_time}
            },
            return_document=True
        )
        return ChatSession(**result)
    else:
        # --- Create new chat ---
        user_message = ChatMessage(role="user", content=prompt, turn_number=1)
        assistant_message = ChatMessage(role="assistant", content=response, turn_number=1)
        
        new_chat_doc = ChatSession(
            user_id=user_id,
            chat_name=prompt[:50], # Use the first 50 chars of the prompt as the initial name
            created_at=current_time,
            updated_at=current_time,
            history=[user_message, assistant_message]
        )
        
        # Insert and get the ID
        insert_result = await chat_collection.insert_one(new_chat_doc.model_dump(exclude={'id'}))
        new_chat_doc.id = insert_result.inserted_id
        return new_chat_doc