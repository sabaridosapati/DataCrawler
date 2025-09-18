from fastapi import APIRouter, Depends, HTTPException, status
from app.core.security import get_current_user
from app.models.user import UserInDB
from app.models.chat import ChatRequest, ChatTurnResponse
from app.db.mongo_handler import get_chat_session, create_or_update_chat_session

router = APIRouter()

@router.post("/", response_model=ChatTurnResponse)
async def handle_query(
    request: ChatRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Handles a user's query, maintaining a continuous conversation.
    1. Fetches the last 5 turns of conversation history if a chat_id is provided.
    2. (Future) Passes the prompt and history to the LLM service.
    3. Saves the new user prompt and AI response to the chat history in MongoDB.
    """
    user_id = str(current_user.id)
    history = []
    
    if request.chat_id:
        chat_session = await get_chat_session(request.chat_id)
        if not chat_session or chat_session.user_id != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat session not found.")
        # Fetch the last 5 turns (10 messages: 5 user, 5 assistant)
        history = chat_session.history[-10:]

    # --- LLM and RAG Logic will go here ---
    # 1. Call LLM service with request.prompt and history
    # 2. Get AI response
    # For now, we will use a placeholder response.
    ai_response_content = f"This is a placeholder response to your query: '{request.prompt}'. The context had {len(history)} messages."
    # --- End of Placeholder ---

    # Save the new turn to the database
    updated_session = await create_or_update_chat_session(
        user_id=user_id,
        prompt=request.prompt,
        response=ai_response_content,
        chat_id=request.chat_id
    )

    # The last message in the history is the new AI response
    new_turn_number = updated_session.history[-1].turn_number if updated_session.history else 1

    return ChatTurnResponse(
        chat_id=str(updated_session.id),
        ai_response=ai_response_content,
        turn_number=new_turn_number
    )