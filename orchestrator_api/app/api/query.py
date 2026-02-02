# orchestrator_api/app/api/query.py

"""
RAG Query API with state-of-the-art hybrid retrieval.

Implements:
- Hybrid retrieval (HyDE + BM25 + Vector + RRF)
- Conversation continuity with chat history
- User-isolated document access
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.core.security import get_current_user
from app.models.user import UserInDB
from app.db.mongo_handler import get_chat_session, create_or_update_chat_session
from app.db.milvus_handler import milvus_db_handler
from app.services.hybrid_retriever import hybrid_retriever
from app.services import gpu_node_client

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    prompt: str = Field(..., description="User's query")
    chat_id: Optional[str] = Field(None, description="Chat session ID for continuity")
    use_hyde: bool = Field(True, description="Use HyDE query expansion")
    use_bm25: bool = Field(True, description="Use BM25 lexical search")
    top_k: int = Field(5, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    ai_response: str
    chat_id: str
    sources: List[Dict[str, Any]] = []
    retrieval_info: Dict[str, Any] = {}


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, ge=1, le=50)
    use_hyde: bool = Field(True)


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int


# --- RAG Prompts ---

RAG_SYSTEM_PROMPT = """You are an intelligent assistant that answers questions based on the provided context from the user's documents.

INSTRUCTIONS:
1. Answer ONLY based on the provided context. Do not use external knowledge.
2. If the context doesn't contain enough information, say "I don't have enough information in your documents to answer this question."
3. Be precise and cite specific details from the context when possible.
4. If the question is unclear, ask for clarification.
5. Format your response clearly with markdown when appropriate.
6. When citing information, reference which document it came from if available.

CONTEXT FROM USER'S DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history}"""


# --- Helper Functions ---

def format_context(chunks: List[Any]) -> str:
    """Format retrieved chunks into context string."""
    if not chunks:
        return "No relevant documents found."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_id = chunk.doc_id[:8] if hasattr(chunk, 'doc_id') else "unknown"
        text = chunk.chunk_text if hasattr(chunk, 'chunk_text') else str(chunk)
        source = chunk.source if hasattr(chunk, 'source') else "vector"
        
        context_parts.append(f"[Source {i} - Doc:{doc_id} via {source}]\n{text}")
    
    return "\n\n---\n\n".join(context_parts)


def format_history(history: List[Any]) -> str:
    """Format chat history for context."""
    if not history:
        return "No previous conversation."
    
    # Take last 6 messages for context window
    recent = history[-6:] if len(history) > 6 else history
    
    formatted = []
    for msg in recent:
        role = msg.role if hasattr(msg, 'role') else msg.get('role', 'unknown')
        content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
        formatted.append(f"{role.upper()}: {content[:500]}")
    
    return "\n".join(formatted)


# --- API Endpoints ---

@router.post("/", response_model=QueryResponse)
async def handle_query(
    request: QueryRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Handle a RAG query with hybrid retrieval.
    
    Pipeline:
    1. Retrieve relevant chunks using hybrid retrieval (HyDE + BM25 + Vector + RRF)
    2. Get conversation history if chat_id provided
    3. Build context from retrieved chunks
    4. Generate response using LLM
    5. Save conversation turn
    """
    user_id = str(current_user.id)
    logger.info(f"RAG query from user {user_id}: '{request.prompt[:50]}...'")
    
    try:
        # Step 1: Hybrid Retrieval
        chunks = await hybrid_retriever.retrieve(
            user_id=user_id,
            query=request.prompt,
            top_k=request.top_k,
            use_hyde=request.use_hyde,
            use_bm25=request.use_bm25,
            use_graph=False  # Graph context optional for now
        )
        
        # Step 2: Get conversation history
        history = []
        if request.chat_id:
            chat_session = await get_chat_session(request.chat_id)
            if chat_session and chat_session.user_id == user_id:
                history = chat_session.history
        
        # Step 3: Build context and prompt
        context = format_context(chunks)
        history_str = format_history(history)
        
        system_prompt = RAG_SYSTEM_PROMPT.format(
            context=context,
            history=history_str
        )
        
        # Step 4: Generate response using LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt}
        ]
        
        ai_response = await gpu_node_client.generate_response(
            messages=messages,
            temperature=0.3,
            max_tokens=2048
        )
        
        logger.info(f"Generated RAG response: {len(ai_response)} chars")
        
        # Step 5: Save conversation
        updated_chat = await create_or_update_chat_session(
            user_id=user_id,
            prompt=request.prompt,
            response=ai_response,
            chat_id=request.chat_id
        )
        
        # Prepare sources for response
        sources = [
            {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "excerpt": chunk.chunk_text[:200] + "..." if len(chunk.chunk_text) > 200 else chunk.chunk_text,
                "score": round(chunk.score, 4),
                "source": chunk.source
            }
            for chunk in chunks[:5]
        ]
        
        return QueryResponse(
            ai_response=ai_response,
            chat_id=str(updated_chat.id) if hasattr(updated_chat, 'id') else str(updated_chat),
            sources=sources,
            retrieval_info={
                "num_chunks": len(chunks),
                "methods_used": ["vector"] + (["hyde"] if request.use_hyde else []) + (["bm25"] if request.use_bm25 else []),
                "fusion": "reciprocal_rank_fusion"
            }
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/search", response_model=SearchResponse)
async def semantic_search(
    query: str,
    top_k: int = 10,
    use_hyde: bool = False,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Perform semantic search across user's documents.
    Returns ranked chunks without LLM generation.
    """
    user_id = str(current_user.id)
    logger.info(f"Search query from user {user_id}: '{query[:50]}...'")
    
    try:
        chunks = await hybrid_retriever.retrieve(
            user_id=user_id,
            query=query,
            top_k=top_k,
            use_hyde=use_hyde,
            use_bm25=True,
            use_graph=False
        )
        
        results = [
            {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "score": round(chunk.score, 4),
                "source": chunk.source
            }
            for chunk in chunks
        ]
        
        return SearchResponse(results=results, total=len(results))
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stats")
async def get_stats(current_user: UserInDB = Depends(get_current_user)):
    """Get vector database stats for current user."""
    user_id = str(current_user.id)
    
    try:
        stats = milvus_db_handler.get_collection_stats()
        return {
            "user_id": user_id,
            **stats
        }
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return {
            "user_id": user_id,
            "num_entities": 0,
            "mode": "unknown",
            "error": str(e)
        }


@router.get("/chats")
async def list_chats(current_user: UserInDB = Depends(get_current_user)):
    """List all chat sessions for current user."""
    from app.db.mongo_handler import chat_collection
    
    user_id = str(current_user.id)
    
    cursor = chat_collection.find(
        {"user_id": user_id},
        {"history": 0}  # Exclude full history for list view
    ).sort("updated_at", -1).limit(50)
    
    chats = await cursor.to_list(length=50)
    
    return {
        "chats": [
            {
                "id": str(c["_id"]),
                "name": c.get("chat_name", "Untitled"),
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at")
            }
            for c in chats
        ]
    }


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str, current_user: UserInDB = Depends(get_current_user)):
    """Get full chat history."""
    user_id = str(current_user.id)
    
    chat = await get_chat_session(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return {
        "id": str(chat.id) if hasattr(chat, 'id') else chat_id,
        "name": chat.chat_name,
        "history": [
            {"role": msg.role, "content": msg.content, "turn": msg.turn_number}
            for msg in chat.history
        ]
    }