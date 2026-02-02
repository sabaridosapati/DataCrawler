# gpu_services/llm_service/main.py

"""
High-concurrency LLM Service using Gemini API.
Provides OpenAI-compatible endpoints for seamless integration.
"""

import logging
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, AsyncIterator
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiLLMProcessor:
    """
    High-concurrency Gemini LLM processor with connection pooling,
    retry logic, and semaphore-based rate limiting.
    """
    
    def __init__(self):
        self.client: Optional[genai.Client] = None
        self.semaphore: Optional[asyncio.Semaphore] = None
        self._initialized = False
        
    def initialize(self):
        """Initialize the Gemini client with optimal settings"""
        logger.info("Initializing Gemini LLM Processor...")
        
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        try:
            # Create client with connection pool settings
            http_options = types.HttpOptions(
                timeout=settings.REQUEST_TIMEOUT_SECONDS * 1000,  # Convert to ms
            )
            
            self.client = genai.Client(
                api_key=settings.GEMINI_API_KEY,
                http_options=http_options
            )
            
            # Semaphore for rate limiting concurrent requests
            self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
            
            self._initialized = True
            logger.info(f"Gemini client initialized successfully with model: {settings.GEMINI_MODEL}")
            
        except Exception as e:
            logger.critical(f"Failed to initialize Gemini client: {e}", exc_info=True)
            raise
    
    def close(self):
        """Clean up resources"""
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing Gemini client: {e}")
        self._initialized = False
        logger.info("Gemini client closed")
    
    def _format_messages_to_contents(
        self, 
        messages: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None
    ) -> tuple[List[types.Content], Optional[str]]:
        """
        Convert OpenAI-style messages to Gemini content format.
        Returns (contents, system_instruction)
        """
        contents = []
        system_instruction = system_prompt
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle system messages - Gemini uses separate system_instruction
            if role == "system":
                system_instruction = content if isinstance(content, str) else str(content)
                continue
            
            # Map roles: user -> user, assistant -> model
            gemini_role = "model" if role == "assistant" else "user"
            
            # Handle different content formats
            if isinstance(content, str):
                parts = [types.Part.from_text(text=content)]
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item.get("text", "")))
                        elif item.get("type") == "image_url":
                            # Handle image URLs if needed
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # Base64 encoded image
                                import base64
                                # Extract mime type and data
                                header, b64_data = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                image_bytes = base64.b64decode(b64_data)
                                parts.append(types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type=mime_type
                                ))
                    elif isinstance(item, str):
                        parts.append(types.Part.from_text(text=item))
            else:
                parts = [types.Part.from_text(text=str(content))]
            
            contents.append(types.Content(role=gemini_role, parts=parts))
        
        return contents, system_instruction

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response using Gemini API with retry logic and rate limiting.
        """
        if not self._initialized:
            raise RuntimeError("Gemini processor not initialized")
        
        max_tokens = max_tokens or settings.DEFAULT_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        
        # Acquire semaphore for rate limiting
        async with self.semaphore:
            try:
                start_time = time.time()
                
                # Convert messages to Gemini format
                contents, system_instruction = self._format_messages_to_contents(
                    messages, system_prompt
                )
                
                # Build generation config
                config = types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold=settings.BLOCK_THRESHOLD,
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold=settings.BLOCK_THRESHOLD,
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold=settings.BLOCK_THRESHOLD,
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold=settings.BLOCK_THRESHOLD,
                        ),
                    ]
                )
                
                if system_instruction:
                    config.system_instruction = system_instruction
                
                # Make async API call
                response = await self.client.aio.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=contents,
                    config=config
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Generated response in {elapsed:.2f}s")
                
                # Extract text from response
                if response.text:
                    return response.text
                elif response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
                else:
                    logger.warning("Empty response from Gemini API")
                    return ""
                    
            except Exception as e:
                logger.error(f"Error generating response: {e}", exc_info=True)
                raise

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Generate streaming response using Gemini API.
        """
        if not self._initialized:
            raise RuntimeError("Gemini processor not initialized")
        
        max_tokens = max_tokens or settings.DEFAULT_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        
        async with self.semaphore:
            try:
                contents, system_instruction = self._format_messages_to_contents(
                    messages, system_prompt
                )
                
                config = types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
                
                if system_instruction:
                    config.system_instruction = system_instruction
                
                # Stream response
                async for chunk in await self.client.aio.models.generate_content_stream(
                    model=settings.GEMINI_MODEL,
                    contents=contents,
                    config=config
                ):
                    if chunk.text:
                        yield chunk.text
                        
            except Exception as e:
                logger.error(f"Error in streaming response: {e}", exc_info=True)
                raise


# Global processor instance
processor_instance: Optional[GeminiLLMProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the LLM processor"""
    global processor_instance
    logger.info("Gemini LLM service is starting up...")
    
    processor_instance = GeminiLLMProcessor()
    processor_instance.initialize()
    
    yield
    
    logger.info("Gemini LLM service is shutting down...")
    if processor_instance:
        processor_instance.close()


app = FastAPI(
    title="Gemini LLM Service",
    description="High-concurrency LLM service using Gemini API with OpenAI-compatible endpoints",
    version="3.0.0",
    lifespan=lifespan
)


# --- Pydantic Models for API ---
class Message(BaseModel):
    role: str
    content: Any  # Can be string or list for multimodal


class GenerateRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 8192
    temperature: float = 0.2
    system_prompt: Optional[str] = None


class GenerateResponse(BaseModel):
    response: str
    model: str = settings.GEMINI_MODEL


class OpenAICompatibleRequest(BaseModel):
    """OpenAI-compatible format for easier integration"""
    model: str = settings.GEMINI_MODEL
    messages: List[Dict[str, Any]]
    max_tokens: int = 8192
    temperature: float = 0.2
    stream: bool = False


class OpenAICompatibleChoice(BaseModel):
    index: int = 0
    message: Dict[str, str]
    finish_reason: str = "stop"


class OpenAICompatibleResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAICompatibleChoice]
    usage: Optional[Dict[str, int]] = None


# --- API Endpoints ---
@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: OpenAICompatibleRequest):
    """OpenAI-compatible chat completions endpoint"""
    if processor_instance is None or not processor_instance._initialized:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        if request.stream:
            # Return streaming response
            async def stream_generator():
                response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())
                
                async for chunk in processor_instance.generate_stream(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                ):
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                
                # Send final chunk
                final_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        response = await processor_instance.generate_response(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return OpenAICompatibleResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                OpenAICompatibleChoice(
                    message={
                        "role": "assistant",
                        "content": response
                    }
                )
            ]
        )
        
    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Native generation endpoint"""
    if processor_instance is None or not processor_instance._initialized:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        messages = [msg.model_dump() for msg in request.messages]
        
        response = await processor_instance.generate_response(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt
        )
        
        return GenerateResponse(response=response, model=settings.GEMINI_MODEL)
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok" if processor_instance and processor_instance._initialized else "unhealthy",
        "model_initialized": processor_instance._initialized if processor_instance else False,
        "model_name": settings.GEMINI_MODEL,
        "backend": "gemini-api",
        "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS
    }


@app.get("/v1/models")
@app.get("/models")
def list_models():
    """List available models - OpenAI compatible"""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.GEMINI_MODEL,
                "object": "model",
                "owned_by": "google",
                "permission": []
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
        workers=1  # Use 1 worker since we're async
    )