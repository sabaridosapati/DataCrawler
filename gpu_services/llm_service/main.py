# gpu_services/llm_service/main.py

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, pipeline
from PIL import Image
import requests
import io
import base64

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self):
        self.model = None
        self.processor = None
        self.pipe = None
        
    def initialize_models(self):
        """Initialize the Gemma 3-4B model with vision capabilities"""
        logger.info("Initializing Gemma 3-4B model...")
        
        try:
            model_id = settings.MODEL_NAME
            
            # Load the model with device mapping for optimal performance
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=settings.TORCH_DTYPE,
                trust_remote_code=True
            ).eval()
            
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Also initialize pipeline for simpler text-only operations
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                device="cuda",
                torch_dtype=settings.TORCH_DTYPE
            )
            
            logger.info(f"Successfully loaded {model_id} with vision capabilities")
            
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load Gemma 3-4B model. Error: {e}", exc_info=True)
            raise

    def format_messages_for_chat(self, messages: List[Dict[str, Any]], system_prompt: str = None) -> List[Dict[str, Any]]:
        """Format messages for Gemma chat template"""
        formatted_messages = []
        
        # Add system message if provided
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Process each message
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if isinstance(content, str):
                # Simple text message
                formatted_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            elif isinstance(content, list):
                # Already formatted content with potential images
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # Convert to text format
                formatted_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": str(content)}]
                })
        
        return formatted_messages

    async def generate_response(
        self, 
        messages: List[Dict[str, Any]], 
        max_tokens: int = 8192,
        temperature: float = 0.1,
        system_prompt: str = None
    ) -> str:
        """Generate response using Gemma 3-4B model with proper chat formatting"""
        try:
            # Format messages for chat template
            formatted_messages = self.format_messages_for_chat(messages, system_prompt)
            
            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                formatted_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=settings.TORCH_DTYPE)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                
                # Extract only the new tokens
                generation = generation[0][input_len:]
            
            # Decode the response
            response = self.processor.decode(generation, skip_special_tokens=True)
            logger.info("Successfully generated response")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}", exc_info=True)
            raise RuntimeError(f"Text generation failed: {e}")

    async def generate_with_images(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 8192,
        temperature: float = 0.7
    ) -> str:
        """Generate response with image inputs using vision capabilities"""
        try:
            # Process messages to handle images
            processed_messages = []
            for msg in messages:
                processed_content = []
                
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item["type"] == "text":
                            processed_content.append(item)
                        elif item["type"] == "image":
                            # Handle image - could be URL or base64
                            if "url" in item:
                                processed_content.append({
                                    "type": "image",
                                    "image": item["url"]
                                })
                            elif "image" in item:
                                processed_content.append({
                                    "type": "image", 
                                    "image": item["image"]
                                })
                elif isinstance(msg.get("content"), str):
                    processed_content.append({
                        "type": "text",
                        "text": msg["content"]
                    })
                
                processed_messages.append({
                    "role": msg.get("role", "user"),
                    "content": processed_content
                })
            
            # Apply chat template for vision
            inputs = self.processor.apply_chat_template(
                processed_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=settings.TORCH_DTYPE)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None
                )
                
                generation = generation[0][input_len:]
            
            response = self.processor.decode(generation, skip_special_tokens=True)
            logger.info("Successfully generated vision response")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error during vision generation: {e}", exc_info=True)
            raise RuntimeError(f"Vision generation failed: {e}")

# Global processor instance
processor_instance: LLMProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the LLM processor when the app starts"""
    global processor_instance
    logger.info("LLM service is starting up...")
    processor_instance = LLMProcessor()
    processor_instance.initialize_models()
    yield
    logger.info("LLM service is shutting down.")

app = FastAPI(
    title="Gemma 3-4B LLM Service",
    description="High-performance LLM service with vision capabilities using Gemma 3-4B model",
    version="2.0.0",
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
    model: str = "google/gemma-3-4b-it"

class OpenAICompatibleRequest(BaseModel):
    """OpenAI-compatible format for easier integration"""
    model: str = "google/gemma-3-4b-it"
    messages: List[Dict[str, Any]]
    max_tokens: int = 8192
    temperature: float = 0.2
    stream: bool = False

class OpenAICompatibleResponse(BaseModel):
    id: str = "gemma-3-4b-response"
    object: str = "chat.completion"
    model: str = "google/gemma-3-4b-it"
    choices: List[Dict[str, Any]]

# --- API Endpoints ---
@app.post("/v1/chat/completions", response_model=OpenAICompatibleResponse)
async def openai_compatible_chat(request: OpenAICompatibleRequest):
    """OpenAI-compatible chat completions endpoint"""
    if processor_instance is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Check if any message contains images
        has_images = any(
            isinstance(msg.get("content"), list) and 
            any(item.get("type") == "image" for item in msg["content"])
            for msg in request.messages
        )
        
        if has_images:
            response = await processor_instance.generate_with_images(
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        else:
            response = await processor_instance.generate_response(
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        
        return OpenAICompatibleResponse(
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }]
        )
        
    except Exception as e:
        logger.error(f"Error in OpenAI compatible endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Native generation endpoint"""
    if processor_instance is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Convert Pydantic models to dictionaries
        messages = [msg.model_dump() for msg in request.messages]
        
        response = await processor_instance.generate_response(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt
        )
        
        return GenerateResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_initialized": processor_instance is not None,
        "model_name": settings.MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
    }

@app.get("/models")
def list_models():
    """List available models - OpenAI compatible"""
    return {
        "object": "list",
        "data": [
            {
                "id": "google/gemma-3-4b-it",
                "object": "model",
                "owned_by": "google",
                "permission": []
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)