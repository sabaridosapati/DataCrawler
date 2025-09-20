# gpu_services/llm_service/config.py

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuration for the Gemma 3-4B LLM service"""
    
    # Model configuration
    MODEL_NAME: str = "google/gemma-3-4b-it"
    
    # Hardware configuration
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    # Generation parameters
    MAX_LENGTH: int = 32000
    DEFAULT_MAX_TOKENS: int = 8192
    DEFAULT_TEMPERATURE: float = 0.2

    # Service configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # Hugging Face settings
    HF_TOKEN: str = ""  # Optional: Set if you need private model access
    
    # Memory optimization
    USE_FLASH_ATTENTION: bool = True
    GPU_MEMORY_FRACTION: float = 0.95
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()

# Log configuration on import
if __name__ == "__main__":
    print(f"Model: {settings.MODEL_NAME}")
    print(f"Device: {settings.DEVICE}")
    print(f"Torch dtype: {settings.TORCH_DTYPE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")