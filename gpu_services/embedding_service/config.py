# gpu_services/embedding_service/config.py

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Manages settings for the Embedding service."""
    # Automatically determine the best device (CUDA if available, otherwise CPU)
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # As per the model card, bfloat16 is preferred on compatible hardware for performance.
    # If your RTX 4070 supports it, this is a good setting. Otherwise, it will default to float32.
    TORCH_DTYPE: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # Model configuration
    MODEL_NAME: str = "google/embeddinggemma-300m"
    EMBEDDING_DIMENSION: int = 768

    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()