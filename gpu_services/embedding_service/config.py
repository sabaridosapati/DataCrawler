# gpu_services/embedding_service/config.py

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuration for Enhanced Embedding Service optimized for RTX 4050 (6GB VRAM)"""
    
    # Model configuration
    MODEL_NAME: str = "google/embeddinggemma-300m"
    EMBEDDING_DIMENSION: int = 768
    
    # Hardware configuration optimized for RTX 4050
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    # Memory optimization for RTX 4050 (6GB VRAM)
    BATCH_SIZE: int = 16 if torch.cuda.is_available() else 8
    MAX_SEQ_LENGTH: int = 512
    GPU_MEMORY_FRACTION: float = 0.85  # Conservative for RTX 4050
    
    # Service configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    
    # Processing optimization
    NORMALIZE_EMBEDDINGS: bool = True
    SHOW_PROGRESS: bool = True
    TRUST_REMOTE_CODE: bool = True
    
    # Text cleaning configuration
    MIN_TEXT_LENGTH: int = 10
    MAX_TEXT_LENGTH: int = 8192
    CLEAN_DOCLING_ARTIFACTS: bool = True
    
    # Hugging Face configuration
    HF_TOKEN: str = ""
    TRANSFORMERS_CACHE: str = "./cache/huggingface"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()

# Log current configuration on import
if __name__ == "__main__":
    print("=== Enhanced Embedding Service Configuration ===")
    print(f"Model: {settings.MODEL_NAME}")
    print(f"Embedding Dimension: {settings.EMBEDDING_DIMENSION}")
    print(f"Device: {settings.DEVICE}")
    print(f"Torch dtype: {settings.TORCH_DTYPE}")
    print(f"Batch size: {settings.BATCH_SIZE}")
    print(f"GPU memory fraction: {settings.GPU_MEMORY_FRACTION}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU: Not available - using CPU")