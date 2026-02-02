# gpu_services/docling_service/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the Docling service.
    Optimized for Linux with NVIDIA GPU (8GB VRAM).
    """
    
    # AssemblyAI API key for audio processing (optional)
    ASSEMBLYAI_API_KEY: str = ""
    
    # Service configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8004
    
    # Model configuration - CUDA/PyTorch backend for Linux
    MODEL_NAME: str = "ds4sd/SmolDocling-256M-preview"  # Lightweight CUDA-compatible model
    MODEL_TYPE: str = "vision-language-model"
    PLATFORM: str = "Linux (CUDA)"
    
    # GPU Memory Optimization for 8GB VRAM
    GPU_MEMORY_FRACTION: float = 0.85  # Conservative to prevent OOM
    CUDA_VISIBLE_DEVICES: str = "0"
    
    # Processing configuration
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.0
    BATCH_SIZE: int = 1  # Process one page at a time for 8GB VRAM
    
    # Chunking configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MERGE_PEERS: bool = True
    
    # Output configuration
    OUTPUT_FORMAT: str = "markdown"
    INCLUDE_IMAGES: bool = True
    INCLUDE_TABLES: bool = True
    PRESERVE_LAYOUT: bool = True
    
    # File handling
    MAX_FILE_SIZE_MB: int = 100
    SUPPORTED_DOCUMENT_FORMATS: list = [
        ".pdf", ".docx", ".pptx", ".html", ".md", ".txt"
    ]
    SUPPORTED_IMAGE_FORMATS: list = [
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"
    ]
    SUPPORTED_AUDIO_FORMATS: list = [
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4"
    ]
    
    # Memory management
    CLEAR_CACHE_AFTER_PROCESSING: bool = True
    USE_FLASH_ATTENTION: bool = True  # If supported
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')


settings = Settings()

# Log current configuration on import
if __name__ == "__main__":
    print("=== Docling Service Configuration (Linux CUDA) ===")
    print(f"Model: {settings.MODEL_NAME}")
    print(f"Platform: {settings.PLATFORM}")
    print(f"Host: {settings.HOST}:{settings.PORT}")
    print(f"GPU Memory Fraction: {settings.GPU_MEMORY_FRACTION}")
    print(f"Batch Size: {settings.BATCH_SIZE}")
    print(f"Max tokens: {settings.MAX_TOKENS}")
    print(f"Temperature: {settings.TEMPERATURE}")
    print(f"Chunk size: {settings.CHUNK_SIZE}")
    print(f"AssemblyAI enabled: {'Yes' if settings.ASSEMBLYAI_API_KEY else 'No'}")
    print(f"Supported document formats: {settings.SUPPORTED_DOCUMENT_FORMATS}")
    print(f"Supported image formats: {settings.SUPPORTED_IMAGE_FORMATS}")
    print(f"Supported audio formats: {settings.SUPPORTED_AUDIO_FORMATS}")