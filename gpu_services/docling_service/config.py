# gpu_services/docling_service/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuration settings for the Granite Docling service.
    Optimized for Apple Silicon Mac with MLX support.
    """
    
    # AssemblyAI API key for audio processing (optional)
    ASSEMBLYAI_API_KEY: str = ""
    
    # Service configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8004
    
    # Model configuration
    MODEL_NAME: str = "granite-docling-258M-mlx"
    MODEL_TYPE: str = "vision-language-model"
    PLATFORM: str = "Apple Silicon (MLX)"
    
    # Processing configuration
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.0
    
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
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()

# Log current configuration on import
if __name__ == "__main__":
    print("=== Granite Docling Service Configuration ===")
    print(f"Model: {settings.MODEL_NAME}")
    print(f"Platform: {settings.PLATFORM}")
    print(f"Host: {settings.HOST}:{settings.PORT}")
    print(f"Max tokens: {settings.MAX_TOKENS}")
    print(f"Temperature: {settings.TEMPERATURE}")
    print(f"Chunk size: {settings.CHUNK_SIZE}")
    print(f"AssemblyAI enabled: {'Yes' if settings.ASSEMBLYAI_API_KEY else 'No'}")
    print(f"Supported document formats: {settings.SUPPORTED_DOCUMENT_FORMATS}")
    print(f"Supported image formats: {settings.SUPPORTED_IMAGE_FORMATS}")
    print(f"Supported audio formats: {settings.SUPPORTED_AUDIO_FORMATS}")