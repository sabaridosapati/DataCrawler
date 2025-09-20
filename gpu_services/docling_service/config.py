# gpu_services/docling_service/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Manages settings for the Docling service."""
    ASSEMBLYAI_API_KEY: str
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()