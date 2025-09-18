# gpu_services/docling_service/config.py

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Manages settings for the Docling service."""
    ASSEMBLYAI_API_KEY: str

    # Automatically determine the best device (CUDA if available, otherwise CPU)
    # This makes the service portable while prioritizing the GPU.
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()