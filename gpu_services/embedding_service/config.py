import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    MODEL_NAME: str = "google/embeddinggemma-300m"
    EMBEDDING_DIMENSION: int = 768
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()