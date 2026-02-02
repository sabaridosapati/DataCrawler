# gpu_services/llm_service/config.py

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the Gemini LLM service using cloud API"""
    
    # Gemini API Configuration
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash-lite"  # gemini-flash-lite-preview
    
    # Generation parameters
    DEFAULT_MAX_TOKENS: int = 8192
    DEFAULT_TEMPERATURE: float = 0.2
    MAX_CONTEXT_LENGTH: int = 1000000  # Gemini 2.0 Flash supports 1M context
    
    # Service configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # Concurrency & Rate Limiting
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT_SECONDS: float = 60.0
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    RETRY_BACKOFF_MULTIPLIER: float = 2.0
    
    # Connection pool settings
    CONNECTION_POOL_SIZE: int = 20
    KEEPALIVE_TIMEOUT: int = 30
    
    # Safety settings
    BLOCK_THRESHOLD: str = "BLOCK_ONLY_HIGH"  # BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')


settings = Settings()

# Log configuration on import
if __name__ == "__main__":
    print("=== Gemini LLM Service Configuration ===")
    print(f"Model: {settings.GEMINI_MODEL}")
    print(f"Max tokens: {settings.DEFAULT_MAX_TOKENS}")
    print(f"Temperature: {settings.DEFAULT_TEMPERATURE}")
    print(f"Max concurrent requests: {settings.MAX_CONCURRENT_REQUESTS}")
    print(f"API Key configured: {'Yes' if settings.GEMINI_API_KEY else 'No'}")