# gpu_services/knowledge_graph_service/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Manages settings for the Knowledge Graph Builder service."""
    # Neo4j Connection Details (mirrors the orchestrator's config)
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # URL for the local LLM service that runs Gemma 3n
    LOCAL_LLM_URL: str = "http://localhost:8001/v1" # Ollama's OpenAI-compatible endpoint

    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()