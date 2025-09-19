from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Manages application settings loaded from a .env file.
    Provides type-hinted, validated configuration across the app.
    """
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # Database Connections
    MONGO_URL: str
    MONGO_DB_NAME: str
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    MILVUS_HOST: str
    MILVUS_PORT: str

    # GPU Node Service URLs
    LLM_SERVICE_URL: str
    EMBEDDING_SERVICE_URL: str
    KNOWLEDGE_GRAPH_SERVICE_URL: str
    DOCLING_SERVICE_URL: str

    # Project Metadata
    PROJECT_NAME: str = "Document Library Orchestrator"
    API_V1_STR: str = "/api/v1"

    # Pydantic settings configuration
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

# --- THIS IS THE CRUCIAL LINE THAT WAS LIKELY MISSING ---
# Create a single, importable instance of the Settings class.
# This is the object that main.py is trying to import.
settings = Settings()