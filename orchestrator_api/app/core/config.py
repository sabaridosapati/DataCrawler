from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Manages application settings loaded from a .env file.
    Provides type-hinted, validated configuration across the app.
    """
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 # 1 day
    # Database URLs and Names
    MONGO_URL: str
    MONGO_DB_NAME: str
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # GPU Node Service URLs (for inter-service communication)
    DOCLING_SERVICE_URL: str
    LANGEXTRACT_SERVICE_URL: str
    LLM_SERVICE_URL: str

    # Project Metadata
    PROJECT_NAME: str = "Document Library Orchestrator"
    API_V1_STR: str = "/api/v1"

    # Pydantic settings configuration
model_config = SettingsConfigDict(env_file=".env", extra='ignore')