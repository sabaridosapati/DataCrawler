# orchestrator_api/app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration for the Document Library System.
    Single-machine deployment with all services on localhost.
    """
    
    # Security
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # MongoDB (Docker container)
    MONGO_URL: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "document_library"
    
    # Neo4j (Running on host)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Milvus Lite (local embedded database)
    MILVUS_DB_PATH: str = "./data/milvus.db"

    # Local Service URLs (All on same machine)
    LLM_SERVICE_URL: str = "http://localhost:8001"
    EMBEDDING_SERVICE_URL: str = "http://localhost:8002"
    KNOWLEDGE_GRAPH_SERVICE_URL: str = "http://localhost:8003"
    DOCLING_SERVICE_URL: str = "http://localhost:8004"

    # Project Metadata
    PROJECT_NAME: str = "Document Library System"
    API_V1_STR: str = "/api/v1"
    
    # File storage paths
    DATA_DIR: str = "./data"
    USER_FILES_DIR: str = "./data/user_files"
    EXTRACTED_DIR: str = "./data/extracted"
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra='ignore')


settings = Settings()