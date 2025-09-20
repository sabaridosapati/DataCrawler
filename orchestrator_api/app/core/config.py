# orchestrator_api/app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Manages application settings for the distributed document library system.
    Updated for multi-machine deployment architecture.
    """
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # Local Database Connections (Mac Mini)
    MONGO_URL: str = "mongodb://admin:password123@localhost:27017"
    MONGO_DB_NAME: str = "document_library"
    
    # Neo4j Desktop connection (already running on Mac Mini)
    # Update these settings to match your Neo4j Desktop instance
    NEO4J_URI: str = "bolt://localhost:7687"  # Default Neo4j Desktop port
    NEO4J_USER: str = "neo4j"                 # Default user
    NEO4J_PASSWORD: str = "your-neo4j-password"  # Update with your actual password

    # Remote Vector Database (Dell Laptop)
    MILVUS_HOST: str = "192.168.100.42"  # Dell laptop IP
    MILVUS_PORT: str = "19530"

    # Remote GPU Services URLs
    LLM_SERVICE_URL: str = "http://192.168.100.43:8001"        # Lenovo laptop
    EMBEDDING_SERVICE_URL: str = "http://192.168.100.42:8002"  # Dell laptop
    KNOWLEDGE_GRAPH_SERVICE_URL: str = "http://192.168.100.42:8003"  # Dell laptop
    
    # Local Service URL (Mac Mini)
    DOCLING_SERVICE_URL: str = "http://localhost:8004"

    # Project Metadata
    PROJECT_NAME: str = "Distributed Document Library System"
    API_V1_STR: str = "/api/v1"
    
    # File storage paths (Mac Mini)
    DATA_DIR: str = "./data"
    USER_FILES_DIR: str = "./data/user_files"
    EXTRACTED_DIR: str = "./data/extracted"
    
    # Network configuration
    ALLOWED_HOSTS: list = ["*"]
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/orchestrator.log"

    # Pydantic settings configuration
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

# Create a single, importable instance of the Settings class
settings = Settings()

# Network topology for reference
NETWORK_TOPOLOGY = {
    "mac_mini": {
        "ip": "192.168.100.41",
        "services": ["orchestrator_api", "docling_service", "mongodb", "neo4j"],
        "ports": {
            "orchestrator_api": 8000,
            "docling_service": 8004,
            "mongodb": 27017,
            "neo4j_http": 7474,
            "neo4j_bolt": 7687
        }
    },
    "lenovo_laptop": {
        "ip": "192.168.100.43",
        "services": ["llm_service"],
        "ports": {
            "llm_service": 8001
        }
    },
    "dell_laptop": {
        "ip": "192.168.100.42",
        "services": ["embedding_service", "knowledge_graph_service", "milvus"],
        "ports": {
            "embedding_service": 8002,
            "knowledge_graph_service": 8003,
            "milvus": 19530,
            "milvus_admin": 9091
        }
    }
}