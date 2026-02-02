# gpu_services/embedding_service/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for Gemini Embedding Service using cloud API"""
    
    # Gemini API Configuration
    GEMINI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    
    # Embedding Configuration
    EMBEDDING_DIMENSION: int = 768  # gemini-embedding-001 output dimension
    
    # Task types for better retrieval performance
    # Options: RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY, 
    #          CLASSIFICATION, CLUSTERING, QUESTION_ANSWERING, FACT_VERIFICATION
    DOCUMENT_TASK_TYPE: str = "RETRIEVAL_DOCUMENT"
    QUERY_TASK_TYPE: str = "RETRIEVAL_QUERY"
    
    # Batch processing settings
    # Gemini API allows up to 100 texts per batch
    MAX_BATCH_SIZE: int = 100
    OPTIMAL_BATCH_SIZE: int = 50  # Balance between throughput and latency
    
    # Rate limiting
    MAX_CONCURRENT_REQUESTS: int = 5
    REQUEST_TIMEOUT_SECONDS: float = 120.0
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    RETRY_BACKOFF_MULTIPLIER: float = 2.0
    
    # Service configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    
    # Text preprocessing
    MIN_TEXT_LENGTH: int = 3
    MAX_TEXT_LENGTH: int = 2048  # Truncate longer texts
    CLEAN_DOCLING_ARTIFACTS: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')


settings = Settings()

# Log configuration on import
if __name__ == "__main__":
    print("=== Gemini Embedding Service Configuration ===")
    print(f"Model: {settings.EMBEDDING_MODEL}")
    print(f"Embedding Dimension: {settings.EMBEDDING_DIMENSION}")
    print(f"Document Task Type: {settings.DOCUMENT_TASK_TYPE}")
    print(f"Query Task Type: {settings.QUERY_TASK_TYPE}")
    print(f"Max Batch Size: {settings.MAX_BATCH_SIZE}")
    print(f"Max Concurrent Requests: {settings.MAX_CONCURRENT_REQUESTS}")
    print(f"API Key configured: {'Yes' if settings.GEMINI_API_KEY else 'No'}")