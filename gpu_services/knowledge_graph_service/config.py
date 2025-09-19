from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # Correctly points to the vLLM container
    LOCAL_LLM_URL: str = "http://llm_service:8001/v1"

    # Correctly points to the embedding container
    EMBEDDING_SERVICE_URL: str = "http://embedding_service:8002/embed-documents"

    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()