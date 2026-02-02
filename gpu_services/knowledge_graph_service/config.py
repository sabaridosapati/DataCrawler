# gpu_services/knowledge_graph_service/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for Knowledge Graph Service - Single Machine Deployment"""
    
    # Gemini API (for direct entity extraction)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-lite-preview"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    
    # Neo4j connection (running on host)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Local LLM Service (fallback)
    LOCAL_LLM_URL: str = "http://localhost:8001/v1"
    
    # Local Embedding Service
    EMBEDDING_SERVICE_URL: str = "http://localhost:8002/embed-documents"

    # Service configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    
    # Graph building parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_ENTITIES_PER_CHUNK: int = 50  # Higher limit for free-form
    BATCH_SIZE: int = 5
    
    # LLM parameters
    EXTRACTION_TEMPERATURE: float = 0.2  # Slightly creative for better extraction
    MAX_EXTRACTION_TOKENS: int = 4096
    
    # Timeouts
    LLM_TIMEOUT: float = 120.0
    EMBEDDING_TIMEOUT: float = 120.0
    NEO4J_TIMEOUT: float = 30.0
    
    # Vector search settings
    VECTOR_DIMENSION: int = 768
    SIMILARITY_THRESHOLD: float = 0.7
    TOP_K_RETRIEVAL: int = 20
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')


settings = Settings()


# Free-form system prompt - LLM decides entity types and relationships
KNOWLEDGE_EXTRACTION_PROMPT = """You are an expert knowledge graph builder. Your task is to extract a comprehensive knowledge graph from text.

INSTRUCTIONS:
1. Identify ALL significant entities (people, organizations, concepts, technologies, places, events, dates, products, etc.)
2. Create meaningful relationships between entities
3. Use descriptive, domain-appropriate labels for both entities and relationships
4. Be comprehensive - capture ALL useful information for later retrieval
5. Each entity should have a clear name and a descriptive type label
6. Relationships should describe how entities are connected

OUTPUT FORMAT - Return ONLY valid JSON:
{
  "entities": [
    {
      "name": "Unique entity name",
      "type": "Descriptive type label (e.g., Person, Technology, Concept, Organization, etc.)",
      "description": "Brief context about this entity"
    }
  ],
  "relationships": [
    {
      "source": "Source entity name",
      "target": "Target entity name", 
      "type": "DESCRIPTIVE_RELATIONSHIP_TYPE (e.g., WORKS_FOR, CREATED_BY, USES, LOCATED_IN, etc.)",
      "description": "Context for this relationship"
    }
  ]
}

GUIDELINES:
- Entity types should be semantic and meaningful (not just "Entity")
- Relationship types should be in UPPER_SNAKE_CASE
- Include temporal relationships when dates/events are mentioned
- Extract hierarchical relationships (part_of, contains, belongs_to)
- Capture causal and dependency relationships
- Be precise with entity names - use full proper names when available

Return ONLY the JSON, no additional text or explanation."""