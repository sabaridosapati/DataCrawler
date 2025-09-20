# gpu_services/knowledge_graph_service/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuration for the Knowledge Graph Service in distributed setup"""
    
    # Neo4j connection (Mac Mini)
    NEO4J_URI: str = "bolt://192.168.100.41:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"

    # Remote LLM Service (Lenovo Laptop) 
    LOCAL_LLM_URL: str = "http://192.168.100.43:8001/v1"
    
    # Local Embedding Service (Dell Laptop)
    EMBEDDING_SERVICE_URL: str = "http://localhost:8002/embed-documents"

    # Service configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    
    # Graph building parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_ENTITIES_PER_CHUNK: int = 20
    
    # LLM parameters for entity extraction
    EXTRACTION_TEMPERATURE: float = 0.0
    MAX_EXTRACTION_TOKENS: int = 2048
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()

# Enhanced schema for knowledge extraction with more entity types
ENHANCED_NODE_LABELS = [
    "Person", "Organization", "Location", "Product", "Technology", "Service",
    "Event", "Concept", "Project", "Skill", "Document", "Software", "Hardware",
    "Company", "Department", "Role", "Process", "Method", "Tool", "Resource",
    "Date", "Number", "Metric", "Goal", "Requirement", "Feature", "Issue",
    "Solution", "Risk", "Opportunity", "Stakeholder", "Customer", "Vendor"
]

ENHANCED_RELATIONSHIP_TYPES = [
    "WORKS_FOR", "LOCATED_IN", "PRODUCES", "USES", "PART_OF", "MEMBER_OF",
    "ATTENDED", "CREATED", "MENTIONS", "HAS_SKILL", "DEVELOPS", "COMPETES_WITH",
    "REPORTS_TO", "MANAGES", "COLLABORATES_WITH", "DEPENDS_ON", "IMPLEMENTS",
    "REQUIRES", "PROVIDES", "SUPPORTS", "CONTAINS", "REFERENCES", "RELATES_TO",
    "PRECEDES", "FOLLOWS", "CAUSES", "AFFECTS", "INFLUENCES", "BELONGS_TO",
    "RESPONSIBLE_FOR", "PARTICIPATES_IN", "ASSOCIATED_WITH", "DERIVED_FROM"
]

# Advanced system prompt for state-of-the-art entity extraction
ADVANCED_SYSTEM_PROMPT = f"""
You are an expert knowledge graph extraction system with deep understanding of business processes, technical systems, and organizational structures.

Your task is to analyze the provided text and extract entities and relationships to build a comprehensive knowledge graph.

**ENTITY EXTRACTION GUIDELINES:**
1. Extract ALL significant entities mentioned in the text
2. Use ONLY these allowed labels: {ENHANCED_NODE_LABELS}
3. Be comprehensive but precise - don't miss important entities
4. Identify implicit entities that are strongly implied
5. Handle abbreviations and acronyms appropriately
6. Extract numerical values, dates, and metrics as separate entities when significant

**RELATIONSHIP EXTRACTION GUIDELINES:**
1. Use ONLY these relationship types: {ENHANCED_RELATIONSHIP_TYPES}
2. Focus on meaningful, actionable relationships
3. Include hierarchical relationships (reports to, part of, etc.)
4. Capture temporal relationships (precedes, follows)
5. Identify causal relationships (causes, affects, influences)
6. Include organizational relationships (works for, manages, collaborates with)

**OUTPUT FORMAT:**
You must respond with a valid JSON object containing exactly two keys: "nodes" and "relationships".

```json
{{
  "nodes": [
    {{ "id": "Entity Name", "label": "EntityType" }},
    {{ "id": "Another Entity", "label": "AnotherType" }}
  ],
  "relationships": [
    {{ "source": "Entity Name", "target": "Another Entity", "type": "RELATIONSHIP_TYPE" }},
    {{ "source": "Source Entity", "target": "Target Entity", "type": "ANOTHER_RELATIONSHIP" }}
  ]
}}
```

**QUALITY REQUIREMENTS:**
- Extract 5-20 entities per text chunk (depending on content richness)
- Create 3-15 relationships per text chunk
- Ensure all relationship source/target entities exist in the nodes list
- Use precise, specific entity names (not generic terms)
- Maintain consistency in entity naming across extractions
- Focus on business-relevant and actionable information

**EXAMPLES:**
Text: "John Smith, the VP of Engineering at TechCorp, is leading the new AI project using Python and TensorFlow."

Response:
```json
{{
  "nodes": [
    {{ "id": "John Smith", "label": "Person" }},
    {{ "id": "VP of Engineering", "label": "Role" }},
    {{ "id": "TechCorp", "label": "Company" }},
    {{ "id": "AI project", "label": "Project" }},
    {{ "id": "Python", "label": "Technology" }},
    {{ "id": "TensorFlow", "label": "Technology" }}
  ],
  "relationships": [
    {{ "source": "John Smith", "target": "VP of Engineering", "type": "HAS_ROLE" }},
    {{ "source": "John Smith", "target": "TechCorp", "type": "WORKS_FOR" }},
    {{ "source": "John Smith", "target": "AI project", "type": "MANAGES" }},
    {{ "source": "AI project", "target": "Python", "type": "USES" }},
    {{ "source": "AI project", "target": "TensorFlow", "type": "USES" }}
  ]
}}
```

Remember: Extract information that is explicitly stated or strongly implied. Don't hallucinate entities or relationships not supported by the text.
"""