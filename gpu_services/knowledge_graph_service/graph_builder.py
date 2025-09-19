import logging
from typing import List, Dict, Any
import httpx

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.base import Embedder

from config import settings

logger = logging.getLogger(__name__)

# --- 1. DEFINE A RICH SCHEMA FOR THE KNOWLEDGE GRAPH ---
# This tells the LLM exactly what kinds of information we care about.
NODE_LABELS = [
    "Person", "Organization", "Location", "Product", "Technology",
    "Event", "Concept", "Project", "Skill", "Document", "Software"
]

RELATIONSHIP_TYPES = [
    "WORKS_FOR", "LOCATED_IN", "PRODUCES", "USES", "PART_OF",
    "MEMBER_OF", "ATTENDED", "CREATED", "MENTIONS", "HAS_SKILL",
    "DEVELOPS", "COMPETES_WITH"
]

# --- 2. ENGINEER A HIGH-QUALITY SYSTEM PROMPT ---
# This is the most critical step for state-of-the-art quality.
# We give the LLM a role, a task, a strict schema, and a required output format.
SYSTEM_PROMPT = f"""
You are a world-class, highly intelligent knowledge graph extraction system.
Your task is to analyze the provided text and extract entities and their relationships.

**INSTRUCTIONS:**
1.  **Identify Entities:** Extract all relevant entities from the text.
2.  **Assign Labels:** Assign a label to each entity using ONLY the following allowed labels: {NODE_LABELS}.
3.  **Identify Relationships:** Identify meaningful, directed relationships between the extracted entities.
4.  **Assign Types:** Assign a relationship type using ONLY the following allowed types: {RELATIONSHIP_TYPES}.
5.  **Output Format:** Your final output MUST be a single, valid JSON object. This object must contain two keys: "nodes" and "relationships".
    - The "nodes" key must have a list of objects, where each object has "id" (the entity name) and "label".
    - The "relationships" key must have a list of objects, where each object has "source" (the 'id' of the source node), "target" (the 'id' of the target node), and "type".

**EXAMPLE OUTPUT:**
{{
  "nodes": [
    {{ "id": "John Doe", "label": "Person" }},
    {{ "id": "Acme Corp", "label": "Organization" }}
  ],
  "relationships": [
    {{ "source": "John Doe", "target": "Acme Corp", "type": "WORKS_FOR" }}
  ]
}}

**CONSTRAINTS:**
- Be precise. Only extract information that is explicitly stated or strongly implied in the text.
- Do not hallucinate or invent information.
- Adhere strictly to the provided node labels and relationship types.
- Ensure your output is a valid JSON object.
"""


# This RemoteEmbeddingClient correctly calls our dedicated embedding_service
class RemoteEmbeddingClient(Embedder):
    def __init__(self, base_url: str):
        self.url = base_url
        self.timeout = httpx.Timeout(300.0, connect=60.0)
        logger.info(f"RemoteEmbeddingClient initialized for URL: {self.url}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.url, json={"texts": texts})
                response.raise_for_status()
                return response.json()["embeddings"]
            except Exception as e:
                logger.error(f"Failed to get embeddings from remote service: {e}")
                raise RuntimeError(f"Embedding service call failed: {e}")

    async def embed_query(self, query: str) -> List[float]:
        embeddings = await self.embed_documents([query])
        return embeddings[0]


class GraphBuilderProcessor:
    def __init__(self):
        logger.info("Initializing State-of-the-Art GraphBuilderProcessor...")

        # --- 3. CONFIGURE THE LLM CLIENT WITH OUR ADVANCED PROMPT ---
        llm_config = LLMConfig(
            api_key="vllm",
            model="google/gemma-3n-E2B-it",
            base_url=settings.LOCAL_LLM_URL,
            # Inject our high-quality prompt into the LLM's configuration
            system_prompt=SYSTEM_PROMPT,
            # Set a low temperature for consistent, factual extraction
            temperature=0.0,
        )
        llm_client = OpenAIClient(config=llm_config)

        # Initialize our custom embedder client
        embedder = RemoteEmbeddingClient(base_url=settings.EMBEDDING_SERVICE_URL)

        # Initialize Graphiti with our powerful, custom-configured clients
        try:
            self.graphiti = Graphiti(
                settings.NEO4J_URI,
                settings.NEO4J_USER,
                settings.NEO4J_PASSWORD,
                llm_client=llm_client,
                embedder=embedder,
            )
            logger.info("Graphiti initialized successfully with custom prompt and remote embedder.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize Graphiti. Error: {e}", exc_info=True)
            raise

    async def build_graph_from_chunks(self, doc_id: str, user_id: str, chunks: List[Dict[str, Any]]):
        """
        Processes text chunks using the Graphiti pipeline, guided by our advanced prompt,
        to build a rich and structured knowledge graph.
        """
        if not chunks:
            logger.warning(f"No chunks provided for doc_id: {doc_id}. Skipping graph build.")
            return

        logger.info(f"Running state-of-the-art Graphiti pipeline for doc_id: {doc_id} with {len(chunks)} chunks.")
        texts_to_process = [chunk['text'] for chunk in chunks]

        try:
            # Graphiti will now use our highly-instructed LLM and remote embedder
            await self.graphiti.add_documents(
                texts_to_process,
                metadata={"doc_id": doc_id, "user_id": user_id}
            )
            logger.info(f"Successfully built graph for doc_id: {doc_id} using advanced pipeline.")
        except Exception as e:
            logger.error(f"Graphiti pipeline failed for doc_id: {doc_id}. Error: {e}", exc_info=True)
            raise RuntimeError(f"Graphiti pipeline failed: {e}")

    def close(self):
        logger.info("Closing GraphBuilderProcessor.")

# Create a single instance to be loaded at startup
graph_builder_processor = GraphBuilderProcessor()