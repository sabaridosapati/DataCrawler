# gpu_services/knowledge_graph_service/graph_builder.py

import logging
import neo4j
from typing import List, Dict, Any

# Import the specific components we need from neo4j-graphrag
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.input_loader.text_loader import TextLoader

from config import settings

logger = logging.getLogger(__name__)

class GraphBuilderProcessor:
    """
    Handles knowledge graph creation using neo4j-graphrag and a local LLM.
    """
    def __init__(self):
        logger.info("Initializing GraphBuilderProcessor...")
        
        # 1. Connect to the Neo4j Database
        self.neo4j_driver = neo4j.GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        
        # 2. Configure the LLM to point to our LOCAL gemma-3n model.
        # The OpenAILLM class works perfectly with any OpenAI-compatible API,
        # which is what our vLLM-powered llm_service provides.
        self.local_llm = OpenAILLM(
            # This model name must match the model being served by vLLM.
            model_name="google/gemma-3n-E2B",
            model_params={
                "temperature": 0.0,
                "response_format": {"type": "json_object"} # Crucial for reliable extraction
            },
            api_key="placeholder-key", # vLLM doesn't require a key
            base_url=settings.LOCAL_LLM_URL # This points to our llm_service container
        )
        
        # 3. Define a generic schema for extraction. This can be expanded later.
        self.node_labels = ["Entity", "Person", "Organization", "Location", "Concept", "Event", "Product", "Technology"]
        self.rel_types = ["RELATES_TO", "LOCATED_IN", "PART_OF", "WORKS_FOR", "PRODUCES", "USES"]

        logger.info(f"GraphBuilderProcessor initialized to use local LLM at {settings.LOCAL_LLM_URL}")

    async def build_graph_from_chunks(self, doc_id: str, user_id: str, chunks: List[Dict[str, Any]]):
        """
        Processes text chunks to extract entities/relations using the local LLM
        and builds the knowledge graph in Neo4j.
        """
        
        full_text = "\n\n---CHUNK SEPARATOR---\n\n".join([chunk['text'] for chunk in chunks])
        
        # We don't need an embedder in this pipeline step.
        kg_builder = SimpleKGPipeline(
            llm=self.local_llm,
            driver=self.neo4j_driver,
            text_splitter=FixedSizeSplitter(chunk_size=1024, chunk_overlap=200),
            embedder=None, # Embedding is a separate service
            entities=self.node_labels,
            relations=self.rel_types,
        )
        
        logger.info(f"Running KG pipeline for doc_id: {doc_id} using local Gemma model.")
        
        text_loader = TextLoader()
        await text_loader.load(full_text, {"doc_id": doc_id, "source": doc_id})
        
        await kg_builder.run_async(data_source=text_loader.get_data_source())
        
        # This final step ensures the newly created graph is owned by the correct user.
        await self._link_graph_to_user(user_id, doc_id)
        
        logger.info(f"Successfully built and linked graph for doc_id: {doc_id}")

    async def _link_graph_to_user(self, user_id: str, doc_id: str):
        """
        Connects the :Document node created by the pipeline to the correct :User node.
        This is the key to our multi-tenant graph strategy.
        """
        # The pipeline creates a Document node with a 'source' property.
        query = """
        MATCH (u:User {username: $user_id})
        MATCH (d:Document {source: $doc_id})
        MERGE (u)-[:OWNS]->(d)
        """
        # Using an async session for Neo4j operations
        async with self.neo4j_driver.session() as session:
            await session.run(query, user_id=user_id, doc_id=doc_id)

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()

# Create a single instance to be loaded at startup
graph_builder_processor = GraphBuilderProcessor()