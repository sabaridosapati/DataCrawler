# gpu_services/knowledge_graph_service/graph_builder.py

import logging
import json
from typing import List, Dict, Any
import httpx
import asyncio

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.base import Embedder

from config import settings, ADVANCED_SYSTEM_PROMPT, ENHANCED_NODE_LABELS, ENHANCED_RELATIONSHIP_TYPES

logger = logging.getLogger(__name__)

class RemoteEmbeddingClient(Embedder):
    """Embedding client that connects to the remote embedding service on Dell laptop"""
    
    def __init__(self, base_url: str):
        self.url = base_url
        self.timeout = httpx.Timeout(300.0, connect=60.0)
        logger.info(f"RemoteEmbeddingClient initialized for URL: {self.url}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of documents"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.url, json={"texts": texts})
                response.raise_for_status()
                result = response.json()
                logger.info(f"Successfully got embeddings for {len(texts)} documents")
                return result["embeddings"]
            except Exception as e:
                logger.error(f"Failed to get embeddings from remote service: {e}")
                raise RuntimeError(f"Embedding service call failed: {e}")

    async def embed_query(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        embeddings = await self.embed_documents([query])
        return embeddings[0]


class AdvancedGraphBuilderProcessor:
    """Enhanced graph builder with improved entity extraction and relationship mapping"""
    
    def __init__(self):
        logger.info("Initializing Advanced GraphBuilderProcessor with distributed services...")

        # Configure LLM client to use remote Gemma 3-4B service
        llm_config = LLMConfig(
            api_key="not-needed-for-local",  # Local service doesn't need API key
            model="google/gemma-3-4b-it",
            base_url=settings.LOCAL_LLM_URL,
            system_prompt=ADVANCED_SYSTEM_PROMPT,
            temperature=settings.EXTRACTION_TEMPERATURE,
        )
        llm_client = OpenAIClient(config=llm_config)

        # Initialize remote embedding client
        embedder = RemoteEmbeddingClient(base_url=settings.EMBEDDING_SERVICE_URL)

        # Initialize Graphiti with enhanced configuration
        try:
            self.graphiti = Graphiti(
                settings.NEO4J_URI,
                settings.NEO4J_USER,
                settings.NEO4J_PASSWORD,
                llm_client=llm_client,
                embedder=embedder,
            )
            logger.info("Graphiti initialized successfully with remote LLM and embedding services")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize Graphiti. Error: {e}", exc_info=True)
            raise

    async def build_graph_from_chunks(self, doc_id: str, user_id: str, chunks: List[Dict[str, Any]]):
        """
        Enhanced graph building process with better error handling and validation
        """
        if not chunks:
            logger.warning(f"No chunks provided for doc_id: {doc_id}. Skipping graph build.")
            return

        logger.info(f"Starting advanced graph building for doc_id: {doc_id} with {len(chunks)} chunks")
        
        # Filter out empty or very short chunks
        valid_chunks = [
            chunk for chunk in chunks 
            if chunk.get('text') and len(chunk['text'].strip()) > 20
        ]
        
        if not valid_chunks:
            logger.warning(f"No valid chunks found for doc_id: {doc_id} after filtering")
            return
            
        logger.info(f"Processing {len(valid_chunks)} valid chunks (filtered from {len(chunks)})")

        try:
            # Process chunks in batches to avoid overwhelming the LLM service
            batch_size = 5
            total_batches = (len(valid_chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(valid_chunks), batch_size):
                batch_chunks = valid_chunks[batch_idx:batch_idx + batch_size]
                current_batch = (batch_idx // batch_size) + 1
                
                logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_chunks)} chunks)")
                
                # Extract texts for this batch
                texts_to_process = [chunk['text'] for chunk in batch_chunks]
                
                # Create metadata for this batch
                batch_metadata = {
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "batch_index": current_batch,
                    "total_batches": total_batches
                }
                
                # Process this batch with Graphiti
                await self.graphiti.add_documents(
                    texts_to_process,
                    metadata=batch_metadata
                )
                
                logger.info(f"Completed batch {current_batch}/{total_batches}")
                
                # Add a small delay between batches to prevent overwhelming the services
                if current_batch < total_batches:
                    await asyncio.sleep(1)

            logger.info(f"Successfully completed advanced graph building for doc_id: {doc_id}")
            
        except Exception as e:
            logger.error(f"Advanced graph building failed for doc_id: {doc_id}. Error: {e}", exc_info=True)
            raise RuntimeError(f"Advanced graph building failed: {e}")

    async def validate_extraction(self, extracted_data: Dict[str, Any]) -> bool:
        """
        Validate the extracted entities and relationships against our schema
        """
        try:
            nodes = extracted_data.get("nodes", [])
            relationships = extracted_data.get("relationships", [])
            
            # Validate nodes
            valid_nodes = set()
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                    
                node_id = node.get("id")
                node_label = node.get("label")
                
                if not node_id or not node_label:
                    continue
                    
                if node_label not in ENHANCED_NODE_LABELS:
                    logger.warning(f"Invalid node label: {node_label}")
                    continue
                    
                valid_nodes.add(node_id)
            
            # Validate relationships
            valid_relationships = 0
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                    
                source = rel.get("source")
                target = rel.get("target")
                rel_type = rel.get("type")
                
                if not source or not target or not rel_type:
                    continue
                    
                if rel_type not in ENHANCED_RELATIONSHIP_TYPES:
                    logger.warning(f"Invalid relationship type: {rel_type}")
                    continue
                    
                if source not in valid_nodes or target not in valid_nodes:
                    logger.warning(f"Relationship references non-existent nodes: {source} -> {target}")
                    continue
                    
                valid_relationships += 1
            
            logger.info(f"Validation: {len(valid_nodes)} valid nodes, {valid_relationships} valid relationships")
            return len(valid_nodes) > 0 and valid_relationships > 0
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for all connected services
        """
        health_status = {
            "graph_builder": "ok",
            "neo4j": "unknown",
            "llm_service": "unknown",
            "embedding_service": "unknown"
        }
        
        # Check Neo4j connection
        try:
            # This is a simple way to test the connection through Graphiti
            await self.graphiti.close()  # This will test the connection
            health_status["neo4j"] = "ok"
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            health_status["neo4j"] = "error"
        
        # Check remote LLM service
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.get(f"{settings.LOCAL_LLM_URL.replace('/v1', '')}/health")
                if response.status_code == 200:
                    health_status["llm_service"] = "ok"
                else:
                    health_status["llm_service"] = "error"
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            health_status["llm_service"] = "error"
        
        # Check remote embedding service
        try:
            embedding_health_url = settings.EMBEDDING_SERVICE_URL.replace('/embed-documents', '/health')
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.get(embedding_health_url)
                if response.status_code == 200:
                    health_status["embedding_service"] = "ok"
                else:
                    health_status["embedding_service"] = "error"
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            health_status["embedding_service"] = "error"
        
        return health_status

    def close(self):
        """Clean up resources"""
        logger.info("Closing AdvancedGraphBuilderProcessor")
        try:
            # Note: Graphiti close should be called in async context
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Create a single instance to be loaded at startup
graph_builder_processor = AdvancedGraphBuilderProcessor()