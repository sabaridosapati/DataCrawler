# orchestrator_api/app/services/gpu_node_client.py

"""
Service client for communicating with local GPU services.
Single-machine deployment with all services on localhost.
"""

import logging
import httpx
from typing import List, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class ServiceClient:
    """
    Unified client for all local services.
    Handles Docling, Embedding, Knowledge Graph, and LLM services.
    """
    
    def __init__(self):
        # Service endpoints (all localhost)
        self.docling_url = f"{settings.DOCLING_SERVICE_URL}/process"
        self.embedding_url = f"{settings.EMBEDDING_SERVICE_URL}/embed-documents"
        self.embedding_query_url = f"{settings.EMBEDDING_SERVICE_URL}/embed-query"
        self.graph_builder_url = f"{settings.KNOWLEDGE_GRAPH_SERVICE_URL}/build-graph"
        self.llm_url = f"{settings.LLM_SERVICE_URL}/v1/chat/completions"
        
        # Timeouts optimized for local services
        self.timeout = httpx.Timeout(300.0, connect=30.0)
        
        logger.info("ServiceClient initialized for single-machine deployment")
        logger.info(f"  Docling: {settings.DOCLING_SERVICE_URL}")
        logger.info(f"  Embedding: {settings.EMBEDDING_SERVICE_URL}")
        logger.info(f"  Knowledge Graph: {settings.KNOWLEDGE_GRAPH_SERVICE_URL}")
        logger.info(f"  LLM: {settings.LLM_SERVICE_URL}")

    async def process_with_docling(self, file_path: str) -> Dict[str, str]:
        """
        Process document with Docling service.
        Returns paths to extracted markdown and chunks.
        """
        from pathlib import Path
        
        logger.info(f"Processing with Docling: {file_path}")
        
        # Use parent directory for output (sibling to 'raw' folder)
        input_path = Path(file_path)
        output_dir = str(input_path.parent.parent / "extracted")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.docling_url, 
                    json={
                        "input_file_path": str(input_path.absolute()), 
                        "output_directory_path": output_dir
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Docling processing complete: {file_path}")
                return {
                    "markdown_path": data.get("extracted_markdown_path", data.get("markdown_path")),
                    "chunks_path": data.get("extracted_chunks_path", data.get("chunks_path"))
                }
                
            except httpx.HTTPStatusError as e:
                error_detail = e.response.json().get("detail", e.response.text)
                logger.error(f"Docling error: {error_detail}")
                raise RuntimeError(f"Document extraction failed: {error_detail}")
            except httpx.RequestError as e:
                logger.error(f"Cannot connect to Docling: {e}")
                raise RuntimeError(f"Cannot connect to Docling service: {e}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of document texts.
        Uses Gemini embedding-001 via local embedding service.
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} document chunks")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.embedding_url, 
                    json={"texts": texts}
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Embedding complete: {data.get('processed_count', len(texts))} chunks")
                return data["embeddings"]
                
            except httpx.HTTPStatusError as e:
                error_detail = e.response.json().get("detail", e.response.text)
                logger.error(f"Embedding error: {error_detail}")
                raise RuntimeError(f"Embedding failed: {error_detail}")
            except httpx.RequestError as e:
                logger.error(f"Cannot connect to Embedding service: {e}")
                raise RuntimeError(f"Cannot connect to Embedding service: {e}")

    async def embed_query(self, text: str) -> List[float]:
        """
        Get embedding for a single query.
        Uses RETRIEVAL_QUERY task type for optimal search performance.
        """
        logger.info(f"Embedding query: '{text[:50]}...'")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.embedding_query_url, 
                    json={"text": text}
                )
                response.raise_for_status()
                data = response.json()
                
                return data["embedding"]
                
            except httpx.HTTPStatusError as e:
                error_detail = e.response.json().get("detail", e.response.text)
                logger.error(f"Query embedding error: {error_detail}")
                raise RuntimeError(f"Query embedding failed: {error_detail}")
            except httpx.RequestError as e:
                logger.error(f"Cannot connect to Embedding service: {e}")
                raise RuntimeError(f"Cannot connect to Embedding service: {e}")

    async def build_knowledge_graph(
        self, 
        doc_id: str, 
        user_id: str, 
        chunks: List[Dict[str, Any]]
    ):
        """
        Build knowledge graph from document chunks.
        Extracts entities and relationships using LLM.
        """
        logger.info(f"Building knowledge graph for doc_id: {doc_id}, chunks: {len(chunks)}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                payload = {
                    "doc_id": doc_id, 
                    "user_id": user_id, 
                    "chunks": chunks
                }
                response = await client.post(self.graph_builder_url, json=payload)
                response.raise_for_status()
                
                logger.info(f"Knowledge graph built for doc_id: {doc_id}")
                return response.json()
                
            except httpx.HTTPStatusError as e:
                error_detail = e.response.json().get("detail", e.response.text)
                logger.error(f"Knowledge Graph error: {error_detail}")
                raise RuntimeError(f"Knowledge Graph failed: {error_detail}")
            except httpx.RequestError as e:
                logger.error(f"Cannot connect to Knowledge Graph service: {e}")
                raise RuntimeError(f"Cannot connect to Knowledge Graph service: {e}")

    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2,
        max_tokens: int = 4096
    ) -> str:
        """
        Generate LLM response using Gemini via local LLM service.
        """
        logger.info(f"Generating LLM response with {len(messages)} messages")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.llm_url,
                    json={
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                generated_text = data["choices"][0]["message"]["content"]
                logger.info(f"LLM response generated: {len(generated_text)} chars")
                return generated_text
                
            except httpx.HTTPStatusError as e:
                error_detail = e.response.json().get("detail", e.response.text)
                logger.error(f"LLM error: {error_detail}")
                raise RuntimeError(f"LLM failed: {error_detail}")
            except httpx.RequestError as e:
                logger.error(f"Cannot connect to LLM service: {e}")
                raise RuntimeError(f"Cannot connect to LLM service: {e}")

    async def check_all_services(self) -> Dict[str, str]:
        """Health check for all services."""
        status = {}
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            for name, url in [
                ("docling", settings.DOCLING_SERVICE_URL),
                ("embedding", settings.EMBEDDING_SERVICE_URL),
                ("knowledge_graph", settings.KNOWLEDGE_GRAPH_SERVICE_URL),
                ("llm", settings.LLM_SERVICE_URL),
            ]:
                try:
                    response = await client.get(f"{url}/health")
                    status[name] = "ok" if response.status_code == 200 else "error"
                except:
                    status[name] = "unreachable"
        
        return status


# Singleton instance
gpu_node_client = ServiceClient()