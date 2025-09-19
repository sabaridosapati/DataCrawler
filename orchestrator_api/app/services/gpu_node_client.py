# orchestrator_api/app/services/gpu_node_client.py

import logging
import httpx
from typing import List, Dict, Any
from fastapi import HTTPException, status

from app.core.config import settings

logger = logging.getLogger(__name__)

class GPUNodeClient:
    def __init__(self):
        self.docling_url = f"{settings.DOCLING_SERVICE_URL}/process"
        # --- UPDATE THIS URL TO POINT TO THE DEDICATED EMBEDDING SERVICE ---
        self.embedding_url = f"{settings.EMBEDDING_SERVICE_URL}/embed-documents"
        self.graph_builder_url = f"{settings.KNOWLEDGE_GRAPH_SERVICE_URL}/build-graph"
        self.timeout = httpx.Timeout(600.0, connect=60.0)

    async def process_with_docling(self, file_path: str) -> Dict[str, str]:
        logger.info(f"Sending request to Docling service for file: {file_path}")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.docling_url, json={"input_file_path": file_path, "output_directory_path": "/app/data"})
                response.raise_for_status()
                data = response.json()
                logger.info(f"Docling service successfully processed {file_path}")
                return {
                    "markdown_path": data["extracted_markdown_path"],
                    "chunks_path": data["extracted_chunks_path"]
                }
            except httpx.HTTPStatusError as e:
                error_detail = e.response.json().get("detail", e.response.text)
                logger.error(f"Docling service returned an error: {e.response.status_code} - {error_detail}")
                raise RuntimeError(f"Document Extraction Service failed: {error_detail}")
            except httpx.RequestError as e:
                logger.error(f"Could not connect to Docling service: {e}")
                raise RuntimeError(f"Could not connect to Document Extraction Service: {e}")

    async def build_knowledge_graph(self, doc_id: str, user_id: str, chunks: List[Dict[str, Any]]):
        logger.info(f"Sending request to Knowledge Graph service for doc_id: {doc_id}")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                payload = {"doc_id": doc_id, "user_id": user_id, "chunks": chunks}
                response = await client.post(self.graph_builder_url, json=payload)
                response.raise_for_status()
                logger.info(f"Knowledge Graph service successfully processed doc_id: {doc_id}")
            except httpx.HTTPStatusError as e:
                error_detail = e.response.json().get("detail", e.response.text)
                logger.error(f"Knowledge Graph service returned an error: {e.response.status_code} - {error_detail}")
                raise RuntimeError(f"Knowledge Graph Service failed: {error_detail}")
            except httpx.RequestError as e:
                logger.error(f"Could not connect to Knowledge Graph service: {e}")
                raise RuntimeError(f"Could not connect to Knowledge Graph Service: {e}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Calls the Embedding service to get vector embeddings for a batch of texts."""
        logger.info(f"Sending request to Embedding service for a batch of {len(texts)} chunks.")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # This call now correctly targets the dedicated embedding service
                response = await client.post(self.embedding_url, json={"texts": texts})
                response.raise_for_status()
                data = response.json()
                logger.info("Embedding service successfully generated vectors.")
                return data["embeddings"]
            except httpx.HTTPStatusError as e:
                error_detail = e.response.json().get("detail", e.response.text)
                logger.error(f"Embedding service returned an error: {e.response.status_code} - {error_detail}")
                raise RuntimeError(f"Embedding Service failed: {error_detail}")
            except httpx.RequestError as e:
                logger.error(f"Could not connect to Embedding service: {e}")
                raise RuntimeError(f"Could not connect to Embedding Service: {e}")

gpu_node_client = GPUNodeClient()