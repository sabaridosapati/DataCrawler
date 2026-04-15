# orchestrator_api/app/db/milvus_handler.py

"""
Milvus Handler - Connects to a Milvus server (Docker).
"""

import logging
from pymilvus import MilvusClient
from typing import List, Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Configuration ---
COLLECTION_NAME = "document_library"
VECTOR_DIMENSION = 3072  # gemini-embedding-001 dimension


class MilvusHandler:
    """
    Milvus handler - connects to a running Milvus server.
    Start Milvus with: docker-compose -f docker-compose.dev.yml up -d milvus
    """

    def __init__(self):
        self.client: Optional[MilvusClient] = None

    def connect(self):
        """Connect to Milvus server."""
        try:
            uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            self.client = MilvusClient(uri=uri)

            logger.info(f"Connected to Milvus server at {uri}")
            
            # Create collection if needed
            self._create_collection_if_not_exists()
            
        except Exception as e:
            logger.critical(f"Failed to initialize Milvus Lite: {e}")
            raise

    def _create_collection_if_not_exists(self):
        """Create collection with optimal index for HNSW search."""
        if self.client.has_collection(COLLECTION_NAME):
            # Check if existing collection has the correct dimension
            schema = self.client.describe_collection(COLLECTION_NAME)
            for field in schema.get("fields", []):
                if field.get("name") == "vector":
                    existing_dim = field.get("params", {}).get("dim")
                    if existing_dim and existing_dim != VECTOR_DIMENSION:
                        logger.warning(
                            f"Collection dim mismatch: existing={existing_dim}, "
                            f"expected={VECTOR_DIMENSION}. Dropping and recreating."
                        )
                        self.client.drop_collection(COLLECTION_NAME)
                        break
            else:
                logger.info(f"Collection '{COLLECTION_NAME}' exists with correct schema")
                return

        logger.info(f"Creating collection '{COLLECTION_NAME}' with dim={VECTOR_DIMENSION}...")

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=VECTOR_DIMENSION,
            metric_type="COSINE",
            auto_id=True,
            id_type="int"
        )

        logger.info(f"Collection '{COLLECTION_NAME}' created with dim={VECTOR_DIMENSION}")

    def insert_chunks(self, data: List[Dict[str, Any]]) -> Any:
        """Insert document chunks with embeddings."""
        if not self.client:
            raise RuntimeError("Milvus not connected")
        
        if not data:
            return None
        
        # Prepare records for insertion
        records = []
        for item in data:
            records.append({
                "user_id": item['user_id'],
                "doc_id": item['doc_id'],
                "chunk_index": item['chunk_index'],
                "chunk_text": item['chunk_text'][:7999],
                "vector": item['embedding']
            })
        
        result = self.client.insert(
            collection_name=COLLECTION_NAME,
            data=records
        )
        
        logger.info(f"Inserted {len(records)} vectors for user '{data[0]['user_id']}'")
        return result

    def search_user_vectors(
        self,
        user_id: str,
        query_vector: List[float],
        top_k: int = 10,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors within a user's documents.

        Args:
            min_similarity: Minimum cosine similarity (0-1). Results below this
                            are irrelevant and filtered out. Default 0.3.
        """
        if not self.client:
            raise RuntimeError("Milvus not connected")

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            filter=f"user_id == '{user_id}'",
            limit=top_k,
            output_fields=["doc_id", "chunk_text", "chunk_index", "user_id"]
        )

        response = []
        for hits in results:
            for hit in hits:
                # Milvus COSINE metric returns cosine similarity directly (higher = more similar)
                similarity = hit['distance']
                if similarity < min_similarity:
                    continue  # Skip results that are not semantically relevant
                response.append({
                    "id": hit['id'],
                    "distance": hit['distance'],
                    "score": similarity,
                    "doc_id": hit['entity'].get('doc_id'),
                    "chunk_text": hit['entity'].get('chunk_text'),
                    "chunk_index": hit['entity'].get('chunk_index')
                })

        logger.info(f"Search for user '{user_id}' found {len(response)} results above similarity {min_similarity}")
        return response

    def hybrid_search(
        self,
        user_id: str,
        query_vector: List[float],
        doc_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Hybrid search with optional document filtering."""
        if not self.client:
            raise RuntimeError("Milvus not connected")
        
        # Build filter
        filter_expr = f"user_id == '{user_id}'"
        if doc_ids:
            doc_ids_str = ", ".join([f"'{d}'" for d in doc_ids])
            filter_expr += f" and doc_id in [{doc_ids_str}]"
        
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            filter=filter_expr,
            limit=top_k,
            output_fields=["doc_id", "chunk_text", "chunk_index"]
        )
        
        response = []
        for hits in results:
            for hit in hits:
                response.append({
                    "id": hit['id'],
                    "distance": hit['distance'],
                    "score": hit['distance'],  # COSINE metric returns similarity directly
                    "doc_id": hit['entity'].get('doc_id'),
                    "chunk_text": hit['entity'].get('chunk_text'),
                    "chunk_index": hit['entity'].get('chunk_index')
                })

        return response

    def delete_document_vectors(self, user_id: str, doc_id: str) -> int:
        """Delete all vectors for a specific document."""
        if not self.client:
            raise RuntimeError("Milvus not connected")
        
        # Get IDs to delete
        results = self.client.query(
            collection_name=COLLECTION_NAME,
            filter=f"user_id == '{user_id}' and doc_id == '{doc_id}'",
            output_fields=["id"]
        )
        
        if results:
            ids = [r['id'] for r in results]
            self.client.delete(
                collection_name=COLLECTION_NAME,
                ids=ids
            )
            logger.info(f"Deleted {len(ids)} vectors for doc_id '{doc_id}'")
            return len(ids)
        
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.client:
            return {"status": "not_connected"}
        
        stats = self.client.get_collection_stats(COLLECTION_NAME)
        
        return {
            "collection_name": COLLECTION_NAME,
            "num_entities": stats.get('row_count', 0),
            "mode": "Milvus Server",
            "uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            "vector_dimension": VECTOR_DIMENSION
        }


# Singleton instance
milvus_db_handler = MilvusHandler()