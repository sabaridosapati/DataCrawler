# orchestrator_api/app/db/milvus_handler.py

import logging
from pymilvus import (
    connections, utility, Collection, CollectionSchema, FieldSchema, DataType
)
from typing import List, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Milvus Configuration ---
COLLECTION_NAME = "document_chunks"
VECTOR_DIMENSION = 768 # Gemini embedding dimension
INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
SEARCH_PARAMS = {"metric_type": "L2", "params": {"nprobe": 10}}

class MilvusHandler:
    def __init__(self, alias="default"):
        self.alias = alias
        self.collection = None

    def connect(self):
        """
        Connects to Milvus and ensures the collection exists.
        This is a synchronous operation, typically done at startup.
        """
        try:
            connections.connect(alias=self.alias, host='localhost', port='19530')
            logger.info("Successfully connected to Milvus.")
            self._create_collection_if_not_exists()
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to connect to Milvus or setup collection. Error: {e}")
            raise

    def _create_collection_if_not_exists(self):
        """
        Defines the schema and creates the Milvus collection if it's missing.
        This is the foundation for our user-wise data handling via metadata.
        """
        if utility.has_collection(COLLECTION_NAME):
            logger.info(f"Milvus collection '{COLLECTION_NAME}' already exists.")
            return

        logger.info(f"Milvus collection '{COLLECTION_NAME}' not found. Creating...")
        
        # Define fields. 'user_id' is our key for multi-tenancy.
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        ]
        schema = CollectionSchema(fields, "Document chunks for user-specific RAG")
        
        collection = Collection(COLLECTION_NAME, schema)
        collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
        logger.info(f"Successfully created Milvus collection '{COLLECTION_NAME}' and its index.")

    def insert_chunks(self, data: List[Dict[str, Any]]):
        """
        Inserts a batch of document chunks into the collection.
        'data' should be a list of dicts, e.g.,
        [{'user_id': 'x', 'doc_id': 'y', 'chunk_text': '...', 'embedding': [...]}]
        """
        if not self.collection:
            raise RuntimeError("Milvus collection not loaded.")
        
        # Pymilvus expects lists of values for each field
        entities = [
            data.get("user_id"),
            data.get("doc_id"),
            data.get("chunk_text"),
            data.get("embedding")
        ]

        mr = self.collection.insert(entities)
        self.collection.flush()
        logger.info(f"Inserted {len(mr.primary_keys)} vectors into Milvus.")
        return mr

    def search_user_vectors(self, user_id: str, query_vector: List[float], top_k: int = 5) -> List[dict]:
        """
        Searches for vectors belonging ONLY to a specific user.
        The 'expr' parameter is crucial for ensuring data isolation.
        """
        if not self.collection:
            raise RuntimeError("Milvus collection not loaded.")

        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=top_k,
            expr=f"user_id == '{user_id}'", # <-- THIS ENFORCES USER-WISE SEARCH
            output_fields=["doc_id", "chunk_text"] # Fields to return alongside the result
        )
        
        hits = results[0]
        response = [
            {
                "id": hit.id,
                "distance": hit.distance,
                "doc_id": hit.entity.get('doc_id'),
                "chunk_text": hit.entity.get('chunk_text')
            }
            for hit in hits
        ]
        logger.info(f"Milvus search for user '{user_id}' found {len(response)} results.")
        return response

# --- Singleton Instance ---
# Note: Connection is established in the main app lifespan event.
milvus_db_handler = MilvusHandler()