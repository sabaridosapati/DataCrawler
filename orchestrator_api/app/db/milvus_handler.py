# orchestrator_api/app/db/milvus_handler.py

import logging
from pymilvus import (
    connections, utility, Collection, CollectionSchema, FieldSchema, DataType
)
from typing import List, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Milvus Configuration ---
COLLECTION_NAME = "document_library_v1"
VECTOR_DIMENSION = 768 # From embeddinggemma-300m

# --- STATE-OF-THE-ART HNSW INDEX PARAMETERS ---
# HNSW is a graph-based index that is much faster and more accurate
# for most real-world use cases than IVF_FLAT.
INDEX_PARAMS = {
    "metric_type": "L2",      # L2 is the standard for measuring distance between embeddings
    "index_type": "HNSW",     # Use the high-performance HNSW index
    "params": {
        "M": 16,              # Number of bi-directional links for each node
        "efConstruction": 256 # Size of the dynamic list for searching during construction
    }
}

# Search parameters for HNSW
SEARCH_PARAMS = {
    "metric_type": "L2",
    "params": {
        "ef": 128             # Size of the dynamic list for searching at query time
    }
}

class MilvusHandler:
    def __init__(self, alias="default"):
        self.alias = alias
        self.collection = None

    def connect(self):
        """
        Connects to a standalone Milvus instance using host and port.
        """
        try:
            # Check if a connection already exists to avoid errors
            if self.alias in connections.list_connections():
                connections.disconnect(self.alias)

            connections.connect(
                alias=self.alias,
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            logger.info(f"Successfully connected to standalone Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}.")
            self._create_collection_if_not_exists()
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to connect to standalone Milvus. Is it running on the Linux machine? Error: {e}")
            raise

    def _create_collection_if_not_exists(self):
        if utility.has_collection(COLLECTION_NAME):
            logger.info(f"Milvus collection '{COLLECTION_NAME}' already exists.")
            return

        # A unified, multi-tenant schema for all document types
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=255, description="ID of the user who owns the data"),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255, description="ID of the source document"),
            FieldSchema(name="chunk_index", dtype=DataType.INT64, description="Order of the chunk within the document"),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=4000, description="The actual text content of the chunk"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        ]
        schema = CollectionSchema(fields, "Unified document library for multi-tenant RAG")
        collection = Collection(COLLECTION_NAME, schema)
        
        logger.info(f"Creating HNSW index for collection '{COLLECTION_NAME}'...")
        collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
        logger.info("HNSW index created successfully.")

    def insert_chunks(self, data: List[Dict[str, Any]]):
        if not self.collection:
            raise RuntimeError("Milvus collection not loaded. Cannot insert.")
        
        # Prepare data for insertion from the pipeline's output
        user_ids = [item['user_id'] for item in data]
        doc_ids = [item['doc_id'] for item in data]
        chunk_indices = [item['chunk_index'] for item in data]
        chunk_texts = [item['chunk_text'] for item in data]
        embeddings = [item['embedding'] for item in data]
        
        entities = [user_ids, doc_ids, chunk_indices, chunk_texts, embeddings]

        mr = self.collection.insert(entities)
        self.collection.flush() # Ensure data is written to disk
        logger.info(f"Inserted {len(mr.primary_keys)} vectors into Milvus for user '{user_ids[0]}'.")
        return mr

    def search_user_vectors(self, user_id: str, query_vector: List[float], top_k: int = 5) -> List[dict]:
        if not self.collection:
            raise RuntimeError("Milvus collection not loaded. Cannot search.")
            
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=top_k,
            # This expression is the key to user-wise data isolation
            expr=f"user_id == '{user_id}'",
            output_fields=["doc_id", "chunk_text", "chunk_index"]
        )
        
        hits = results[0]
        response = [
            {
                "id": hit.id,
                "distance": hit.distance,
                "doc_id": hit.entity.get('doc_id'),
                "chunk_text": hit.entity.get('chunk_text'),
                "chunk_index": hit.entity.get('chunk_index')
            }
            for hit in hits
        ]
        logger.info(f"Milvus search for user '{user_id}' found {len(response)} results.")
        return response

# Create a single, importable instance of the handler
milvus_db_handler = MilvusHandler()