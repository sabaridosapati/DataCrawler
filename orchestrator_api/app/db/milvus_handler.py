# orchestrator_api/app/db/milvus_handler.py

import logging
from pymilvus import (
    connections, utility, Collection, CollectionSchema, FieldSchema, DataType
)
from typing import List, Dict, Any

from app.core.config import settings # <-- Import settings

logger = logging.getLogger(__name__)

# --- Milvus Configuration ---
COLLECTION_NAME = "document_chunks"
VECTOR_DIMENSION = 768 # EmbeddingGemma's dimension
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
        Connects to a native Milvus instance using host and port.
        """
        try:
            # Use host and port from settings for native connection
            connections.connect(
                alias=self.alias, 
                host=settings.MILVUS_HOST, 
                port=settings.MILVUS_PORT
            )
            logger.info(f"Successfully connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}.")
            self._create_collection_if_not_exists()
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to connect to Milvus. Error: {e}")
            raise

    def _create_collection_if_not_exists(self):
        if utility.has_collection(COLLECTION_NAME):
            return
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        ]
        schema = CollectionSchema(fields, "Document chunks for user-specific RAG")
        collection = Collection(COLLECTION_NAME, schema)
        collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
        logger.info(f"Successfully created Milvus collection '{COLLECTION_NAME}' and its index.")

    def insert_chunks(self, data: List[Dict[str, Any]]):
        if not self.collection:
            raise RuntimeError("Milvus collection not loaded.")
        
        # Prepare data for insertion
        user_ids = [item['user_id'] for item in data]
        doc_ids = [item['doc_id'] for item in data]
        chunk_indices = [item['chunk_index'] for item in data]
        chunk_texts = [item['chunk_text'] for item in data]
        embeddings = [item['embedding'] for item in data]
        
        entities = [user_ids, doc_ids, chunk_indices, chunk_texts, embeddings]

        mr = self.collection.insert(entities)
        self.collection.flush()
        logger.info(f"Inserted {len(mr.primary_keys)} vectors into Milvus.")
        return mr

    def search_user_vectors(self, user_id: str, query_vector: List[float], top_k: int = 5) -> List[dict]:
        if not self.collection:
            raise RuntimeError("Milvus collection not loaded.")
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=top_k,
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

milvus_db_handler = MilvusHandler()