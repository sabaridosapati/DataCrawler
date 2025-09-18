# gpu_services/embedding_service/embedding.py

import logging
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """
    A singleton class to handle all embedding tasks.
    It initializes the EmbeddingGemma model once to be reused across API calls.
    """
    def __init__(self):
        logger.info(f"Initializing EmbeddingProcessor on device: {settings.DEVICE} with dtype: {settings.TORCH_DTYPE}")
        try:
            # Load the model from Hugging Face. SentenceTransformer handles caching.
            # We pass the device and torch_dtype for optimal performance.
            self.model = SentenceTransformer(
                settings.MODEL_NAME,
                device=settings.DEVICE,
                torch_dtype=settings.TORCH_DTYPE
            )
            # Ensure the output dimension matches our expectation for Milvus.
            assert self.model.get_sentence_embedding_dimension() == settings.EMBEDDING_DIMENSION
            logger.info(f"EmbeddingGemma model '{settings.MODEL_NAME}' loaded successfully.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load EmbeddingGemma model. Error: {e}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of document chunks using the recommended document prompt format.
        """
        # Prepend the document-specific prompt to each text chunk for better performance.
        # We assume no title is available for chunks, so we use "none".
        prefixed_texts = [f"title: none | text: {text}" for text in texts]
        
        logger.info(f"Embedding a batch of {len(prefixed_texts)} document chunks...")
        
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=32,  # Tune this based on VRAM
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info("Batch embedding complete.")
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single user query using the recommended retrieval query prompt format.
        """
        # Prepend the query-specific prompt for optimal retrieval performance.
        prefixed_text = f"task: search result | query: {text}"
        
        logger.info(f"Embedding single query...")
        
        embedding = self.model.encode(
            prefixed_text,
            convert_to_numpy=True
        )
        
        logger.info("Query embedding complete.")
        return embedding.tolist()

# Create a single instance to be loaded at startup
embedding_processor = EmbeddingProcessor()