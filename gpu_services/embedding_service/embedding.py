# gpu_services/embedding_service/embedding.py

import logging
import re
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

class EnhancedEmbeddingProcessor:
    """
    Enhanced embedding processor specifically designed to handle data from docling service.
    Optimized for RTX 4050 (6GB VRAM) and EmbeddingGemma-300M model.
    """
    def __init__(self):
        logger.info(f"Initializing Enhanced EmbeddingProcessor on device: {settings.DEVICE}")
        logger.info(f"Torch dtype: {settings.TORCH_DTYPE}")
        logger.info(f"Model: {settings.MODEL_NAME}")
        
        try:
            # Load the EmbeddingGemma model with optimized settings for RTX 4050
            self.model = SentenceTransformer(
                settings.MODEL_NAME,
                device=settings.DEVICE,
                torch_dtype=settings.TORCH_DTYPE,
                trust_remote_code=True
            )
            
            # Verify embedding dimension
            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim != settings.EMBEDDING_DIMENSION:
                logger.warning(f"Embedding dimension mismatch: Expected {settings.EMBEDDING_DIMENSION}, got {actual_dim}")
                settings.EMBEDDING_DIMENSION = actual_dim
            
            logger.info(f"EmbeddingGemma model loaded successfully")
            logger.info(f"Embedding dimension: {settings.EMBEDDING_DIMENSION}")
            
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to load EmbeddingGemma model. Error: {e}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of document chunks using the recommended document prompt format.
        Specifically handles data from Docling service which provides clean, contextual text chunks.
        """
        if not texts:
            logger.warning("Empty text list provided to embed_documents")
            return []
        
        # Filter out empty or very short texts that might come from docling processing
        valid_texts = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if text and len(text.strip()) > 10:
                cleaned_text = self._clean_docling_text(text.strip())
                if len(cleaned_text) > 10:  # Double-check after cleaning
                    valid_texts.append(cleaned_text)
                    text_indices.append(i)
        
        if not valid_texts:
            logger.warning("No valid texts found after filtering and cleaning")
            return [[0.0] * settings.EMBEDDING_DIMENSION] * len(texts)
        
        logger.info(f"Embedding {len(valid_texts)} valid document chunks (filtered from {len(texts)} total)")
        
        # Prepend the document-specific prompt to each text chunk for better performance
        prefixed_texts = [f"title: none | text: {text}" for text in valid_texts]
        
        try:
            # Batch size optimized for RTX 4050 (6GB VRAM)
            batch_size = 16 if settings.DEVICE == "cuda" else 8
            
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for better similarity search
                device=settings.DEVICE,
                convert_to_tensor=False  # Return numpy arrays
            )
            
            logger.info(f"Successfully generated embeddings: shape {embeddings.shape}")
            
            # Create results array with correct indexing
            results = [[0.0] * settings.EMBEDDING_DIMENSION] * len(texts)
            
            # Fill in the valid embeddings at their original indices
            for i, embedding in enumerate(embeddings):
                original_index = text_indices[i]
                results[original_index] = embedding.tolist()
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch embedding: {e}", exc_info=True)
            # Return zero vectors as fallback
            zero_vector = [0.0] * settings.EMBEDDING_DIMENSION
            return [zero_vector] * len(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single user query using the recommended retrieval query prompt format.
        """
        if not text or len(text.strip()) < 3:
            logger.warning("Empty or very short query provided")
            return [0.0] * settings.EMBEDDING_DIMENSION
        
        # Clean the query text
        cleaned_text = self._clean_docling_text(text.strip())
        
        # Prepend the query-specific prompt for optimal retrieval performance
        prefixed_text = f"task: search result | query: {cleaned_text}"
        
        logger.info(f"Embedding query: '{cleaned_text[:50]}...'")
        
        try:
            embedding = self.model.encode(
                prefixed_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=settings.DEVICE
            )
            
            logger.info("Query embedding complete")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error during query embedding: {e}", exc_info=True)
            return [0.0] * settings.EMBEDDING_DIMENSION

    def _clean_docling_text(self, text: str) -> str:
        """
        Clean text that comes from docling processing to optimize for embedding.
        Removes artifacts and formatting that might interfere with embedding quality.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common docling artifacts
        # Table separators and formatting
        cleaned = re.sub(r'\|{2,}', ' ', cleaned)  # Multiple pipe characters
        cleaned = re.sub(r'-{3,}', ' ', cleaned)  # Horizontal lines
        cleaned = re.sub(r'_{3,}', ' ', cleaned)  # Underline separators
        cleaned = re.sub(r'={3,}', ' ', cleaned)  # Equal sign separators
        cleaned = re.sub(r'\*{3,}', ' ', cleaned)  # Asterisk separators
        
        # Remove page numbers and common headers/footers
        cleaned = re.sub(r'Page\s+\d+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\d+\s*/\s*\d+', '', cleaned)  # Page x/y format
        cleaned = re.sub(r'Page\s+\d+\s+of\s+\d+', '', cleaned, flags=re.IGNORECASE)
        
        # Remove common PDF artifacts
        cleaned = re.sub(r'Figure\s+\d+:', 'Figure:', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Table\s+\d+:', 'Table:', cleaned, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        cleaned = re.sub(r'\.{3,}', '...', cleaned)  # Multiple dots
        cleaned = re.sub(r',{2,}', ',', cleaned)     # Multiple commas
        
        # Remove markdown artifacts that might come from docling
        cleaned = re.sub(r'#{1,6}\s*', '', cleaned)  # Headers
        cleaned = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', cleaned)  # Bold/italic
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)  # Code blocks
        
        # Clean up multiple spaces again after all replacements
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove very short "words" that are likely artifacts
        words = cleaned.split()
        filtered_words = [word for word in words if len(word) > 1 or word.isalnum()]
        cleaned = ' '.join(filtered_words)
        
        return cleaned

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": settings.MODEL_NAME,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "device": settings.DEVICE,
            "torch_dtype": str(settings.TORCH_DTYPE),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown')
        }

    def health_check(self) -> bool:
        """Perform a health check by embedding a test text"""
        try:
            test_embedding = self.embed_query("test health check")
            return len(test_embedding) == settings.EMBEDDING_DIMENSION
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Create a single instance to be loaded at startup
embedding_processor = EnhancedEmbeddingProcessor()