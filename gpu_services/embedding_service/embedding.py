# gpu_services/embedding_service/embedding.py

"""
High-performance embedding processor using Gemini Embedding API.
Optimized for batch processing with rate limiting and retry logic.
"""

import logging
import re
import asyncio
from typing import List, Optional
import numpy as np

from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from config import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddingProcessor:
    """
    Gemini-based embedding processor with async support,
    batch processing, and docling text cleaning.
    """
    
    def __init__(self):
        self.client: Optional[genai.Client] = None
        self.semaphore: Optional[asyncio.Semaphore] = None
        self._initialized = False
        
        logger.info(f"Initializing Gemini Embedding Processor")
        logger.info(f"Model: {settings.EMBEDDING_MODEL}")
        logger.info(f"Embedding dimension: {settings.EMBEDDING_DIMENSION}")
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini client"""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        try:
            http_options = types.HttpOptions(
                timeout=settings.REQUEST_TIMEOUT_SECONDS * 1000,
            )
            
            self.client = genai.Client(
                api_key=settings.GEMINI_API_KEY,
                http_options=http_options
            )
            
            self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
            self._initialized = True
            
            logger.info("Gemini Embedding client initialized successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize Gemini client: {e}", exc_info=True)
            raise
    
    def _clean_docling_text(self, text: str) -> str:
        """
        Clean text from docling processing to optimize for embedding.
        Removes artifacts and formatting that might interfere with embedding quality.
        """
        if not text:
            return ""
        
        if not settings.CLEAN_DOCLING_ARTIFACTS:
            return text.strip()
        
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common docling artifacts
        cleaned = re.sub(r'\|{2,}', ' ', cleaned)  # Multiple pipe characters
        cleaned = re.sub(r'-{3,}', ' ', cleaned)   # Horizontal lines
        cleaned = re.sub(r'_{3,}', ' ', cleaned)   # Underline separators
        cleaned = re.sub(r'={3,}', ' ', cleaned)   # Equal sign separators
        cleaned = re.sub(r'\*{3,}', ' ', cleaned)  # Asterisk separators
        
        # Remove page numbers and common headers/footers
        cleaned = re.sub(r'Page\s+\d+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\d+\s*/\s*\d+', '', cleaned)
        cleaned = re.sub(r'Page\s+\d+\s+of\s+\d+', '', cleaned, flags=re.IGNORECASE)
        
        # Remove markdown artifacts
        cleaned = re.sub(r'#{1,6}\s*', '', cleaned)  # Headers
        cleaned = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', cleaned)  # Bold/italic
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)  # Code blocks
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Truncate if too long
        if len(cleaned) > settings.MAX_TEXT_LENGTH:
            cleaned = cleaned[:settings.MAX_TEXT_LENGTH]
        
        return cleaned

    def _prepare_texts(self, texts: List[str]) -> tuple[List[str], List[int]]:
        """
        Prepare and filter texts for embedding.
        Returns (valid_texts, original_indices)
        """
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and len(text.strip()) >= settings.MIN_TEXT_LENGTH:
                cleaned = self._clean_docling_text(text.strip())
                if len(cleaned) >= settings.MIN_TEXT_LENGTH:
                    valid_texts.append(cleaned)
                    valid_indices.append(i)
        
        return valid_texts, valid_indices

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    async def _embed_batch_async(
        self, 
        texts: List[str], 
        task_type: str
    ) -> List[List[float]]:
        """
        Embed a single batch of texts using Gemini API.
        """
        async with self.semaphore:
            try:
                config = types.EmbedContentConfig(
                    task_type=task_type
                )
                
                result = await self.client.aio.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=texts,
                    config=config
                )
                
                # Extract embeddings
                embeddings = [emb.values for emb in result.embeddings]
                return embeddings
                
            except Exception as e:
                logger.error(f"Error embedding batch: {e}", exc_info=True)
                raise

    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of document chunks using async processing.
        Handles batching for large inputs.
        """
        if not texts:
            logger.warning("Empty text list provided to embed_documents")
            return []
        
        # Prepare and filter texts
        valid_texts, valid_indices = self._prepare_texts(texts)
        
        if not valid_texts:
            logger.warning("No valid texts found after filtering")
            return [[0.0] * settings.EMBEDDING_DIMENSION] * len(texts)
        
        logger.info(f"Embedding {len(valid_texts)} valid texts (filtered from {len(texts)})")
        
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(valid_texts), settings.OPTIMAL_BATCH_SIZE):
                batch = valid_texts[i:i + settings.OPTIMAL_BATCH_SIZE]
                batch_num = (i // settings.OPTIMAL_BATCH_SIZE) + 1
                total_batches = (len(valid_texts) + settings.OPTIMAL_BATCH_SIZE - 1) // settings.OPTIMAL_BATCH_SIZE
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                batch_embeddings = await self._embed_batch_async(
                    batch, 
                    settings.DOCUMENT_TASK_TYPE
                )
                all_embeddings.extend(batch_embeddings)
            
            # Reconstruct results with zero vectors for filtered texts
            results = [[0.0] * settings.EMBEDDING_DIMENSION] * len(texts)
            for i, embedding in enumerate(all_embeddings):
                original_idx = valid_indices[i]
                # Normalize embeddings for consistency
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (np.array(embedding) / norm).tolist()
                results[original_idx] = embedding
            
            logger.info(f"Successfully generated {len(valid_texts)} embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error during batch embedding: {e}", exc_info=True)
            return [[0.0] * settings.EMBEDDING_DIMENSION] * len(texts)

    async def embed_query_async(self, text: str) -> List[float]:
        """
        Embed a single query using async processing.
        Uses RETRIEVAL_QUERY task type for optimal search performance.
        """
        if not text or len(text.strip()) < settings.MIN_TEXT_LENGTH:
            logger.warning("Empty or very short query provided")
            return [0.0] * settings.EMBEDDING_DIMENSION
        
        cleaned_text = self._clean_docling_text(text.strip())
        
        logger.info(f"Embedding query: '{cleaned_text[:50]}...'")
        
        try:
            embeddings = await self._embed_batch_async(
                [cleaned_text],
                settings.QUERY_TASK_TYPE
            )
            
            embedding = embeddings[0]
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            
            logger.info("Query embedding complete")
            return embedding
            
        except Exception as e:
            logger.error(f"Error during query embedding: {e}", exc_info=True)
            return [0.0] * settings.EMBEDDING_DIMENSION

    # Synchronous wrappers for backward compatibility
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for embed_documents_async"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.embed_documents_async(texts)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.embed_documents_async(texts))
        except RuntimeError:
            # No event loop exists
            return asyncio.run(self.embed_documents_async(texts))

    def embed_query(self, text: str) -> List[float]:
        """Synchronous wrapper for embed_query_async"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.embed_query_async(text)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.embed_query_async(text))
        except RuntimeError:
            return asyncio.run(self.embed_query_async(text))

    def get_model_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            "model_name": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "backend": "gemini-api",
            "document_task_type": settings.DOCUMENT_TASK_TYPE,
            "query_task_type": settings.QUERY_TASK_TYPE,
            "max_batch_size": settings.MAX_BATCH_SIZE,
            "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS
        }

    async def health_check(self) -> bool:
        """Perform a health check by embedding a test text"""
        try:
            test_embedding = await self.embed_query_async("health check test")
            return len(test_embedding) == settings.EMBEDDING_DIMENSION
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Create a single instance to be loaded at startup
embedding_processor = GeminiEmbeddingProcessor()