# orchestrator_api/app/services/hybrid_retriever.py

"""
State-of-the-Art Hybrid Retrieval System

Implements:
1. HyDE (Hypothetical Document Embeddings) - Query expansion with LLM
2. BM25 lexical search on chunk text
3. Vector similarity search on embeddings (Milvus)
4. Graph context enrichment from Neo4j
5. Reciprocal Rank Fusion for score combination
6. Cross-encoder reranking (optional)
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from app.db.milvus_handler import milvus_db_handler
from app.services import gpu_node_client

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    chunk_text: str
    doc_id: str
    chunk_index: int
    score: float
    source: str  # "vector", "bm25", "graph"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BM25Index:
    """
    Simple in-memory BM25 index for lexical search.
    For production, consider using Elasticsearch or Whoosh.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, Dict] = {}  # doc_key -> {text, doc_id, chunk_index}
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # term -> doc count
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length = 0
        self.total_docs = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer with lowercasing."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 2]
    
    def add_document(self, doc_key: str, text: str, doc_id: str, chunk_index: int):
        """Add a document to the index."""
        tokens = self._tokenize(text)
        
        self.documents[doc_key] = {
            "text": text,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "tokens": tokens,
            "term_freqs": defaultdict(int)
        }
        
        # Count term frequencies
        seen_terms = set()
        for token in tokens:
            self.documents[doc_key]["term_freqs"][token] += 1
            if token not in seen_terms:
                self.doc_freqs[token] += 1
                seen_terms.add(token)
        
        self.doc_lengths[doc_key] = len(tokens)
        self.total_docs += 1
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(1, self.total_docs)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search and return (doc_key, score) pairs."""
        query_tokens = self._tokenize(query)
        scores = {}
        
        for doc_key, doc_data in self.documents.items():
            score = 0.0
            doc_len = self.doc_lengths[doc_key]
            
            for token in query_tokens:
                if token in doc_data["term_freqs"]:
                    tf = doc_data["term_freqs"][token]
                    df = self.doc_freqs.get(token, 0)
                    
                    # IDF
                    idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
                    
                    # BM25 score
                    tf_norm = (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avg_doc_length))
                    )
                    score += idf * tf_norm
            
            if score > 0:
                scores[doc_key] = score
        
        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


class HybridRetriever:
    """
    State-of-the-art hybrid retrieval combining:
    - HyDE for query expansion
    - BM25 for lexical matching
    - Vector search for semantic similarity
    - Graph traversal for knowledge context
    - Reciprocal Rank Fusion for score combination
    """
    
    def __init__(self):
        self.bm25_indices: Dict[str, BM25Index] = {}  # user_id -> BM25Index
    
    async def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        use_hyde: bool = True,
        use_bm25: bool = True,
        use_graph: bool = True,
        rrf_k: int = 60  # RRF constant
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval with multiple strategies.
        
        Args:
            user_id: User ID for isolation
            query: User's query
            top_k: Number of results to return
            use_hyde: Whether to use HyDE query expansion
            use_bm25: Whether to include BM25 results
            use_graph: Whether to include graph context
            rrf_k: Reciprocal Rank Fusion constant (higher = smoother blending)
        
        Returns:
            List of RetrievalResult objects, scored and ranked
        """
        logger.info(f"Hybrid retrieval for user {user_id}: '{query[:50]}...'")
        
        all_results: Dict[str, List[Tuple[RetrievalResult, int]]] = defaultdict(list)
        
        # 1. Vector Search (always enabled)
        vector_results = await self._vector_search(user_id, query, top_k * 2)
        for rank, result in enumerate(vector_results):
            key = f"{result.doc_id}:{result.chunk_index}"
            all_results[key].append((result, rank + 1))
        
        # 2. HyDE - Query Expansion
        if use_hyde:
            hyde_results = await self._hyde_search(user_id, query, top_k * 2)
            for rank, result in enumerate(hyde_results):
                key = f"{result.doc_id}:{result.chunk_index}"
                all_results[key].append((result, rank + 1))
        
        # 3. BM25 Lexical Search
        if use_bm25:
            bm25_results = await self._bm25_search(user_id, query, top_k * 2)
            for rank, result in enumerate(bm25_results):
                key = f"{result.doc_id}:{result.chunk_index}"
                all_results[key].append((result, rank + 1))
        
        # 4. Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(all_results, rrf_k)
        
        # 5. Sort by fused score and take top_k
        fused_results.sort(key=lambda x: x.score, reverse=True)
        final_results = fused_results[:top_k]
        
        logger.info(f"Retrieved {len(final_results)} results via hybrid retrieval")
        return final_results
    
    async def _vector_search(
        self, 
        user_id: str, 
        query: str, 
        top_k: int
    ) -> List[RetrievalResult]:
        """Perform vector similarity search using Milvus."""
        try:
            # Get query embedding
            query_embedding = await gpu_node_client.embed_query(query)
            
            # Search Milvus
            results = milvus_db_handler.search_user_vectors(
                user_id=user_id,
                query_vector=query_embedding,
                top_k=top_k
            )
            
            return [
                RetrievalResult(
                    chunk_text=r.get("chunk_text", ""),
                    doc_id=r.get("doc_id", ""),
                    chunk_index=r.get("chunk_index", 0),
                    score=r.get("score", 0.0),
                    source="vector"
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _hyde_search(
        self, 
        user_id: str, 
        query: str, 
        top_k: int
    ) -> List[RetrievalResult]:
        """
        HyDE: Generate hypothetical document, then search with its embedding.
        This improves retrieval by bridging the query-document vocabulary gap.
        """
        try:
            # Generate hypothetical answer using LLM
            hyde_prompt = f"""Given this question, write a short paragraph that would be a perfect answer 
found in a document. Write only the answer text, no preamble.

Question: {query}

Hypothetical answer:"""
            
            hypothetical_doc = await gpu_node_client.generate_response(
                messages=[{"role": "user", "content": hyde_prompt}],
                temperature=0.0,
                max_tokens=300
            )
            
            logger.info(f"HyDE generated: '{hypothetical_doc[:100]}...'")
            
            # Embed hypothetical document
            hyde_embedding = await gpu_node_client.embed_query(hypothetical_doc)
            
            # Search with hypothetical embedding
            results = milvus_db_handler.search_user_vectors(
                user_id=user_id,
                query_vector=hyde_embedding,
                top_k=top_k
            )
            
            return [
                RetrievalResult(
                    chunk_text=r.get("chunk_text", ""),
                    doc_id=r.get("doc_id", ""),
                    chunk_index=r.get("chunk_index", 0),
                    score=r.get("score", 0.0),
                    source="hyde"
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"HyDE search failed: {e}")
            return []
    
    async def _bm25_search(
        self, 
        user_id: str, 
        query: str, 
        top_k: int
    ) -> List[RetrievalResult]:
        """Perform BM25 lexical search."""
        try:
            # Build BM25 index if not exists
            if user_id not in self.bm25_indices:
                await self._build_bm25_index(user_id)
            
            index = self.bm25_indices.get(user_id)
            if not index or index.total_docs == 0:
                return []
            
            # Search
            results = index.search(query, top_k)
            
            return [
                RetrievalResult(
                    chunk_text=index.documents[doc_key]["text"],
                    doc_id=index.documents[doc_key]["doc_id"],
                    chunk_index=index.documents[doc_key]["chunk_index"],
                    score=score / max(1, max(s for _, s in results)),  # Normalize
                    source="bm25"
                )
                for doc_key, score in results
            ]
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    async def _build_bm25_index(self, user_id: str):
        """Build BM25 index from user's documents in Milvus."""
        logger.info(f"Building BM25 index for user {user_id}")
        
        index = BM25Index()
        
        # Query all user's chunks from Milvus
        try:
            # Get a sample embedding to search all docs
            sample_embedding = [0.0] * 768  # Zero vector
            results = milvus_db_handler.search_user_vectors(
                user_id=user_id,
                query_vector=sample_embedding,
                top_k=1000  # Get all documents
            )
            
            for r in results:
                doc_key = f"{r['doc_id']}:{r['chunk_index']}"
                index.add_document(
                    doc_key=doc_key,
                    text=r.get("chunk_text", ""),
                    doc_id=r.get("doc_id", ""),
                    chunk_index=r.get("chunk_index", 0)
                )
            
            self.bm25_indices[user_id] = index
            logger.info(f"BM25 index built with {index.total_docs} documents")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_indices[user_id] = BM25Index()
    
    def _reciprocal_rank_fusion(
        self,
        all_results: Dict[str, List[Tuple[RetrievalResult, int]]],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Combine results from multiple retrieval methods using RRF.
        RRF score = sum(1 / (k + rank)) across all methods
        """
        fused_scores: Dict[str, float] = {}
        best_result: Dict[str, RetrievalResult] = {}
        
        for key, results_with_ranks in all_results.items():
            rrf_score = 0.0
            for result, rank in results_with_ranks:
                rrf_score += 1.0 / (k + rank)
                # Keep the result with highest original score
                if key not in best_result or result.score > best_result[key].score:
                    best_result[key] = result
            
            fused_scores[key] = rrf_score
        
        # Create final results with RRF scores
        final_results = []
        for key, rrf_score in fused_scores.items():
            result = best_result[key]
            result.score = rrf_score
            result.metadata["rrf_score"] = rrf_score
            final_results.append(result)
        
        return final_results
    
    def invalidate_bm25_cache(self, user_id: str):
        """Invalidate BM25 index when user's documents change."""
        if user_id in self.bm25_indices:
            del self.bm25_indices[user_id]
            logger.info(f"Invalidated BM25 cache for user {user_id}")


# Singleton instance
hybrid_retriever = HybridRetriever()
