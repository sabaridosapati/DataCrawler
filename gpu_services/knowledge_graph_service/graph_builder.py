# gpu_services/knowledge_graph_service/graph_builder.py

"""
Advanced Knowledge Graph Builder using the NEW google-genai SDK.
FREE-FORM extraction - LLM determines entity types and relationships dynamically.
Uses vector similarity search for graph retrieval.
"""

import logging
import json
import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple

from neo4j import AsyncGraphDatabase
from google import genai
from google.genai import types

from config import settings, KNOWLEDGE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class GeminiKnowledgeExtractor:
    """
    Free-form entity/relationship extraction using NEW google-genai SDK.
    LLM decides entity types and relationship types dynamically.
    """
    
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", settings.GEMINI_API_KEY)
        self.client = genai.Client(api_key=api_key)
        
        self.extraction_model = settings.GEMINI_MODEL
        self.embedding_model = settings.EMBEDDING_MODEL
        
        logger.info(f"GeminiKnowledgeExtractor initialized")
        logger.info(f"  Extraction: {self.extraction_model}")
        logger.info(f"  Embeddings: {self.embedding_model}")
    
    async def extract_knowledge(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships with FREE-FORM types.
        LLM decides the best labels based on content.
        """
        try:
            prompt = f"""{KNOWLEDGE_EXTRACTION_PROMPT}

TEXT TO ANALYZE:
{text}

JSON OUTPUT:"""

            response = await self.client.aio.models.generate_content(
                model=self.extraction_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.EXTRACTION_TEMPERATURE,
                    max_output_tokens=settings.MAX_EXTRACTION_TOKENS,
                )
            )
            
            result_text = response.text.strip()
            
            # Clean markdown code blocks
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            elif result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            result = json.loads(result_text.strip())
            
            entities = result.get("entities", result.get("nodes", []))
            relationships = result.get("relationships", [])
            
            logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")
            return {"entities": entities, "relationships": relationships}
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {"entities": [], "relationships": []}
    
    async def embed_text(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            response = await self.client.aio.models.embed_content(
                model=self.embedding_model,
                contents=text[:2000],  # Truncate for embedding
            )
            if response.embeddings and len(response.embeddings) > 0:
                return list(response.embeddings[0].values)
            return []
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts."""
        if not texts:
            return []
        try:
            # Truncate each text
            truncated = [t[:2000] for t in texts]
            response = await self.client.aio.models.embed_content(
                model=self.embedding_model,
                contents=truncated,
            )
            if response.embeddings:
                return [list(emb.values) for emb in response.embeddings]
            return [[] for _ in texts]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fallback to individual
            results = []
            for text in texts:
                emb = await self.embed_text(text)
                results.append(emb)
            return results
    
    def close(self):
        try:
            self.client.close()
        except:
            pass


class Neo4jGraphStore:
    """
    Neo4j graph storage with VECTOR SIMILARITY SEARCH.
    Supports free-form entity types and relationship types.
    """
    
    def __init__(self, driver):
        self.driver = driver
    
    async def ensure_indexes(self):
        """Create vector indexes for similarity search."""
        async with self.driver.session() as session:
            try:
                # Vector index on Entity nodes
                await session.run("""
                    CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
                    FOR (e:Entity)
                    ON e.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                """, dim=settings.VECTOR_DIMENSION)
                
                # Vector index on Chunk nodes
                await session.run("""
                    CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
                    FOR (c:Chunk)
                    ON c.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                """, dim=settings.VECTOR_DIMENSION)
                
                logger.info("Vector indexes ensured")
            except Exception as e:
                logger.warning(f"Index creation: {e}")
    
    async def store_document(self, doc_id: str, user_id: str):
        """Create document and user nodes."""
        async with self.driver.session() as session:
            await session.run("""
                MERGE (u:User {id: $user_id})
                ON CREATE SET u.createdAt = timestamp()
                WITH u
                MERGE (d:Document {id: $doc_id})
                SET d.user_id = $user_id, d.updatedAt = timestamp()
                MERGE (u)-[:OWNS]->(d)
            """, doc_id=doc_id, user_id=user_id)
    
    async def store_chunk(
        self, 
        doc_id: str, 
        user_id: str,
        chunk_index: int, 
        text: str, 
        embedding: List[float]
    ):
        """Store chunk with embedding for vector search."""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (d:Document {id: $doc_id})
                MERGE (c:Chunk {doc_id: $doc_id, index: $chunk_index})
                SET c.text = $text, 
                    c.embedding = $embedding,
                    c.user_id = $user_id,
                    c.updatedAt = timestamp()
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, 
                doc_id=doc_id, 
                chunk_index=chunk_index, 
                text=text[:3000],
                embedding=embedding,
                user_id=user_id
            )
    
    async def store_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        embedding: List[float],
        doc_id: str,
        user_id: str,
        chunk_index: int
    ):
        """Store entity with dynamic type and embedding."""
        async with self.driver.session() as session:
            # Sanitize entity type for Neo4j label
            safe_type = "".join(c if c.isalnum() else "_" for c in entity_type)
            if not safe_type or safe_type[0].isdigit():
                safe_type = "Entity"
            
            # Create entity with both generic Entity label and specific type
            await session.run(f"""
                MERGE (e:Entity:{safe_type} {{name: $name}})
                SET e.type = $entity_type,
                    e.description = $description,
                    e.embedding = $embedding,
                    e.doc_id = $doc_id,
                    e.user_id = $user_id,
                    e.updatedAt = timestamp()
                WITH e
                MATCH (c:Chunk {{doc_id: $doc_id, index: $chunk_index}})
                MERGE (c)-[:MENTIONS]->(e)
                WITH e
                MATCH (d:Document {{id: $doc_id}})
                MERGE (d)-[:CONTAINS]->(e)
            """,
                name=name,
                entity_type=entity_type,
                description=description,
                embedding=embedding,
                doc_id=doc_id,
                user_id=user_id,
                chunk_index=chunk_index
            )
    
    async def store_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        description: str,
        doc_id: str
    ):
        """Store relationship with dynamic type."""
        async with self.driver.session() as session:
            # Sanitize relationship type
            safe_type = "".join(c if c.isalnum() or c == "_" else "_" for c in rel_type.upper())
            if not safe_type:
                safe_type = "RELATED_TO"
            
            try:
                await session.run(f"""
                    MATCH (s:Entity {{name: $source}})
                    MATCH (t:Entity {{name: $target}})
                    MERGE (s)-[r:{safe_type}]->(t)
                    SET r.description = $description, r.doc_id = $doc_id
                """,
                    source=source,
                    target=target,
                    description=description,
                    doc_id=doc_id
                )
            except Exception as e:
                logger.debug(f"Relationship skipped: {source}->{target}: {e}")
    
    async def vector_search_entities(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 20,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        VECTOR SIMILARITY SEARCH on entity embeddings.
        Returns entities similar to query with their relationships.
        """
        async with self.driver.session() as session:
            result = await session.run("""
                CALL db.index.vector.queryNodes('entity_vector_index', $top_k, $embedding)
                YIELD node, score
                WHERE node.user_id = $user_id AND score >= $threshold
                OPTIONAL MATCH (node)-[r]-(related:Entity)
                WHERE related.user_id = $user_id
                RETURN node.name AS entity,
                       node.type AS type,
                       node.description AS description,
                       score,
                       collect(DISTINCT {
                           related: related.name,
                           relType: type(r),
                           relDesc: r.description
                       })[..5] AS relationships
                ORDER BY score DESC
            """,
                embedding=query_embedding,
                user_id=user_id,
                top_k=top_k,
                threshold=threshold
            )
            return await result.data()
    
    async def vector_search_chunks(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """VECTOR SIMILARITY SEARCH on chunk embeddings."""
        async with self.driver.session() as session:
            result = await session.run("""
                CALL db.index.vector.queryNodes('chunk_vector_index', $top_k, $embedding)
                YIELD node, score
                WHERE node.user_id = $user_id
                RETURN node.text AS text,
                       node.doc_id AS doc_id,
                       node.index AS chunk_index,
                       score
                ORDER BY score DESC
            """,
                embedding=query_embedding,
                user_id=user_id,
                top_k=top_k
            )
            return await result.data()


class AdvancedGraphBuilderProcessor:
    """
    Complete graph builder with:
    - Free-form LLM extraction (no hardcoded types)
    - Vector embeddings on all nodes
    - Similarity search for retrieval
    """
    
    def __init__(self):
        logger.info("Initializing AdvancedGraphBuilderProcessor...")
        
        self.neo4j_driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_pool_size=10
        )
        
        self.extractor = GeminiKnowledgeExtractor()
        self.graph_store = Neo4jGraphStore(self.neo4j_driver)
        
        logger.info("GraphBuilderProcessor initialized")
    
    async def build_graph_from_chunks(
        self, 
        doc_id: str, 
        user_id: str, 
        chunks: List[Dict[str, Any]]
    ):
        """Build complete knowledge graph with embeddings."""
        if not chunks:
            logger.warning(f"No chunks for doc: {doc_id}")
            return
        
        logger.info(f"Building graph: doc={doc_id}, chunks={len(chunks)}")
        
        # Ensure indexes
        await self.graph_store.ensure_indexes()
        
        # Store document
        await self.graph_store.store_document(doc_id, user_id)
        
        # Filter valid chunks
        valid_chunks = []
        for chunk in chunks:
            text = chunk.get('text', chunk.get('content', ''))
            if text and len(text.strip()) > 30:
                idx = chunk.get('index', chunk.get('chunk_index', 
                    chunk.get('metadata', {}).get('chunk_index', len(valid_chunks))))
                valid_chunks.append({'text': text, 'index': idx})
        
        if not valid_chunks:
            logger.warning(f"No valid chunks for doc: {doc_id}")
            return
        
        logger.info(f"Processing {len(valid_chunks)} chunks")
        
        # Process with rate limiting
        semaphore = asyncio.Semaphore(2)
        
        async def process_chunk(chunk, idx):
            async with semaphore:
                try:
                    text = chunk['text']
                    
                    # 1. Extract knowledge (LLM free-form)
                    knowledge = await self.extractor.extract_knowledge(text)
                    entities = knowledge.get("entities", [])
                    relationships = knowledge.get("relationships", [])
                    
                    # 2. Get chunk embedding
                    chunk_embedding = await self.extractor.embed_text(text)
                    
                    # 3. Store chunk
                    await self.graph_store.store_chunk(
                        doc_id, user_id, idx, text, chunk_embedding
                    )
                    
                    # 4. Get entity embeddings
                    if entities:
                        entity_texts = [
                            f"{e.get('name', '')} ({e.get('type', '')}): {e.get('description', '')}"
                            for e in entities
                        ]
                        entity_embeddings = await self.extractor.embed_batch(entity_texts)
                        
                        # Store entities
                        for entity, embedding in zip(entities, entity_embeddings):
                            await self.graph_store.store_entity(
                                name=entity.get('name', ''),
                                entity_type=entity.get('type', 'Entity'),
                                description=entity.get('description', ''),
                                embedding=embedding,
                                doc_id=doc_id,
                                user_id=user_id,
                                chunk_index=idx
                            )
                    
                    # 5. Store relationships
                    for rel in relationships:
                        await self.graph_store.store_relationship(
                            source=rel.get('source', ''),
                            target=rel.get('target', ''),
                            rel_type=rel.get('type', 'RELATED_TO'),
                            description=rel.get('description', ''),
                            doc_id=doc_id
                        )
                    
                    logger.info(f"Chunk {idx}: {len(entities)} entities, {len(relationships)} rels")
                    
                except Exception as e:
                    logger.error(f"Chunk {idx} failed: {e}")
        
        # Process all
        tasks = [process_chunk(c, i) for i, c in enumerate(valid_chunks)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Graph complete: {doc_id}")
    
    async def similarity_search(
        self,
        query: str,
        user_id: str,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        VECTOR SIMILARITY SEARCH on graph.
        Returns similar entities AND chunks for RAG context.
        """
        # Get query embedding
        query_embedding = await self.extractor.embed_text(query)
        
        if not query_embedding:
            return {"entities": [], "chunks": []}
        
        # Search entities
        entities = await self.graph_store.vector_search_entities(
            query_embedding, user_id, top_k, settings.SIMILARITY_THRESHOLD
        )
        
        # Search chunks
        chunks = await self.graph_store.vector_search_chunks(
            query_embedding, user_id, min(top_k, 10)
        )
        
        return {
            "entities": entities,
            "chunks": chunks
        }
    
    async def health_check(self) -> Dict[str, str]:
        """Health check."""
        status = {"graph_builder": "ok", "neo4j": "unknown", "gemini": "unknown"}
        
        try:
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            status["neo4j"] = "ok"
        except Exception as e:
            status["neo4j"] = f"error: {e}"
        
        try:
            await self.extractor.embed_text("test")
            status["gemini"] = "ok"
        except Exception as e:
            status["gemini"] = f"error: {e}"
        
        return status
    
    async def close(self):
        """Cleanup."""
        await self.neo4j_driver.close()
        self.extractor.close()


# Singleton
graph_builder_processor = None


def get_graph_builder() -> AdvancedGraphBuilderProcessor:
    """Factory function."""
    global graph_builder_processor
    if graph_builder_processor is None:
        graph_builder_processor = AdvancedGraphBuilderProcessor()
    return graph_builder_processor