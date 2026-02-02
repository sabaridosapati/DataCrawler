

import logging
from neo4j import AsyncGraphDatabase, EagerResult
from typing import List, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

class Neo4jHandler:
    def __init__(self, uri, user, password):
        self._uri = uri
        self._auth = (user, password)
        self._driver = None

    async def connect(self):
        """Establishes the connection to the Neo4j database."""
        if not self._driver:
            try:
                self._driver = AsyncGraphDatabase.driver(self._uri, auth=self._auth)
                await self._driver.verify_connectivity()
                logger.info("Successfully connected to Neo4j.")
            except Exception as e:
                logger.critical(f"CRITICAL: Failed to connect to Neo4j. Error: {e}")
                self._driver = None

    async def close(self):
        """Closes the connection to the Neo4j database."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j connection closed.")

    async def execute_query(self, query: str, parameters: dict = None) -> EagerResult:
        """A generic utility to execute a Cypher query."""
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized. Cannot execute query.")
        
        async with self._driver.session() as session:
            result = await session.run(query, parameters)
            return await result.consume() # consume() gets all results and summary

# --- Singleton Instance and Dependency ---
db_handler = Neo4jHandler(
    uri=settings.NEO4J_URI,
    user=settings.NEO4J_USER,
    password=settings.NEO4J_PASSWORD
)

# --- Graph Creation Functions ---
async def create_user_node_in_graph(email: str):
    """Creates a User node in Neo4j if it doesn't already exist."""
    query = """
    MERGE (u:User {email: $email})
    ON CREATE SET u.createdAt = timestamp()
    """
    await db_handler.execute_query(query, {"email": email})
    logger.info(f"Ensured User node exists for: {email}")

async def add_document_node_and_link_to_user(user_id: str, doc_id: str, filename: str):
    """
    Creates a Document node and links it to its owner (:User) with an :OWNS relationship.
    This is the core of user-wise data segregation in the graph.
    """
    query = """
    MATCH (u:User {email: $user_id})
    MERGE (d:Document {id: $doc_id})
    ON CREATE SET d.filename = $filename, d.createdAt = timestamp()
    MERGE (u)-[:OWNS]->(d)
    """
    params = {"user_id": user_id, "doc_id": doc_id, "filename": filename}
    await db_handler.execute_query(query, params)
    logger.info(f"Linked Document {doc_id} to User {user_id} in graph.")

async def add_entities_and_link_to_document(doc_id: str, entities: List[Dict[str, Any]]):
    """
    Batch-creates Entity nodes and links them to their parent Document.
    Uses UNWIND for efficient batch processing.
    """
    if not entities:
        return

    query = """
    MATCH (d:Document {id: $doc_id})
    UNWIND $entities AS entity_data
    MERGE (e:Entity {name: entity_data.name, type: entity_data.type})
    MERGE (d)-[:CONTAINS_ENTITY]->(e)
    """
    params = {"doc_id": doc_id, "entities": entities}
    await db_handler.execute_query(query, params)
    logger.info(f"Added {len(entities)} entities to Document {doc_id} in graph.")


async def add_relationships_to_graph(doc_id: str, relationships: List[Dict[str, Any]]):
    """
    Add extracted relationships between entities.
    Each relationship has: source, target, relationship_type
    """
    if not relationships:
        return
    
    query = """
    MATCH (d:Document {id: $doc_id})
    UNWIND $relationships AS rel
    MERGE (source:Entity {name: rel.source})
    MERGE (target:Entity {name: rel.target})
    MERGE (source)-[r:RELATES_TO {type: rel.relationship_type, doc_id: $doc_id}]->(target)
    """
    params = {"doc_id": doc_id, "relationships": relationships}
    await db_handler.execute_query(query, params)
    logger.info(f"Added {len(relationships)} relationships for Document {doc_id}")


async def get_graph_context_for_query(user_id: str, keywords: List[str], max_nodes: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieve relevant graph context for RAG based on keywords.
    Returns entities and their relationships from user's documents.
    """
    if not keywords:
        return []
    
    # Build regex pattern for keyword matching
    keyword_pattern = "|".join([f"(?i).*{k}.*" for k in keywords[:5]])  # Limit to 5 keywords
    
    query = """
    MATCH (u:User {email: $user_id})-[:OWNS]->(d:Document)-[:CONTAINS_ENTITY]->(e:Entity)
    WHERE e.name =~ $pattern
    OPTIONAL MATCH (e)-[r:RELATES_TO]-(related:Entity)
    RETURN DISTINCT 
        e.name AS entity,
        e.type AS entity_type,
        collect(DISTINCT {
            related: related.name,
            relationship: r.type
        })[..5] AS relationships,
        d.id AS doc_id
    LIMIT $max_nodes
    """
    
    try:
        async with db_handler._driver.session() as session:
            result = await session.run(query, {
                "user_id": user_id,
                "pattern": keyword_pattern,
                "max_nodes": max_nodes
            })
            records = await result.data()
            
            logger.info(f"Retrieved {len(records)} graph nodes for query context")
            return records
    except Exception as e:
        logger.error(f"Graph context retrieval failed: {e}")
        return []


async def get_entity_neighborhood(user_id: str, entity_name: str, depth: int = 2) -> List[Dict[str, Any]]:
    """
    Get the neighborhood of an entity for detailed exploration.
    Returns connected entities up to specified depth.
    """
    query = """
    MATCH (u:User {email: $user_id})-[:OWNS]->(d:Document)-[:CONTAINS_ENTITY]->(e:Entity {name: $entity_name})
    CALL apoc.neighbors.byhop(e, "RELATES_TO", $depth) YIELD nodes
    UNWIND nodes AS neighbor
    RETURN DISTINCT neighbor.name AS name, neighbor.type AS type
    LIMIT 50
    """
    
    try:
        async with db_handler._driver.session() as session:
            result = await session.run(query, {
                "user_id": user_id,
                "entity_name": entity_name,
                "depth": depth
            })
            records = await result.data()
            return records
    except Exception as e:
        logger.warning(f"Entity neighborhood query failed: {e}")
        return []