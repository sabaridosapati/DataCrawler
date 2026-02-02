import logging
import json
from app.db.mongo_handler import get_document_by_id, update_document_status
from app.db.milvus_handler import milvus_db_handler
from app.services.gpu_node_client import gpu_node_client
from app.models.document import DocumentStatus

logger = logging.getLogger(__name__)

async def start_document_processing(doc_id: str):
    """
    The final, complete orchestration pipeline. This is the "conductor" that
    calls all GPU services in sequence and stores the results in the databases.
    """
    doc = None
    try:
        logger.info(f"PIPELINE STARTED for document_id: {doc_id}")
        doc = await get_document_by_id(doc_id)
        if not doc:
            logger.error(f"Document {doc_id} not found in DB. Aborting pipeline.")
            return

        # STEP 1: Update status to PROCESSING
        await update_document_status(doc_id, DocumentStatus.PROCESSING)

        # STEP 2: Call Docling Service for content extraction and chunking.
        # The docling service reads the raw file and writes the output files.
        docling_results = await gpu_node_client.process_with_docling(doc.original_file_path)
        chunks_path = docling_results["chunks_path"]
        
        # Load the resulting chunks into memory for the next steps.
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        if not chunks:
            raise RuntimeError("Docling service returned no chunks. Cannot proceed.")

        # STEP 3: Call Knowledge Graph Service.
        # This service uses the local Gemma 3n model to extract entities/relations
        # and writes them directly to Neo4j.
        await gpu_node_client.build_knowledge_graph(
            doc_id=doc.id,
            user_id=doc.user_id,
            chunks=chunks
        )

        # STEP 4: Call Embedding Service.
        # This service uses the local EmbeddingGemma model to vectorize the text chunks.
        texts_to_embed = [chunk['text'] for chunk in chunks]
        embeddings = await gpu_node_client.embed_documents(texts_to_embed)

        # STEP 5: Store vectors in Milvus.
        # We enrich the vectors with metadata for user-specific filtering.
        data_to_insert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data_to_insert.append({
                "user_id": doc.user_id,
                "doc_id": doc.id,
                "chunk_index": i,
                "chunk_text": chunk['text'],
                "embedding": embedding
            })
        
        milvus_db_handler.insert_chunks(data_to_insert)
        
        # Invalidate BM25 cache for this user (new content added)
        from app.services.hybrid_retriever import hybrid_retriever
        hybrid_retriever.invalidate_bm25_cache(doc.user_id)

        # STEP 6: If all steps succeed, mark the document as COMPLETED.
        await update_document_status(doc_id, DocumentStatus.COMPLETED)
        logger.info(f"PIPELINE COMPLETED successfully for document_id: {doc_id}")

    except Exception as e:
        error_message = f"Pipeline failed: {e}"
        logger.error(f"PIPELINE FAILED for document_id: {doc_id}. Error: {error_message}", exc_info=True)
        if doc:
            # If any step fails, mark the document as FAILED and record the error.
            await update_document_status(doc_id, DocumentStatus.FAILED, error_message=error_message)