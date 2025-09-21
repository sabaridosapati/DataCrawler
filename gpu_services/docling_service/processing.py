# gpu_services/docling_service/processing.py

import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import assemblyai as aai
from transformers import AutoTokenizer

# Import the new MLX-based granite docling components
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import (
    PictureItem,
    PictureDescriptionData,
    PictureClassificationData,
    PictureMoleculeData,
)
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownPictureSerializer, MarkdownTableSerializer
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.vlm_model_specs import VlmModelSpecs
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from config import settings

logger = logging.getLogger(__name__)

class EnhancedPictureSerializer(MarkdownPictureSerializer):
    """Enhanced picture serializer that handles all types of image annotations"""
    
    def serialize(self, *, item: PictureItem, doc_serializer: BaseDocSerializer, doc: DoclingDocument, **kwargs: Any) -> SerializationResult:
        text_parts: List[str] = []
        
        # Process all annotations from the granite-docling MLX model
        for annotation in item.annotations:
            if isinstance(annotation, PictureDescriptionData):
                text_parts.append(f"Image Description: {annotation.text}")
            elif isinstance(annotation, PictureClassificationData):
                if annotation.predicted_classes:
                    predicted_class = annotation.predicted_classes[0].class_name
                    text_parts.append(f"Image Type: {predicted_class}")
            elif isinstance(annotation, PictureMoleculeData):
                text_parts.append(f"Chemical Structure (SMILES): {annotation.smi}")
        
        # If we have annotations, use them; otherwise fall back to parent behavior
        if text_parts:
            text_result = "\n".join(text_parts)
            text_result = doc_serializer.post_process(text=text_result)
            return create_ser_result(text=text_result, span_source=item)
        else:
            return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)

class AdvancedSerializerProvider(ChunkingSerializerProvider):
    """Advanced serializer provider with enhanced image and table handling"""
    
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=EnhancedPictureSerializer()
        )

class GraniteDoclingProcessor:
    """
    Enhanced document processor using the granite-docling-258M-mlx model.
    Optimized for Apple Silicon Mac with full document type support.
    """
    
    def __init__(self):
        logger.info("Initializing GraniteDoclingProcessor with granite-docling-258M-mlx for Apple Silicon")
        
        try:
            # Configure VLM pipeline to use the Granite-Docling MLX model
            # This model is specifically optimized for Apple Silicon
            vlm_pipeline_options = VlmPipelineOptions(
                vlm_options=VlmModelSpecs(
                    model="granite_docling_mlx",  # Use the MLX version
                    max_tokens=4096,
                    temperature=0.0,
                )
            )
            
            # Initialize document converter with granite docling MLX model
            self.converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=vlm_pipeline_options
                    )
                }
            )
            
            # Initialize AssemblyAI for audio processing (optional)
            if settings.ASSEMBLYAI_API_KEY:
                aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
                self.transcriber = aai.Transcriber()
                logger.info("AssemblyAI transcriber initialized")
            else:
                self.transcriber = None
                logger.warning("AssemblyAI API key not provided - audio processing disabled")
            
            # Initialize advanced chunking with contextual serializers
            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                max_tokens=512
            )
            
            self.chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,  # Merge related chunks for better context
                serializer_provider=AdvancedSerializerProvider()
            )
            
            logger.info("GraniteDoclingProcessor initialized successfully with granite-docling-258M-mlx")
            logger.info("Supported formats: PDF, DOCX, PPTX, HTML, MD, Images (PNG, JPG, JPEG, BMP), Audio (MP3, WAV, M4A, FLAC, OGG)")
            
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize GraniteDoclingProcessor. Error: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize document processor: {e}")

    async def process_file(self, input_path: str, output_dir: str) -> Dict[str, str]:
        """
        Process any supported file type and return paths to extracted content.
        Handles documents, images, and audio files.
        """
        file_path = Path(input_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_extension = file_path.suffix.lower()
        
        # Define supported formats
        document_formats = ['.pdf', '.docx', '.pptx', '.html', '.md', '.txt']
        image_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4']
        
        logger.info(f"Processing file: {input_path} (type: {file_extension})")
        
        if file_extension in document_formats or file_extension in image_formats:
            return await self._process_document(file_path, output_path)
        elif file_extension in audio_formats and self.transcriber:
            return await self._process_audio(file_path, output_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    async def _process_document(self, input_path: Path, output_dir: Path) -> Dict[str, str]:
        """
        Process documents and images using granite-docling-258M-mlx model.
        Extracts text, tables, images, and maintains document structure.
        """
        logger.info(f"Processing document/image with Granite-Docling MLX: {input_path}")
        
        try:
            # Convert document using granite-docling MLX
            conversion_result = self.converter.convert(source=str(input_path))
            docling_document = conversion_result.document
            
            logger.info(f"Successfully converted document: {input_path}")
            logger.info(f"Document contains {len(docling_document.texts)} text elements")
            logger.info(f"Document contains {len(docling_document.tables)} tables")
            logger.info(f"Document contains {len(docling_document.pictures)} images")
            
            # Generate base filename for outputs
            base_name = input_path.stem
            
            # 1. Save as markdown (preserves structure and formatting)
            markdown_path = output_dir / f"{base_name}.md"
            docling_document.save_as_markdown(markdown_path)
            logger.info(f"Saved markdown to: {markdown_path}")
            
            # 2. Create contextual chunks for embedding
            logger.info("Creating contextual chunks with advanced serialization...")
            chunks_data = []
            
            try:
                docling_chunks = self.chunker.chunk(dl_doc=docling_document)
                
                for i, chunk in enumerate(docling_chunks):
                    # Get contextualized text that preserves relationships
                    contextual_text = self.chunker.contextualize(chunk=chunk)
                    
                    # Extract metadata from the chunk
                    chunk_metadata = {
                        "chunk_index": i,
                        "source_file": str(input_path),
                        "chunk_type": "contextual",
                        "length": len(contextual_text),
                    }
                    
                    # Add chunk metadata if available
                    if hasattr(chunk, 'meta') and chunk.meta:
                        chunk_metadata.update(chunk.meta.model_dump())
                    
                    chunks_data.append({
                        "chunk_index": i,
                        "text": contextual_text,
                        "metadata": chunk_metadata
                    })
                
                logger.info(f"Created {len(chunks_data)} contextual chunks")
                
            except Exception as chunk_error:
                logger.warning(f"Advanced chunking failed, using basic chunking: {chunk_error}")
                # Fallback to basic text extraction
                full_text = docling_document.export_to_markdown()
                chunks_data = [{
                    "chunk_index": 0,
                    "text": full_text,
                    "metadata": {
                        "source_file": str(input_path),
                        "chunk_type": "full_document",
                        "length": len(full_text)
                    }
                }]
            
            # 3. Save chunks as JSON
            chunks_path = output_dir / f"{base_name}_chunks.json"
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks_data)} chunks to: {chunks_path}")
            
            return {
                "markdown_path": str(markdown_path),
                "chunks_path": str(chunks_path)
            }
            
        except Exception as e:
            logger.error(f"Document processing failed for {input_path}: {e}", exc_info=True)
            raise RuntimeError(f"Document processing failed: {e}")

    async def _process_audio(self, input_path: Path, output_dir: Path) -> Dict[str, str]:
        """
        Process audio files using AssemblyAI transcription.
        Fallback for audio content when document processing isn't applicable.
        """
        if not self.transcriber:
            raise RuntimeError("Audio processing not available - AssemblyAI API key not configured")
        
        logger.info(f"Processing audio with AssemblyAI: {input_path}")
        
        try:
            # Transcribe audio file
            transcript = self.transcriber.transcribe(str(input_path))
            
            if transcript.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
            
            # Get transcribed text
            full_text = transcript.text
            base_name = input_path.stem
            
            # Save as markdown
            markdown_path = output_dir / f"{base_name}.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(f"# Transcription of {input_path.name}\n\n{full_text}")
            
            # Create chunks for the transcription
            chunks_data = [{
                "chunk_index": 0,
                "text": full_text,
                "metadata": {
                    "source_file": str(input_path),
                    "source_type": "audio_transcript",
                    "length": len(full_text),
                    "duration": getattr(transcript, 'audio_duration', 'unknown')
                }
            }]
            
            # Save chunks
            chunks_path = output_dir / f"{base_name}_chunks.json"
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Audio transcription completed: {len(full_text)} characters")
            
            return {
                "markdown_path": str(markdown_path),
                "chunks_path": str(chunks_path)
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed for {input_path}: {e}", exc_info=True)
            raise RuntimeError(f"Audio processing failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model and processor"""
        return {
            "model_name": "granite-docling-258M-mlx",
            "model_type": "vision-language-model",
            "platform": "Apple Silicon (MLX)",
            "supported_formats": {
                "documents": ["pdf", "docx", "pptx", "html", "md", "txt"],
                "images": ["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
                "audio": ["mp3", "wav", "m4a", "flac", "ogg", "mp4"] if self.transcriber else []
            },
            "features": [
                "OCR and text extraction",
                "Table detection and parsing", 
                "Image analysis and description",
                "Document structure preservation",
                "Contextual chunking",
                "Multi-modal understanding"
            ]
        }

# Create global processor instance
granite_processor = GraniteDoclingProcessor()