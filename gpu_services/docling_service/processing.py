# gpu_services/docling_service/processing.py

import logging
import json
from pathlib import Path
from typing import Dict, Any

import assemblyai as aai
from transformers import AutoTokenizer

# --- IMPORTS FOR ADVANCED CHUNKING ---
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import (
    PictureClassificationData,
    PictureDescriptionData,
    PictureItem,
    PictureMoleculeData,
)
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownPictureSerializer, MarkdownTableSerializer
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider
from typing_extensions import override

# --- STANDARD DOCLING IMPORTS ---
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from config import settings

logger = logging.getLogger(__name__)


# --- 1. DEFINE ADVANCED SERIALIZERS AS PER YOUR DOCUMENTATION ---

class AnnotationPictureSerializer(MarkdownPictureSerializer):
    """
    Custom picture serialization strategy that leverages picture annotations
    to create rich, descriptive text instead of a simple placeholder.
    """
    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        text_parts: list[str] = []
        for annotation in item.annotations:
            if isinstance(annotation, PictureClassificationData):
                predicted_class = (
                    annotation.predicted_classes[0].class_name
                    if annotation.predicted_classes
                    else None
                )
                if predicted_class is not None:
                    text_parts.append(f"Picture type: {predicted_class}")
            elif isinstance(annotation, PictureMoleculeData):
                text_parts.append(f"SMILES: {annotation.smi}")
            elif isinstance(annotation, PictureDescriptionData):
                text_parts.append(f"Picture description: {annotation.text}")

        if not text_parts:
            return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)

        text_res = "\n".join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)
        return create_ser_result(text=text_res, span_source=item)


# --- 2. CREATE A COMBINED PROVIDER FOR ALL ADVANCED SERIALIZERS ---

class AdvancedSerializerProvider(ChunkingSerializerProvider):
    """
    A single, powerful provider that configures the chunker to use our
    desired state-of-the-art serializers for different document elements.
    """
    def get_serializer(self, doc: DoclingDocument):
        return ChunkingDocSerializer(
            doc=doc,
            # Use the Markdown serializer for tables for better context
            table_serializer=MarkdownTableSerializer(),
            # Use our custom annotation-based serializer for pictures
            picture_serializer=AnnotationPictureSerializer(),
        )


class DoclingProcessor:
    """
    A singleton class to handle all document and audio processing tasks.
    - Uses Granite-Docling for documents with advanced chunking.
    - Uses AssemblyAI for audio.
    """
    def __init__(self):
        logger.info(f"Initializing DoclingProcessor on device: {settings.DEVICE}")

        # --- 3. Configure the VLM Pipeline for Documents ---
        vlm_pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
        )
        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=vlm_pipeline_options,
                ),
            }
        )

        # --- 4. Configure the AssemblyAI client for Audio ---
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        self.transcriber = aai.Transcriber()

        # --- 5. CONFIGURE THE HYBRID CHUNKER WITH THE ADVANCED PROVIDER ---
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            max_tokens=512,
        )
        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
            # This is the crucial step to enable advanced serialization
            serializer_provider=AdvancedSerializerProvider(),
        )

        logger.info("DoclingProcessor initialized successfully with Granite-Docling, AssemblyAI, and ADVANCED CHUNKING.")

    async def process_file(self, input_path: str, output_dir: str) -> Dict[str, str]:
        """
        Main dispatcher function. Determines file type and routes to the correct processor.
        """
        file_path = Path(input_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found at: {input_path}")

        ext = file_path.suffix.lower()
        
        doc_formats = ['.pdf', '.docx', '.pptx', '.html', '.md', '.png', '.jpg', '.jpeg', '.bmp']
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']

        if ext in doc_formats:
            return self._process_document(file_path, Path(output_dir))
        elif ext in audio_formats:
            return await self._process_audio(file_path, Path(output_dir))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _process_document(self, input_path: Path, output_dir: Path) -> Dict[str, str]:
        """Handles PDFs, DOCX, images, etc., using the Docling VLM pipeline."""
        logger.info(f"Processing document with Granite-Docling VLM: {input_path}")
        
        result = self.converter.convert(source=str(input_path))
        doc = result.document
        
        logger.info("Chunking extracted content with advanced serializers...")
        chunks_data = []
        docling_chunks = self.chunker.chunk(dl_doc=doc)
        
        for i, chunk in enumerate(docling_chunks):
            # 'contextualize' now produces the rich, serialized text
            contextual_text = self.chunker.contextualize(chunk=chunk)
            chunks_data.append({
                "chunk_index": i,
                "text": contextual_text,
                "metadata": chunk.meta.model_dump()
            })
        
        base_name = input_path.stem
        
        md_output_path = output_dir / f"{base_name}.md"
        doc.save_as_markdown(md_output_path)
        logger.info(f"Saved full Markdown output to: {md_output_path}")

        chunks_output_path = output_dir / f"{base_name}_chunks.json"
        with open(chunks_output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2)
        logger.info(f"Saved {len(chunks_data)} context-rich chunks to: {chunks_output_path}")

        return {
            "markdown_path": str(md_output_path),
            "chunks_path": str(chunks_output_path)
        }

    async def _process_audio(self, input_path: Path, output_dir: Path) -> Dict[str, str]:
        """Handles audio files using the AssemblyAI API for transcription."""
        logger.info(f"Processing audio with AssemblyAI: {input_path}")
        
        transcript = self.transcriber.transcribe(str(input_path))

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

        full_text = transcript.text
        
        base_name = input_path.stem
        
        md_output_path = output_dir / f"{base_name}.md"
        with open(md_output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Transcription of {input_path.name}\n\n")
            f.write(full_text)
        logger.info(f"Saved full transcript to: {md_output_path}")

        chunks_data = [{
            "chunk_index": 0,
            "text": full_text,
            "metadata": {"source_type": "audio_transcript"}
        }]
        chunks_output_path = output_dir / f"{base_name}_chunks.json"
        with open(chunks_output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2)
        logger.info(f"Saved transcript as a single chunk to: {chunks_output_path}")

        return {
            "markdown_path": str(md_output_path),
            "chunks_path": str(chunks_output_path)
        }

# Create a single instance to be loaded at startup
docling_processor = DoclingProcessor()