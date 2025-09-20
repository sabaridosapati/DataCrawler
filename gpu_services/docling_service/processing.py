# gpu_services/docling_service/processing.py

import logging
import json
from pathlib import Path
from typing import Dict, Any

import assemblyai as aai
from transformers import AutoTokenizer

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

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from config import settings

logger = logging.getLogger(__name__)

class AnnotationPictureSerializer(MarkdownPictureSerializer):
    @override
    def serialize(self, *, item: PictureItem, doc_serializer: BaseDocSerializer, doc: DoclingDocument, **kwargs: Any) -> SerializationResult:
        text_parts: list[str] = []
        for annotation in item.annotations:
            if isinstance(annotation, PictureClassificationData):
                predicted_class = (annotation.predicted_classes[0].class_name if annotation.predicted_classes else None)
                if predicted_class is not None: text_parts.append(f"Picture type: {predicted_class}")
            elif isinstance(annotation, PictureMoleculeData): text_parts.append(f"SMILES: {annotation.smi}")
            elif isinstance(annotation, PictureDescriptionData): text_parts.append(f"Picture description: {annotation.text}")
        if not text_parts: return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
        text_res = "\n".join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)
        return create_ser_result(text=text_res, span_source=item)

class AdvancedSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument):
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer(), picture_serializer=AnnotationPictureSerializer())


class DoclingProcessor:
    def __init__(self):
        # The DEVICE setting is no longer needed, as MLX handles it automatically.
        logger.info("Initializing DoclingProcessor for Apple Silicon (MLX)")

        # --- THIS IS THE CRITICAL CHANGE ---
        # Configure the VLM Pipeline to use the MLX-optimized Granite-Docling model.
        vlm_pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
        )
        self.converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=vlm_pipeline_options)}
        )

        # --- (The rest of the __init__ method is unchanged) ---
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        self.transcriber = aai.Transcriber()
        tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"), max_tokens=512)
        self.chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True, serializer_provider=AdvancedSerializerProvider())
        logger.info("DoclingProcessor initialized successfully with GRANITEDOCLING_MLX, AssemblyAI, and ADVANCED CHUNKING.")

    # --- (The process_file, _process_document, and _process_audio methods are completely unchanged) ---
    async def process_file(self, input_path: str, output_dir: str) -> Dict[str, str]:
        file_path = Path(input_path)
        if not file_path.exists(): raise FileNotFoundError(f"Input file not found at: {input_path}")
        ext = file_path.suffix.lower()
        doc_formats = ['.pdf', '.docx', '.pptx', '.html', '.md', '.png', '.jpg', '.jpeg', '.bmp']
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        if ext in doc_formats: return self._process_document(file_path, Path(output_dir))
        elif ext in audio_formats: return await self._process_audio(file_path, Path(output_dir))
        else: raise ValueError(f"Unsupported file type: {ext}")

    def _process_document(self, input_path: Path, output_dir: Path) -> Dict[str, str]:
        logger.info(f"Processing document with Granite-Docling MLX: {input_path}")
        result = self.converter.convert(source=str(input_path))
        doc = result.document
        logger.info("Chunking extracted content with advanced serializers...")
        chunks_data = []
        docling_chunks = self.chunker.chunk(dl_doc=doc)
        for i, chunk in enumerate(docling_chunks):
            contextual_text = self.chunker.contextualize(chunk=chunk)
            chunks_data.append({"chunk_index": i, "text": contextual_text, "metadata": chunk.meta.model_dump()})
        base_name = input_path.stem
        md_output_path = output_dir / f"{base_name}.md"
        doc.save_as_markdown(md_output_path)
        chunks_output_path = output_dir / f"{base_name}_chunks.json"
        with open(chunks_output_path, 'w', encoding='utf-8') as f: json.dump(chunks_data, f, indent=2)
        logger.info(f"Saved {len(chunks_data)} context-rich chunks to: {chunks_output_path}")
        return {"markdown_path": str(md_output_path), "chunks_path": str(chunks_output_path)}

    async def _process_audio(self, input_path: Path, output_dir: Path) -> Dict[str, str]:
        # ... (implementation is unchanged) ...
        logger.info(f"Processing audio with AssemblyAI: {input_path}")
        transcript = self.transcriber.transcribe(str(input_path))
        if transcript.status == aai.TranscriptStatus.error: raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
        full_text = transcript.text
        base_name = input_path.stem
        md_output_path = output_dir / f"{base_name}.md"
        with open(md_output_path, 'w', encoding='utf-8') as f: f.write(f"# Transcription of {input_path.name}\n\n{full_text}")
        chunks_data = [{"chunk_index": 0, "text": full_text, "metadata": {"source_type": "audio_transcript"}}]
        chunks_output_path = output_dir / f"{base_name}_chunks.json"
        with open(chunks_output_path, 'w', encoding='utf-8') as f: json.dump(chunks_data, f, indent=2)
        return {"markdown_path": str(md_output_path), "chunks_path": str(chunks_output_path)}

docling_processor = DoclingProcessor()