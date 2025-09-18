# gpu_services/docling_service/processing.py

import logging
import json
from pathlib import Path
from typing import Dict, List, Any

import assemblyai as aai
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from config import settings

logger = logging.getLogger(__name__)

class DoclingProcessor:
    """
    A singleton class to handle all document processing tasks.
    It initializes heavy models once to be reused across API calls.
    """
    def __init__(self):
        logger.info(f"Initializing DoclingProcessor on device: {settings.DEVICE}")
        
        # 1. Configure the VLM Pipeline with Granite-Docling
        # This uses the powerful vision-language model for end-to-end conversion.
        # It's excellent for complex layouts, tables, and embedded images.
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
        )
        
        # 2. Configure the DocumentConverter
        # This is the main entry point into the docling library.
        self.converter = DocumentConverter(
            format_options={
                # We specify that for PDFs, we want to use the VLM pipeline.
                # Docling will automatically handle other formats like DOCX, images, etc.
                "pdf": PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        
        # 3. Configure the Hybrid Chunker for intelligent, overlapping chunks
        # We use the tokenizer from a popular embedding model to ensure chunk sizes
        # are optimized for the subsequent embedding step.
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            max_tokens=512, # Max tokens for the embedding model
        )
        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True, # Merges small, adjacent chunks for better context
        )
        
        # 4. Configure AssemblyAI client
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        self.transcriber = aai.Transcriber()

        logger.info("DoclingProcessor initialized successfully.")

    async def process_file(self, input_path: str, output_dir: str) -> Dict[str, str]:
        """
        Main dispatcher function. Determines file type and routes to the correct processor.
        """
        file_path = Path(input_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found at: {input_path}")

        ext = file_path.suffix.lower()
        
        # Supported document/image formats for Docling
        doc_formats = ['.pdf', '.docx', '.pptx', '.html', '.md', '.png', '.jpg', '.jpeg', '.bmp']
        # Supported audio formats for AssemblyAI
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']

        if ext in doc_formats:
            return self._process_document(file_path, Path(output_dir))
        elif ext in audio_formats:
            return await self._process_audio(file_path, Path(output_dir))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _process_document(self, input_path: Path, output_dir: Path) -> Dict[str, str]:
        """Handles PDFs, DOCX, images, etc., using the Docling VLM pipeline."""
        logger.info(f"Processing document with Docling VLM: {input_path}")
        
        # Convert the source file into a structured DoclingDocument
        result = self.converter.convert(source=str(input_path))
        doc = result.document
        
        # --- Chunking ---
        logger.info("Chunking extracted content...")
        chunks_data = []
        docling_chunks = self.chunker.chunk(dl_doc=doc)
        
        for i, chunk in enumerate(docling_chunks):
            # 'contextualize' adds headers/titles to the chunk for better embedding context
            contextual_text = self.chunker.contextualize(chunk=chunk)
            chunks_data.append({
                "chunk_index": i,
                "text": contextual_text,
                "metadata": chunk.meta.model_dump() # Store rich metadata
            })
        
        # --- Serialization (Saving Outputs) ---
        base_name = input_path.stem
        
        # 1. Save the full, extracted content as a rich Markdown file
        md_output_path = output_dir / f"{base_name}.md"
        doc.save_as_markdown(md_output_path)
        logger.info(f"Saved full Markdown output to: {md_output_path}")

        # 2. Save the structured chunks as a JSON file for the next pipeline step
        chunks_output_path = output_dir / f"{base_name}_chunks.json"
        with open(chunks_output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2)
        logger.info(f"Saved {len(chunks_data)} chunks to: {chunks_output_path}")

        return {
            "markdown_path": str(md_output_path),
            "chunks_path": str(chunks_output_path)
        }

    async def _process_audio(self, input_path: Path, output_dir: Path) -> Dict[str, str]:
        """Handles audio files using AssemblyAI for transcription."""
        logger.info(f"Processing audio with AssemblyAI: {input_path}")
        
        transcript = self.transcriber.transcribe(str(input_path))

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

        full_text = transcript.text
        
        # --- Serialization (Saving Outputs) ---
        base_name = input_path.stem
        
        # 1. Save the full transcript as a simple Markdown file
        md_output_path = output_dir / f"{base_name}.md"
        with open(md_output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Transcription of {input_path.name}\n\n")
            f.write(full_text)
        logger.info(f"Saved full transcript to: {md_output_path}")

        # 2. Create a single chunk containing the full text
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