# gpu_services/docling_service/processing.py

"""
Document processing service using Docling.
Updated for Docling 2.x API.
"""

import logging
import json
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except ImportError:
    ASSEMBLYAI_AVAILABLE = False

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

from config import settings

logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clear GPU memory cache to prevent OOM errors"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("GPU memory cache cleared")


class DoclingProcessor:
    """
    Document processor using Docling.
    Handles PDF, DOCX, and other document formats.
    """
    
    def __init__(self):
        logger.info(f"Initializing DoclingProcessor")
        
        # Check GPU availability
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU available: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            self.device = "cpu"
            logger.info("Running on CPU")
        
        # Initialize converter with default settings
        self.converter = DocumentConverter()
        
        # Initialize chunker
        self.chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=settings.CHUNK_SIZE,
            merge_peers=True
        )
        
        logger.info("DoclingProcessor initialized successfully")

    def process_document(
        self, 
        input_file: str, 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Process a document and extract content.
        
        Args:
            input_file: Path to input document
            output_dir: Directory for output files
            
        Returns:
            Dict with paths to extracted content
        """
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing document: {input_path.name}")
        
        try:
            # Clear GPU memory before processing
            clear_gpu_memory()
            
            # Convert document
            result = self.converter.convert(str(input_path))
            
            # Get markdown content
            markdown_content = result.document.export_to_markdown()
            
            # Save markdown
            markdown_file = output_path / f"{input_path.stem}.md"
            markdown_file.write_text(markdown_content, encoding="utf-8")
            logger.info(f"Saved markdown to: {markdown_file}")
            
            # Create chunks
            chunks = self._create_chunks(result.document)
            
            # Save chunks as JSON
            chunks_file = output_path / f"{input_path.stem}_chunks.json"
            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(chunks)} chunks to: {chunks_file}")
            
            # Clear GPU memory after processing
            clear_gpu_memory()
            
            return {
                "success": True,
                "extracted_markdown_path": str(markdown_file),
                "extracted_chunks_path": str(chunks_file),
                "num_chunks": len(chunks),
                "document_name": input_path.name
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            clear_gpu_memory()
            raise

    def _create_chunks(self, document) -> List[Dict[str, Any]]:
        """Create semantic chunks from document."""
        chunks = []
        
        try:
            chunk_iter = self.chunker.chunk(document)
            for i, chunk in enumerate(chunk_iter):
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                
                # Clean up the text
                chunk_text = self._clean_text(chunk_text)
                
                if chunk_text.strip():
                    chunks.append({
                        "index": i,
                        "text": chunk_text,
                        "metadata": {
                            "chunk_index": i
                        }
                    })
        except Exception as e:
            logger.warning(f"Chunking failed, using fallback: {e}")
            # Fallback: simple text splitting
            markdown = document.export_to_markdown() if hasattr(document, 'export_to_markdown') else str(document)
            chunks = self._simple_chunk(markdown)
        
        return chunks

    def _simple_chunk(self, text: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """Simple fallback chunking by paragraphs."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        "index": chunk_index,
                        "text": self._clean_text(current_chunk),
                        "metadata": {"chunk_index": chunk_index}
                    })
                    chunk_index += 1
                current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append({
                "index": chunk_index,
                "text": self._clean_text(current_chunk),
                "metadata": {"chunk_index": chunk_index}
            })
        
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean up extracted text."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove docling artifacts
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links artifacts
        
        return text.strip()

    def get_info(self) -> Dict[str, Any]:
        """Get processor information."""
        info = {
            "processor": "DoclingProcessor",
            "device": self.device,
            "chunk_size": settings.CHUNK_SIZE,
            "supported_formats": ["pdf", "docx", "pptx", "xlsx", "html", "md", "txt"]
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
        
        return info


class AudioProcessor:
    """Process audio files using AssemblyAI."""
    
    def __init__(self, api_key: str = None):
        if not ASSEMBLYAI_AVAILABLE:
            logger.warning("AssemblyAI not available")
            self.available = False
            return
            
        api_key = api_key or settings.ASSEMBLYAI_API_KEY
        if not api_key:
            logger.warning("No AssemblyAI API key provided")
            self.available = False
            return
            
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        self.available = True
        logger.info("AudioProcessor initialized with AssemblyAI")

    def transcribe(self, audio_file: str, output_dir: str) -> Dict[str, Any]:
        """Transcribe audio file to text."""
        if not self.available:
            raise RuntimeError("AssemblyAI not available")
        
        input_path = Path(audio_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Transcribing audio: {input_path.name}")
        
        # Transcribe
        transcript = self.transcriber.transcribe(str(input_path))
        
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")
        
        # Save transcript
        text_file = output_path / f"{input_path.stem}.txt"
        text_file.write_text(transcript.text, encoding="utf-8")
        
        # Create chunks from transcript
        chunks = []
        for i, utterance in enumerate(transcript.utterances or []):
            chunks.append({
                "index": i,
                "text": utterance.text,
                "metadata": {
                    "speaker": utterance.speaker,
                    "start": utterance.start,
                    "end": utterance.end
                }
            })
        
        # If no utterances, chunk the full text
        if not chunks:
            processor = DoclingProcessor()
            chunks = processor._simple_chunk(transcript.text)
        
        chunks_file = output_path / f"{input_path.stem}_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "extracted_text_path": str(text_file),
            "extracted_chunks_path": str(chunks_file),
            "num_chunks": len(chunks)
        }


# Create singleton instances
docling_processor = DoclingProcessor()

try:
    audio_processor = AudioProcessor()
except Exception as e:
    logger.warning(f"AudioProcessor initialization failed: {e}")
    audio_processor = None


# Backward compatibility - alias for granite_processor
granite_processor = docling_processor


class GraniteDoclingProcessor:
    """Backward compatibility wrapper."""
    
    def __init__(self):
        self.processor = docling_processor
    
    def process_document(self, *args, **kwargs):
        return self.processor.process_document(*args, **kwargs)
    
    def get_info(self):
        return self.processor.get_info()