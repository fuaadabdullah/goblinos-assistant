"""
Document Chunking
Split documents into optimal chunks for embedding
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class ChunkingStrategy(str, Enum):
    """Chunking strategies."""
    FIXED_SIZE = "fixed_size"           # Fixed character/token count
    SENTENCE = "sentence"                # Sentence-based
    PARAGRAPH = "paragraph"              # Paragraph-based
    SEMANTIC = "semantic"                # Semantic similarity (advanced)
    MARKDOWN = "markdown"                # Markdown structure-aware


@dataclass
class Chunk:
    """A document chunk."""
    content: str
    index: int
    metadata: dict[str, Any]
    
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())


class DocumentChunker:
    """
    Document chunker with multiple strategies.
    
    Best practices:
    - Use overlap to preserve context
    - Match chunk size to embedding model's optimal input
    - Consider document structure for better retrieval
    """
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[Chunk]:
        """
        Chunk text using configured strategy.
        
        Args:
            text: Input text
            metadata: Base metadata for all chunks
            
        Returns:
            List of chunks
        """
        base_metadata = metadata or {}
        
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text, base_metadata)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text, base_metadata)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text, base_metadata)
        elif self.strategy == ChunkingStrategy.MARKDOWN:
            return self._chunk_markdown(text, base_metadata)
        else:
            return self._chunk_fixed_size(text, base_metadata)
    
    def _chunk_fixed_size(
        self,
        text: str,
        base_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk by fixed character count with overlap."""
        chunks = []
        start = 0
        index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look for last space in chunk
                last_space = text.rfind(" ", start, end)
                if last_space > start + self.min_chunk_size:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    index=index,
                    metadata={
                        **base_metadata,
                        "chunk_strategy": "fixed_size",
                        "char_start": start,
                        "char_end": end,
                    }
                ))
                index += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text) - self.min_chunk_size:
                break
        
        return chunks
    
    def _chunk_by_sentence(
        self,
        text: str,
        base_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk by sentences, grouping to target size."""
        # Simple sentence splitting (can use spacy for better results)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        content=chunk_text,
                        index=index,
                        metadata={
                            **base_metadata,
                            "chunk_strategy": "sentence",
                            "sentence_count": len(current_chunk),
                        }
                    ))
                    index += 1
                
                # Start new chunk with overlap (last sentence)
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_length = len(current_chunk[0]) if current_chunk else 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    index=index,
                    metadata={
                        **base_metadata,
                        "chunk_strategy": "sentence",
                        "sentence_count": len(current_chunk),
                    }
                ))
        
        return chunks
    
    def _chunk_by_paragraph(
        self,
        text: str,
        base_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk by paragraphs, merging small ones."""
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            if current_length + para_length > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        content=chunk_text,
                        index=index,
                        metadata={
                            **base_metadata,
                            "chunk_strategy": "paragraph",
                            "paragraph_count": len(current_chunk),
                        }
                    ))
                    index += 1
                
                current_chunk = []
                current_length = 0
            
            current_chunk.append(para)
            current_length += para_length + 2
        
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    index=index,
                    metadata={
                        **base_metadata,
                        "chunk_strategy": "paragraph",
                        "paragraph_count": len(current_chunk),
                    }
                ))
        
        return chunks
    
    def _chunk_markdown(
        self,
        text: str,
        base_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk markdown respecting headers and structure."""
        chunks = []
        index = 0
        
        # Split by headers (##, ###, etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        
        current_chunk = []
        current_header = None
        current_length = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous chunk if exists
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            content=chunk_text,
                            index=index,
                            metadata={
                                **base_metadata,
                                "chunk_strategy": "markdown",
                                "header": current_header,
                            }
                        ))
                        index += 1
                
                current_header = header_match.group(2)
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line) + 1
                
                # Check if we need to split
                if current_length > self.chunk_size:
                    chunk_text = '\n'.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            content=chunk_text,
                            index=index,
                            metadata={
                                **base_metadata,
                                "chunk_strategy": "markdown",
                                "header": current_header,
                            }
                        ))
                        index += 1
                    
                    current_chunk = []
                    current_length = 0
        
        # Last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    index=index,
                    metadata={
                        **base_metadata,
                        "chunk_strategy": "markdown",
                        "header": current_header,
                    }
                ))
        
        return chunks


def create_chunker(
    strategy: str = "fixed_size",
    **kwargs,
) -> DocumentChunker:
    """Factory function to create a chunker."""
    try:
        selected_strategy = ChunkingStrategy(strategy)
    except ValueError:
        selected_strategy = ChunkingStrategy.FIXED_SIZE
    
    return DocumentChunker(strategy=selected_strategy, **kwargs)
