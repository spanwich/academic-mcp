"""
Semantic chunking for academic papers.
Preserves meaning across chunk boundaries.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    """A text chunk with metadata."""
    content: str
    section_type: str
    section_title: Optional[str]
    chunk_index: int
    page_numbers: Optional[str]
    token_count: int
    has_equations: bool = False
    has_tables: bool = False


class SemanticChunker:
    """
    Create semantic chunks from extracted documents.
    Respects paragraph and sentence boundaries.
    """
    
    def __init__(
        self,
        target_chunk_size: int = 800,
        max_chunk_size: int = 1200,
        min_chunk_size: int = 200,
        overlap_sentences: int = 2
    ):
        """
        Initialize chunker.
        
        Args:
            target_chunk_size: Target tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
            overlap_sentences: Sentences to overlap between chunks
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences
    
    def chunk_document(self, extracted) -> list[Chunk]:
        """
        Create semantic chunks from extracted document.
        
        Args:
            extracted: ExtractedDocument from PDFProcessor
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        global_index = 0
        
        for section in extracted.sections:
            section_chunks = self._chunk_section(
                content=section["content"],
                section_type=section.get("type", "other"),
                section_title=section.get("title"),
                page_start=section.get("page_start", 1),
                start_index=global_index
            )
            
            chunks.extend(section_chunks)
            global_index += len(section_chunks)
        
        return chunks
    
    def _chunk_section(
        self,
        content: str,
        section_type: str,
        section_title: Optional[str],
        page_start: int,
        start_index: int
    ) -> list[Chunk]:
        """Chunk a single section."""
        
        paragraphs = self._split_paragraphs(content)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = start_index
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            # Handle oversized paragraphs
            if para_tokens > self.max_chunk_size:
                # Flush current chunk first
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        current_chunk.strip(),
                        section_type,
                        section_title,
                        chunk_index,
                        page_start
                    ))
                    chunk_index += 1
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentence_chunks = self._split_by_sentences(para)
                for sc in sentence_chunks:
                    chunks.append(self._create_chunk(
                        sc.strip(),
                        section_type,
                        section_title,
                        chunk_index,
                        page_start
                    ))
                    chunk_index += 1
                continue
            
            # Check if adding paragraph exceeds target
            if current_tokens + para_tokens > self.target_chunk_size:
                if current_tokens >= self.min_chunk_size:
                    # Save current chunk
                    chunks.append(self._create_chunk(
                        current_chunk.strip(),
                        section_type,
                        section_title,
                        chunk_index,
                        page_start
                    ))
                    chunk_index += 1
                    
                    # Add overlap
                    overlap = self._get_overlap_text(current_chunk)
                    current_chunk = overlap + "\n\n" + para if overlap else para
                    current_tokens = self._estimate_tokens(current_chunk)
                else:
                    # Chunk too small, keep accumulating
                    current_chunk = current_chunk + "\n\n" + para if current_chunk else para
                    current_tokens += para_tokens
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
        
        # Final chunk
        if current_chunk.strip() and current_tokens >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                current_chunk.strip(),
                section_type,
                section_title,
                chunk_index,
                page_start
            ))
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text into sentence-based chunks."""
        # Sentence boundary detection
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        chunks = []
        current = ""
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)
            
            if current_tokens + sent_tokens > self.target_chunk_size:
                if current.strip():
                    chunks.append(current.strip())
                current = sentence
                current_tokens = sent_tokens
            else:
                current = (current + " " + sentence).strip()
                current_tokens += sent_tokens
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get last N sentences for overlap."""
        if not self.overlap_sentences:
            return ""
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        overlap_sentences = sentences[-self.overlap_sentences:]
        return " ".join(overlap_sentences)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~4 chars per token for English)."""
        return len(text) // 4
    
    def _create_chunk(
        self,
        content: str,
        section_type: str,
        section_title: Optional[str],
        chunk_index: int,
        page_start: int
    ) -> Chunk:
        """Create a Chunk object."""
        return Chunk(
            content=content,
            section_type=section_type,
            section_title=section_title,
            chunk_index=chunk_index,
            page_numbers=str(page_start),
            token_count=self._estimate_tokens(content),
            has_equations=bool(re.search(r'\$.*\$|\\begin\{equation\}', content)),
            has_tables=bool(re.search(r'Table \d|\\begin\{table\}', content))
        )
