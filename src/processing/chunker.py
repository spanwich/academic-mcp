"""
Page-based chunking for document retrieval.

v3.2: 1 page = 1 chunk. Simple, predictable, PDF-aligned.
"""

from dataclasses import dataclass
from typing import Optional

from .pdf_processor import ExtractedDocument
from .section_detector import DetectedSection


@dataclass
class DocumentChunk:
    """A page-based chunk of document content."""
    chunk_id: str
    paper_id: str
    page_number: int
    
    # Content
    content: str
    word_count: int
    
    # Position in full_text
    char_start: int
    char_end: int
    
    # Section assignment (may be None)
    section_id: Optional[str] = None
    section_type: Optional[str] = None


class PageChunker:
    """
    Create page-based chunks from documents.
    
    Each page becomes one chunk for simple, predictable retrieval.
    """
    
    def chunk_document(
        self,
        doc: ExtractedDocument,
        paper_id: str,
        sections: Optional[list[DetectedSection]] = None
    ) -> list[DocumentChunk]:
        """
        Create chunks from document pages.
        
        Args:
            doc: Extracted document with pages
            paper_id: Paper identifier
            sections: Optional detected sections for assignment
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        for page in doc.pages:
            # Determine section for this page
            section_id = None
            section_type = None
            
            if sections:
                for i, section in enumerate(sections):
                    if section.page_start <= page.page_number <= section.page_end:
                        section_id = f"{paper_id}_sec_{i}"
                        section_type = section.section_type
                        break
            
            chunk = DocumentChunk(
                chunk_id=f"{paper_id}_page_{page.page_number}",
                paper_id=paper_id,
                page_number=page.page_number,
                content=page.text,
                word_count=page.word_count,
                char_start=page.char_start,
                char_end=page.char_end,
                section_id=section_id,
                section_type=section_type
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_chunk_for_page(
        self, 
        chunks: list[DocumentChunk], 
        page_number: int
    ) -> Optional[DocumentChunk]:
        """Get chunk for specific page."""
        for chunk in chunks:
            if chunk.page_number == page_number:
                return chunk
        return None
    
    def get_chunks_for_section(
        self, 
        chunks: list[DocumentChunk], 
        section_type: str
    ) -> list[DocumentChunk]:
        """Get all chunks for a section type."""
        return [c for c in chunks if c.section_type == section_type]
    
    def get_chunks_in_range(
        self, 
        chunks: list[DocumentChunk], 
        start_page: int, 
        end_page: int
    ) -> list[DocumentChunk]:
        """Get chunks for page range."""
        return [c for c in chunks if start_page <= c.page_number <= end_page]


# Convenience function
def chunk_by_pages(
    doc: ExtractedDocument,
    paper_id: str,
    sections: Optional[list[DetectedSection]] = None
) -> list[DocumentChunk]:
    """Create page-based chunks from document."""
    chunker = PageChunker()
    return chunker.chunk_document(doc, paper_id, sections)
