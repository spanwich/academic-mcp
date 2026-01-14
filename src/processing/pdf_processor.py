"""
PDF text extraction with page boundaries.

v3.2: Extracts text per page for page-based chunking.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


@dataclass
class PageText:
    """Text content from a single page."""
    page_number: int  # 1-indexed
    text: str
    char_start: int   # Position in full_text
    char_end: int
    word_count: int


@dataclass
class ExtractedDocument:
    """Complete extracted document with page boundaries."""
    file_path: str
    full_text: str
    pages: list[PageText]
    page_count: int
    word_count: int
    
    def get_page(self, page_number: int) -> Optional[PageText]:
        """Get specific page (1-indexed)."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None
    
    def get_pages(self, start: int, end: int) -> list[PageText]:
        """Get page range (1-indexed, inclusive)."""
        return [p for p in self.pages if start <= p.page_number <= end]
    
    def get_text_range(self, start_page: int, end_page: int) -> str:
        """Get combined text for page range."""
        pages = self.get_pages(start_page, end_page)
        return "\n\n".join(p.text for p in pages)


class PDFProcessor:
    """
    Extract text from PDFs with page boundaries.
    
    Uses PyMuPDF (fitz) for reliable text extraction.
    """
    
    def __init__(self):
        pass
    
    def extract_with_pages(self, pdf_path: str) -> ExtractedDocument:
        """
        Extract text from PDF with page boundaries.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractedDocument with full_text and per-page content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        
        pages: list[PageText] = []
        full_text_parts: list[str] = []
        current_char_pos = 0
        total_words = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text("text")
            text = self._clean_text(text)
            
            # Calculate positions
            char_start = current_char_pos
            char_end = current_char_pos + len(text)
            word_count = len(text.split())
            
            # Create page record
            page_text = PageText(
                page_number=page_num + 1,  # 1-indexed
                text=text,
                char_start=char_start,
                char_end=char_end,
                word_count=word_count
            )
            pages.append(page_text)
            
            # Update tracking
            full_text_parts.append(text)
            current_char_pos = char_end + 2  # +2 for "\n\n" separator
            total_words += word_count
        
        doc.close()
        
        # Combine full text
        full_text = "\n\n".join(full_text_parts)
        
        return ExtractedDocument(
            file_path=str(pdf_path),
            full_text=full_text,
            pages=pages,
            page_count=len(pages),
            word_count=total_words
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = text.replace('\x00', '')  # Null bytes
        text = re.sub(r'-\s+', '', text)  # Hyphenation at line breaks
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def extract_page_previews(
        self, 
        doc: ExtractedDocument, 
        max_chars: int = 1000
    ) -> list[dict]:
        """
        Get page previews for section detection.
        
        Args:
            doc: Extracted document
            max_chars: Max characters per page preview
            
        Returns:
            List of {"page": int, "preview": str}
        """
        previews = []
        for page in doc.pages:
            preview = page.text[:max_chars]
            if len(page.text) > max_chars:
                # Try to end at sentence boundary
                last_period = preview.rfind('.')
                if last_period > max_chars // 2:
                    preview = preview[:last_period + 1]
            
            previews.append({
                "page": page.page_number,
                "preview": preview,
                "word_count": page.word_count
            })
        
        return previews


# Convenience function
def extract_pdf(pdf_path: str) -> ExtractedDocument:
    """Extract PDF with page boundaries."""
    processor = PDFProcessor()
    return processor.extract_with_pages(pdf_path)
