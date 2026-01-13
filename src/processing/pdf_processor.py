"""
PDF text extraction with structure preservation.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import fitz  # PyMuPDF


@dataclass
class ExtractedDocument:
    """Extracted document with structure."""
    full_text: str
    pages: list[dict]
    sections: list[dict]
    metadata: dict
    page_count: int
    word_count: int


class PDFProcessor:
    """Extract text from PDFs while preserving structure."""
    
    # Common section header patterns
    SECTION_PATTERNS = [
        r"^\d+\.?\s+[A-Z]",  # "1. Introduction" or "1 Introduction"
        r"^[IVX]+\.?\s+[A-Z]",  # Roman numerals
        r"^(Abstract|Introduction|Background|Methods?|Methodology|Results?|Discussion|Conclusion|References|Acknowledgments?)\s*$",
    ]
    
    # Section type classification
    SECTION_KEYWORDS = {
        "abstract": ["abstract", "summary"],
        "introduction": ["introduction", "background", "overview"],
        "literature_review": ["literature", "related work", "prior work"],
        "methodology": ["method", "methodology", "approach", "materials", "procedure", "experimental", "design"],
        "results": ["result", "finding", "experiment", "evaluation", "analysis"],
        "discussion": ["discussion", "interpretation"],
        "conclusion": ["conclusion", "concluding", "future work", "summary"],
        "references": ["reference", "bibliography", "citation"],
        "appendix": ["appendix", "supplementary", "supplemental"]
    }
    
    def extract_text_with_structure(self, pdf_path: str) -> ExtractedDocument:
        """
        Extract text from PDF preserving document structure.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractedDocument with full text and sections
        """
        doc = fitz.open(pdf_path)
        
        pages = []
        sections = []
        full_text = ""
        
        current_section = {
            "title": "Preamble",
            "content": "",
            "page_start": 1,
            "type": "other"
        }
        
        for page_num, page in enumerate(doc, 1):
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            page_content = ""
            
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    block_text, is_header, header_info = self._process_text_block(block)
                    
                    if is_header and header_info:
                        # Save current section if it has content
                        if current_section["content"].strip():
                            current_section["page_end"] = page_num
                            sections.append(current_section.copy())
                        
                        # Start new section
                        current_section = {
                            "title": header_info["text"],
                            "content": "",
                            "page_start": page_num,
                            "type": self._classify_section(header_info["text"])
                        }
                    else:
                        current_section["content"] += block_text + "\n"
                    
                    page_content += block_text + "\n"
            
            pages.append({
                "page_num": page_num,
                "content": page_content
            })
            full_text += page_content + "\n\n"
        
        # Don't forget the last section
        if current_section["content"].strip():
            current_section["page_end"] = len(pages)
            sections.append(current_section)
        
        # Clean up sections
        sections = self._clean_sections(sections)
        
        result = ExtractedDocument(
            full_text=full_text,
            pages=pages,
            sections=sections,
            metadata=dict(doc.metadata),
            page_count=len(pages),
            word_count=len(full_text.split())
        )
        
        doc.close()
        return result
    
    def _process_text_block(self, block: dict) -> tuple[str, bool, Optional[dict]]:
        """
        Process a text block and detect headers.
        
        Returns:
            (block_text, is_header, header_info)
        """
        lines = block.get("lines", [])
        block_text = ""
        
        for line in lines:
            line_text = ""
            max_font_size = 0
            is_bold = False
            
            for span in line.get("spans", []):
                line_text += span.get("text", "")
                max_font_size = max(max_font_size, span.get("size", 0))
                if "bold" in span.get("font", "").lower():
                    is_bold = True
            
            line_stripped = line_text.strip()
            
            # Check if this line is a section header
            if self._is_section_header(line_stripped, max_font_size, is_bold):
                return block_text, True, {
                    "text": line_stripped,
                    "font_size": max_font_size,
                    "is_bold": is_bold
                }
            
            block_text += line_text
        
        return block_text, False, None
    
    def _is_section_header(self, text: str, font_size: float, is_bold: bool) -> bool:
        """Determine if text is a section header."""
        if len(text) > 100 or len(text) < 2:
            return False
        
        # Check against patterns
        for pattern in self.SECTION_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Font-based detection (headers typically larger/bold)
        if is_bold and font_size > 11:
            # Also check it's not just emphasized text
            if len(text.split()) <= 6:
                return True
        
        return False
    
    def _classify_section(self, title: str) -> str:
        """Classify section type from title."""
        title_lower = title.lower()
        
        for section_type, keywords in self.SECTION_KEYWORDS.items():
            if any(kw in title_lower for kw in keywords):
                return section_type
        
        return "other"
    
    def _clean_sections(self, sections: list[dict]) -> list[dict]:
        """Clean up extracted sections."""
        cleaned = []
        
        for section in sections:
            # Skip very short sections (likely noise)
            if len(section["content"].strip()) < 50:
                continue
            
            # Clean content
            section["content"] = self._clean_text(section["content"])
            cleaned.append(section)
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Fix common OCR/extraction issues
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'(?<=[a-z])-\s+(?=[a-z])', '', text)  # Fix hyphenation
        text = text.strip()
        return text
