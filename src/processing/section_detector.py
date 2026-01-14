"""
LLM-based section detection for academic papers.

v3.2: Identifies document structure (intro, methodology, results, etc.)
Falls back to page-based grouping if detection fails.
"""

import json
import re
import sys
from dataclasses import dataclass
from typing import Optional

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from .pdf_processor import ExtractedDocument


@dataclass
class DetectedSection:
    """A detected section in the document."""
    section_type: str
    section_title: Optional[str]
    page_start: int
    page_end: int
    
    # Calculated after linking to document
    char_start: int = 0
    char_end: int = 0


@dataclass 
class SectionDetectionResult:
    """Result of section detection."""
    sections: list[DetectedSection]
    detection_method: str  # "llm" or "page_fallback"
    confidence: float


class SectionDetector:
    """
    Detect document sections using LLM.
    
    Analyzes page content to identify section boundaries
    like introduction, methodology, results, etc.
    """
    
    VALID_SECTION_TYPES = {
        "abstract",
        "introduction", 
        "background",
        "literature_review",
        "related_work",
        "methodology",
        "methods",
        "design",
        "implementation",
        "results",
        "evaluation",
        "experiments",
        "discussion",
        "analysis",
        "conclusion",
        "conclusions",
        "future_work",
        "references",
        "bibliography",
        "appendix",
        "acknowledgments",
        "unknown"
    }
    
    # Normalize section types to standard names
    TYPE_NORMALIZATION = {
        "methods": "methodology",
        "related_work": "literature_review",
        "experiments": "results",
        "evaluation": "results",
        "design": "methodology",
        "implementation": "methodology",
        "conclusions": "conclusion",
        "analysis": "discussion",
        "bibliography": "references",
        "acknowledgments": "appendix",
        "future_work": "conclusion",
    }
    
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.client = ollama.Client(host=host)
    
    def detect_sections(
        self, 
        doc: ExtractedDocument,
        max_pages_for_context: int = 50
    ) -> SectionDetectionResult:
        """
        Detect sections in document.
        
        Args:
            doc: Extracted document with pages
            max_pages_for_context: Max pages to send to LLM
            
        Returns:
            SectionDetectionResult with sections and method used
        """
        # Build page previews for LLM
        previews = self._build_page_previews(doc, max_pages_for_context)
        
        # Try LLM detection
        try:
            sections = self._llm_detect(previews, doc.page_count)
            if sections and len(sections) >= 1:
                # Link sections to document positions
                sections = self._link_to_document(sections, doc)
                return SectionDetectionResult(
                    sections=sections,
                    detection_method="llm",
                    confidence=0.7
                )
        except Exception as e:
            # Get the actual underlying error from RetryError
            actual_error = e
            if hasattr(e, '__cause__') and e.__cause__:
                actual_error = e.__cause__
            elif hasattr(e, 'last_attempt'):
                try:
                    actual_error = e.last_attempt.exception()
                except:
                    pass
            # Use stderr to avoid corrupting MCP JSON-RPC on stdout
            print(f"LLM section detection failed: {actual_error}", file=sys.stderr)
        
        # Fallback: group by pages
        sections = self._page_fallback(doc)
        sections = self._link_to_document(sections, doc)
        return SectionDetectionResult(
            sections=sections,
            detection_method="page_fallback",
            confidence=0.3
        )
    
    def _build_page_previews(
        self, 
        doc: ExtractedDocument, 
        max_pages: int
    ) -> str:
        """Build page previews for LLM prompt."""
        lines = []
        
        # If document is short, include all pages
        if doc.page_count <= max_pages:
            pages_to_include = range(doc.page_count)
        else:
            # Sample pages: first 20, middle samples, last 10
            first_pages = list(range(min(20, doc.page_count)))
            last_pages = list(range(max(0, doc.page_count - 10), doc.page_count))
            
            # Middle samples
            middle_count = max_pages - len(first_pages) - len(last_pages)
            if middle_count > 0:
                middle_start = len(first_pages)
                middle_end = doc.page_count - len(last_pages)
                step = max(1, (middle_end - middle_start) // middle_count)
                middle_pages = list(range(middle_start, middle_end, step))[:middle_count]
            else:
                middle_pages = []
            
            pages_to_include = sorted(set(first_pages + middle_pages + last_pages))
        
        for page_idx in pages_to_include:
            page = doc.pages[page_idx]
            preview = page.text[:800]  # First 800 chars
            lines.append(f"=== PAGE {page.page_number} ===")
            lines.append(preview)
            lines.append("")
        
        return "\n".join(lines)
    
    def _llm_detect(
        self, 
        page_previews: str, 
        total_pages: int
    ) -> list[DetectedSection]:
        """Use LLM to detect sections."""
        
        prompt = f"""Analyze this academic document and identify the major sections.

Document has {total_pages} pages. Here are previews:

{page_previews}

Identify the section boundaries and return JSON in this EXACT format:

{{"sections": [
  {{"type": "abstract", "title": "Abstract", "page_start": 1, "page_end": 1}},
  {{"type": "introduction", "title": "1. Introduction", "page_start": 1, "page_end": 3}},
  {{"type": "methodology", "title": "3. Methods", "page_start": 8, "page_end": 15}}
]}}

Valid section types:
- abstract
- introduction
- background / literature_review
- methodology (or methods, design, implementation)
- results (or evaluation, experiments)  
- discussion
- conclusion
- references
- appendix
- unknown (if unclear)

IMPORTANT:
- page_start and page_end are inclusive (1-indexed)
- Sections should not overlap
- Cover all pages from 1 to {total_pages}
- If a section spans multiple chapters, group them logically

Return ONLY valid JSON with "sections" array."""

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            format="json",
            options={"temperature": 0.1, "num_predict": 2000}
        )
        
        # Parse response (handle both dict and object response formats)
        try:
            if hasattr(response, 'response'):
                text = response.response
            elif isinstance(response, dict) and 'response' in response:
                text = response['response']
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")
            
            if text is None:
                raise ValueError("Response text is None")
            
            text = str(text).strip()
        except Exception as e:
            raise AttributeError(f"Failed to extract response text: {e}")
        
        # Handle markdown code blocks
        if text.startswith("```"):
            text = re.sub(r'^```json?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        
        data = json.loads(text)
        
        # Handle case where LLM wraps array in a dict like {"sections": [...]}
        if isinstance(data, dict):
            # Try common keys that might contain the list
            for key in ["sections", "data", "results", "items"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                # If no known key, try first list value found
                for value in data.values():
                    if isinstance(value, list):
                        data = value
                        break
        
        # Ensure data is a list
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}: {str(data)[:200]}")
        
        # Validate and convert
        sections = []
        for item in data:
            if not isinstance(item, dict):
                continue  # Skip non-dict items
                
            section_type = str(item.get("type", "unknown")).lower()
            section_type = self.TYPE_NORMALIZATION.get(section_type, section_type)
            
            if section_type not in self.VALID_SECTION_TYPES:
                section_type = "unknown"
            
            try:
                page_start = int(item.get("page_start", 1))
                page_end = int(item.get("page_end", 1))
            except (ValueError, TypeError):
                page_start = 1
                page_end = total_pages
            
            sections.append(DetectedSection(
                section_type=section_type,
                section_title=item.get("title"),
                page_start=page_start,
                page_end=page_end
            ))
        
        # Sort by page_start
        sections.sort(key=lambda s: s.page_start)
        
        # Validate page ranges
        sections = self._validate_sections(sections, total_pages)
        
        return sections
    
    def _validate_sections(
        self, 
        sections: list[DetectedSection], 
        total_pages: int
    ) -> list[DetectedSection]:
        """Validate and fix section boundaries."""
        if not sections:
            return sections
        
        # Fix out-of-bounds pages
        for section in sections:
            section.page_start = max(1, min(section.page_start, total_pages))
            section.page_end = max(section.page_start, min(section.page_end, total_pages))
        
        # Fix overlaps by adjusting boundaries
        for i in range(1, len(sections)):
            prev = sections[i - 1]
            curr = sections[i]
            if curr.page_start <= prev.page_end:
                curr.page_start = prev.page_end + 1
                if curr.page_start > curr.page_end:
                    curr.page_end = curr.page_start
        
        # Fill gaps with "unknown" sections
        filled = []
        expected_start = 1
        
        for section in sections:
            if section.page_start > expected_start:
                # Gap detected, fill with unknown
                filled.append(DetectedSection(
                    section_type="unknown",
                    section_title=f"Pages {expected_start}-{section.page_start - 1}",
                    page_start=expected_start,
                    page_end=section.page_start - 1
                ))
            filled.append(section)
            expected_start = section.page_end + 1
        
        # Fill trailing gap
        if expected_start <= total_pages:
            filled.append(DetectedSection(
                section_type="unknown",
                section_title=f"Pages {expected_start}-{total_pages}",
                page_start=expected_start,
                page_end=total_pages
            ))
        
        return filled
    
    def _page_fallback(
        self, 
        doc: ExtractedDocument, 
        pages_per_section: int = 10
    ) -> list[DetectedSection]:
        """Fallback: group pages into sections."""
        sections = []
        
        for i in range(0, doc.page_count, pages_per_section):
            page_start = i + 1  # 1-indexed
            page_end = min(i + pages_per_section, doc.page_count)
            
            sections.append(DetectedSection(
                section_type="unknown",
                section_title=f"Pages {page_start}-{page_end}",
                page_start=page_start,
                page_end=page_end
            ))
        
        return sections
    
    def _link_to_document(
        self, 
        sections: list[DetectedSection], 
        doc: ExtractedDocument
    ) -> list[DetectedSection]:
        """Link sections to character positions in full_text."""
        for section in sections:
            # Find char positions from pages
            start_page = doc.get_page(section.page_start)
            end_page = doc.get_page(section.page_end)
            
            if start_page and end_page:
                section.char_start = start_page.char_start
                section.char_end = end_page.char_end
            else:
                # Fallback if pages not found
                section.char_start = 0
                section.char_end = len(doc.full_text)
        
        return sections
