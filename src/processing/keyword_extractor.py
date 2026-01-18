"""
Keyword extraction for Academic Paper MCP v3.3

Extracts keywords from papers with two strategies:
1. Parse explicit keywords from PDF (trustworthy, source="paper")
2. LLM extraction from abstract (fallback, source="llm")
"""

import re
import sys
from typing import Optional

import ollama


class KeywordExtractor:
    """
    Extract keywords from academic papers.
    
    Prioritizes author-provided keywords (from PDF) over LLM inference.
    """
    
    # Patterns to find explicit keywords in papers
    KEYWORD_PATTERNS = [
        r"Keywords?[:\s—–\-]+([^\n]+)",
        r"Index\s+Terms?[:\s—–\-]+([^\n]+)",
        r"Key\s+words?[:\s—–\-]+([^\n]+)",
        r"Categories\s+and\s+Subject\s+Descriptors[:\s—–\-]+([^\n]+)",
    ]
    
    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
    
    def extract(
        self, 
        full_text: str, 
        abstract: Optional[str] = None,
        title: Optional[str] = None
    ) -> tuple[list[str], str]:
        """
        Extract keywords from paper.
        
        Args:
            full_text: Complete paper text
            abstract: Paper abstract (for LLM fallback)
            title: Paper title (for LLM context)
            
        Returns:
            (keywords, source) where source is "paper" or "llm"
        """
        # Phase 1: Try to extract explicit keywords from PDF
        keywords = self._extract_from_text(full_text)
        if keywords:
            print(f"  Keywords from paper: {keywords}", file=sys.stderr)
            return keywords, "paper"
        
        # Phase 2: LLM fallback
        if abstract:
            keywords = self._extract_with_llm(abstract, title)
            if keywords:
                print(f"  Keywords from LLM: {keywords}", file=sys.stderr)
                return keywords, "llm"
        
        # No keywords found
        return [], "none"
    
    def _extract_from_text(self, full_text: str) -> list[str]:
        """
        Extract explicit keywords from paper text.
        
        Looks for "Keywords:", "Index Terms:", etc.
        """
        # Search in first portion of paper (keywords usually appear early)
        search_text = full_text[:20000]
        
        for pattern in self.KEYWORD_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                keywords_text = match.group(1)
                
                # Clean up the keywords text
                # Stop at common section starters
                keywords_text = re.split(
                    r'\n\s*\n|\n\s*[A-Z][A-Z\s]+:|\n\s*\d+\s*\.?\s*[A-Z]',
                    keywords_text
                )[0]
                
                # Split by common separators
                keywords = re.split(r'[,;•·]|\s{2,}', keywords_text)
                
                # Clean each keyword
                cleaned = []
                for kw in keywords:
                    kw = kw.strip().strip('.')
                    # Remove common noise
                    kw = re.sub(r'^\d+\s*', '', kw)  # Leading numbers
                    kw = kw.lower()
                    
                    # Filter valid keywords
                    if (kw and 
                        len(kw) >= 2 and 
                        len(kw) <= 100 and
                        not kw.startswith('http') and
                        not re.match(r'^[\d\s]+$', kw)):  # Not just numbers
                        cleaned.append(kw)
                
                if cleaned:
                    return cleaned[:10]  # Max 10 keywords
        
        return []
    
    def _extract_with_llm(
        self, 
        abstract: str, 
        title: Optional[str] = None
    ) -> list[str]:
        """
        Extract keywords using LLM (fallback).
        """
        context = f"Title: {title}\n\n" if title else ""
        
        prompt = f"""Extract 3-5 keywords from this academic paper.

{context}Abstract:
{abstract}

Return ONLY a JSON array of keywords, nothing else:
["keyword1", "keyword2", "keyword3"]

Rules:
- Be specific and technical
- Use terms that appear in the abstract
- Lowercase only
- No general terms like "research" or "study"
"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            # Handle both dict and object response
            if hasattr(response, 'response'):
                text = response.response
            elif isinstance(response, dict) and 'response' in response:
                text = response['response']
            else:
                return []
            
            # Parse JSON array
            keywords = self._parse_json_array(text)
            
            # Clean and validate
            cleaned = []
            for kw in keywords:
                if isinstance(kw, str):
                    kw = kw.strip().lower()
                    if kw and len(kw) >= 2 and len(kw) <= 100:
                        cleaned.append(kw)
            
            return cleaned[:5]  # Max 5 from LLM
            
        except Exception as e:
            print(f"  LLM keyword extraction failed: {e}", file=sys.stderr)
            return []
    
    def _parse_json_array(self, text: str) -> list:
        """Parse JSON array from LLM response."""
        import json
        
        text = text.strip()
        
        # Try direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        
        # Try to extract array from text
        match = re.search(r'\[([^\]]+)\]', text)
        if match:
            try:
                return json.loads(f"[{match.group(1)}]")
            except json.JSONDecodeError:
                pass
        
        # Try to parse comma-separated values
        if ',' in text:
            items = []
            for item in text.split(','):
                item = item.strip().strip('"\'[]')
                if item:
                    items.append(item)
            return items
        
        return []
