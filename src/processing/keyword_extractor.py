"""
Keyword extraction for Academic Paper MCP v3.3

Extracts keywords from papers with two strategies:
1. Parse explicit keywords from PDF (trustworthy, source="paper")
2. LLM extraction from abstract (fallback, source="llm")

v3.3.1: Fixed extraction to handle PDF text without newlines.
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

    # LLM context limit for abstracts (handles edge cases: books/sections misclassified)
    # ~25K tokens safe for Qwen 32K context window
    LLM_MAX_CHARS = 100000

    # Patterns to find explicit keywords in papers
    KEYWORD_PATTERNS = [
        r"Keywords?[:\s—–\-]+",
        r"Index\s+Terms?[:\s—–\-]+",
        r"Key\s+words?[:\s—–\-]+",
        r"Categories\s+and\s+Subject\s+Descriptors[:\s—–\-]+",
    ]

    # Stop patterns - don't rely on newlines (PDFs often have none)
    STOP_PATTERNS = [
        r'\s+\d+\.\s+[A-Z]',           # "1. Introduction"
        r'\s+ACM\s+Reference',          # "ACM Reference Format"
        r'\s+CCS\s+Concepts',           # "CCS Concepts:"
        r'\s+Article\s+\d+',            # "Article 59 (May 2023)"
        r'\s+https?://',                # URLs
        r'\s+doi[\.:]\s*\d',            # DOIs
        r'\s+\d{4}\s+ACM',              # "2023 ACM"
        r'\s+pp\.\s*\d+',               # "pp. 65-86"
        r'\s+LNCS\s+\d+',               # "LNCS 16055"
        r'\s+Vol\.\s*\d+',              # "Vol. 56"
        r'\s+Permission\s+to\s+make',   # Copyright notice
        r'\s+©\s*\d{4}',                # Copyright symbol
        r'\s+I\.\s+[A-Z]',              # Roman numeral section "I. INTRODUCTION"
        r'\s+Abstract[:\s]',            # Abstract section
        r'\s+ABSTRACT[:\s]',            # ABSTRACT section
        r'\s+Introduction[:\s]',        # Introduction section
        r'\s+INTRODUCTION[:\s]',        # INTRODUCTION section
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
        Uses stop patterns to detect end of keyword section (not newlines).
        """
        # Search in first portion of paper (keywords usually appear early)
        search_text = full_text[:20000]

        for pattern in self.KEYWORD_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                # Get text after the "Keywords:" marker
                start_pos = match.end()
                remaining_text = search_text[start_pos:]

                # Find the earliest stop pattern
                end_pos = len(remaining_text)
                for stop_pattern in self.STOP_PATTERNS:
                    stop_match = re.search(stop_pattern, remaining_text, re.IGNORECASE)
                    if stop_match and stop_match.start() < end_pos:
                        end_pos = stop_match.start()

                # Also stop at double spaces followed by capital letters (new section)
                section_break = re.search(r'\s{2,}[A-Z][a-z]+\s+[a-z]', remaining_text)
                if section_break and section_break.start() < end_pos:
                    end_pos = section_break.start()

                keywords_text = remaining_text[:end_pos].strip()

                # Limit length (safeguard)
                if len(keywords_text) > 500:
                    keywords_text = keywords_text[:500]

                # Split by common separators
                keywords = re.split(r'[,;•·]|\s{2,}', keywords_text)

                # Clean and validate each keyword
                cleaned = []
                for kw in keywords:
                    kw = kw.strip().strip('.')
                    # Remove leading numbers
                    kw = re.sub(r'^\d+\s*', '', kw)
                    kw = kw.lower().strip()

                    if self._is_valid_keyword(kw):
                        cleaned.append(kw)

                if cleaned:
                    return cleaned[:10]  # Max 10 keywords

        return []

    def _is_valid_keyword(self, kw: str) -> bool:
        """
        Validate individual keyword.

        Filters out garbage that got captured with the keywords.
        """
        # Length checks
        if len(kw) < 2:
            return False
        if len(kw) > 50:
            return False

        # Citation patterns [1], [2,3], etc.
        if re.search(r'\[\d+\]', kw):
            return False

        # Contains year (likely citation or metadata)
        if re.search(r'\b(19|20)\d{2}\b', kw):
            return False

        # URLs
        if re.search(r'https?://', kw):
            return False

        # DOIs
        if re.search(r'doi[\.:]', kw, re.I):
            return False

        # Likely author name: "firstname lastname" pattern (all lowercase, 2 words, both < 15 chars)
        if re.match(r'^[a-z]{2,15}\s+[a-z]{2,15}$', kw) and not self._is_known_term(kw):
            return False

        # ACM metadata
        if 'acm' in kw.lower():
            return False

        # Article references
        if kw.startswith('article '):
            return False

        # Page numbers
        if re.match(r'^pp?\.\s*\d+', kw):
            return False

        # Just numbers/special chars
        if re.match(r'^[\d\s\-\.:]+$', kw):
            return False

        # Email addresses
        if '@' in kw:
            return False

        # Copyright notice
        if '©' in kw or 'copyright' in kw.lower():
            return False

        # Permission notices
        if 'permission' in kw.lower():
            return False

        return True

    def _is_known_term(self, term: str) -> bool:
        """
        Check if a two-word phrase is a known technical term (not a name).
        """
        known_terms = {
            'machine learning', 'deep learning', 'neural networks',
            'formal verification', 'model checking', 'side channel',
            'access control', 'information flow', 'real time',
            'operating system', 'operating systems', 'file system',
            'memory safety', 'type safety', 'control flow',
            'data flow', 'program analysis', 'static analysis',
            'dynamic analysis', 'symbolic execution', 'fuzzy testing',
            'network security', 'web security', 'mobile security',
            'binary analysis', 'reverse engineering', 'malware analysis',
            'intrusion detection', 'anomaly detection', 'threat modeling',
        }
        return term.lower() in known_terms

    def _extract_with_llm(
        self,
        abstract: str,
        title: Optional[str] = None
    ) -> list[str]:
        """
        Extract keywords using LLM (fallback).

        Truncates abstract at LLM context limit to handle edge cases
        (e.g., books/sections misclassified as papers with huge "abstracts").
        """
        # Truncate at LLM context limit to handle edge cases
        abstract_truncated = abstract[:self.LLM_MAX_CHARS] if abstract else ""

        context = f"Title: {title}\n\n" if title else ""

        prompt = f"""Extract 3-5 keywords from this academic paper.

{context}Abstract:
{abstract_truncated}

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
