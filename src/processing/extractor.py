"""
Per-section verbatim extraction for academic papers.

v3.2: 
- Extracts from each section separately (better for long docs)
- Copies original text verbatim (prevents hallucination)
- Tracks source page/section for verification
"""

import json
import re
import sys
from typing import Optional

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from .pdf_processor import ExtractedDocument
from .section_detector import DetectedSection


class SectionExtractor:
    """
    Extract content from document sections.
    
    Philosophy: Copy original text, don't summarize.
    - Verbatim fields contain exact quotes (trustworthy)
    - Summary fields are LLM-generated (use with caution)
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.client = ollama.Client(host=host)
    
    def extract_all(
        self,
        doc: ExtractedDocument,
        sections: list[DetectedSection]
    ) -> dict:
        """
        Extract content from all sections.
        
        Args:
            doc: Extracted document
            sections: Detected sections
            
        Returns:
            Dictionary with all extractions
        """
        results = {
            # Verbatim extractions
            "methodology_verbatim": None,
            "evaluation_setup_verbatim": None,
            "contributions_verbatim": [],
            "results_verbatim": [],
            "statistics_verbatim": [],
            "limitations_verbatim": [],
            "future_work_verbatim": [],
            
            # Summaries
            "methodology_summary": None,
            
            # Classification
            "research_domain": None,
            "subdomain": None,
            "methodology_type": None,
            "paper_type": None,
            "keywords": [],
            "software_tools": [],
            
            # Per-section summaries and key points
            "section_summaries": {},
            "section_key_points": {},
        }
        
        # Process each section based on type
        for i, section in enumerate(sections):
            section_text = doc.full_text[section.char_start:section.char_end]
            section_id = f"sec_{i}"
            
            # Limit section text for very large sections (e.g., book chapters)
            max_section_chars = 20000  # ~5000 words
            if len(section_text) > max_section_chars:
                # Take beginning and end for context
                section_text = section_text[:max_section_chars//2] + "\n\n[...truncated...]\n\n" + section_text[-max_section_chars//2:]
            
            # Get section summary and key points
            summary, key_points = self._extract_section_content(
                section_text, 
                section.section_type,
                section.page_start,
                section.page_end
            )
            results["section_summaries"][section_id] = summary
            results["section_key_points"][section_id] = key_points
            
            # Extract type-specific content
            if section.section_type in ["abstract", "introduction"]:
                contributions = self._extract_contributions(
                    section_text, section.section_type, section.page_start
                )
                results["contributions_verbatim"].extend(contributions)
            
            elif section.section_type == "methodology":
                methodology = self._extract_methodology_verbatim(
                    section_text, section.page_start
                )
                if not results["methodology_verbatim"]:
                    results["methodology_verbatim"] = methodology.get("methodology")
                    results["evaluation_setup_verbatim"] = methodology.get("evaluation_setup")
                    results["software_tools"].extend(methodology.get("tools", []))
                    results["methodology_summary"] = methodology.get("summary")
            
            elif section.section_type == "results":
                findings = self._extract_results_verbatim(
                    section_text, section.page_start
                )
                results["results_verbatim"].extend(findings.get("results", []))
                results["statistics_verbatim"].extend(findings.get("statistics", []))
            
            elif section.section_type in ["discussion", "conclusion"]:
                analysis = self._extract_analysis_verbatim(
                    section_text, section.section_type, section.page_start
                )
                results["limitations_verbatim"].extend(analysis.get("limitations", []))
                results["future_work_verbatim"].extend(analysis.get("future_work", []))
                results["contributions_verbatim"].extend(analysis.get("contributions", []))
        
        # Extract classification from abstract/intro
        intro_text = self._get_intro_text(doc, sections)
        if intro_text:
            classification = self._extract_classification(intro_text)
            results.update(classification)
        
        return results
    
    def _get_intro_text(
        self, 
        doc: ExtractedDocument, 
        sections: list[DetectedSection]
    ) -> Optional[str]:
        """Get abstract + introduction text for classification."""
        intro_text = ""
        for section in sections[:3]:  # First 3 sections
            if section.section_type in ["abstract", "introduction", "background"]:
                intro_text += doc.full_text[section.char_start:section.char_end]
                intro_text += "\n\n"
        
        return intro_text[:15000] if intro_text else None
    
    def _repair_json(self, text: str) -> str:
        """Attempt to repair malformed JSON."""
        import re
        
        # Remove control characters that break JSON
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
        
        # Replace newlines inside strings (common issue with verbatim text)
        # This is a rough heuristic - find strings and replace their newlines
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Try to fix truncated JSON by closing brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        # If truncated, try to find last complete item
        if open_braces > 0 or open_brackets > 0:
            # Find last complete structure
            last_good = max(
                text.rfind('},'),
                text.rfind('}]'),
                text.rfind(']'),
            )
            if last_good > len(text) // 2:  # Only if we have substantial content
                text = text[:last_good + 1]
                # Re-count after truncation
                open_braces = text.count('{') - text.count('}')
                open_brackets = text.count('[') - text.count(']')
        
        # Close any remaining open brackets
        text = text + ']' * open_brackets + '}' * open_braces
        
        return text
    
    def _safe_json_loads(self, text: str) -> dict | list | None:
        """Try multiple strategies to parse JSON."""
        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Repair and parse
        try:
            repaired = self._repair_json(text)
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Try to extract just the JSON object/array
        try:
            # Find first { or [ and matching end
            start = -1
            for i, c in enumerate(text):
                if c in '{[':
                    start = i
                    break
            
            if start >= 0:
                bracket = text[start]
                end_bracket = '}' if bracket == '{' else ']'
                depth = 0
                end = -1
                
                for i in range(start, len(text)):
                    if text[i] == bracket:
                        depth += 1
                    elif text[i] == end_bracket:
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                
                if end > start:
                    extracted = text[start:end+1]
                    return json.loads(extracted)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Return empty result
        return None
    
    def _llm_extract(self, prompt: str, expect_json: bool = True) -> dict | str:
        """Run LLM extraction."""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json" if expect_json else None,
                options={"temperature": 0.1, "num_predict": 4000}
            )
            
            # Handle both dict and object response formats
            try:
                if hasattr(response, 'response'):
                    text = response.response
                elif isinstance(response, dict) and 'response' in response:
                    text = response['response']
                else:
                    print(f"Unexpected response format: {type(response)}", file=sys.stderr)
                    return {} if expect_json else ""
                
                if text is None:
                    return {} if expect_json else ""
                    
                text = str(text).strip()
            except Exception as e:
                print(f"Failed to extract response text: {e}", file=sys.stderr)
                return {} if expect_json else ""
            
            if expect_json:
                if text.startswith("```"):
                    text = re.sub(r'^```json?\s*', '', text)
                    text = re.sub(r'\s*```$', '', text)
                
                # Use safe JSON loading with multiple fallback strategies
                result = self._safe_json_loads(text)
                if result is not None:
                    return result
                else:
                    # Non-fatal: extraction will return empty, import continues
                    return {}
            
            return text
            
        except Exception as e:
            print(f"LLM extraction error: {e}", file=sys.stderr)
            return {} if expect_json else ""
    
    def _extract_section_content(
        self,
        text: str,
        section_type: str,
        page_start: int,
        page_end: int
    ) -> tuple[str, list[dict]]:
        """Extract summary and key points from a section."""
        
        prompt = f"""Analyze this {section_type} section (pages {page_start}-{page_end}).

Text:
{text[:12000]}

Return JSON:
{{
  "summary": "2-3 sentence summary of this section",
  "key_points": [
    {{"text": "Copy an important sentence exactly as written", "page": {page_start}}},
    {{"text": "Another key sentence copied verbatim", "page": {page_start}}}
  ]
}}

RULES:
- summary: Your own words, brief overview
- key_points: EXACT quotes from the text (copy verbatim)
- Include 3-5 key points
- Estimate which page each quote is from (between {page_start}-{page_end})"""

        result = self._llm_extract(prompt)
        
        summary = result.get("summary", "")
        key_points = result.get("key_points", [])
        
        return summary, key_points
    
    def _extract_contributions(
        self,
        text: str,
        section_type: str,
        page_start: int
    ) -> list[dict]:
        """Extract contribution statements verbatim."""
        
        prompt = f"""Find sentences stating the paper's contributions in this {section_type} section.

Text:
{text[:10000]}

Return JSON:
{{
  "contributions": [
    {{"text": "Copy the exact sentence stating a contribution", "section": "{section_type}", "page": {page_start}}},
    {{"text": "Another contribution sentence copied verbatim", "section": "{section_type}", "page": {page_start}}}
  ]
}}

RULES:
- Only include sentences that explicitly state what this paper contributes
- Copy EXACTLY as written
- Look for phrases like "we present", "we propose", "this paper introduces", "our contribution"
- If no clear contribution statements, return empty array"""

        result = self._llm_extract(prompt)
        return result.get("contributions", [])
    
    def _extract_methodology_verbatim(
        self,
        text: str,
        page_start: int
    ) -> dict:
        """Extract methodology content verbatim."""
        
        prompt = f"""Extract methodology content from this section. COPY text verbatim, don't summarize.

Text:
{text[:15000]}

Return JSON:
{{
  "methodology": "Copy the main paragraphs describing the methodology/approach exactly as written",
  "evaluation_setup": "Copy the paragraphs describing experimental/evaluation setup exactly as written",
  "tools": ["tool1", "tool2"],
  "summary": "Brief 2-sentence summary in your own words"
}}

RULES:
- methodology: Copy 2-4 paragraphs verbatim (the core approach/design)
- evaluation_setup: Copy paragraphs about experiments/evaluation setup
- tools: List software, platforms, hardware mentioned
- summary: Your own brief summary

If no methodology found, return empty strings."""

        return self._llm_extract(prompt)
    
    def _extract_results_verbatim(
        self,
        text: str,
        page_start: int
    ) -> dict:
        """Extract results and statistics verbatim."""
        
        prompt = f"""Extract results and statistics from this section. COPY sentences exactly.

Text:
{text[:15000]}

Return JSON:
{{
  "results": [
    {{"text": "Copy exact sentence reporting a result", "section": "results", "page": {page_start}}},
    {{"text": "Another result sentence copied verbatim", "section": "results", "page": {page_start}}}
  ],
  "statistics": [
    {{"text": "Copy exact sentence with numbers/measurements", "section": "results", "page": {page_start}}}
  ]
}}

RULES:
- results: Sentences stating findings, outcomes, or conclusions
- statistics: ONLY sentences with explicit numbers, percentages, measurements
- Copy EXACTLY as written
- If no statistics exist in the text, return empty array for statistics
- Do NOT invent or estimate numbers"""

        result = self._llm_extract(prompt)
        return {
            "results": result.get("results", []),
            "statistics": result.get("statistics", [])
        }
    
    def _extract_analysis_verbatim(
        self,
        text: str,
        section_type: str,
        page_start: int
    ) -> dict:
        """Extract limitations and future work verbatim."""
        
        prompt = f"""Extract limitations and future work from this {section_type} section. COPY exactly.

Text:
{text[:12000]}

Return JSON:
{{
  "limitations": [
    {{"text": "Copy exact sentence where authors state a limitation", "section": "{section_type}", "page": {page_start}}}
  ],
  "future_work": [
    {{"text": "Copy exact sentence about future work", "section": "{section_type}", "page": {page_start}}}
  ],
  "contributions": [
    {{"text": "Copy sentence summarizing a contribution (if restated here)", "section": "{section_type}", "page": {page_start}}}
  ]
}}

RULES:
- limitations: What the AUTHORS say are limitations (not your inference)
- future_work: What AUTHORS suggest for future research
- contributions: If authors restate their contributions in conclusion
- Copy EXACTLY as written
- If nothing found, return empty arrays"""

        result = self._llm_extract(prompt)
        return {
            "limitations": result.get("limitations", []),
            "future_work": result.get("future_work", []),
            "contributions": result.get("contributions", [])
        }
    
    def _extract_classification(self, text: str) -> dict:
        """Extract document classification."""
        
        prompt = f"""Classify this academic document.

Text (from abstract/introduction):
{text[:8000]}

Return JSON:
{{
  "research_domain": "Main field (e.g., Computer Science, Operating Systems)",
  "subdomain": "Specific area (e.g., Microkernels, Formal Verification)",
  "methodology_type": "empirical|theoretical|design-science|survey|formal-methods|mixed",
  "paper_type": "systems-paper|empirical-study|survey|position-paper|formal-verification|thesis|report",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}"""

        result = self._llm_extract(prompt)
        return {
            "research_domain": result.get("research_domain"),
            "subdomain": result.get("subdomain"),
            "methodology_type": result.get("methodology_type"),
            "paper_type": result.get("paper_type"),
            "keywords": result.get("keywords", [])
        }
    
    def extract_custom(
        self, 
        text: str, 
        question: str,
        page_context: Optional[str] = None
    ) -> str:
        """Ask a custom question about the paper."""
        
        context = f" (from {page_context})" if page_context else ""
        
        prompt = f"""Based on this text{context}, answer the question.

Question: {question}

Text:
{text[:30000]}

Instructions:
- Provide a detailed, accurate answer
- Quote relevant passages to support your answer
- If information is not in the text, say so clearly
- Do not make up information"""
        
        return self._llm_extract(prompt, expect_json=False)
