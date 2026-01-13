"""
Quality LLM extraction for academic papers.
v3: Reduced to 3 passes since Zotero provides metadata.

Pass 1: Methodology
Pass 2: Findings & Statistics  
Pass 3: Critical Analysis
"""

import json
import re
from typing import Optional
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential


class QualityExtractor:
    """
    LLM extraction for academic papers.
    
    Only extracts methodology, findings, and critical analysis.
    Basic metadata (title, authors, abstract) comes from Zotero.
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434"
    ):
        """
        Initialize extractor.
        
        Args:
            model: Ollama model name
            host: Ollama host URL
        """
        self.model = model
        self.client = ollama.Client(host=host)
    
    def extract_all(self, extracted_doc) -> dict:
        """
        Run all extraction passes on document.
        
        Args:
            extracted_doc: ExtractedDocument from PDFProcessor
            
        Returns:
            Dictionary with all extracted fields
        """
        # Limit text to avoid token limits (~30k chars ≈ 7.5k tokens)
        text = extracted_doc.full_text[:30000]
        
        results = {}
        
        # Pass 1: Methodology (detailed)
        methodology = self._extract_methodology(text)
        results.update(methodology)
        
        # Pass 2: Findings & Statistics
        findings = self._extract_findings(text)
        results.update(findings)
        
        # Pass 3: Critical Analysis (limitations, contributions)
        analysis = self._extract_critical_analysis(text)
        results.update(analysis)
        
        # Quick classification (simpler, combined with analysis)
        classification = self._extract_classification(text)
        results.update(classification)
        
        return results
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _llm_extract(self, prompt: str, expect_json: bool = True) -> dict | str:
        """Run LLM extraction with retry logic."""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json" if expect_json else None,
                options={
                    "temperature": 0.1,
                    "num_predict": 4000
                }
            )
            
            text = response["response"]
            
            if expect_json:
                text = text.strip()
                # Remove markdown code blocks if present
                if text.startswith("```"):
                    text = re.sub(r'^```json?\s*', '', text)
                    text = re.sub(r'\s*```$', '', text)
                return json.loads(text)
            
            return text
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {}
        except Exception as e:
            print(f"LLM error: {e}")
            raise
    
    def _extract_methodology(self, text: str) -> dict:
        """
        Pass 1: Extract detailed methodology.
        """
        prompt = f"""Extract detailed methodology from this academic paper. Return JSON only.

Paper text:
{text}

Return JSON with these fields:
{{
  "methodology_summary": "2-3 sentence summary of methods",
  "methodology_detailed": "detailed description of methodology including all steps",
  "study_design": "experimental/observational/survey/case study/simulation/etc",
  "sample_description": "who/what was studied, population, context",
  "sample_size": "N=X or specific numbers",
  "data_collection_methods": ["method1", "method2"],
  "analysis_methods": ["method1", "method2"],
  "statistical_tests": ["test1 with parameters", "test2"],
  "software_tools": ["tool1", "tool2"]
}}"""
        
        result = self._llm_extract(prompt)
        return {
            "methodology_summary": result.get("methodology_summary"),
            "methodology_detailed": result.get("methodology_detailed"),
            "study_design": result.get("study_design"),
            "sample_description": result.get("sample_description"),
            "sample_size": result.get("sample_size"),
            "data_collection_methods": result.get("data_collection_methods", []),
            "analysis_methods": result.get("analysis_methods", []),
            "statistical_tests": result.get("statistical_tests", []),
            "software_tools": result.get("software_tools", [])
        }
    
    def _extract_findings(self, text: str) -> dict:
        """
        Pass 2: Extract key findings and statistics.
        """
        prompt = f"""Extract key findings and results from this academic paper. Return JSON only.

Paper text:
{text}

Return JSON with these fields:
{{
  "key_findings": [
    {{
      "finding": "main finding statement",
      "evidence": "supporting evidence/data",
      "confidence": "high/medium/low"
    }}
  ],
  "quantitative_results": {{
    "metric1": "value1",
    "metric2": "value2"
  }},
  "qualitative_themes": [
    {{
      "theme": "theme name",
      "description": "theme description"
    }}
  ],
  "effect_sizes": [
    {{
      "measure": "Cohen's d / OR / eta squared / etc",
      "value": "0.5",
      "context": "what comparison this measures"
    }}
  ]
}}"""
        
        result = self._llm_extract(prompt)
        return {
            "key_findings": result.get("key_findings", []),
            "quantitative_results": result.get("quantitative_results", {}),
            "qualitative_themes": result.get("qualitative_themes", []),
            "effect_sizes": result.get("effect_sizes", [])
        }
    
    def _extract_critical_analysis(self, text: str) -> dict:
        """
        Pass 3: Extract limitations and critical analysis.
        """
        prompt = f"""Extract critical analysis from this academic paper. Return JSON only.

Paper text:
{text}

Return JSON with these fields:
{{
  "main_arguments": [
    {{
      "argument": "main argument or claim",
      "support": "how it's supported in the paper"
    }}
  ],
  "theoretical_contributions": "what theoretical contribution does this make",
  "practical_implications": "real-world applications and implications",
  "limitations": [
    {{
      "limitation": "specific limitation",
      "impact": "how it affects the findings",
      "acknowledged": true or false (did authors mention it)
    }}
  ],
  "future_research": ["suggestion1", "suggestion2"]
}}"""
        
        result = self._llm_extract(prompt)
        return {
            "main_arguments": result.get("main_arguments", []),
            "theoretical_contributions": result.get("theoretical_contributions"),
            "practical_implications": result.get("practical_implications"),
            "limitations": result.get("limitations", []),
            "future_research": result.get("future_research", [])
        }
    
    def _extract_classification(self, text: str) -> dict:
        """
        Quick classification pass.
        """
        # Use shorter text for classification
        prompt = f"""Classify this academic paper. Return JSON only.

Paper text (beginning):
{text[:8000]}

Return JSON with these fields:
{{
  "research_domain": "e.g., Computer Science, Medicine, Psychology, Engineering",
  "subdomain": "more specific area within the domain",
  "methodology_type": "quantitative/qualitative/mixed/theoretical/review",
  "paper_type": "empirical/review/meta_analysis/theoretical/case_study/design_science",
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
    
    def extract_custom(self, text: str, question: str) -> str:
        """
        Ask a custom question about the paper.
        
        Args:
            text: Paper text
            question: Custom question to answer
            
        Returns:
            Answer text
        """
        prompt = f"""Based on this academic paper, answer the following question.

Question: {question}

Paper text:
{text[:25000]}

Provide a detailed, accurate answer based on the paper content. 
If the information is not in the paper, say so clearly."""
        
        return self._llm_extract(prompt, expect_json=False)
    
    def reextract_field(self, text: str, field: str, focus: str = "") -> str:
        """
        Re-extract a specific field with more detail.
        
        Args:
            text: Paper text
            field: Field to extract (methodology, findings, limitations, etc.)
            focus: Optional specific aspect to focus on
            
        Returns:
            Detailed extraction
        """
        prompts = {
            "methodology": f"""Extract DETAILED methodology from this paper{f' focusing on: {focus}' if focus else ''}.

Include:
- Complete study design and approach
- All participants/sample details (who, how many, selection criteria)
- Step-by-step data collection procedures
- All instruments and tools used
- Complete analysis methods
- All statistical tests with parameters
- Software and versions
- Validity and reliability measures

Paper text:
{text[:35000]}

Provide comprehensive methodology details:""",

            "findings": f"""Extract ALL findings from this paper{f' focusing on: {focus}' if focus else ''}.

Include:
- All main results with exact numbers
- Secondary findings
- Effect sizes with confidence intervals
- All p-values and significance levels
- Qualitative themes with quotes
- Unexpected findings
- Null results

Paper text:
{text[:35000]}

List all findings with supporting data:""",

            "limitations": f"""Extract ALL limitations from this paper{f' focusing on: {focus}' if focus else ''}.

Include:
- Explicitly stated limitations
- Implicit limitations (not stated but evident)
- Methodological weaknesses
- Sample limitations
- Generalizability issues
- Measurement problems
- Threats to validity (internal and external)
- Statistical power issues

Paper text:
{text[:35000]}

List all limitations with their implications:""",

            "statistics": f"""Extract ALL statistical information from this paper{f' focusing on: {focus}' if focus else ''}.

Include:
- Sample sizes (n) for all groups
- Descriptive statistics (M, SD, range)
- All test statistics (t, F, χ², r, β, etc.)
- All p-values
- Effect sizes (d, η², r², OR, RR, etc.)
- Confidence intervals
- Power analysis if mentioned
- Reliability coefficients

Paper text:
{text[:35000]}

List all statistics in structured format:"""
        }
        
        prompt = prompts.get(
            field, 
            f"Extract detailed {field} information from this paper:\n{text[:35000]}"
        )
        
        return self._llm_extract(prompt, expect_json=False)
