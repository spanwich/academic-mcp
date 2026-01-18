"""
Domain classification for Academic Paper MCP v3.3

Self-organizing taxonomy that:
1. Reuses existing domains when paper fits
2. Creates new specific domains when needed
3. Keeps domains research-actionable (not too broad)
"""

import json
import re
import sys
from typing import Optional

import ollama


class DomainClassifier:
    """
    Classify papers into specific research domains.
    
    Uses self-organizing taxonomy: LLM checks existing domains
    before creating new ones.
    """
    
    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
    
    def classify(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        existing_domains: Optional[list[str]] = None
    ) -> tuple[str, bool]:
        """
        Classify paper into a specific domain.
        
        Args:
            abstract: Paper abstract
            title: Paper title (optional)
            keywords: Paper keywords (optional)
            existing_domains: List of existing domains in database
            
        Returns:
            (domain_name, is_new) - domain and whether it's newly created
        """
        existing_domains = existing_domains or []
        
        # Build context
        context_parts = []
        if title:
            context_parts.append(f"Title: {title}")
        if keywords:
            context_parts.append(f"Keywords: {', '.join(keywords)}")
        context_parts.append(f"Abstract: {abstract}")
        
        paper_context = "\n".join(context_parts)
        
        # Format existing domains
        if existing_domains:
            domains_list = "\n".join(f"- {d}" for d in existing_domains[:50])  # Limit for context
            domains_note = f"""
EXISTING DOMAINS IN DATABASE ({len(existing_domains)} total):
{domains_list}
"""
        else:
            domains_note = "EXISTING DOMAINS: (none yet - you'll create the first one)"
        
        prompt = f"""Classify this academic paper into ONE specific research domain.

{domains_note}

PAPER:
{paper_context}

INSTRUCTIONS:
1. If an existing domain fits semantically, REUSE it (strongly preferred)
   - Even if wording differs slightly, reuse if same research area
   - Example: "microkernel verification" fits "microkernel formal verification using Isabelle"
   
2. Only create NEW domain if paper covers truly different research area

3. Be SPECIFIC (research-actionable), not broad:
   BAD (too broad): "security", "formal verification", "real-time systems"
   GOOD (specific): "microkernel formal verification using Isabelle"
   GOOD: "acoustic side-channel attacks on input devices"
   GOOD: "mixed-criticality scheduling with temporal isolation"
   GOOD: "ICS/SCADA protocol security analysis"

4. Domain should answer: "What specific research problem/method does this paper address?"

Return JSON only:
{{"domain": "the specific domain name", "is_new": true/false, "reason": "brief explanation"}}
"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.2}
            )
            
            # Handle both dict and object response
            if hasattr(response, 'response'):
                text = response.response
            elif isinstance(response, dict) and 'response' in response:
                text = response['response']
            else:
                return self._fallback_domain(abstract, title), True
            
            result = self._parse_json_response(text)
            
            if result and 'domain' in result:
                domain = result['domain'].strip()
                is_new = result.get('is_new', True)
                
                # Validate domain
                if self._is_valid_domain(domain):
                    # Check if it matches an existing domain (fuzzy match)
                    matched = self._find_matching_domain(domain, existing_domains)
                    if matched:
                        print(f"  Domain matched existing: {matched}", file=sys.stderr)
                        return matched, False
                    
                    print(f"  Domain {'(new)' if is_new else '(existing)'}: {domain}", file=sys.stderr)
                    return domain, is_new
            
            # Fallback
            return self._fallback_domain(abstract, title), True
            
        except Exception as e:
            print(f"  Domain classification failed: {e}", file=sys.stderr)
            return self._fallback_domain(abstract, title), True
    
    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        text = text.strip()
        
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON object
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to extract just the domain
        match = re.search(r'"domain"\s*:\s*"([^"]+)"', text)
        if match:
            return {"domain": match.group(1), "is_new": True}
        
        return None
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Check if domain is valid (not too broad, not too long)."""
        if not domain or len(domain) < 5:
            return False
        if len(domain) > 100:
            return False
        
        # Reject overly broad domains
        too_broad = [
            "security", "verification", "systems", "computing",
            "software", "hardware", "research", "analysis",
            "computer science", "engineering", "technology"
        ]
        
        domain_lower = domain.lower()
        
        # Single word domains are too broad
        if ' ' not in domain and domain_lower in too_broad:
            return False
        
        # Very short domains are likely too broad
        if len(domain_lower.split()) == 1 and len(domain_lower) < 15:
            return False
        
        return True
    
    def _find_matching_domain(self, domain: str, existing_domains: list[str]) -> Optional[str]:
        """
        Find if domain matches an existing one (fuzzy match).
        
        Returns the existing domain if matched, None otherwise.
        """
        domain_lower = domain.lower()
        domain_words = set(domain_lower.split())
        
        for existing in existing_domains:
            existing_lower = existing.lower()
            
            # Exact match
            if domain_lower == existing_lower:
                return existing
            
            # High word overlap (>70%)
            existing_words = set(existing_lower.split())
            overlap = len(domain_words & existing_words)
            total = max(len(domain_words), len(existing_words))
            
            if total > 0 and overlap / total > 0.7:
                return existing
            
            # Key phrase match
            if domain_lower in existing_lower or existing_lower in domain_lower:
                return existing
        
        return None
    
    def _fallback_domain(self, abstract: str, title: Optional[str] = None) -> str:
        """Generate fallback domain from title/abstract."""
        # Use title if available
        if title:
            # Extract key phrases
            title_clean = re.sub(r'[^\w\s]', ' ', title.lower())
            words = title_clean.split()
            
            # Remove common words
            stopwords = {'a', 'an', 'the', 'of', 'for', 'in', 'on', 'to', 'and', 'with', 'using'}
            words = [w for w in words if w not in stopwords and len(w) > 2]
            
            if len(words) >= 3:
                return ' '.join(words[:5])
        
        # Fall back to generic domain
        return "uncategorized research"
