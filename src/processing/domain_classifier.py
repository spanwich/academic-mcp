"""
Domain classification for Academic Paper MCP v3.3.1

Embedding-based self-organizing taxonomy that:
1. LLM suggests a creative domain (without seeing existing domains)
2. Embedding search finds similar existing domains
3. Uses existing domain if similarity > threshold, else creates new

This avoids the bias problem where LLM over-matches to examples in prompt.
"""

import json
import re
import sys
from typing import Optional, TYPE_CHECKING

import ollama

if TYPE_CHECKING:
    from ..models.vectors import VectorStore


class DomainClassifier:
    """
    Classify papers into specific research domains using embedding similarity.

    Flow:
    1. LLM suggests domain name + description (creative, no existing domains shown)
    2. Create embedding of suggestion
    3. Search for similar existing domains
    4. Use existing if similarity > 0.7, else create new
    """

    # Similarity threshold for reusing existing domain
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def classify(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        vector_store: Optional["VectorStore"] = None,
        existing_domains: Optional[list[str]] = None  # Kept for backward compat, ignored
    ) -> tuple[str, bool, str]:
        """
        Classify paper into a specific domain using embedding similarity.

        Args:
            abstract: Paper abstract
            title: Paper title (optional)
            keywords: Paper keywords (optional)
            vector_store: VectorStore instance for domain embedding search
            existing_domains: IGNORED - kept for backward compatibility

        Returns:
            (domain_name, is_new, description)
            - domain_name: The assigned domain
            - is_new: True if this is a new domain
            - description: Description of the domain
        """
        # Step 1: LLM suggests domain (creative, no existing domains shown)
        suggested_domain, description = self._llm_suggest_domain(
            abstract=abstract,
            title=title,
            keywords=keywords
        )

        # Validate the suggestion
        if not self._is_valid_domain(suggested_domain):
            fallback = self._fallback_domain(abstract, title)
            print(f"  Domain suggestion invalid, using fallback: {fallback}", file=sys.stderr)
            return fallback, True, "Fallback domain based on title"

        # If no vector store, just return the suggestion as new
        if vector_store is None:
            print(f"  Domain (new, no vector store): {suggested_domain}", file=sys.stderr)
            return suggested_domain, True, description

        # Step 2: Create query text for embedding search
        query_text = f"{suggested_domain}: {description}"
        if keywords:
            query_text += f". Keywords: {', '.join(keywords[:10])}"

        # Step 3: Search for similar existing domains
        matches = vector_store.search_similar_domains(
            query_text=query_text,
            threshold=self.SIMILARITY_THRESHOLD,
            top_k=3
        )

        # Step 4: Decision
        if matches:
            best_match, similarity = matches[0]
            print(
                f"  Domain matched existing (sim={similarity:.3f}): {best_match}",
                file=sys.stderr
            )
            return best_match, False, description
        else:
            print(f"  Domain (new): {suggested_domain}", file=sys.stderr)
            return suggested_domain, True, description

    def classify_legacy(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        existing_domains: Optional[list[str]] = None
    ) -> tuple[str, bool]:
        """
        Legacy classify method for backward compatibility.

        Returns only (domain_name, is_new) without description.
        Does not use embedding search.
        """
        domain, is_new, _ = self.classify(
            abstract=abstract,
            title=title,
            keywords=keywords,
            vector_store=None,
            existing_domains=existing_domains
        )
        return domain, is_new

    def _llm_suggest_domain(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[list[str]] = None
    ) -> tuple[str, str]:
        """
        Use LLM to suggest a domain name and description.

        IMPORTANT: Does NOT show existing domains to avoid bias.
        Let the LLM be creative, then use embeddings to match.
        """
        # Build context
        context_parts = []
        if title:
            context_parts.append(f"Title: {title}")
        if keywords:
            context_parts.append(f"Keywords: {', '.join(keywords)}")
        context_parts.append(f"Abstract: {abstract[:2000]}")  # Limit length

        paper_context = "\n".join(context_parts)

        # Neutral prompt without biased examples
        prompt = f"""Classify this academic paper into a specific research domain.

PAPER:
{paper_context}

INSTRUCTIONS:
1. Identify the SPECIFIC research area this paper addresses
2. Be specific and research-actionable (not broad categories)
   - BAD: "security", "machine learning", "systems"
   - GOOD: "network protocol fuzzing for vulnerability discovery"
   - GOOD: "transformer models for code completion"
   - GOOD: "acoustic side-channel attacks on input devices"
3. Domain should answer: "What specific research problem/method does this paper address?"

Return JSON:
{{"domain": "specific domain name", "description": "2-3 sentence description of what papers in this domain study"}}
"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.3}  # Slightly higher for creativity
            )

            # Handle both dict and object response
            if hasattr(response, 'response'):
                text = response.response
            elif isinstance(response, dict) and 'response' in response:
                text = response['response']
            else:
                return self._fallback_domain(abstract, title), "Fallback domain"

            result = self._parse_json_response(text)

            if result and 'domain' in result:
                domain = result['domain'].strip()
                description = result.get('description', '').strip()

                if self._is_valid_domain(domain):
                    return domain, description or f"Research on {domain}"

            return self._fallback_domain(abstract, title), "Fallback domain based on title"

        except Exception as e:
            print(f"  Domain suggestion failed: {e}", file=sys.stderr)
            return self._fallback_domain(abstract, title), "Fallback domain"

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
            desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', text)
            return {
                "domain": match.group(1),
                "description": desc_match.group(1) if desc_match else ""
            }

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
            "computer science", "engineering", "technology",
            "machine learning", "artificial intelligence",
            "networking", "programming", "development"
        ]

        domain_lower = domain.lower()

        # Single word domains are too broad
        if ' ' not in domain and domain_lower in too_broad:
            return False

        # Very short domains are likely too broad
        words = domain_lower.split()
        if len(words) == 1 and len(domain_lower) < 15:
            return False

        # Two-word domains that are just broad terms
        if len(words) == 2:
            broad_pairs = [
                "machine learning", "deep learning", "computer security",
                "network security", "system security", "software security",
                "formal verification", "model checking", "static analysis"
            ]
            if domain_lower in broad_pairs:
                return False

        return True

    def _fallback_domain(self, abstract: str, title: Optional[str] = None) -> str:
        """Generate fallback domain from title/abstract."""
        # Use title if available
        if title:
            # Extract key phrases
            title_clean = re.sub(r'[^\w\s]', ' ', title.lower())
            words = title_clean.split()

            # Remove common words
            stopwords = {
                'a', 'an', 'the', 'of', 'for', 'in', 'on', 'to', 'and',
                'with', 'using', 'based', 'toward', 'towards', 'via'
            }
            words = [w for w in words if w not in stopwords and len(w) > 2]

            if len(words) >= 3:
                return ' '.join(words[:5])

        # Fall back to generic domain
        return "uncategorized research"
