"""
Vector store for semantic search.

Uses ChromaDB with embeddings for retrieval.

v3.3.1: Added domain embedding collection for similarity-based domain matching.
v3.3.2: Added Ollama embedding support (no PyTorch/CUDA dependency).
v3.3.3: Added sub-page chunking to prevent truncation of long pages.
"""

import os
from typing import Optional, Literal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    ChromaDB-based vector store for chunk retrieval and domain matching.

    Collections:
    - paper_chunks: Page-level chunks for semantic search
    - domain_embeddings: Domain embeddings for similarity-based classification

    Embedding backends:
    - "ollama": Uses Ollama API (recommended, no PyTorch dependency)
    - "sentence-transformers": Uses sentence-transformers library (requires PyTorch)

    v3.3.3: Long pages are now split into sub-chunks at natural boundaries
    to prevent truncation. Sub-chunks are stored with metadata linking to
    the parent chunk (parent_chunk_id, sub_index, is_subchunk).
    """

    # Model-specific CHARACTER limits (not tokens)
    # nomic-embed-text has 2048 token context. With retry logic, we can be more generous.
    # Target ~75% of actual limit for natural splits, retry handles edge cases.
    MODEL_CHAR_LIMITS = {
        "nomic": 6000,     # nomic-embed-text: 2048 tokens * ~4 chars * 0.75
        "bge": 1500,       # BGE models: 512 tokens * ~4 chars * 0.75
        "all-minilm": 800, # all-MiniLM: 256 tokens * ~4 chars * 0.75
        "default": 2000    # Conservative default
    }

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "paper_chunks",
        embedding_backend: Literal["ollama", "sentence-transformers"] = "ollama",
        ollama_host: str = "http://localhost:11434"
    ):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self.embedding_backend = embedding_backend
        self.ollama_host = ollama_host

        # Calculate max chars for this embedding model
        self.max_chars = self._get_max_chars(embedding_model)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create paper chunks collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Get or create domain embeddings collection
        self.domain_collection = self.client.get_or_create_collection(
            name="domain_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

        # Lazy load embedding model (for sentence-transformers backend)
        self._st_model = None

    def _get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for text using configured backend.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        if self.embedding_backend == "ollama":
            return self._embed_ollama(text)
        else:
            return self._embed_sentence_transformers(text)

    def _embed_ollama(self, text: str) -> list[float]:
        """Get embedding using Ollama API with automatic truncation on overflow."""
        import ollama
        from ollama._types import ResponseError

        # Try with full text first, truncate on context overflow
        attempt_text = text
        for attempt in range(3):  # Try up to 3 times with progressively shorter text
            try:
                response = ollama.embed(
                    model=self.embedding_model_name,
                    input=attempt_text
                )

                # Handle response format
                if hasattr(response, 'embeddings'):
                    return response.embeddings[0]
                elif isinstance(response, dict) and 'embeddings' in response:
                    return response['embeddings'][0]
                else:
                    raise ValueError(f"Unexpected Ollama response format: {type(response)}")

            except ResponseError as e:
                if "context length" in str(e).lower() and attempt < 2:
                    # Truncate by half and retry
                    attempt_text = attempt_text[:len(attempt_text) // 2]
                    continue
                raise

    def _embed_sentence_transformers(self, text: str) -> list[float]:
        """Get embedding using sentence-transformers."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.embedding_model_name)

        return self._st_model.encode(text).tolist()

    def _get_max_chars(self, model: str) -> int:
        """
        Get max characters for embedding model.

        Args:
            model: Model name/identifier

        Returns:
            Maximum characters that fit within model's context limit
        """
        model_lower = model.lower()
        for prefix, limit in self.MODEL_CHAR_LIMITS.items():
            if prefix in model_lower:
                return limit
        return self.MODEL_CHAR_LIMITS["default"]

    def _split_for_embedding(self, text: str) -> list[str]:
        """
        Split text into chunks that fit embedding model limit.

        Splits on natural boundaries (paragraphs, then sentences) to preserve
        semantic coherence. Returns list of text chunks, each guaranteed to be
        within max_chars limit.

        Args:
            text: Text to split

        Returns:
            List of text chunks, each within max_chars limit
        """
        if len(text) <= self.max_chars:
            return [text]

        chunks = []
        current_chunk = ""

        # Split by paragraphs (double newline) first
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            # If single paragraph exceeds limit, split by sentences
            if len(para) > self.max_chars:
                # Split by sentence boundaries
                sentences = para.replace(". ", ".\n").split("\n")
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current_chunk) + len(sent) + 2 <= self.max_chars:
                        if current_chunk:
                            current_chunk += " " + sent
                        else:
                            current_chunk = sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip()[:self.max_chars])
                        # If single sentence exceeds limit, truncate it
                        if len(sent) > self.max_chars:
                            current_chunk = sent[:self.max_chars]
                        else:
                            current_chunk = sent
            elif len(current_chunk) + len(para) + 2 <= self.max_chars:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip()[:self.max_chars])
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip()[:self.max_chars])

        # Fallback: if no chunks created, return truncated original
        # Final safety: ensure all chunks are within limit
        result = chunks if chunks else [text[:self.max_chars]]
        return [c[:self.max_chars] for c in result]

    # ==================== Paper Chunk Methods ====================

    def add_chunk(
        self,
        chunk_id: str,
        content: str,
        metadata: dict
    ):
        """
        Add chunk(s) to vector store, splitting if necessary.

        If content exceeds embedding model's limit, splits into sub-chunks
        at natural boundaries and stores each with metadata linking to parent.

        Args:
            chunk_id: Unique identifier for the chunk (e.g., "smith_2023_page_5")
            content: Text content to embed
            metadata: Metadata dict (paper_id, page_number, etc.)
        """
        # Skip empty or whitespace-only content
        if not content or not content.strip():
            return

        sub_chunks = self._split_for_embedding(content)
        # Filter out any empty sub-chunks
        sub_chunks = [c for c in sub_chunks if c and c.strip()]
        if not sub_chunks:
            return

        if len(sub_chunks) == 1:
            # Single chunk - store normally (use sub_chunks[0] in case of truncation)
            text = sub_chunks[0]
            embedding = self._get_embedding(text)
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{**metadata, "is_subchunk": False}]
            )
        else:
            # Multiple sub-chunks - store each with parent reference
            for i, sub_text in enumerate(sub_chunks):
                sub_id = f"{chunk_id}_sub_{i}"
                embedding = self._get_embedding(sub_text)
                self.collection.add(
                    ids=[sub_id],
                    embeddings=[embedding],
                    documents=[sub_text],
                    metadatas=[{
                        **metadata,
                        "parent_chunk_id": chunk_id,
                        "sub_index": i,
                        "is_subchunk": True
                    }]
                )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_paper_id: Optional[str] = None,
        filter_section_type: Optional[str] = None
    ) -> list[dict]:
        """
        Semantic search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results
            filter_paper_id: Limit to specific paper
            filter_section_type: Limit to section type

        Returns:
            List of results with chunk_id, score, content, metadata
        """
        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Build filter
        where_filter = None
        if filter_paper_id or filter_section_type:
            conditions = []
            if filter_paper_id:
                conditions.append({"paper_id": filter_paper_id})
            if filter_section_type:
                conditions.append({"section_type": filter_section_type})

            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}

        # Query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                formatted.append({
                    "chunk_id": chunk_id,
                    "content": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0
                })

        return formatted

    def delete_paper(self, paper_id: str):
        """Delete all chunks for a paper."""
        # Get chunk IDs for this paper
        results = self.collection.get(
            where={"paper_id": paper_id},
            include=[]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Get a specific chunk by ID."""
        results = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )

        if results["ids"]:
            return {
                "chunk_id": results["ids"][0],
                "content": results["documents"][0] if results["documents"] else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else {}
            }

        return None

    def count(self) -> int:
        """Get total number of chunks."""
        return self.collection.count()

    def count_paper(self, paper_id: str) -> int:
        """Get number of chunks for a paper."""
        results = self.collection.get(
            where={"paper_id": paper_id},
            include=[]
        )
        return len(results["ids"]) if results["ids"] else 0

    # ==================== Domain Embedding Methods ====================

    def add_domain_embedding(
        self,
        domain_name: str,
        keywords: list[str],
        description: Optional[str] = None
    ):
        """
        Create or update embedding for a domain.

        The embedding is created from: "{domain_name}: {description}. Keywords: {keywords}"
        This allows semantic matching of papers to domains.

        Args:
            domain_name: Name of the domain
            keywords: List of keywords associated with this domain
            description: Optional description of the domain
        """
        # Build text for embedding
        parts = [domain_name]
        if description:
            parts.append(description)
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords[:20])}")  # Limit keywords

        text = ". ".join(parts)

        # Generate embedding
        embedding = self._get_embedding(text)

        # Use domain name as ID (sanitized)
        domain_id = self._sanitize_id(domain_name)

        # Check if domain already exists
        existing = self.domain_collection.get(ids=[domain_id])
        if existing["ids"]:
            # Update existing
            self.domain_collection.update(
                ids=[domain_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "domain_name": domain_name,
                    "keyword_count": len(keywords),
                    "has_description": bool(description)
                }]
            )
        else:
            # Add new
            self.domain_collection.add(
                ids=[domain_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "domain_name": domain_name,
                    "keyword_count": len(keywords),
                    "has_description": bool(description)
                }]
            )

    def search_similar_domains(
        self,
        query_text: str,
        threshold: float = 0.7,
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Find domains similar to query text.

        Args:
            query_text: Text to match (e.g., "{suggested_domain}: {description}")
            threshold: Minimum similarity score (0-1, cosine similarity)
            top_k: Maximum number of results

        Returns:
            List of (domain_name, similarity_score) tuples, sorted by score descending
        """
        # Check if we have any domains
        if self.domain_collection.count() == 0:
            return []

        # Generate query embedding
        query_embedding = self._get_embedding(query_text)

        # Query domain collection
        results = self.domain_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )

        # Filter by threshold and format results
        matches = []
        if results["ids"] and results["ids"][0]:
            for i, _ in enumerate(results["ids"][0]):
                # Convert distance to similarity (cosine distance -> similarity)
                similarity = 1 - results["distances"][0][i]
                if similarity >= threshold:
                    domain_name = results["metadatas"][0][i].get("domain_name", "")
                    if domain_name:
                        matches.append((domain_name, similarity))

        # Sort by similarity (should already be sorted, but ensure)
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    def update_domain_keywords(
        self,
        domain_name: str,
        new_keywords: list[str],
        description: Optional[str] = None
    ):
        """
        Add keywords to domain and re-embed.

        This should be called when a paper is assigned to a domain,
        to incorporate its keywords into the domain embedding.

        Args:
            domain_name: Name of the domain
            new_keywords: Keywords to add (will be merged with existing)
            description: Optional new description
        """
        domain_id = self._sanitize_id(domain_name)

        # Get existing domain data
        existing = self.domain_collection.get(
            ids=[domain_id],
            include=["documents", "metadatas"]
        )

        existing_keywords = []
        existing_description = description

        if existing["ids"]:
            # Parse existing keywords from document if possible
            doc = existing["documents"][0] if existing["documents"] else ""
            if "Keywords:" in doc:
                kw_part = doc.split("Keywords:")[-1].strip()
                existing_keywords = [k.strip() for k in kw_part.split(",")]

        # Merge keywords (deduplicate, lowercase)
        all_keywords = list(set(
            [k.lower() for k in existing_keywords] +
            [k.lower() for k in new_keywords]
        ))

        # Re-embed with updated keywords
        self.add_domain_embedding(
            domain_name=domain_name,
            keywords=all_keywords,
            description=existing_description
        )

    def delete_domain(self, domain_name: str):
        """Delete a domain embedding."""
        domain_id = self._sanitize_id(domain_name)
        try:
            self.domain_collection.delete(ids=[domain_id])
        except Exception:
            pass  # Domain may not exist

    def get_domain_embedding(self, domain_name: str) -> Optional[dict]:
        """Get domain embedding info."""
        domain_id = self._sanitize_id(domain_name)
        results = self.domain_collection.get(
            ids=[domain_id],
            include=["documents", "metadatas"]
        )

        if results["ids"]:
            return {
                "domain_name": domain_name,
                "document": results["documents"][0] if results["documents"] else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else {}
            }

        return None

    def count_domains(self) -> int:
        """Get total number of domain embeddings."""
        return self.domain_collection.count()

    def list_domains(self) -> list[str]:
        """List all domain names with embeddings."""
        results = self.domain_collection.get(include=["metadatas"])
        domains = []
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "domain_name" in meta:
                    domains.append(meta["domain_name"])
        return domains

    def _sanitize_id(self, text: str) -> str:
        """
        Sanitize text for use as ChromaDB ID.

        ChromaDB IDs must be strings, max 512 chars.
        """
        # Replace problematic characters
        sanitized = text.lower()
        sanitized = sanitized.replace(" ", "_")
        sanitized = sanitized.replace("/", "_")
        sanitized = sanitized.replace("\\", "_")

        # Remove other special chars
        import re
        sanitized = re.sub(r'[^a-z0-9_\-]', '', sanitized)

        # Truncate if too long
        if len(sanitized) > 500:
            sanitized = sanitized[:500]

        return sanitized
