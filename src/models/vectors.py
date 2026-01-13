"""
Pydantic models and typed wrapper for ChromaDB vector storage.
Provides type safety and validation for vector operations.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Union
from enum import Enum
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class SectionType(str, Enum):
    """Section types matching database enum."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    OTHER = "other"


# === Pydantic Models ===

class ChunkDocument(BaseModel):
    """Type-safe model for chunk documents stored in ChromaDB."""
    model_config = ConfigDict(use_enum_values=True)
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    paper_id: str = Field(..., description="Parent paper ID")
    content: str = Field(..., description="Chunk text content")
    section_type: SectionType = Field(..., description="Section classification")
    section_title: Optional[str] = Field(None, description="Original section heading")
    chunk_index: int = Field(..., ge=0, description="Order within paper")
    token_count: int = Field(..., gt=0, description="Estimated token count")
    page_numbers: Optional[str] = Field(None, description="Source page numbers")
    
    def to_chroma_metadata(self) -> dict:
        """Convert to ChromaDB metadata (excludes content)."""
        return {
            "paper_id": self.paper_id,
            "section_type": self.section_type if isinstance(self.section_type, str) else self.section_type.value,
            "section_title": self.section_title or "",
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "page_numbers": self.page_numbers or ""
        }
    
    @classmethod
    def from_chroma_result(
        cls,
        document: str,
        metadata: dict,
        chunk_id: str
    ) -> "ChunkDocument":
        """Create from ChromaDB query result."""
        return cls(
            chunk_id=chunk_id,
            content=document,
            paper_id=metadata["paper_id"],
            section_type=metadata["section_type"],
            section_title=metadata.get("section_title") or None,
            chunk_index=metadata["chunk_index"],
            token_count=metadata["token_count"],
            page_numbers=metadata.get("page_numbers") or None
        )


class PaperSummary(BaseModel):
    """Type-safe model for paper-level summaries stored in ChromaDB."""
    model_config = ConfigDict(use_enum_values=True)
    
    paper_id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    key_findings_text: Optional[str] = Field(None, description="Concatenated findings")
    keywords: list[str] = Field(default_factory=list, description="Paper keywords")
    research_domain: Optional[str] = Field(None, description="Research domain")
    methodology_type: Optional[str] = Field(None, description="Methodology type")
    paper_type: Optional[str] = Field(None, description="Paper type")
    
    def to_embedding_text(self) -> str:
        """Create rich text for embedding."""
        parts = [f"Title: {self.title}"]
        
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        
        if self.key_findings_text:
            parts.append(f"Key Findings: {self.key_findings_text}")
        
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        
        return "\n\n".join(parts)
    
    def to_chroma_metadata(self) -> dict:
        """Convert to ChromaDB metadata."""
        return {
            "title": self.title,
            "research_domain": self.research_domain or "",
            "methodology_type": self.methodology_type or "",
            "paper_type": self.paper_type or "",
            "keywords": ",".join(self.keywords)
        }
    
    @classmethod
    def from_chroma_result(
        cls,
        document: str,
        metadata: dict,
        paper_id: str
    ) -> "PaperSummary":
        """Create from ChromaDB query result."""
        keywords = metadata.get("keywords", "")
        return cls(
            paper_id=paper_id,
            title=metadata.get("title", ""),
            abstract=None,
            key_findings_text=document,
            keywords=keywords.split(",") if keywords else [],
            research_domain=metadata.get("research_domain") or None,
            methodology_type=metadata.get("methodology_type") or None,
            paper_type=metadata.get("paper_type") or None
        )


class SearchResult(BaseModel):
    """Search result with relevance scoring."""
    document: Union[ChunkDocument, PaperSummary]
    distance: float = Field(..., description="Vector distance (lower = more similar)")
    relevance_score: Optional[float] = Field(None, description="LLM-assigned relevance 0-10")
    
    @property
    def similarity(self) -> float:
        """Convert distance to similarity score (0-1)."""
        return 1 - min(self.distance, 1.0)


class SearchQuery(BaseModel):
    """Search query parameters."""
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    section_filter: Optional[SectionType] = None
    paper_filter: Optional[str] = None
    domain_filter: Optional[str] = None
    rerank: bool = Field(default=True, description="Apply LLM reranking")
    synthesize: bool = Field(default=True, description="Generate synthesized answer")


# === Typed ChromaDB Wrapper ===

class VectorStore:
    """
    Type-safe wrapper around ChromaDB operations.
    Handles embedding generation and provides typed query results.
    """
    
    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        embedding_model: str = "BAAI/bge-large-en-v1.5"
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collections
        self.chunks_collection = self.client.get_or_create_collection(
            name="paper_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.papers_collection = self.client.get_or_create_collection(
            name="paper_summaries",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10
        )
        return embeddings.tolist()
    
    def _embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        return self._embed([text])[0]
    
    # === Chunk Operations ===
    
    def add_chunks(self, chunks: list[ChunkDocument]) -> None:
        """Add chunks to the vector store."""
        if not chunks:
            return
        
        embeddings = self._embed([c.content for c in chunks])
        
        self.chunks_collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.to_chroma_metadata() for c in chunks],
            embeddings=embeddings
        )
    
    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        section_filter: Optional[SectionType] = None,
        paper_filter: Optional[str] = None
    ) -> list[SearchResult]:
        """Search for relevant chunks."""
        query_embedding = self._embed_single(query)
        
        # Build where filter
        where = {}
        if section_filter:
            where["section_type"] = section_filter.value if isinstance(section_filter, SectionType) else section_filter
        if paper_filter:
            where["paper_id"] = paper_filter
        
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Handle empty results
        if not results["ids"] or not results["ids"][0]:
            return []
        
        # Convert to typed results
        search_results = []
        for doc, meta, dist, id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0]
        ):
            chunk = ChunkDocument.from_chroma_result(doc, meta, id)
            search_results.append(SearchResult(document=chunk, distance=dist))
        
        return search_results
    
    def get_chunk(self, chunk_id: str) -> Optional[ChunkDocument]:
        """Get a specific chunk by ID."""
        result = self.chunks_collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if not result["ids"]:
            return None
        
        return ChunkDocument.from_chroma_result(
            result["documents"][0],
            result["metadatas"][0],
            chunk_id
        )
    
    def delete_paper_chunks(self, paper_id: str) -> None:
        """Delete all chunks for a paper."""
        # ChromaDB requires fetching IDs first for where-based deletion
        results = self.chunks_collection.get(
            where={"paper_id": paper_id},
            include=[]
        )
        if results["ids"]:
            self.chunks_collection.delete(ids=results["ids"])
    
    # === Paper Summary Operations ===
    
    def add_paper_summary(self, summary: PaperSummary) -> None:
        """Add paper summary to the vector store."""
        embedding_text = summary.to_embedding_text()
        embedding = self._embed_single(embedding_text)
        
        # Delete existing if present
        try:
            self.papers_collection.delete(ids=[summary.paper_id])
        except Exception:
            pass
        
        self.papers_collection.add(
            ids=[summary.paper_id],
            documents=[embedding_text],
            metadatas=[summary.to_chroma_metadata()],
            embeddings=[embedding]
        )
    
    def search_papers(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None
    ) -> list[SearchResult]:
        """Search for relevant papers."""
        query_embedding = self._embed_single(query)
        
        where = {}
        if domain_filter:
            where["research_domain"] = domain_filter
        
        results = self.papers_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Handle empty results
        if not results["ids"] or not results["ids"][0]:
            return []
        
        search_results = []
        for doc, meta, dist, id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0]
        ):
            summary = PaperSummary.from_chroma_result(doc, meta, id)
            search_results.append(SearchResult(document=summary, distance=dist))
        
        return search_results
    
    def delete_paper_summary(self, paper_id: str) -> None:
        """Delete paper summary."""
        try:
            self.papers_collection.delete(ids=[paper_id])
        except Exception:
            pass
    
    # === Utility Methods ===
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collections."""
        return {
            "chunks_count": self.chunks_collection.count(),
            "papers_count": self.papers_collection.count(),
            "embedding_model": self.embedding_model_name,
            "persist_dir": self.persist_dir
        }
    
    def reset(self) -> None:
        """Reset all collections (use with caution)."""
        self.client.delete_collection("paper_chunks")
        self.client.delete_collection("paper_summaries")
        
        self.chunks_collection = self.client.get_or_create_collection(
            name="paper_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.papers_collection = self.client.get_or_create_collection(
            name="paper_summaries",
            metadata={"hnsw:space": "cosine"}
        )
