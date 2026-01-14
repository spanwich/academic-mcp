"""
Vector store for semantic search.

Uses ChromaDB with BGE embeddings for retrieval.
"""

import os
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    ChromaDB-based vector store for chunk retrieval.
    
    Stores embeddings of page chunks for semantic search.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        collection_name: str = "paper_chunks"
    ):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Lazy load embedding model
        self._embedding_model = None
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def add_chunk(
        self,
        chunk_id: str,
        content: str,
        metadata: dict
    ):
        """Add a chunk to the vector store."""
        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()
        
        # Add to collection
        self.collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
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
        query_embedding = self.embedding_model.encode(query).tolist()
        
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
