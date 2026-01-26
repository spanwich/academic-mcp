#!/usr/bin/env python3
"""
Clean re-index all embeddings from SQLite source.

This script:
1. Deletes ALL ChromaDB collections (paper_chunks, domain_embeddings)
2. Re-creates collections with configured embedding model
3. Re-indexes all chunks from SQLite with sub-chunking
4. Re-indexes all domains from SQLite

Why: SQLite has the full page content. Old ChromaDB may have truncated embeddings.
     Sub-chunking ensures long pages are fully indexed without truncation.

Usage:
    python reindex_all.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from src.config import get_config
from src.models.database import Database, Chunk, Domain
from src.models.vectors import VectorStore


def reindex_all():
    config = get_config()

    print("=" * 60)
    print("Clean Re-Index All Embeddings")
    print("=" * 60)
    print(f"Embedding model: {config.embedding_model}")
    print(f"Embedding backend: {config.embedding_backend}")
    print(f"ChromaDB path: {config.chroma_persist_dir}")
    print()

    # Step 1: Delete ChromaDB collections
    print("[1/4] Deleting old ChromaDB collections...")
    client = chromadb.PersistentClient(
        path=config.chroma_persist_dir, settings=Settings(anonymized_telemetry=False)
    )

    for coll_name in ["paper_chunks", "domain_embeddings"]:
        try:
            client.delete_collection(coll_name)
            print(f"  Deleted: {coll_name}")
        except Exception:
            print(f"  Not found: {coll_name}")

    # Step 2: Initialize fresh VectorStore
    print("\n[2/4] Creating fresh VectorStore...")
    vector_store = VectorStore(
        persist_directory=config.chroma_persist_dir,
        embedding_model=config.embedding_model,
        embedding_backend=config.embedding_backend,
        ollama_host=config.ollama_host,
    )
    print(f"  Max chars per sub-chunk: {vector_store.max_chars}")

    # Step 3: Re-index chunks
    print("\n[3/4] Re-indexing chunks from SQLite...")
    db = Database(config.database_url)

    with db.get_session() as session:
        chunks = (
            session.query(Chunk).order_by(Chunk.paper_id, Chunk.page_number).all()
        )
        print(f"  Found {len(chunks)} chunks in SQLite")

        if not chunks:
            print("  No chunks to index.")
        else:
            for chunk in tqdm(chunks, desc="  Indexing chunks"):
                # VectorStore.add_chunk() now handles sub-chunking automatically
                vector_store.add_chunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata={
                        "paper_id": chunk.paper_id,
                        "page_number": chunk.page_number,
                        "section_id": chunk.section_id or "",
                        "word_count": chunk.word_count,
                    },
                )

    chunk_count = vector_store.count()
    print(f"  Created {chunk_count} embeddings (including sub-chunks)")

    # Step 4: Re-index domains
    print("\n[4/4] Re-indexing domains from SQLite...")
    with db.get_session() as session:
        domains = session.query(Domain).all()
        print(f"  Found {len(domains)} domains in SQLite")

        if not domains:
            print("  No domains to index.")
        else:
            for domain in tqdm(domains, desc="  Indexing domains"):
                # Parse aggregated_keywords (may be None or empty)
                keywords = domain.aggregated_keywords or []
                vector_store.add_domain_embedding(
                    domain_name=domain.name,
                    keywords=keywords,
                    description=domain.description,
                )

    domain_count = vector_store.count_domains()
    print(f"  Created {domain_count} domain embeddings")

    # Summary
    print()
    print("=" * 60)
    print("Re-Index Complete")
    print("=" * 60)
    print(f"  Chunk embeddings: {chunk_count}")
    print(f"  Domain embeddings: {domain_count}")
    print(f"  Embedding model: {config.embedding_model}")
    print()
    print("Next steps:")
    print("  - Run 'python test_setup.py' to verify setup")
    print("  - Test search with: python -c \"from src.models.vectors import VectorStore; ...\"")


if __name__ == "__main__":
    reindex_all()
