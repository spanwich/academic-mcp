#!/usr/bin/env python3
"""
Migrate paper_chunks collection to Ollama embeddings.

This script:
1. Reads all chunks from SQLite (content already stored)
2. Deletes old paper_chunks collection in ChromaDB
3. Re-embeds each chunk using Ollama (nomic-embed-text)
4. Creates new paper_chunks collection with 768-dim embeddings

Why: The paper_chunks collection was created with BGE (1024 dim) but
the config now uses Ollama (768 dim). This causes dimension mismatch
errors during search_content operations.

Usage:
    python migrate_chunk_embeddings.py

Expected time: ~30 minutes for 10,000 chunks
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import ollama
import chromadb
from chromadb.config import Settings
from sqlalchemy import select
from tqdm import tqdm

from config import get_config
from models.database import Database, Chunk


def migrate_embeddings():
    """Migrate paper_chunks to Ollama embeddings."""
    config = get_config()

    print("=" * 60)
    print("Paper Chunks Embedding Migration")
    print("=" * 60)
    print(f"Database: {config.database_url}")
    print(f"ChromaDB: {config.chroma_persist_dir}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Embedding backend: {config.embedding_backend}")
    print(f"Ollama host: {config.ollama_host}")
    print()

    # 1. Connect to SQLite and count chunks
    print("[1/5] Connecting to SQLite database...")
    db = Database(config.database_url)

    with db.get_session() as session:
        chunk_count = session.query(Chunk).count()

    print(f"      Found {chunk_count:,} chunks to migrate")

    if chunk_count == 0:
        print("      No chunks to migrate. Exiting.")
        return

    # 2. Test Ollama connection
    print("\n[2/5] Testing Ollama connection...")
    try:
        test_response = ollama.embed(
            model=config.embedding_model,
            input="test embedding"
        )
        if hasattr(test_response, 'embeddings'):
            dim = len(test_response.embeddings[0])
        else:
            dim = len(test_response['embeddings'][0])
        print(f"      Ollama OK - embedding dimension: {dim}")
    except Exception as e:
        print(f"      ERROR: Cannot connect to Ollama: {e}")
        print(f"      Make sure Ollama is running with the {config.embedding_model} model")
        sys.exit(1)

    # 3. Connect to ChromaDB and check current state
    print("\n[3/5] Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path=config.chroma_persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    # Check if paper_chunks exists and get info
    try:
        old_collection = chroma_client.get_collection("paper_chunks")
        old_count = old_collection.count()

        # Sample to check dimensions
        sample = old_collection.get(limit=1, include=['embeddings'])
        if sample['embeddings'] and len(sample['embeddings']) > 0:
            old_dim = len(sample['embeddings'][0])
            print(f"      Existing collection: {old_count:,} embeddings, {old_dim} dimensions")
        else:
            print(f"      Existing collection: {old_count:,} embeddings")

    except Exception:
        print("      No existing paper_chunks collection found")
        old_count = 0

    # 4. Delete old collection and create new one
    print("\n[4/5] Recreating paper_chunks collection...")
    try:
        chroma_client.delete_collection("paper_chunks")
        print("      Deleted old collection")
    except Exception:
        print("      No existing collection to delete")

    new_collection = chroma_client.create_collection(
        name="paper_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"      Created new collection")

    # 5. Re-embed all chunks
    print(f"\n[5/5] Re-embedding {chunk_count:,} chunks...")
    print("      This may take a while. Progress:")

    start_time = time.time()
    batch_size = 50  # Process in batches for efficiency
    error_count = 0

    with db.get_session() as session:
        # Get all chunks ordered by paper_id for better cache locality
        chunks = session.query(Chunk).order_by(Chunk.paper_id, Chunk.page_number).all()

        # Process in batches
        batch_ids = []
        batch_embeddings = []
        batch_documents = []
        batch_metadatas = []

        for chunk in tqdm(chunks, desc="      Embedding"):
            try:
                # Generate embedding
                response = ollama.embed(
                    model=config.embedding_model,
                    input=chunk.content
                )

                if hasattr(response, 'embeddings'):
                    embedding = response.embeddings[0]
                else:
                    embedding = response['embeddings'][0]

                # Prepare batch data
                batch_ids.append(chunk.chunk_id)
                batch_embeddings.append(embedding)
                batch_documents.append(chunk.content)
                batch_metadatas.append({
                    "paper_id": chunk.paper_id,
                    "page_number": chunk.page_number,
                    "section_id": chunk.section_id or "",
                    "word_count": chunk.word_count
                })

                # Add batch when full
                if len(batch_ids) >= batch_size:
                    new_collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    batch_ids = []
                    batch_embeddings = []
                    batch_documents = []
                    batch_metadatas = []

            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"\n      Error on chunk {chunk.chunk_id}: {e}")
                elif error_count == 6:
                    print("      (suppressing further error messages)")

        # Add remaining batch
        if batch_ids:
            new_collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

    elapsed = time.time() - start_time
    final_count = new_collection.count()

    # Summary
    print("\n" + "=" * 60)
    print("Migration Complete")
    print("=" * 60)
    print(f"Chunks migrated: {final_count:,}")
    print(f"Errors: {error_count}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Rate: {final_count/elapsed:.1f} chunks/second")

    # Verify dimensions
    sample = new_collection.get(limit=1, include=['embeddings'])
    if sample['embeddings'] is not None and len(sample['embeddings']) > 0:
        new_dim = len(sample['embeddings'][0])
        print(f"New embedding dimension: {new_dim}")

    print("\nNext steps:")
    print("  1. Test search: python -c \"from src.models.vectors import VectorStore; ...\"")
    print("  2. Start server: ./start_server.sh")


if __name__ == "__main__":
    migrate_embeddings()
