#!/usr/bin/env python3
"""
Migration script for Academic Paper MCP v3.3.1

Migrates existing domains to have embeddings in ChromaDB.
This is a one-time migration for existing data.

Run this after upgrading to v3.3.1:
    python migrate_domain_embeddings.py

What it does:
1. Reads all existing domains from SQLite
2. For each domain, aggregates keywords from all papers in that domain
3. Creates domain embedding in ChromaDB
4. Updates Domain.aggregated_keywords field

This enables the new embedding-based domain classification.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import get_config
settings = get_config()
from src.models.database import Database, Domain, Paper
from src.models.vectors import VectorStore
from src.processing.keyword_extractor import KeywordExtractor


def migrate_domains():
    """Migrate existing domains to have embeddings."""
    print("=" * 60)
    print("Domain Embedding Migration for Academic MCP v3.3.1")
    print("=" * 60)
    print()

    # Initialize components
    print("Initializing database and vector store...")
    db = Database(settings.database_url)
    vector_store = VectorStore(
        persist_directory=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
        embedding_backend=settings.embedding_backend,
        ollama_host=settings.ollama_host
    )
    keyword_extractor = KeywordExtractor(
        model=settings.llm_model,
        host=settings.ollama_host
    )

    print(f"  Database: {settings.database_url}")
    print(f"  ChromaDB: {settings.chroma_persist_dir}")
    print(f"  Embedding: {settings.embedding_model} ({settings.embedding_backend})")
    print()

    # Check current state
    print("Current state:")
    print(f"  Domain embeddings in ChromaDB: {vector_store.count_domains()}")

    with db.get_session() as session:
        domain_count = session.query(Domain).count()
        paper_count = session.query(Paper).count()
        print(f"  Domains in database: {domain_count}")
        print(f"  Papers in database: {paper_count}")
    print()

    if domain_count == 0:
        print("No domains to migrate. Exiting.")
        return

    # Get user confirmation
    print(f"This will create/update embeddings for {domain_count} domains.")
    print("This may take a few minutes depending on the number of domains.")
    print()
    response = input("Continue? [y/N] ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    print()
    print("Starting migration...")
    print("-" * 60)

    # Process each domain
    migrated = 0
    skipped = 0
    errors = 0

    with db.get_session() as session:
        domains = session.query(Domain).all()

        for i, domain in enumerate(domains):
            print(f"\n[{i+1}/{len(domains)}] {domain.name}")

            try:
                # Get all papers in this domain
                papers = session.query(Paper).filter(
                    Paper.domain == domain.name
                ).all()

                print(f"  Papers: {len(papers)}")

                # Aggregate keywords from all papers
                all_keywords = []
                for paper in papers:
                    if paper.keywords:
                        all_keywords.extend(paper.keywords)

                # Deduplicate and filter
                unique_keywords = list(set(kw.lower() for kw in all_keywords if kw))

                # Validate keywords (remove garbage using new extractor)
                valid_keywords = [
                    kw for kw in unique_keywords
                    if keyword_extractor._is_valid_keyword(kw)
                ]

                print(f"  Aggregated keywords: {len(valid_keywords)}")
                if valid_keywords:
                    print(f"    Sample: {valid_keywords[:5]}")

                # Update domain's aggregated_keywords field
                domain.aggregated_keywords = valid_keywords

                # Create domain embedding
                # Use domain name as description if we don't have one
                description = domain.description or f"Research papers about {domain.name}"

                vector_store.add_domain_embedding(
                    domain_name=domain.name,
                    keywords=valid_keywords,
                    description=description
                )

                print(f"  ✓ Created embedding")
                migrated += 1

            except Exception as e:
                print(f"  ✗ Error: {e}")
                errors += 1

        # Commit all changes
        session.commit()

    print()
    print("-" * 60)
    print("Migration complete!")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print()
    print(f"Domain embeddings in ChromaDB: {vector_store.count_domains()}")


def verify_migration():
    """Verify the migration by testing a sample search."""
    print()
    print("=" * 60)
    print("Verification")
    print("=" * 60)
    print()

    db = Database(settings.database_url)
    vector_store = VectorStore(
        persist_directory=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
        embedding_backend=settings.embedding_backend,
        ollama_host=settings.ollama_host
    )

    print(f"Domain embeddings: {vector_store.count_domains()}")
    print()

    # Test search with a sample query
    test_queries = [
        "formal verification of operating systems",
        "machine learning for security",
        "network protocol analysis",
        "side-channel attacks on embedded systems"
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        matches = vector_store.search_similar_domains(query, threshold=0.5, top_k=3)
        if matches:
            for domain_name, score in matches:
                print(f"  {score:.3f}: {domain_name}")
        else:
            print("  (no matches above threshold)")
        print()


def list_domains_with_embeddings():
    """List all domains that have embeddings."""
    print()
    print("=" * 60)
    print("Domains with Embeddings")
    print("=" * 60)
    print()

    vector_store = VectorStore(
        persist_directory=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
        embedding_backend=settings.embedding_backend,
        ollama_host=settings.ollama_host
    )

    domains = vector_store.list_domains()
    print(f"Total: {len(domains)}")
    print()

    for domain in sorted(domains):
        info = vector_store.get_domain_embedding(domain)
        keyword_count = info["metadata"].get("keyword_count", 0) if info else 0
        print(f"  {domain} ({keyword_count} keywords)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate existing domains to have embeddings"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify migration with sample searches"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all domains with embeddings"
    )

    args = parser.parse_args()

    if args.verify:
        verify_migration()
    elif args.list:
        list_domains_with_embeddings()
    else:
        migrate_domains()
        verify_migration()
