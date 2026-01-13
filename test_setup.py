#!/usr/bin/env python3
"""Validate setup for Academic Paper MCP v3."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_database():
    """Test database connection."""
    print("Testing database...", end=" ")
    try:
        from src.models.database import Database, Paper
        db = Database("sqlite:///data/papers.db")
        db.create_tables()
        
        with db.get_session() as session:
            count = session.query(Paper).count()
        print(f"OK ({count} papers)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_vectors():
    """Test vector store."""
    print("Testing vector store...", end=" ")
    try:
        from src.models.vectors import VectorStore
        store = VectorStore(persist_dir="./data/chroma")
        stats = store.get_collection_stats()
        print(f"OK ({stats['chunks_count']} chunks)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_ollama():
    """Test Ollama connection."""
    print("Testing Ollama...", end=" ")
    try:
        import ollama
        response = ollama.list()
        
        models = response.get('models', [])
        model_names = []
        for m in models:
            name = m.get('name') or m.get('model') or str(m)
            model_names.append(name)
        
        if any('qwen' in m.lower() for m in model_names):
            print("OK (qwen model found)")
        else:
            print(f"OK (connected, models: {model_names[:3]})")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_pdf():
    """Test PDF processing."""
    print("Testing PDF processor...", end=" ")
    try:
        import fitz
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_embeddings():
    """Test embedding model."""
    print("Testing embedding model...", end=" ")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        embedding = model.encode(["test"], normalize_embeddings=True)
        print(f"OK (dim={len(embedding[0])})")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_zotero():
    """Test Zotero access."""
    print("Testing Zotero access...", end=" ")
    try:
        from src.zotero import ZoteroReader
        reader = ZoteroReader()
        stats = reader.get_stats()
        bbt = "✓" if stats['better_bibtex_installed'] else "✗"
        print(f"OK ({stats['total_items']} items, BBT: {bbt})")
        return True
    except FileNotFoundError as e:
        print(f"SKIPPED (Zotero not found)")
        return True  # Not a failure, just not available
    except Exception as e:
        print(f"FAILED: {e}")
        return False


if __name__ == "__main__":
    print("\n=== Academic Paper MCP v3 - Setup Validation ===\n")
    
    results = [
        test_database(),
        test_vectors(),
        test_ollama(),
        test_pdf(),
        test_embeddings(),
        test_zotero(),
    ]
    
    print("\n" + "=" * 45)
    passed = results.count(True)
    total = len(results)
    
    if all(results):
        print(f"All tests passed! ({passed}/{total}) ✓")
        sys.exit(0)
    else:
        print(f"Some tests failed: {passed}/{total}")
        sys.exit(1)
