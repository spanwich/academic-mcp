#!/usr/bin/env python3
"""
Setup validation for Academic Paper MCP v3.3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_database():
    """Test database connection."""
    try:
        from src.models.database import Database, Paper
        from src.config import get_config, ensure_directories
        
        ensure_directories()
        config = get_config()
        db = Database(config.database_url)
        db.create_tables()
        
        with db.get_session() as session:
            count = session.query(Paper).count()
        
        print(f"Testing database... OK ({count} papers)")
        return True
    except Exception as e:
        print(f"Testing database... FAILED: {e}")
        return False


def test_vector_store():
    """Test vector store."""
    try:
        from src.models.vectors import VectorStore
        from src.config import get_config
        
        config = get_config()
        store = VectorStore(
            persist_directory=config.chroma_persist_dir,
            embedding_model=config.embedding_model,
            embedding_backend=config.embedding_backend,
            ollama_host=config.ollama_host
        )
        count = store.count()
        
        print(f"Testing vector store... OK ({count} chunks)")
        return True
    except Exception as e:
        print(f"Testing vector store... FAILED: {e}")
        return False


def test_ollama():
    """Test Ollama connection."""
    try:
        import ollama
        from src.config import get_config
        
        config = get_config()
        client = ollama.Client(host=config.ollama_host)
        models = client.list()
        
        qwen_found = any('qwen' in m['name'].lower() for m in models.get('models', []))
        
        print(f"Testing Ollama... OK ({'qwen model found' if qwen_found else 'no qwen model'})")
        return True
    except Exception as e:
        print(f"Testing Ollama... FAILED: {e}")
        return False


def test_pdf_processor():
    """Test PDF processor."""
    try:
        from src.processing.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        print("Testing PDF processor... OK")
        return True
    except Exception as e:
        print(f"Testing PDF processor... FAILED: {e}")
        return False


def test_embedding_model():
    """Test embedding model loading."""
    try:
        from src.config import get_config

        config = get_config()

        if config.embedding_backend == "ollama":
            import ollama
            response = ollama.embed(
                model=config.embedding_model,
                input="test embedding"
            )
            if hasattr(response, 'embeddings'):
                dim = len(response.embeddings[0])
            elif isinstance(response, dict) and 'embeddings' in response:
                dim = len(response['embeddings'][0])
            else:
                raise ValueError(f"Unexpected Ollama response format: {type(response)}")
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.embedding_model)
            dim = model.get_sentence_embedding_dimension()

        print(f"Testing embedding model... OK (dim={dim}, backend={config.embedding_backend})")
        return True
    except Exception as e:
        print(f"Testing embedding model... FAILED: {e}")
        return False


def test_zotero():
    """Test Zotero access."""
    try:
        from src.zotero.reader import ZoteroReader
        from src.config import get_config
        
        config = get_config()
        reader = ZoteroReader(config.zotero_path)
        stats = reader.get_stats()
        
        bbt = "✓" if stats.get("has_bbt") else "✗"
        print(f"Testing Zotero access... OK ({stats['total_items']} items, BBT: {bbt})")
        return True
    except FileNotFoundError:
        print("Testing Zotero access... SKIPPED (Zotero not found)")
        return True  # Optional test
    except RuntimeError as e:
        if "locked" in str(e).lower():
            print("Testing Zotero access... SKIPPED (database locked, close Zotero)")
            return True  # Optional test
        print(f"Testing Zotero access... FAILED: {e}")
        return False
    except Exception as e:
        print(f"Testing Zotero access... FAILED: {e}")
        return False


def main():
    print("=== Academic Paper MCP v3.3 - Setup Validation ===")
    
    tests = [
        test_database,
        test_vector_store,
        test_ollama,
        test_pdf_processor,
        test_embedding_model,
        test_zotero,
    ]
    
    passed = sum(1 for test in tests if test())
    total = len(tests)
    
    print("=" * 45)
    if passed == total:
        print(f"All tests passed! ({passed}/{total}) ✓")
    else:
        print(f"Some tests failed: ({passed}/{total})")
        sys.exit(1)


if __name__ == "__main__":
    main()
