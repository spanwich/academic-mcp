#!/usr/bin/env python3
"""
Zotero Import CLI for Academic Paper MCP v3.2

Usage:
    python zotero_import.py --stats
    python zotero_import.py --list-collections
    python zotero_import.py --list-items "seL4"
    python zotero_import.py --collection "seL4"
    python zotero_import.py --item "klein_2009_sel4"
    python zotero_import.py --all
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config, ensure_directories
from src.models.database import Database
from src.models.vectors import VectorStore
from src.processing.pdf_processor import PDFProcessor
from src.processing.section_detector import SectionDetector
from src.processing.chunker import PageChunker
from src.processing.extractor import SectionExtractor
from src.zotero.reader import ZoteroReader
from src.zotero.sync import ZoteroSync


def print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def cmd_stats(reader: ZoteroReader):
    """Show Zotero statistics."""
    stats = reader.get_stats()
    
    print_header("Zotero Statistics")
    print(f"  Total items:     {stats['total_items']}")
    print(f"  With PDFs:       {stats['with_pdfs']}")
    print(f"  Collections:     {stats['collection_count']}")
    print(f"  Better BibTeX:   {'✓' if stats['has_bbt'] else '✗'}")
    
    if stats.get('item_types'):
        print("\n  Item types:")
        for item_type, count in sorted(stats['item_types'].items(), key=lambda x: -x[1]):
            print(f"    {item_type}: {count}")


def cmd_list_collections(reader: ZoteroReader):
    """List all collections."""
    collections = reader.get_collections()
    
    print_header("Zotero Collections")
    for c in collections:
        print(f"  • {c.name} ({c.item_count} items)")


def cmd_list_items(reader: ZoteroReader, collection: str | None, limit: int):
    """List items in a collection."""
    items = reader.get_items(collection_name=collection if collection else None, limit=limit)
    
    title = f"Items in '{collection}'" if collection else "All Items"
    print_header(title)
    
    for item in items:
        pdf = "📎" if item.has_pdf() else "  "
        key = item.citation_key or "(no key)"
        print(f"  {pdf} {key}")
        print(f"      {item.title[:55]}{'...' if len(item.title or '') > 55 else ''}")
        print(f"      {item.get_formatted_authors()} ({item.year or 'n.d.'})")
        print()


def cmd_import_collection(sync: ZoteroSync, reader: ZoteroReader, collection: str, force: bool):
    """Import a collection."""
    items = reader.get_items(collection_name=collection)
    
    if not items:
        print(f"  No items found in collection '{collection}'")
        return
    
    print_header(f"Importing Collection: {collection}")
    print(f"  Found {len(items)} items\n")
    
    results = sync.import_collection(collection, force_reprocess=force)
    
    # Summary
    print()
    imported = sum(1 for r in results if r.status == "imported")
    updated = sum(1 for r in results if r.status == "updated")
    merged = sum(1 for r in results if r.status == "merged")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    
    print_header("Import Summary")
    print(f"  ✓ Imported: {imported}")
    print(f"  ↻ Updated:  {updated}")
    print(f"  ⊕ Merged:   {merged}")
    print(f"  - Skipped:  {skipped}")
    print(f"  ✗ Failed:   {failed}")
    
    # Show skip reasons
    skipped_results = [r for r in results if r.status == "skipped"]
    if skipped_results:
        skip_reasons = {}
        for r in skipped_results:
            reason = r.message or "Unknown"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        
        print("\n  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    • {reason}: {count}")
    
    # Show failures
    failures = [r for r in results if r.status == "failed"]
    if failures:
        print("\n  Failures:")
        for r in failures:
            print(f"    • {r.paper_id}: {r.message}")


def cmd_import_item(sync: ZoteroSync, reader: ZoteroReader, citation_key: str, force: bool):
    """Import a single item."""
    item = reader.get_item_by_citation_key(citation_key)
    
    print_header(f"Importing: {citation_key}")
    
    if not item:
        print(f"  Item not found: {citation_key}")
        return
    
    print(f"  Title: {item.title}")
    print(f"  Authors: {item.get_formatted_authors()}")
    print(f"  PDF: {'✓' if item.has_pdf() else '✗'}")
    print()
    
    result = sync.import_item(item, force_reprocess=force)
    
    print(f"  Status: {result.status}")
    if result.message:
        print(f"  Message: {result.message}")
    if result.page_count:
        print(f"  Pages: {result.page_count}")
    if result.section_count:
        print(f"  Sections: {result.section_count}")
    if result.time_seconds:
        print(f"  Time: {result.time_seconds:.1f}s")


def cmd_import_all(sync: ZoteroSync, reader: ZoteroReader, force: bool):
    """Import all papers."""
    items = reader.get_items()
    
    print_header("Importing All Papers")
    print(f"  Found {len(items)} items\n")
    
    # Confirmation
    response = input("  Continue? [y/N] ")
    if response.lower() != 'y':
        print("  Aborted.")
        return
    
    print()
    results = sync.import_all(force_reprocess=force)
    
    # Summary
    print()
    imported = sum(1 for r in results if r.status == "imported")
    updated = sum(1 for r in results if r.status == "updated")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    
    print_header("Import Summary")
    print(f"  ✓ Imported: {imported}")
    print(f"  ↻ Updated:  {updated}")
    print(f"  - Skipped:  {skipped}")
    print(f"  ✗ Failed:   {failed}")
    
    # Show skip reasons
    skipped_results = [r for r in results if r.status == "skipped"]
    if skipped_results:
        skip_reasons = {}
        for r in skipped_results:
            reason = r.message or "Unknown"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        
        print("\n  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    • {reason}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Import papers from Zotero into MCP database (v3.2)"
    )
    
    # Commands
    parser.add_argument("--stats", action="store_true", help="Show Zotero statistics")
    parser.add_argument("--list-collections", action="store_true", help="List all collections")
    parser.add_argument("--list-items", type=str, metavar="COLLECTION", nargs="?", const="", help="List items (optionally in collection)")
    parser.add_argument("--collection", "-c", type=str, help="Import specific collection")
    parser.add_argument("--item", "-i", type=str, help="Import single item by citation key")
    parser.add_argument("--all", action="store_true", help="Import all papers")
    
    # Options
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocess existing papers")
    parser.add_argument("--zotero-path", type=Path, help="Path to Zotero data directory")
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--limit", type=int, default=20, help="Limit for --list-items")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = get_config()
    ensure_directories()
    
    # Initialize Zotero reader
    try:
        reader = ZoteroReader(args.zotero_path or config.zotero_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please specify Zotero path with --zotero-path or set ACADEMIC_ZOTERO_PATH in .env")
        sys.exit(1)
    except RuntimeError as e:
        # Database lock error
        print(f"\n{'='*60}")
        print("  ERROR: Zotero Database Locked")
        print(f"{'='*60}")
        print(f"\n  {e}")
        print("\n  Please close Zotero application and try again.")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # Commands that only need reader
    if args.stats:
        cmd_stats(reader)
        return
    
    if args.list_collections:
        cmd_list_collections(reader)
        return
    
    if args.list_items is not None:
        collection = args.list_items if args.list_items else None
        cmd_list_items(reader, collection, args.limit)
        return
    
    # Commands that need full sync infrastructure
    if not any([args.collection, args.item, args.all]):
        parser.print_help()
        return
    
    print("Initializing components...")
    
    # Initialize database
    db = Database(config.database_url)
    db.create_tables()
    
    # Initialize vector store
    vectors = VectorStore(
        persist_directory=config.chroma_persist_dir,
        embedding_model=config.embedding_model
    )
    
    # Initialize processing components
    model = args.model or config.llm_model
    print(f"Using LLM model: {model}")
    
    pdf_processor = PDFProcessor()
    section_detector = SectionDetector(model=model, host=config.ollama_host)
    chunker = PageChunker()
    extractor = SectionExtractor(model=model, host=config.ollama_host)
    
    # Initialize sync
    sync = ZoteroSync(
        reader=reader,
        database=db,
        vector_store=vectors,
        section_detector=section_detector,
        extractor=extractor,
        pdf_processor=pdf_processor,
        chunker=chunker
    )
    
    # Run command
    try:
        if args.collection:
            cmd_import_collection(sync, reader, args.collection, args.force)
        elif args.item:
            cmd_import_item(sync, reader, args.item, args.force)
        elif args.all:
            cmd_import_all(sync, reader, args.force)
        else:
            parser.print_help()
    except RuntimeError as e:
        # Database lock error during operation
        print(f"\n{'='*60}")
        print("  ERROR: Zotero Database Locked")
        print(f"{'='*60}")
        print(f"\n  {e}")
        print("\n  Please close Zotero application and try again.")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
