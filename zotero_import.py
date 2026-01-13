#!/usr/bin/env python3
"""
Zotero Import CLI - Import papers from Zotero into MCP database.

Usage:
    python zotero_import.py --list-collections
    python zotero_import.py --collection "PhD Research"
    python zotero_import.py --collection "PhD Research" --force
    python zotero_import.py --all
    python zotero_import.py --sync
    python zotero_import.py --stats
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.models.database import Database
from src.models.vectors import VectorStore
from src.processing.chunker import SemanticChunker
from src.processing.extractor import QualityExtractor
from src.processing.pdf_processor import PDFProcessor
from src.zotero import ZoteroReader, ZoteroSync


def print_header(text: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_progress(current: int, total: int, name: str):
    """Print progress callback."""
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\r  [{bar}] {current}/{total} - {name[:40]:<40}", end="", flush=True)
    if current == total:
        print()


def cmd_stats(reader: ZoteroReader):
    """Show Zotero statistics."""
    print_header("Zotero Statistics")
    
    stats = reader.get_stats()
    
    print(f"  Zotero Path:        {stats['zotero_path']}")
    print(f"  Total Items:        {stats['total_items']}")
    print(f"  Total Collections:  {stats['total_collections']}")
    print(f"  Better BibTeX:      {'✓ Installed' if stats['better_bibtex_installed'] else '✗ Not found'}")
    print(f"  Citation Keys:      {stats['items_with_citation_key']}")


def cmd_list_collections(reader: ZoteroReader):
    """List all collections."""
    print_header("Zotero Collections")
    
    collections = reader.get_collections()
    
    if not collections:
        print("  No collections found.")
        return
    
    for c in collections:
        print(f"  • {c.name} ({c.item_count} items)")


def cmd_list_items(reader: ZoteroReader, collection_name: str = None, limit: int = 20):
    """List items in a collection or library."""
    title = f"Items in '{collection_name}'" if collection_name else "All Items"
    print_header(title)
    
    items = reader.get_items(collection_name=collection_name, limit=limit)
    
    if not items:
        print("  No items found.")
        return
    
    for item in items:
        status = "📎" if item.has_pdf() else "📄"
        key = item.citation_key or "(no cite key)"
        print(f"  {status} {key}")
        print(f"     {item.title[:60]}{'...' if len(item.title or '') > 60 else ''}")
        print(f"     {item.get_formatted_authors()} ({item.year or 'n.d.'})")
        print()


def cmd_import_collection(
    sync: ZoteroSync,
    reader: ZoteroReader,
    collection_name: str,
    force: bool = False
):
    """Import a collection."""
    print_header(f"Importing Collection: {collection_name}")
    
    items = reader.get_items(collection_name=collection_name)
    
    if not items:
        print(f"  No items found in collection '{collection_name}'")
        return
    
    print(f"  Found {len(items)} items")
    print(f"  Force reprocess: {force}")
    print()
    
    results = sync.import_collection(
        collection_name=collection_name,
        force_reprocess=force,
        progress_callback=print_progress
    )
    
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
    
    # Show skip reasons breakdown
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
        print("\n  Failed items:")
        for r in failures:
            print(f"    • {r.paper_id}: {r.message}")
    
    # Show skipped (no PDF)
    no_pdf = [r for r in results if r.status == "skipped" and "No PDF" in (r.message or "")]
    if no_pdf:
        print(f"\n  Items without PDF ({len(no_pdf)}):")
        for r in no_pdf[:5]:
            print(f"    • {r.paper_id or r.citation_key}")
        if len(no_pdf) > 5:
            print(f"    ... and {len(no_pdf) - 5} more")


def cmd_import_all(sync: ZoteroSync, reader: ZoteroReader, force: bool = False):
    """Import all papers from Zotero."""
    print_header("Importing All Papers")
    
    items = reader.get_items()
    
    print(f"  Found {len(items)} items in library")
    print(f"  Force reprocess: {force}")
    
    confirm = input("\n  Proceed? (y/N): ")
    if confirm.lower() != 'y':
        print("  Cancelled.")
        return
    
    print()
    
    results = sync.import_all(
        force_reprocess=force,
        progress_callback=print_progress
    )
    
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
    
    # Show skip reasons breakdown
    skipped_results = [r for r in results if r.status == "skipped"]
    if skipped_results:
        skip_reasons = {}
        for r in skipped_results:
            reason = r.message or "Unknown"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        
        print("\n  Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    • {reason}: {count}")


def cmd_sync_metadata(sync: ZoteroSync):
    """Sync metadata for all imported papers."""
    print_header("Syncing Metadata from Zotero")
    
    results = sync.sync_metadata()
    
    updated = sum(1 for r in results if r.status == "updated")
    skipped = sum(1 for r in results if r.status == "skipped")
    
    print(f"  ↻ Updated: {updated}")
    print(f"  - Skipped: {skipped}")


def cmd_import_item(sync: ZoteroSync, reader: ZoteroReader, citation_key: str, force: bool = False):
    """Import a single item by citation key."""
    print_header(f"Importing: {citation_key}")
    
    item = reader.get_item_by_citation_key(citation_key)
    
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
    if result.time_seconds:
        print(f"  Time: {result.time_seconds:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Import papers from Zotero into MCP database"
    )
    
    # Commands
    parser.add_argument("--stats", action="store_true", help="Show Zotero statistics")
    parser.add_argument("--list-collections", action="store_true", help="List all collections")
    parser.add_argument("--list-items", type=str, metavar="COLLECTION", nargs="?", const="", help="List items (optionally in collection)")
    parser.add_argument("--collection", "-c", type=str, help="Import specific collection")
    parser.add_argument("--item", "-i", type=str, help="Import single item by citation key")
    parser.add_argument("--all", action="store_true", help="Import all papers")
    parser.add_argument("--sync", action="store_true", help="Sync metadata for imported papers")
    
    # Options
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocess existing papers")
    parser.add_argument("--zotero-path", type=Path, help="Path to Zotero data directory")
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--limit", type=int, default=20, help="Limit for --list-items")
    
    args = parser.parse_args()
    
    # Initialize components
    config = get_config()
    
    try:
        reader = ZoteroReader(args.zotero_path or config.zotero_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please specify Zotero path with --zotero-path or set ZOTERO_PATH in .env")
        sys.exit(1)
    
    # Check Better BibTeX
    if not reader.bbt_db_path.exists():
        print("Warning: Better BibTeX not found. Citation keys will not be available.")
        print("Install from: https://retorque.re/zotero-better-bibtex/")
    
    # Commands that don't need full initialization
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
    
    # Commands that need full initialization
    print("Initializing components...")
    
    db = Database(config.database_url)
    db.create_tables()
    
    vectors = VectorStore(
        persist_dir=config.chroma_persist_dir,
        embedding_model=config.embedding_model
    )
    
    processor = PDFProcessor()
    chunker = SemanticChunker()
    
    model = args.model or config.llm_model
    print(f"Using LLM model: {model}")
    
    extractor = QualityExtractor(model=model)
    
    sync = ZoteroSync(
        reader=reader,
        database=db,
        vector_store=vectors,
        extractor=extractor,
        pdf_processor=processor,
        chunker=chunker
    )
    
    # Run command
    if args.collection:
        cmd_import_collection(sync, reader, args.collection, args.force)
    elif args.item:
        cmd_import_item(sync, reader, args.item, args.force)
    elif args.all:
        cmd_import_all(sync, reader, args.force)
    elif args.sync:
        cmd_sync_metadata(sync)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()