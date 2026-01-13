#!/usr/bin/env python3
"""
Academic Paper Processing MCP Server
v3: Zotero-integrated with citation key as paper_id
"""

import asyncio
import json
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .config import get_config
from .models.database import Database, Paper, Extraction, Chunk, ProcessingStatus
from .models.vectors import VectorStore, ChunkDocument, PaperSummary, SectionType
from .utils.ollama_utils import ensure_ollama_ready

# Initialize
config = get_config()
server = Server("academic-papers")

# Auto-start Ollama
print("Checking Ollama status...")
if ensure_ollama_ready(model=config.llm_model, auto_start=True):
    print(f"✓ Ollama ready with model: {config.llm_model}")
else:
    print(f"⚠ Ollama not available - LLM features will fail")

# Initialize database and vectors
database = Database(config.database_url)
database.create_tables()

vector_store = VectorStore(
    persist_dir=config.chroma_persist_dir,
    embedding_model=config.embedding_model
)

# Lazy-loaded components
_zotero_reader = None
_zotero_sync = None


def get_zotero_reader():
    """Get or create ZoteroReader."""
    global _zotero_reader
    if _zotero_reader is None:
        from .zotero import ZoteroReader
        _zotero_reader = ZoteroReader(config.zotero_path)
    return _zotero_reader


def get_zotero_sync():
    """Get or create ZoteroSync."""
    global _zotero_sync
    if _zotero_sync is None:
        from .zotero import ZoteroSync
        from .processing import PDFProcessor, SemanticChunker, QualityExtractor
        
        _zotero_sync = ZoteroSync(
            reader=get_zotero_reader(),
            database=database,
            vector_store=vector_store,
            extractor=QualityExtractor(model=config.llm_model),
            pdf_processor=PDFProcessor(),
            chunker=SemanticChunker()
        )
    return _zotero_sync


@server.list_tools()
async def list_tools():
    """List available MCP tools."""
    return [
        # === Zotero Tools ===
        Tool(
            name="zotero_list_collections",
            description="List all Zotero collections with paper counts.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        Tool(
            name="zotero_list_items",
            description="List papers in a Zotero collection or entire library.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection name (empty for all)"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20
                    }
                }
            }
        ),
        
        Tool(
            name="zotero_import_collection",
            description="Import all papers from a Zotero collection. Extracts text, runs LLM analysis, and indexes for search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection name to import"
                    },
                    "force": {
                        "type": "boolean",
                        "default": False,
                        "description": "Force re-import existing papers"
                    }
                },
                "required": ["collection"]
            }
        ),
        
        Tool(
            name="zotero_import_item",
            description="Import a single paper by citation key (e.g., 'lyons_2023_mixedcriticality').",
            inputSchema={
                "type": "object",
                "properties": {
                    "citation_key": {
                        "type": "string",
                        "description": "BibTeX citation key"
                    },
                    "force": {
                        "type": "boolean",
                        "default": False
                    }
                },
                "required": ["citation_key"]
            }
        ),
        
        Tool(
            name="zotero_sync",
            description="Sync metadata from Zotero for all imported papers (doesn't re-run LLM extraction).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        # === Search Tools ===
        Tool(
            name="search_papers",
            description="Search for papers by topic. Returns paper summaries ranked by relevance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collection": {"type": "string", "description": "Limit to Zotero collection"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="search_content",
            description="Search within paper content using semantic search + LLM reranking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "section_filter": {
                        "type": "string",
                        "enum": ["abstract", "introduction", "methodology", "results", "discussion", "conclusion"]
                    },
                    "paper_id": {"type": "string", "description": "Limit to specific paper (citation key)"},
                    "top_k": {"type": "integer", "default": 5},
                    "synthesize": {"type": "boolean", "default": True}
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="search_by_citation",
            description="Find a paper by its BibTeX citation key (e.g., from \\cite{key} in LaTeX).",
            inputSchema={
                "type": "object",
                "properties": {
                    "citation_key": {"type": "string"}
                },
                "required": ["citation_key"]
            }
        ),
        
        # === Retrieval Tools ===
        Tool(
            name="list_papers",
            description="List all imported papers in the database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "Filter by Zotero collection"},
                    "limit": {"type": "integer", "default": 50}
                }
            }
        ),
        
        Tool(
            name="get_paper_metadata",
            description="Get full metadata for a paper (title, authors, abstract, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Citation key"}
                },
                "required": ["paper_id"]
            }
        ),
        
        Tool(
            name="get_methodology",
            description="Get detailed methodology extraction for a paper.",
            inputSchema={
                "type": "object",
                "properties": {"paper_id": {"type": "string"}},
                "required": ["paper_id"]
            }
        ),
        
        Tool(
            name="get_findings",
            description="Get key findings and results for a paper.",
            inputSchema={
                "type": "object",
                "properties": {"paper_id": {"type": "string"}},
                "required": ["paper_id"]
            }
        ),
        
        Tool(
            name="get_limitations",
            description="Get limitations and critical analysis for a paper.",
            inputSchema={
                "type": "object",
                "properties": {"paper_id": {"type": "string"}},
                "required": ["paper_id"]
            }
        ),
        
        # === On-demand Analysis Tools ===
        Tool(
            name="get_paper_text",
            description="Get full raw text of a paper. Use when extracted info isn't enough.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 50000}
                },
                "required": ["paper_id"]
            }
        ),
        
        Tool(
            name="query_paper",
            description="Ask a custom question about a specific paper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "question": {"type": "string"}
                },
                "required": ["paper_id", "question"]
            }
        ),
        
        Tool(
            name="reextract_field",
            description="Re-extract a specific field with more detail.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "field": {
                        "type": "string",
                        "enum": ["methodology", "findings", "limitations", "statistics"]
                    },
                    "focus": {"type": "string", "description": "Specific aspect to focus on"}
                },
                "required": ["paper_id", "field"]
            }
        ),
        
        Tool(
            name="compare_papers",
            description="Compare multiple papers on specific aspects.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of citation keys"
                    },
                    "aspect": {
                        "type": "string",
                        "enum": ["methodology", "findings", "limitations", "all"]
                    }
                },
                "required": ["paper_ids", "aspect"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle MCP tool calls."""
    
    try:
        # Zotero tools
        if name == "zotero_list_collections":
            return await _zotero_list_collections()
        elif name == "zotero_list_items":
            return await _zotero_list_items(arguments)
        elif name == "zotero_import_collection":
            return await _zotero_import_collection(arguments)
        elif name == "zotero_import_item":
            return await _zotero_import_item(arguments)
        elif name == "zotero_sync":
            return await _zotero_sync_metadata()
        
        # Search tools
        elif name == "search_papers":
            return await _search_papers(arguments)
        elif name == "search_content":
            return await _search_content(arguments)
        elif name == "search_by_citation":
            return await _search_by_citation(arguments)
        
        # Retrieval tools
        elif name == "list_papers":
            return await _list_papers(arguments)
        elif name == "get_paper_metadata":
            return await _get_paper_metadata(arguments)
        elif name == "get_methodology":
            return await _get_extraction_field(arguments["paper_id"], "methodology")
        elif name == "get_findings":
            return await _get_extraction_field(arguments["paper_id"], "findings")
        elif name == "get_limitations":
            return await _get_extraction_field(arguments["paper_id"], "limitations")
        
        # On-demand tools
        elif name == "get_paper_text":
            return await _get_paper_text(arguments)
        elif name == "query_paper":
            return await _query_paper(arguments)
        elif name == "reextract_field":
            return await _reextract_field(arguments)
        elif name == "compare_papers":
            return await _compare_papers(arguments)
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error: {str(e)}\n{traceback.format_exc()}")]


# === Zotero Tool Implementations ===

async def _zotero_list_collections():
    """List Zotero collections."""
    reader = get_zotero_reader()
    collections = reader.get_collections()
    
    if not collections:
        return [TextContent(type="text", text="No collections found in Zotero.")]
    
    output = "## Zotero Collections\n\n"
    for c in collections:
        output += f"• **{c.name}** ({c.item_count} items)\n"
    
    return [TextContent(type="text", text=output)]


async def _zotero_list_items(args: dict):
    """List items in a collection."""
    reader = get_zotero_reader()
    collection = args.get("collection")
    limit = args.get("limit", 20)
    
    items = reader.get_items(collection_name=collection if collection else None, limit=limit)
    
    if not items:
        return [TextContent(type="text", text="No items found.")]
    
    # Check which are already imported
    with database.get_session() as session:
        imported_ids = {p.paper_id for p in session.query(Paper.paper_id).all()}
    
    title = f"## Items in '{collection}'" if collection else "## All Items"
    output = f"{title}\n\n"
    
    for item in items:
        status = "✓" if item.citation_key in imported_ids else "○"
        pdf = "📎" if item.has_pdf() else ""
        key = item.citation_key or "(no key)"
        output += f"{status} {pdf} **{key}**\n"
        output += f"   {item.title[:60]}{'...' if len(item.title or '') > 60 else ''}\n"
        output += f"   {item.get_formatted_authors()} ({item.year or 'n.d.'})\n\n"
    
    output += f"\n✓ = imported, ○ = not imported, 📎 = has PDF"
    
    return [TextContent(type="text", text=output)]


async def _zotero_import_collection(args: dict):
    """Import a collection."""
    sync = get_zotero_sync()
    reader = get_zotero_reader()
    
    collection = args["collection"]
    force = args.get("force", False)
    
    items = reader.get_items(collection_name=collection)
    
    if not items:
        return [TextContent(type="text", text=f"No items found in collection '{collection}'")]
    
    output = f"## Importing '{collection}'\n\n"
    output += f"Found {len(items)} items. Processing...\n\n"
    
    results = sync.import_collection(collection, force_reprocess=force)
    
    imported = sum(1 for r in results if r.status == "imported")
    updated = sum(1 for r in results if r.status == "updated")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    
    output += f"**Results:**\n"
    output += f"• ✓ Imported: {imported}\n"
    output += f"• ↻ Updated: {updated}\n"
    output += f"• − Skipped: {skipped}\n"
    output += f"• ✗ Failed: {failed}\n"
    
    failures = [r for r in results if r.status == "failed"]
    if failures:
        output += f"\n**Failures:**\n"
        for r in failures[:5]:
            output += f"• {r.paper_id}: {r.message}\n"
    
    return [TextContent(type="text", text=output)]


async def _zotero_import_item(args: dict):
    """Import single item."""
    sync = get_zotero_sync()
    reader = get_zotero_reader()
    
    citation_key = args["citation_key"]
    force = args.get("force", False)
    
    item = reader.get_item_by_citation_key(citation_key)
    
    if not item:
        return [TextContent(type="text", text=f"Item not found: {citation_key}")]
    
    result = sync.import_item(item, force_reprocess=force)
    
    output = f"## Import: {citation_key}\n\n"
    output += f"**Title:** {item.title}\n"
    output += f"**Status:** {result.status}\n"
    if result.message:
        output += f"**Message:** {result.message}\n"
    if result.time_seconds:
        output += f"**Time:** {result.time_seconds:.1f}s\n"
    
    return [TextContent(type="text", text=output)]


async def _zotero_sync_metadata():
    """Sync metadata from Zotero."""
    sync = get_zotero_sync()
    results = sync.sync_metadata()
    
    updated = sum(1 for r in results if r.status == "updated")
    skipped = sum(1 for r in results if r.status == "skipped")
    
    return [TextContent(
        type="text",
        text=f"## Metadata Sync Complete\n\n• Updated: {updated}\n• Skipped: {skipped}"
    )]


# === Search Tool Implementations ===

async def _search_papers(args: dict):
    """Search for papers."""
    query = args["query"]
    top_k = args.get("top_k", 5)
    collection = args.get("collection")
    
    results = vector_store.search_papers(query=query, top_k=top_k * 2)
    
    if not results:
        return [TextContent(type="text", text="No relevant papers found.")]
    
    # Filter by collection if specified
    if collection:
        with database.get_session() as session:
            papers_in_collection = {
                p.paper_id for p in session.query(Paper).filter(
                    Paper.zotero_collections.contains(collection)
                ).all()
            }
        results = [r for r in results if r.document.paper_id in papers_in_collection]
    
    results = results[:top_k]
    
    output = f"## Papers matching: '{query}'\n\n"
    for i, r in enumerate(results, 1):
        doc = r.document
        output += f"{i}. **{doc.title}** (`{doc.paper_id}`)\n"
        output += f"   Domain: {doc.research_domain or 'unknown'}\n"
        output += f"   Similarity: {r.similarity:.2f}\n\n"
    
    return [TextContent(type="text", text=output)]


async def _search_content(args: dict):
    """Search within paper content."""
    import ollama
    
    query = args["query"]
    top_k = args.get("top_k", 5)
    section_filter = args.get("section_filter")
    paper_filter = args.get("paper_id")
    synthesize = args.get("synthesize", True)
    
    # Convert section filter
    section_enum = None
    if section_filter:
        section_enum = SectionType(section_filter)
    
    results = vector_store.search_chunks(
        query=query,
        top_k=top_k * 3,
        section_filter=section_enum,
        paper_filter=paper_filter
    )
    
    if not results:
        return [TextContent(type="text", text="No relevant content found.")]
    
    results = results[:top_k]
    
    output = ""
    
    # Synthesize answer
    if synthesize:
        context = "\n\n---\n\n".join([
            f"[{r.document.paper_id} - {r.document.section_type.value}]\n{r.document.content[:1500]}"
            for r in results
        ])
        
        prompt = f"""Based on these excerpts from academic papers, answer the query.

Query: {query}

Excerpts:
{context}

Provide a clear, comprehensive answer citing the sources by paper_id."""
        
        response = ollama.generate(
            model=config.llm_model,
            prompt=prompt,
            options={"temperature": 0.3, "num_predict": 2000}
        )
        
        output += f"## Answer\n\n{response['response']}\n\n"
    
    output += "## Sources\n\n"
    for i, r in enumerate(results, 1):
        doc = r.document
        output += f"{i}. `{doc.paper_id}` ({doc.section_type.value}) - sim: {r.similarity:.2f}\n"
    
    return [TextContent(type="text", text=output)]


async def _search_by_citation(args: dict):
    """Find paper by citation key."""
    citation_key = args["citation_key"]
    
    with database.get_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == citation_key).first()
        
        if not paper:
            # Try to find in Zotero
            reader = get_zotero_reader()
            item = reader.get_item_by_citation_key(citation_key)
            
            if item:
                return [TextContent(
                    type="text",
                    text=f"Paper `{citation_key}` found in Zotero but not imported.\n\n"
                         f"**Title:** {item.title}\n"
                         f"**Authors:** {item.get_formatted_authors()}\n\n"
                         f"Use `zotero_import_item` to import it."
                )]
            
            return [TextContent(type="text", text=f"Paper not found: {citation_key}")]
        
        output = f"## {paper.title}\n\n"
        output += f"**Citation key:** `{paper.paper_id}`\n"
        output += f"**Authors:** {', '.join(paper.authors) if paper.authors else 'Unknown'}\n"
        output += f"**Year:** {paper.year or 'Unknown'}\n"
        output += f"**Collections:** {', '.join(paper.zotero_collections) if paper.zotero_collections else 'None'}\n\n"
        output += f"**Abstract:** {paper.abstract[:500] if paper.abstract else 'Not available'}...\n"
        
        return [TextContent(type="text", text=output)]


# === Retrieval Tool Implementations ===

async def _list_papers(args: dict):
    """List imported papers."""
    limit = args.get("limit", 50)
    collection = args.get("collection")
    
    with database.get_session() as session:
        query = session.query(Paper)
        
        if collection:
            # Filter by collection (JSON contains)
            query = query.filter(Paper.zotero_collections.contains([collection]))
        
        papers = query.limit(limit).all()
        
        if not papers:
            return [TextContent(type="text", text="No papers imported yet.")]
        
        output = f"## Imported Papers ({len(papers)})\n\n"
        for p in papers:
            collections = ", ".join(p.zotero_collections) if p.zotero_collections else ""
            output += f"• **{p.paper_id}**: {p.title[:50]}...\n"
            output += f"  {p.year or 'n.d.'} | {collections}\n\n"
        
        return [TextContent(type="text", text=output)]


async def _get_paper_metadata(args: dict):
    """Get paper metadata."""
    paper_id = args["paper_id"]
    
    with database.get_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == paper_id).first()
        
        if not paper:
            return [TextContent(type="text", text=f"Paper not found: {paper_id}")]
        
        output = f"# {paper.title}\n\n"
        output += f"**Citation key:** `{paper.paper_id}`\n"
        output += f"**Authors:** {', '.join(paper.authors) if paper.authors else 'Unknown'}\n"
        output += f"**Year:** {paper.year or 'Unknown'}\n"
        output += f"**Venue:** {paper.journal_or_venue or 'Unknown'}\n"
        output += f"**DOI:** {paper.doi or 'None'}\n"
        output += f"**Collections:** {', '.join(paper.zotero_collections) if paper.zotero_collections else 'None'}\n"
        output += f"**Pages:** {paper.page_count} | **Words:** {paper.word_count}\n\n"
        output += f"## Abstract\n\n{paper.abstract or 'Not available'}\n"
        
        if paper.extraction:
            e = paper.extraction
            output += f"\n**Domain:** {e.research_domain or 'Unknown'}\n"
            output += f"**Type:** {e.paper_type or 'Unknown'}\n"
            output += f"**Keywords:** {', '.join(e.keywords) if e.keywords else 'None'}\n"
        
        return [TextContent(type="text", text=output)]


async def _get_extraction_field(paper_id: str, field_type: str):
    """Get specific extraction field."""
    with database.get_session() as session:
        extraction = session.query(Extraction).filter(
            Extraction.paper_id == paper_id
        ).first()
        
        if not extraction:
            return [TextContent(type="text", text=f"Paper not found or not processed: {paper_id}")]
        
        if field_type == "methodology":
            output = f"# Methodology: {paper_id}\n\n"
            output += f"## Summary\n{extraction.methodology_summary or 'Not available'}\n\n"
            output += f"## Study Design\n{extraction.study_design or 'Not available'}\n\n"
            output += f"## Sample\n{extraction.sample_description or 'Not available'}\n"
            output += f"Size: {extraction.sample_size or 'Not specified'}\n\n"
            output += f"## Data Collection\n{', '.join(extraction.data_collection_methods) if extraction.data_collection_methods else 'Not available'}\n\n"
            output += f"## Analysis Methods\n{', '.join(extraction.analysis_methods) if extraction.analysis_methods else 'Not available'}\n\n"
            output += f"## Statistical Tests\n{', '.join(extraction.statistical_tests) if extraction.statistical_tests else 'Not available'}\n\n"
            output += f"## Detailed Methodology\n{extraction.methodology_detailed or 'Not available'}\n"
        
        elif field_type == "findings":
            output = f"# Findings: {paper_id}\n\n"
            
            if extraction.key_findings:
                output += "## Key Findings\n\n"
                for i, finding in enumerate(extraction.key_findings, 1):
                    if isinstance(finding, dict):
                        output += f"{i}. **{finding.get('finding', 'N/A')}**\n"
                        output += f"   Evidence: {finding.get('evidence', 'N/A')}\n\n"
                    else:
                        output += f"{i}. {finding}\n"
            
            if extraction.effect_sizes:
                output += "\n## Effect Sizes\n\n"
                for es in extraction.effect_sizes:
                    if isinstance(es, dict):
                        output += f"• {es.get('measure', 'N/A')}: {es.get('value', 'N/A')} ({es.get('context', '')})\n"
        
        elif field_type == "limitations":
            output = f"# Limitations: {paper_id}\n\n"
            
            if extraction.limitations:
                for i, lim in enumerate(extraction.limitations, 1):
                    if isinstance(lim, dict):
                        output += f"{i}. **{lim.get('limitation', 'N/A')}**\n"
                        output += f"   Impact: {lim.get('impact', 'N/A')}\n"
                        ack = "Yes" if lim.get('acknowledged') else "No"
                        output += f"   Acknowledged by authors: {ack}\n\n"
                    else:
                        output += f"{i}. {lim}\n"
            else:
                output += "No limitations extracted.\n"
            
            output += f"\n## Future Research\n"
            if extraction.future_research:
                for fr in extraction.future_research:
                    output += f"• {fr}\n"
            else:
                output += "Not available\n"
        
        return [TextContent(type="text", text=output)]


# === On-demand Analysis Implementations ===

async def _get_paper_text(args: dict):
    """Get full paper text."""
    paper_id = args["paper_id"]
    max_chars = args.get("max_chars", 50000)
    
    with database.get_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == paper_id).first()
        
        if not paper:
            return [TextContent(type="text", text=f"Paper not found: {paper_id}")]
        
        if paper.full_text:
            text = paper.full_text[:max_chars]
            truncated = len(paper.full_text) > max_chars
        else:
            chunks = session.query(Chunk).filter(
                Chunk.paper_id == paper_id
            ).order_by(Chunk.chunk_index).all()
            
            text = "\n\n".join(c.content for c in chunks)[:max_chars]
            truncated = len("\n\n".join(c.content for c in chunks)) > max_chars
        
        header = f"# Full Text: {paper.title}\n\n"
        if truncated:
            header += f"*Truncated to {max_chars:,} characters*\n\n"
        header += "---\n\n"
        
        return [TextContent(type="text", text=header + text)]


async def _query_paper(args: dict):
    """Ask custom question about paper."""
    import ollama
    
    paper_id = args["paper_id"]
    question = args["question"]
    
    with database.get_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == paper_id).first()
        
        if not paper:
            return [TextContent(type="text", text=f"Paper not found: {paper_id}")]
        
        text = paper.full_text[:25000] if paper.full_text else ""
        
        if not text:
            chunks = session.query(Chunk).filter(
                Chunk.paper_id == paper_id
            ).order_by(Chunk.chunk_index).all()
            text = "\n\n".join(c.content for c in chunks)[:25000]
    
    prompt = f"""Answer this question about the academic paper.

Paper: {paper.title}
Question: {question}

Paper text:
{text}

Provide a detailed, accurate answer. If the information is not in the paper, say so."""

    response = ollama.generate(
        model=config.llm_model,
        prompt=prompt,
        options={"temperature": 0.2, "num_predict": 2000}
    )
    
    return [TextContent(type="text", text=f"## {question}\n\n{response['response']}")]


async def _reextract_field(args: dict):
    """Re-extract field with more detail."""
    from .processing import QualityExtractor
    
    paper_id = args["paper_id"]
    field = args["field"]
    focus = args.get("focus", "")
    
    with database.get_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == paper_id).first()
        
        if not paper:
            return [TextContent(type="text", text=f"Paper not found: {paper_id}")]
        
        text = paper.full_text[:35000] if paper.full_text else ""
    
    extractor = QualityExtractor(model=config.llm_model)
    result = extractor.reextract_field(text, field, focus)
    
    return [TextContent(type="text", text=f"## Re-extracted: {field}\n\n{result}")]


async def _compare_papers(args: dict):
    """Compare multiple papers."""
    import ollama
    
    paper_ids = args["paper_ids"]
    aspect = args["aspect"]
    
    papers_data = []
    
    with database.get_session() as session:
        for pid in paper_ids:
            paper = session.query(Paper).filter(Paper.paper_id == pid).first()
            if paper and paper.extraction:
                e = paper.extraction
                data = {
                    "paper_id": pid,
                    "title": paper.title,
                    "year": paper.year
                }
                
                if aspect in ["methodology", "all"]:
                    data["methodology"] = e.methodology_summary
                    data["study_design"] = e.study_design
                    data["sample_size"] = e.sample_size
                
                if aspect in ["findings", "all"]:
                    data["key_findings"] = e.key_findings[:3] if e.key_findings else []
                
                if aspect in ["limitations", "all"]:
                    data["limitations"] = e.limitations[:3] if e.limitations else []
                
                papers_data.append(data)
    
    if len(papers_data) < 2:
        return [TextContent(type="text", text="Need at least 2 papers to compare.")]
    
    prompt = f"""Compare these academic papers on {aspect}:

{json.dumps(papers_data, indent=2)}

Provide a structured comparison:
1. Key similarities
2. Key differences  
3. Relative strengths and weaknesses
4. Summary recommendation"""

    response = ollama.generate(
        model=config.llm_model,
        prompt=prompt,
        options={"temperature": 0.3, "num_predict": 2500}
    )
    
    return [TextContent(type="text", text=f"## Comparison: {aspect}\n\n{response['response']}")]


# === Main Entry Point ===

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
