"""
Academic Paper MCP Server v3.3

MCP server for academic paper analysis with:
- Zotero integration
- Page-based chunking
- Section-aware retrieval
- Verbatim extraction
- Keywords & domain search (NEW in v3.3)
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Suppress library warnings (not stdout, just warnings)
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, ensure_directories
from src.models.database import Database, Paper, Section, Chunk, Extraction, Domain
from src.models.vectors import VectorStore
from src.zotero.reader import ZoteroReader

# Initialize components
config = get_config()
ensure_directories()
database = Database(config.database_url)
database.create_tables()
vector_store = VectorStore(
    persist_directory=config.chroma_persist_dir,
    embedding_model=config.embedding_model,
    embedding_backend=config.embedding_backend,
    ollama_host=config.ollama_host
)

# Lazy-loaded components
_zotero_reader = None


def get_zotero_reader() -> ZoteroReader:
    global _zotero_reader
    if _zotero_reader is None:
        _zotero_reader = ZoteroReader(config.zotero_path)
    return _zotero_reader


# Create server
server = Server("academic-papers")


# === Tool Definitions ===

TOOLS = [
    # Zotero Tools
    Tool(
        name="zotero_list_collections",
        description="List all Zotero collections with paper counts",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="zotero_list_items",
        description="List papers in a Zotero collection (from Zotero database)",
        inputSchema={
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "Collection name (optional)"},
                "limit": {"type": "integer", "description": "Max items to return", "default": 100}
            }
        }
    ),
    Tool(
        name="list_imported_papers",
        description="List all papers imported into the MCP database (with extraction status)",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max items to return", "default": 200}
            }
        }
    ),
    
    # Search Tools
    Tool(
        name="search_papers",
        description="""Find papers by multiple criteria. All filters are optional and combinable.
        
Examples:
- search_papers(query="verification") - text search in title/abstract
- search_papers(venue="NDSS") - papers from NDSS conference
- search_papers(domain="side-channel") - papers in domains containing "side-channel"
- search_papers(author="Heiser") - papers by author
- search_papers(year=2024) - papers from 2024
- search_papers(venue="RTSS", year_from=2020) - RTSS papers from 2020+
- search_papers(keywords=["seL4", "microkernel"]) - papers with these keywords""",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text search in title/abstract"},
                "venue": {"type": "string", "description": "Filter by venue (partial match, e.g., 'NDSS' matches full conference name)"},
                "domain": {"type": "string", "description": "Filter by domain (partial match)"},
                "author": {"type": "string", "description": "Filter by author name (partial match)"},
                "year": {"type": "integer", "description": "Filter by exact publication year"},
                "year_from": {"type": "integer", "description": "Filter by year range start (inclusive)"},
                "year_to": {"type": "integer", "description": "Filter by year range end (inclusive)"},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "Filter by keywords (matches if paper has ANY of these)"},
                "limit": {"type": "integer", "description": "Max results to return", "default": 50}
            }
        }
    ),
    
    # Discovery Tools (NEW in v3.3)
    Tool(
        name="list_venues",
        description="List all venues (conferences/journals) in the database with paper counts. Use this to discover venue names for filtering.",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="list_domains",
        description="List all research domains in the database with paper counts. Domains are specific and research-actionable (e.g., 'microkernel formal verification using Isabelle').",
        inputSchema={"type": "object", "properties": {}}
    ),
    
    Tool(
        name="search_content",
        description="Semantic search within paper content. Returns relevant chunks with page numbers.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "paper_id": {"type": "string", "description": "Limit to specific paper (optional)"},
                "section_type": {"type": "string", "description": "Limit to section type (optional)"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 10}
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="search_by_citation",
        description="Find paper by citation key (e.g., from \\cite{key})",
        inputSchema={
            "type": "object",
            "properties": {
                "citation_key": {"type": "string", "description": "Citation key like 'klein_2009_sel4'"}
            },
            "required": ["citation_key"]
        }
    ),
    
    # Section Tools
    Tool(
        name="list_sections",
        description="Get document structure (sections with page ranges)",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"}
            },
            "required": ["paper_id"]
        }
    ),
    Tool(
        name="get_section_summary",
        description="Get LLM summary of a specific section",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"},
                "section_type": {"type": "string", "description": "Section type (introduction, methodology, results, etc.)"}
            },
            "required": ["paper_id", "section_type"]
        }
    ),
    Tool(
        name="get_section_text",
        description="Get full original text of a section",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"},
                "section_type": {"type": "string", "description": "Section type"}
            },
            "required": ["paper_id", "section_type"]
        }
    ),
    Tool(
        name="get_section_key_points",
        description="Get verbatim key points extracted from a section",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"},
                "section_type": {"type": "string", "description": "Section type"}
            },
            "required": ["paper_id", "section_type"]
        }
    ),
    
    # Page Tools
    Tool(
        name="get_page",
        description="Get text of a single page",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"},
                "page": {"type": "integer", "description": "Page number (1-indexed)"}
            },
            "required": ["paper_id", "page"]
        }
    ),
    Tool(
        name="get_pages",
        description="Get text of a page range",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"},
                "start_page": {"type": "integer", "description": "Start page (1-indexed)"},
                "end_page": {"type": "integer", "description": "End page (inclusive)"}
            },
            "required": ["paper_id", "start_page", "end_page"]
        }
    ),
    
    # Retrieval Tools
    Tool(
        name="get_paper_metadata",
        description="Get paper metadata (title, authors, abstract, etc.)",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"}
            },
            "required": ["paper_id"]
        }
    ),
    Tool(
        name="get_methodology",
        description="Get methodology extraction (verbatim + summary)",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"}
            },
            "required": ["paper_id"]
        }
    ),
    Tool(
        name="get_findings",
        description="Get results and findings (verbatim quotes with page numbers)",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"}
            },
            "required": ["paper_id"]
        }
    ),
    Tool(
        name="get_limitations",
        description="Get limitations and future work (author-stated only)",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"}
            },
            "required": ["paper_id"]
        }
    ),
    Tool(
        name="get_statistics",
        description="Get statistics (only those actually in the paper)",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"}
            },
            "required": ["paper_id"]
        }
    ),
    
    # On-Demand Analysis
    Tool(
        name="query_paper",
        description="Ask a custom question about a paper",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Paper citation key"},
                "question": {"type": "string", "description": "Your question"}
            },
            "required": ["paper_id", "question"]
        }
    ),
]


@server.list_tools()
async def list_tools():
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    
    # Zotero tools
    if name == "zotero_list_collections":
        return await _zotero_list_collections()
    elif name == "zotero_list_items":
        return await _zotero_list_items(arguments)
    elif name == "list_imported_papers":
        return await _list_imported_papers(arguments)
    
    # Search tools
    elif name == "search_papers":
        return await _search_papers(arguments)
    elif name == "search_content":
        return await _search_content(arguments)
    elif name == "search_by_citation":
        return await _search_by_citation(arguments)
    
    # Discovery tools (NEW in v3.3)
    elif name == "list_venues":
        return await _list_venues()
    elif name == "list_domains":
        return await _list_domains()
    
    # Section tools
    elif name == "list_sections":
        return await _list_sections(arguments)
    elif name == "get_section_summary":
        return await _get_section_summary(arguments)
    elif name == "get_section_text":
        return await _get_section_text(arguments)
    elif name == "get_section_key_points":
        return await _get_section_key_points(arguments)
    
    # Page tools
    elif name == "get_page":
        return await _get_page(arguments)
    elif name == "get_pages":
        return await _get_pages(arguments)
    
    # Retrieval tools
    elif name == "get_paper_metadata":
        return await _get_paper_metadata(arguments)
    elif name == "get_methodology":
        return await _get_methodology(arguments)
    elif name == "get_findings":
        return await _get_findings(arguments)
    elif name == "get_limitations":
        return await _get_limitations(arguments)
    elif name == "get_statistics":
        return await _get_statistics(arguments)
    
    # Analysis tools
    elif name == "query_paper":
        return await _query_paper(arguments)
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# === Tool Implementations ===

async def _zotero_list_collections():
    """List Zotero collections."""
    try:
        reader = get_zotero_reader()
        collections = reader.get_collections()
        
        if not collections:
            return [TextContent(type="text", text="No collections found in Zotero.")]
        
        output = "## Zotero Collections\n\n"
        for c in collections:
            output += f"• **{c.name}** ({c.item_count} items)\n"
        
        return [TextContent(type="text", text=output)]
    except RuntimeError as e:
        return [TextContent(type="text", text=f"Error: {e}\n\nPlease close Zotero and try again.")]


async def _zotero_list_items(args: dict):
    """List items in a collection."""
    try:
        reader = get_zotero_reader()
        collection = args.get("collection")
        limit = args.get("limit", 100)  # Increased default from 20 to 100
        
        items = reader.get_items(collection_name=collection if collection else None, limit=limit)
        
        if not items:
            return [TextContent(type="text", text="No items found.")]
        
        # Check which are imported
        with database.get_session() as session:
            imported_ids = {p.paper_id for p in session.query(Paper.paper_id).all()}
        
        title = f"## Items in '{collection}'" if collection else "## All Items"
        output = f"{title} (showing {len(items)}, {len(imported_ids)} imported in database)\n\n"
        
        for item in items:
            status = "✓" if item.citation_key in imported_ids else "○"
            pdf = "📎" if item.has_pdf() else ""
            key = item.citation_key or "(no key)"
            output += f"{status} {pdf} **{key}**\n"
            output += f"   {item.title[:60]}{'...' if len(item.title or '') > 60 else ''}\n"
            output += f"   {item.get_formatted_authors()} ({item.year or 'n.d.'})\n\n"
        
        output += "\n✓ = imported, ○ = not imported, 📎 = has PDF"
        
        return [TextContent(type="text", text=output)]
    except RuntimeError as e:
        return [TextContent(type="text", text=f"Error: {e}\n\nPlease close Zotero and try again.")]


async def _list_imported_papers(args: dict):
    """List all imported papers from database."""
    limit = args.get("limit", 200)
    
    with database.get_session() as session:
        papers = session.query(Paper).order_by(Paper.paper_id).limit(limit).all()
        
        if not papers:
            return [TextContent(type="text", text="No papers imported yet.")]
        
        output = f"## Imported Papers ({len(papers)} total)\n\n"
        
        for p in papers:
            sections = len(p.sections) if p.sections else 0
            pages = p.page_count or 0
            output += f"• **{p.paper_id}** ({pages} pages, {sections} sections)\n"
            output += f"  {p.title[:60]}{'...' if len(p.title or '') > 60 else ''}\n"
        
        return [TextContent(type="text", text=output)]


async def _search_papers(args: dict):
    """Search papers by multiple criteria."""
    query = args.get("query")
    venue = args.get("venue")
    domain = args.get("domain")
    author = args.get("author")
    year = args.get("year")
    year_from = args.get("year_from")
    year_to = args.get("year_to")
    keywords = args.get("keywords", [])
    limit = args.get("limit", 50)
    
    with database.get_session() as session:
        # Start with base query
        q = session.query(Paper)
        
        # Apply filters
        filters_applied = []
        
        # Text search on title/abstract
        if query:
            q = q.filter(
                (Paper.title.ilike(f"%{query}%")) |
                (Paper.abstract.ilike(f"%{query}%"))
            )
            filters_applied.append(f"query='{query}'")
        
        # Venue filter (partial match)
        if venue:
            q = q.filter(Paper.journal_or_venue.ilike(f"%{venue}%"))
            filters_applied.append(f"venue='{venue}'")
        
        # Domain filter (partial match)
        if domain:
            q = q.filter(Paper.domain.ilike(f"%{domain}%"))
            filters_applied.append(f"domain='{domain}'")
        
        # Author filter (search in JSON array)
        if author:
            # SQLite JSON search - check if any author matches
            q = q.filter(Paper.authors.cast(str).ilike(f"%{author}%"))
            filters_applied.append(f"author='{author}'")
        
        # Year filters
        if year:
            q = q.filter(Paper.year == year)
            filters_applied.append(f"year={year}")
        if year_from:
            q = q.filter(Paper.year >= year_from)
            filters_applied.append(f"year_from={year_from}")
        if year_to:
            q = q.filter(Paper.year <= year_to)
            filters_applied.append(f"year_to={year_to}")
        
        # Keywords filter (match any)
        if keywords:
            keyword_conditions = []
            for kw in keywords:
                keyword_conditions.append(Paper.keywords.cast(str).ilike(f"%{kw}%"))
            if keyword_conditions:
                from sqlalchemy import or_
                q = q.filter(or_(*keyword_conditions))
            filters_applied.append(f"keywords={keywords}")
        
        # Execute query
        papers = q.order_by(Paper.year.desc().nullslast()).limit(limit).all()
        
        if not papers:
            filters_str = ", ".join(filters_applied) if filters_applied else "no filters"
            return [TextContent(type="text", text=f"No papers found with filters: {filters_str}")]
        
        # Format output
        filters_str = ", ".join(filters_applied) if filters_applied else "none"
        output = f"## Papers ({len(papers)} results)\n**Filters:** {filters_str}\n\n"
        
        for p in papers:
            output += f"### {p.paper_id}\n"
            output += f"**{p.title}**\n"
            output += f"*{', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}* ({p.year or 'n.d.'})\n"
            
            if p.journal_or_venue:
                output += f"**Venue:** {p.journal_or_venue}\n"
            if p.domain:
                output += f"**Domain:** {p.domain}\n"
            if p.keywords:
                source = f" (from {p.keywords_source})" if p.keywords_source else ""
                output += f"**Keywords:** {', '.join(p.keywords[:5])}{source}\n"
            
            output += "\n"
        
        return [TextContent(type="text", text=output)]


async def _list_venues():
    """List all venues with paper counts."""
    with database.get_session() as session:
        from sqlalchemy import func
        
        # Get venues with counts, excluding nulls
        results = session.query(
            Paper.journal_or_venue,
            func.count(Paper.paper_id).label('count')
        ).filter(
            Paper.journal_or_venue.isnot(None),
            Paper.journal_or_venue != ''
        ).group_by(
            Paper.journal_or_venue
        ).order_by(
            func.count(Paper.paper_id).desc()
        ).all()
        
        if not results:
            return [TextContent(type="text", text="No venues found in database.")]
        
        output = f"## Venues in Database ({len(results)} total)\n\n"
        output += "Use `search_papers(venue=\"...\")` to find papers from a venue.\n\n"
        
        for venue, count in results:
            output += f"• **{venue}** ({count} paper{'s' if count != 1 else ''})\n"
        
        return [TextContent(type="text", text=output)]


async def _list_domains():
    """List all domains with paper counts."""
    with database.get_session() as session:
        # Get domains from Domain table
        domains = session.query(Domain).order_by(Domain.paper_count.desc()).all()
        
        if not domains:
            # Fallback: get from papers directly
            from sqlalchemy import func
            results = session.query(
                Paper.domain,
                func.count(Paper.paper_id).label('count')
            ).filter(
                Paper.domain.isnot(None),
                Paper.domain != ''
            ).group_by(
                Paper.domain
            ).order_by(
                func.count(Paper.paper_id).desc()
            ).all()
            
            if not results:
                return [TextContent(type="text", text="No domains found. Papers may need re-import for domain classification.")]
            
            output = f"## Research Domains ({len(results)} total)\n\n"
            output += "Use `search_papers(domain=\"...\")` to find papers in a domain.\n\n"
            
            for domain, count in results:
                output += f"• **{domain}** ({count} paper{'s' if count != 1 else ''})\n"
            
            return [TextContent(type="text", text=output)]
        
        output = f"## Research Domains ({len(domains)} total)\n\n"
        output += "Use `search_papers(domain=\"...\")` to find papers in a domain.\n"
        output += "Domains are specific and research-actionable.\n\n"
        
        for d in domains:
            output += f"• **{d.name}** ({d.paper_count} paper{'s' if d.paper_count != 1 else ''})\n"
        
        return [TextContent(type="text", text=output)]


async def _search_content(args: dict):
    """Semantic search within papers."""
    query = args["query"]
    paper_id = args.get("paper_id")
    section_type = args.get("section_type")
    top_k = args.get("top_k", 10)
    
    results = vector_store.search(
        query=query,
        top_k=top_k,
        filter_paper_id=paper_id,
        filter_section_type=section_type
    )
    
    if not results:
        return [TextContent(type="text", text=f"No results for '{query}'")]
    
    output = f"## Search results for '{query}'\n\n"
    
    for r in results:
        meta = r["metadata"]
        output += f"### {meta.get('paper_id', 'unknown')} - Page {meta.get('page_number', '?')}\n"
        output += f"*Section: {meta.get('section_type', 'unknown')} | Score: {r['score']:.2f}*\n\n"
        
        # Show content preview
        content = r.get("content", "")[:500]
        output += f"{content}{'...' if len(r.get('content', '')) > 500 else ''}\n\n"
        output += "---\n\n"
    
    return [TextContent(type="text", text=output)]


async def _search_by_citation(args: dict):
    """Find paper by citation key."""
    key = args["citation_key"]
    
    with database.get_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == key).first()
        
        if not paper:
            return [TextContent(type="text", text=f"Paper not found: {key}")]
        
        output = f"## {key}\n\n"
        output += f"**Title:** {paper.title}\n\n"
        output += f"**Authors:** {', '.join(paper.authors)}\n\n"
        output += f"**Year:** {paper.year or 'n.d.'}\n\n"
        output += f"**Venue:** {paper.journal_or_venue or 'Unknown'}\n\n"
        output += f"**Pages:** {paper.page_count}\n\n"
        
        if paper.abstract:
            output += f"**Abstract:**\n{paper.abstract}\n"
        
        return [TextContent(type="text", text=output)]


async def _list_sections(args: dict):
    """List sections of a paper."""
    paper_id = args["paper_id"]
    
    with database.get_session() as session:
        sections = session.query(Section).filter(
            Section.paper_id == paper_id
        ).order_by(Section.section_index).all()
        
        if not sections:
            return [TextContent(type="text", text=f"No sections found for {paper_id}")]
        
        output = f"## Document Structure: {paper_id}\n\n"
        
        for s in sections:
            output += f"• **{s.section_type}** (pages {s.page_start}-{s.page_end})\n"
            if s.section_title:
                output += f"  *{s.section_title}*\n"
        
        return [TextContent(type="text", text=output)]


async def _get_section_summary(args: dict):
    """Get section summary."""
    paper_id = args["paper_id"]
    section_type = args["section_type"]
    
    with database.get_session() as session:
        section = session.query(Section).filter(
            Section.paper_id == paper_id,
            Section.section_type == section_type
        ).first()
        
        if not section:
            return [TextContent(type="text", text=f"Section '{section_type}' not found in {paper_id}")]
        
        output = f"## {section_type.title()} Summary: {paper_id}\n\n"
        output += f"*Pages {section.page_start}-{section.page_end}*\n\n"
        output += section.summary or "*No summary available*"
        
        return [TextContent(type="text", text=output)]


async def _get_section_text(args: dict):
    """Get full section text."""
    paper_id = args["paper_id"]
    section_type = args["section_type"]
    
    with database.get_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == paper_id).first()
        section = session.query(Section).filter(
            Section.paper_id == paper_id,
            Section.section_type == section_type
        ).first()
        
        if not paper or not section:
            return [TextContent(type="text", text=f"Section '{section_type}' not found in {paper_id}")]
        
        text = paper.full_text[section.char_start:section.char_end]
        
        output = f"## {section_type.title()}: {paper_id}\n\n"
        output += f"*Pages {section.page_start}-{section.page_end}*\n\n"
        output += text
        
        return [TextContent(type="text", text=output)]


async def _get_section_key_points(args: dict):
    """Get section key points."""
    paper_id = args["paper_id"]
    section_type = args["section_type"]
    
    with database.get_session() as session:
        section = session.query(Section).filter(
            Section.paper_id == paper_id,
            Section.section_type == section_type
        ).first()
        
        if not section:
            return [TextContent(type="text", text=f"Section '{section_type}' not found in {paper_id}")]
        
        output = f"## Key Points from {section_type.title()}: {paper_id}\n\n"
        
        if section.key_points_verbatim:
            for i, point in enumerate(section.key_points_verbatim, 1):
                text = point.get("text", str(point))
                page = point.get("page", "?")
                output += f"{i}. \"{text}\"\n   *— Page {page}*\n\n"
        else:
            output += "*No key points extracted*"
        
        return [TextContent(type="text", text=output)]


async def _get_page(args: dict):
    """Get single page text."""
    paper_id = args["paper_id"]
    page = args["page"]
    
    with database.get_session() as session:
        chunk = session.query(Chunk).filter(
            Chunk.paper_id == paper_id,
            Chunk.page_number == page
        ).first()
        
        if not chunk:
            return [TextContent(type="text", text=f"Page {page} not found in {paper_id}")]
        
        output = f"## Page {page}: {paper_id}\n\n"
        output += chunk.content
        
        return [TextContent(type="text", text=output)]


async def _get_pages(args: dict):
    """Get page range text."""
    paper_id = args["paper_id"]
    start = args["start_page"]
    end = args["end_page"]
    
    with database.get_session() as session:
        chunks = session.query(Chunk).filter(
            Chunk.paper_id == paper_id,
            Chunk.page_number >= start,
            Chunk.page_number <= end
        ).order_by(Chunk.page_number).all()
        
        if not chunks:
            return [TextContent(type="text", text=f"Pages {start}-{end} not found in {paper_id}")]
        
        output = f"## Pages {start}-{end}: {paper_id}\n\n"
        
        for chunk in chunks:
            output += f"### Page {chunk.page_number}\n\n"
            output += chunk.content
            output += "\n\n"
        
        return [TextContent(type="text", text=output)]


async def _get_paper_metadata(args: dict):
    """Get paper metadata."""
    paper_id = args["paper_id"]
    
    with database.get_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == paper_id).first()
        
        if not paper:
            return [TextContent(type="text", text=f"Paper not found: {paper_id}")]
        
        output = f"## {paper_id}\n\n"
        output += f"**Title:** {paper.title}\n\n"
        output += f"**Authors:** {', '.join(paper.authors)}\n\n"
        output += f"**Year:** {paper.year or 'n.d.'}\n\n"
        output += f"**Venue:** {paper.journal_or_venue or 'Unknown'}\n\n"
        
        # NEW in v3.3: Domain and keywords
        if paper.domain:
            output += f"**Domain:** {paper.domain}\n\n"
        
        if paper.keywords:
            source_note = f" (from {paper.keywords_source})" if paper.keywords_source else ""
            output += f"**Keywords:** {', '.join(paper.keywords)}{source_note}\n\n"
        
        output += f"**DOI:** {paper.doi or 'None'}\n\n"
        output += f"**Pages:** {paper.page_count}\n\n"
        output += f"**Words:** {paper.word_count}\n\n"
        output += f"**Collections:** {', '.join(paper.zotero_collections) if paper.zotero_collections else 'None'}\n\n"
        
        if paper.abstract:
            output += f"**Abstract:**\n{paper.abstract}\n"
        
        return [TextContent(type="text", text=output)]


async def _get_methodology(args: dict):
    """Get methodology extraction."""
    paper_id = args["paper_id"]
    
    with database.get_session() as session:
        extraction = session.query(Extraction).filter(
            Extraction.paper_id == paper_id
        ).first()
        
        if not extraction:
            return [TextContent(type="text", text=f"No extraction found for {paper_id}")]
        
        output = f"## Methodology: {paper_id}\n\n"
        
        if extraction.methodology_verbatim:
            output += "### Methodology (from paper)\n\n"
            output += f"{extraction.methodology_verbatim}\n\n"
        
        if extraction.evaluation_setup_verbatim:
            output += "### Evaluation Setup (from paper)\n\n"
            output += f"{extraction.evaluation_setup_verbatim}\n\n"
        
        if extraction.software_tools:
            output += f"### Tools\n{', '.join(extraction.software_tools)}\n\n"
        
        if extraction.methodology_summary:
            output += f"---\n*LLM Summary:* {extraction.methodology_summary}\n"
        
        return [TextContent(type="text", text=output)]


async def _get_findings(args: dict):
    """Get findings extraction."""
    paper_id = args["paper_id"]
    
    with database.get_session() as session:
        extraction = session.query(Extraction).filter(
            Extraction.paper_id == paper_id
        ).first()
        
        if not extraction:
            return [TextContent(type="text", text=f"No extraction found for {paper_id}")]
        
        output = f"## Findings: {paper_id}\n\n"
        
        if extraction.contributions_verbatim:
            output += "### Contributions (from paper)\n\n"
            for i, item in enumerate(extraction.contributions_verbatim, 1):
                text = item.get("text", str(item))
                section = item.get("section", "?")
                page = item.get("page", "?")
                output += f"{i}. \"{text}\"\n   *— {section}, Page {page}*\n\n"
        
        if extraction.results_verbatim:
            output += "### Results (from paper)\n\n"
            for i, item in enumerate(extraction.results_verbatim, 1):
                text = item.get("text", str(item))
                section = item.get("section", "?")
                page = item.get("page", "?")
                output += f"{i}. \"{text}\"\n   *— {section}, Page {page}*\n\n"
        
        return [TextContent(type="text", text=output)]


async def _get_limitations(args: dict):
    """Get limitations extraction."""
    paper_id = args["paper_id"]
    
    with database.get_session() as session:
        extraction = session.query(Extraction).filter(
            Extraction.paper_id == paper_id
        ).first()
        
        if not extraction:
            return [TextContent(type="text", text=f"No extraction found for {paper_id}")]
        
        output = f"## Limitations: {paper_id}\n\n"
        
        if extraction.limitations_verbatim:
            output += "### Limitations (from paper)\n\n"
            for i, item in enumerate(extraction.limitations_verbatim, 1):
                text = item.get("text", str(item))
                section = item.get("section", "?")
                page = item.get("page", "?")
                output += f"{i}. \"{text}\"\n   *— {section}, Page {page}*\n\n"
        else:
            output += "*No limitations explicitly stated by authors*\n\n"
        
        if extraction.future_work_verbatim:
            output += "### Future Work (from paper)\n\n"
            for item in extraction.future_work_verbatim:
                text = item.get("text", str(item))
                section = item.get("section", "?")
                page = item.get("page", "?")
                output += f"• \"{text}\" *— {section}, Page {page}*\n"
        
        return [TextContent(type="text", text=output)]


async def _get_statistics(args: dict):
    """Get statistics extraction."""
    paper_id = args["paper_id"]
    
    with database.get_session() as session:
        extraction = session.query(Extraction).filter(
            Extraction.paper_id == paper_id
        ).first()
        
        if not extraction:
            return [TextContent(type="text", text=f"No extraction found for {paper_id}")]
        
        output = f"## Statistics: {paper_id}\n\n"
        
        if extraction.statistics_verbatim:
            output += "*These are exact quotes from the paper:*\n\n"
            for item in extraction.statistics_verbatim:
                text = item.get("text", str(item))
                section = item.get("section", "?")
                page = item.get("page", "?")
                output += f"• \"{text}\"\n  *— {section}, Page {page}*\n\n"
        else:
            output += "*No quantitative statistics found in paper*"
        
        return [TextContent(type="text", text=output)]


async def _query_paper(args: dict):
    """Custom query about a paper."""
    paper_id = args["paper_id"]
    question = args["question"]
    
    with database.get_session() as session:
        paper = session.query(Paper).filter(Paper.paper_id == paper_id).first()
        
        if not paper or not paper.full_text:
            return [TextContent(type="text", text=f"Paper not found or no text: {paper_id}")]
        
        # Use extractor for custom query
        from src.processing.extractor import SectionExtractor
        extractor = SectionExtractor(model=config.llm_model, host=config.ollama_host)
        
        answer = extractor.extract_custom(paper.full_text, question)
        
        output = f"## Answer: {paper_id}\n\n"
        output += f"**Question:** {question}\n\n"
        output += f"**Answer:**\n{answer}"
        
        return [TextContent(type="text", text=output)]


# === Main ===

import signal

# Track signal count for force exit
_signal_count = 0


def _cleanup():
    """Cleanup resources on shutdown."""
    import sys
    print("\nShutting down Academic Paper MCP Server...", file=sys.stderr)
    
    # Close database connections
    try:
        if database.engine:
            database.engine.dispose()
            print("  ✓ Database connections closed", file=sys.stderr)
    except Exception as e:
        print(f"  ✗ Database cleanup error: {e}", file=sys.stderr)
    
    print("Goodbye!", file=sys.stderr)


def _signal_handler(signum, frame):
    """Handle shutdown signals."""
    global _signal_count
    _signal_count += 1
    
    import sys
    sig_name = signal.Signals(signum).name
    
    if _signal_count == 1:
        print(f"\n{sig_name} received. Shutting down... (press again to force)", file=sys.stderr)
        _cleanup()
        sys.exit(0)
    else:
        # Force exit on second signal
        print(f"\nForce exit.", file=sys.stderr)
        os._exit(1)


async def main():
    # Register signal handlers (simple approach that works)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except Exception:
        pass
    finally:
        _cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Already handled by signal handler
