# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic Paper MCP v3.3 is an MCP server that provides Claude Code with intelligent, chunked access to academic papers from a Zotero library. It solves the problem of LLMs being unable to process large documents (>30K tokens) by providing page-based chunking, semantic search, and pre-extracted content.

**Core capabilities:**
- Import papers from Zotero with citation keys as IDs
- Page-level access and semantic search across papers
- LLM-powered extraction of methodology, findings, limitations
- Self-organizing domain taxonomy for research classification

## Commands

```bash
# Start server (auto-starts Ollama if needed)
./start_server.sh

# Validate setup (database, vectors, Ollama, embeddings, Zotero)
python test_setup.py

# Import papers from Zotero
python zotero_import.py --collection "Collection Name"
python zotero_import.py --all

# Run tests
pytest tests/

# Format code
black src/
ruff check src/
```

## Architecture

```
src/
├── server.py              # MCP server entry point (20 tools)
├── config.py              # Pydantic settings from .env
├── models/
│   ├── database.py        # SQLAlchemy ORM (Paper, Section, Chunk, Domain, Extraction)
│   └── vectors.py         # ChromaDB wrapper for semantic search
├── processing/
│   ├── pdf_processor.py   # PyMuPDF text extraction with page boundaries
│   ├── section_detector.py# LLM-based document structure detection
│   ├── chunker.py         # Page-based chunking (1 page = 1 chunk)
│   ├── extractor.py       # Verbatim extraction of key content
│   ├── keyword_extractor.py # Keywords from PDF or LLM fallback
│   └── domain_classifier.py # Self-organizing taxonomy classification
└── zotero/
    ├── reader.py          # Read-only SQLite access to Zotero DB
    ├── sync.py            # Import/sync papers into MCP database
    └── models.py          # Zotero data structures
```

**Data flow:** Zotero → PDF Processing → SQLite + ChromaDB → MCP Tools → Claude Code

## MCP Tools (20 total)

| Category | Tools | Purpose |
|----------|-------|---------|
| Zotero | `zotero_list_collections`, `zotero_list_items`, `list_imported_papers` | Browse Zotero library |
| Discovery | `list_venues`, `list_domains` | Discover available filters |
| Search | `search_papers`, `search_content`, `search_by_citation` | Find papers/content |
| Section | `list_sections`, `get_section_summary`, `get_section_text`, `get_section_key_points` | Navigate document structure |
| Page | `get_page`, `get_pages` | Access specific pages |
| Retrieval | `get_paper_metadata`, `get_methodology`, `get_findings`, `get_limitations`, `get_statistics` | Get extracted content |
| Analysis | `query_paper` | Ask custom questions via LLM |

## Key Design Principles

1. **Citation key = paper_id** — IDs match BibTeX `\cite{}` keys for direct integration
2. **1 page = 1 chunk** — Simple, predictable, PDF-aligned chunking
3. **Verbatim extraction** — Copy original text to prevent hallucination
4. **Keywords trust tracking** — `keywords_source` field indicates "paper" (trustworthy) or "llm" (verify before citing)
5. **Self-organizing domains** — Specific, research-actionable domains created by LLM during import

## Configuration

Key environment variables in `.env`:

```bash
ACADEMIC_ZOTERO_PATH=~/Zotero
ACADEMIC_DATABASE_URL=sqlite:///data/papers.db
ACADEMIC_CHROMA_PERSIST_DIR=./data/chroma
ACADEMIC_LLM_MODEL=qwen2.5:3b
ACADEMIC_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
ACADEMIC_OLLAMA_HOST=http://localhost:11434
```

## Dependencies

- **MCP**: mcp>=1.0.0
- **Database**: SQLAlchemy 2.0+, ChromaDB
- **PDF**: PyMuPDF
- **LLM**: Ollama (local), sentence-transformers for embeddings
- **Python**: 3.10+

## Claude Code Integration

Add to MCP config (`~/.config/claude-code/mcp.json`):

```json
{
  "mcpServers": {
    "academic-papers": {
      "command": "/path/to/academic-mcp/start_server.sh",
      "cwd": "/path/to/academic-mcp"
    }
  }
}
```
