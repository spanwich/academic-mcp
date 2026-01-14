# Academic Paper MCP Server v3

**Zotero-integrated MCP server for academic paper analysis with Claude Code.**

## Architecture Overview

This MCP server provides **chunked access** to academic papers, solving the problem that LLMs cannot process very large documents (>30K tokens) in a single context window.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Academic Paper Database                      │
├─────────────────────────────────────────────────────────────────┤
│  Papers (119+)                                                   │
│  ├── Metadata (title, authors, abstract, DOI)                   │
│  ├── Full text (page-by-page chunks)                            │
│  ├── Sections (auto-detected with page ranges)                  │
│  ├── Extractions (findings, methodology, limitations)           │
│  └── Vector embeddings (semantic search)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Tools (20 tools)                        │
├─────────────────────────────────────────────────────────────────┤
│  Search      │ Page Access  │ Extractions  │ Analysis           │
│  ──────────  │ ───────────  │ ───────────  │ ────────           │
│  search_     │ get_page     │ get_findings │ query_paper        │
│  content     │ get_pages    │ get_method-  │ list_sections      │
│  search_     │              │ ology        │ get_section_       │
│  papers      │              │ get_limit-   │ summary            │
│              │              │ ations       │                    │
└─────────────────────────────────────────────────────────────────┘
```

### Complementary Systems

| System | Purpose | Best For |
|--------|---------|----------|
| **MCP Database** | Raw text + chunked access | Large documents, page-level quotes, semantic search |
| **Summaries Folder** (`papers/_shared/summaries/`) | Claude-synthesized notes | Quick reference, "Cite For" guidance, curated key points |

**Why both exist:**
- Claude Code cannot process documents >30K tokens at once
- Example: Anna Lyons' thesis (214 pages, 69K words) cannot be summarized in one pass
- The MCP database allows accessing such documents in chunks

## Features

- **Zotero Integration** — Import papers directly from your Zotero library
- **Citation Key as ID** — Paper IDs match BibTeX `\cite{}` keys
- **Page-Level Access** — Read specific pages or page ranges
- **Semantic Search** — Find relevant content across all papers
- **Section Detection** — Auto-detected document structure with page ranges
- **LLM Extraction** — Local Ollama extracts methodology, findings, limitations
- **On-demand Analysis** — Ask custom questions about any paper

## Prerequisites

1. **Zotero** with papers in your library
2. **Better BibTeX** plugin for Zotero (recommended)
3. **Ollama** for local LLM inference
4. **Python 3.10+**

## Installation

```bash
# Clone/copy the project
cd academic-paper-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (if not already)
curl -fsSL https://ollama.com/install.sh | sh

# Pull LLM model
ollama pull qwen2.5:3b

# Copy environment config
cp .env.example .env

# Test setup
python test_setup.py
```

## Starting the Server

```bash
./start_server.sh
```

The script will:
1. Check if Ollama is running, start it if needed
2. Verify the LLM model is available
3. Activate the Python virtual environment
4. Start the MCP server

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

## Usage Patterns

### Pattern 1: Small-Medium Papers (<50 pages)

For typical conference/journal papers, use high-level extraction tools:

```
# Get paper overview
get_paper_metadata("klein_2009_sel4formalverification")

# Get pre-extracted content
get_findings("klein_2009_sel4formalverification")
get_methodology("klein_2009_sel4formalverification")
get_limitations("klein_2009_sel4formalverification")

# Ask custom questions
query_paper("klein_2009_sel4formalverification", "What is the proof size?")
```

### Pattern 2: Large Documents (50+ pages)

For theses, books, or handbooks that exceed LLM context limits:

```
# Step 1: Understand document structure
list_sections("lyons_2018_mixedcriticalityschedulingresource")
→ Shows 10 sections across 214 pages

# Step 2: Use semantic search to find relevant content
search_content("scheduling context donation IPC", paper_id="lyons_2018_...")
→ Returns relevant chunks with page numbers and scores

# Step 3: Read specific pages of interest
get_pages("lyons_2018_...", start_page=73, end_page=75)
→ Full text of pages 73-75

# Step 4: Get section-level summaries
get_section_summary("lyons_2018_...", "methodology")
```

### Pattern 3: Cross-Paper Research

```
# Find papers on a topic
search_papers("microkernel scheduling")

# Semantic search across ALL papers
search_content("temporal isolation mechanisms")
→ Returns chunks from multiple papers with relevance scores

# Compare specific papers
get_findings("paper_a")
get_findings("paper_b")
```

### Pattern 4: Citation Workflow

```
# When you see \cite{key} in LaTeX
search_by_citation("klein_2009_sel4")
→ Returns full metadata and abstract

# Get citable quotes with page numbers
get_findings("klein_2009_sel4formalverification")
→ Returns verbatim quotes with "Page X" citations
```

## MCP Tools Reference

### Zotero Tools
| Tool | Description |
|------|-------------|
| `zotero_list_collections` | List all Zotero collections with paper counts |
| `zotero_list_items` | List papers in a collection (shows import status) |
| `list_imported_papers` | List all papers in MCP database |

### Search Tools
| Tool | Description | Use When |
|------|-------------|----------|
| `search_papers` | Text search in titles/abstracts | Finding papers by topic |
| `search_content` | Semantic search in full text | Finding specific concepts across papers |
| `search_by_citation` | Find by citation key | Looking up `\cite{key}` references |

### Page Access Tools
| Tool | Description | Use When |
|------|-------------|----------|
| `get_page` | Get single page text | Reading a specific page |
| `get_pages` | Get page range text | Reading a section of a large document |

### Section Tools
| Tool | Description | Use When |
|------|-------------|----------|
| `list_sections` | Document structure with page ranges | Understanding document organization |
| `get_section_summary` | LLM summary of a section | Quick overview of a section |
| `get_section_text` | Full original text of section | Reading complete section |
| `get_section_key_points` | Verbatim key points from section | Getting quotable content |

### Extraction Tools
| Tool | Description | Use When |
|------|-------------|----------|
| `get_paper_metadata` | Title, authors, abstract, DOI, etc. | Paper overview |
| `get_methodology` | Methodology extraction with quotes | Understanding approach |
| `get_findings` | Results and contributions | Key paper outcomes |
| `get_limitations` | Author-stated limitations | Critical analysis |
| `get_statistics` | Quantitative results | Performance data |

### Analysis Tools
| Tool | Description | Use When |
|------|-------------|----------|
| `query_paper` | Ask custom question about a paper | Specific queries not covered by extractions |

## Example: Working with a Large Thesis

The Anna Lyons thesis (214 pages, 69,088 words) demonstrates the chunked access pattern:

```python
# 1. Get overview
get_paper_metadata("lyons_2018_mixedcriticalityschedulingresource")
→ Title: Mixed-criticality scheduling and resource sharing...
→ Pages: 214, Words: 69088

# 2. Understand structure
list_sections("lyons_2018_mixedcriticalityschedulingresource")
→ abstract (pages 1-1)
→ introduction (pages 2-3)
→ background (pages 4-7)
→ methodology (pages 8-15)
→ ... (10 sections total)

# 3. Search for specific concepts
search_content("scheduling context donation", paper_id="lyons_2018_...")
→ Page 73: "timeslice donation...avoided the scheduler" (score: 0.76)
→ Page 111: "timeout exception handler" (score: 0.75)
→ Page 74: "scheduling contexts with donation over IPC" (score: 0.74)

# 4. Read the relevant pages
get_pages("lyons_2018_...", start_page=73, end_page=75)
→ Full text of pages discussing scheduling context donation

# 5. Get specific extraction
get_findings("lyons_2018_mixedcriticalityschedulingresource")
→ Key contributions with page citations
```

## Known Limitations

1. **Section Detection**: May misclassify sections in unusually structured documents
2. **Extraction Quality**: Depends on Ollama model size (qwen2.5:3b vs 7b)
3. **Large Documents**: Pre-extracted findings may be sparse; use page access instead
4. **Venue Detection**: Often shows "Unknown" - metadata comes from Zotero

## Configuration

Edit `.env`:

```bash
# LLM model for extraction (smaller = faster, larger = better quality)
ACADEMIC_LLM_MODEL=qwen2.5:3b

# Ollama host
OLLAMA_HOST=http://localhost:11434

# Embedding model for semantic search
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

## Hardware Requirements

| Config | LLM | Embeddings | Total VRAM |
|--------|-----|------------|------------|
| Minimal | qwen2.5:3b | bge-base | ~3.5GB |
| **Recommended** | qwen2.5:3b | bge-large | ~5.5GB |
| High Quality | qwen2.5:7b | bge-base | ~6GB |

## Directory Structure

```
academic-mcp/
├── src/
│   ├── models/         # Database + vector models
│   ├── processing/     # PDF, chunking, extraction
│   ├── zotero/         # Zotero integration
│   └── server.py       # MCP server (20 tools)
├── data/
│   ├── papers.db       # SQLite database
│   └── chroma/         # Vector embeddings
├── start_server.sh     # Server startup script
└── README.md           # This file
```

## Better BibTeX Setup

For consistent citation keys:

1. Install from: https://retorque.re/zotero-better-bibtex/
2. Configure citation key format:
   - Zotero → Edit → Settings → Better BibTeX
   - Set formula: `auth.lower + "_" + year + "_" + shorttitle(3, 3).lower`
3. Refresh all keys: Select all → Right-click → Better BibTeX → Refresh BibTeX key

## License

MIT
