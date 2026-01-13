# Academic Paper MCP Server v3

**Zotero-integrated MCP server for academic paper analysis with Claude Code.**

## Features

- **Zotero Integration** — Import papers directly from your Zotero library
- **Citation Key as ID** — Paper IDs match BibTeX `\cite{}` keys
- **Quality Extraction** — Local LLM extracts methodology, findings, limitations
- **Semantic Search** — Find relevant content across all papers
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

## Better BibTeX Setup

1. Install from: https://retorque.re/zotero-better-bibtex/
2. Configure citation key format:
   - Zotero → Edit → Settings → Better BibTeX
   - Set formula: `auth.lower + "_" + year + "_" + shorttitle(3, 3).lower`
3. Refresh all keys: Select all → Right-click → Better BibTeX → Refresh BibTeX key

## Usage

### Command Line

```bash
# List Zotero collections
python zotero_import.py --list-collections

# List items in a collection
python zotero_import.py --list-items "PhD Research"

# Import a collection
python zotero_import.py --collection "PhD Research"

# Import single paper
python zotero_import.py --item "lyons_2023_mixedcriticality"

# Sync metadata from Zotero
python zotero_import.py --sync
```

### Claude Code Integration

Add to MCP config (`~/.config/claude-code/mcp.json`):

```json
{
  "mcpServers": {
    "academic-papers": {
      "command": "/path/to/academic-paper-mcp/start_server.sh",
      "cwd": "/path/to/academic-paper-mcp"
    }
  }
}
```

### Example Queries in Claude Code

```
"List my Zotero collections"
→ zotero_list_collections

"Import my PhD Research collection"  
→ zotero_import_collection

"What methodology did \cite{lyons_2023_mixedcriticality} use?"
→ search_by_citation + get_methodology

"Compare findings between lyons_2023 and belt_2023"
→ compare_papers

"Search for papers about microkernel scheduling"
→ search_papers

"What does the results section say about performance?"
→ search_content with section_filter="results"
```

## MCP Tools

### Zotero Tools
| Tool | Description |
|------|-------------|
| `zotero_list_collections` | List all collections |
| `zotero_list_items` | List papers in collection |
| `zotero_import_collection` | Import entire collection |
| `zotero_import_item` | Import single paper |
| `zotero_sync` | Sync metadata |

### Search Tools
| Tool | Description |
|------|-------------|
| `search_papers` | Find papers by topic |
| `search_content` | Search within paper text |
| `search_by_citation` | Find by citation key |

### Retrieval Tools
| Tool | Description |
|------|-------------|
| `get_paper_metadata` | Get paper info |
| `get_methodology` | Get methodology extraction |
| `get_findings` | Get key findings |
| `get_limitations` | Get limitations |

### Analysis Tools
| Tool | Description |
|------|-------------|
| `query_paper` | Ask custom question |
| `reextract_field` | Re-extract with detail |
| `compare_papers` | Compare multiple papers |
| `get_paper_text` | Get raw text |

## Configuration

Edit `.env`:

```bash
# Use smaller model for less VRAM
ACADEMIC_LLM_MODEL=qwen2.5:3b

# Use larger model for better quality
ACADEMIC_LLM_MODEL=qwen2.5:7b
```

## Hardware Requirements

| Config | LLM | Embeddings | Total VRAM |
|--------|-----|------------|------------|
| Minimal | qwen2.5:3b | bge-base | ~3.5GB |
| **Recommended** | qwen2.5:3b | bge-large | ~5.5GB |
| High Quality | qwen2.5:7b | bge-base | ~6GB |

## Directory Structure

```
academic-paper-mcp/
├── src/
│   ├── models/         # Database + vector models
│   ├── processing/     # PDF, chunking, extraction
│   ├── zotero/        # Zotero integration
│   └── server.py      # MCP server
├── data/
│   ├── papers.db      # SQLite database
│   └── chroma/        # Vector embeddings
├── zotero_import.py   # CLI import tool
└── start_server.sh    # Server startup script
```

## License

MIT
