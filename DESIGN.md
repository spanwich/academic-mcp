# Academic Paper Processing MCP Server

## Design Document v3: Zotero-Integrated Quality System

**Goal:** Reduce Claude Code context consumption when analyzing academic papers by pre-processing, indexing, and retrieving relevant content on demand. Integrate with Zotero as the single source of truth for bibliography management.

**Design Priorities:**
1. Zotero as single source of truth
2. Citation key = paper_id (direct `\cite{}` integration)
3. Quality over speed
4. On-demand import (user controls what gets processed)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 CLAUDE CODE                                      │
│                                                                                  │
│   "What methodology did \cite{lyons_2023_mixedcriticality} use?"                │
│   "Compare findings in my 'PhD Research' collection"                            │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │ MCP Protocol
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            MCP SERVER (Python)                                   │
│                                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │    Zotero     │  │    Search     │  │   Retrieve    │  │   Analysis    │    │
│  │    Tools      │  │    Tools      │  │   Tools       │  │   Tools       │    │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘    │
│                                                                                  │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│     ZOTERO      │         │   MCP Database  │         │     Ollama      │
│  (Read Only)    │         │   (SQLAlchemy)  │         │   Local LLM     │
│                 │         │                 │         │                 │
│ • zotero.sqlite │         │ • papers.db     │         │ • qwen2.5:3b    │
│ • better-bibtex │         │ • ChromaDB      │         │ • bge-large     │
│ • storage/      │         │                 │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
        │                           │
        │    PDF Files              │    Processed Data
        ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│ ~/Zotero/       │         │ data/           │
│ storage/        │         │ ├── papers.db   │
│ XXXXXXXX/       │         │ └── chroma/     │
│ paper.pdf       │         │                 │
└─────────────────┘         └─────────────────┘
```

---

## Zotero Integration Details

### Database Locations

| Database | Path | Purpose |
|----------|------|---------|
| Zotero Main | `~/Zotero/zotero.sqlite` | Items, metadata, collections |
| Better BibTeX | `~/Zotero/better-bibtex.sqlite` | Citation keys |
| PDF Storage | `~/Zotero/storage/{itemKey}/` | PDF files |

### Better BibTeX Configuration

**Citation Key Format:**
```
auth.lower + "_" + year + "_" + shorttitle(3, 3).lower
```

**Example Output:**
```
lyons_2023_mixedcriticality
belt_2023_modeldrivendevelopment
brusilovsky_2020_5gneedplatform
```

### Data Flow: Zotero → MCP

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ZOTERO DATABASES                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  better-bibtex.sqlite                      zotero.sqlite                │
│  ┌─────────────────────┐                  ┌─────────────────────┐       │
│  │ citationkey table   │                  │ items               │       │
│  │                     │                  │ itemData            │       │
│  │ itemID ─────────────┼──────────────────│ itemDataValues      │       │
│  │ itemKey             │                  │ itemAttachments     │       │
│  │ citationKey ────────┼──► paper_id      │ collections         │       │
│  └─────────────────────┘                  │ collectionItems     │       │
│                                           └─────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MCP DATABASE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  papers table                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ paper_id (= citation_key)     ← Links to \cite{} in LaTeX       │   │
│  │ zotero_key (itemKey)          ← Links back to Zotero            │   │
│  │ zotero_item_id                ← Internal Zotero ID              │   │
│  │ file_hash                     ← Deduplication key               │   │
│  │ zotero_collections            ← ["PhD Research", "seL4"]        │   │
│  │                                                                  │   │
│  │ title, authors, abstract      ← FROM ZOTERO (trusted)           │   │
│  │ doi, year, journal            ← FROM ZOTERO (trusted)           │   │
│  │                                                                  │   │
│  │ methodology_*, findings_*     ← FROM LLM (3 passes)             │   │
│  │ limitations_*, statistics_*   ← FROM LLM (3 passes)             │   │
│  │                                                                  │   │
│  │ full_text                     ← For on-demand queries           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model (SQLAlchemy ORM)

### papers table

```python
class Paper(Base):
    __tablename__ = "papers"
    
    # Primary identifier (= BibTeX citation key)
    paper_id: Mapped[str] = mapped_column(primary_key=True)
    # e.g., "lyons_2023_mixedcriticality"
    
    # Zotero linking
    zotero_key: Mapped[Optional[str]] = mapped_column(index=True)
    # e.g., "HTJHSCCZ" (storage folder)
    zotero_item_id: Mapped[Optional[int]]
    # e.g., 1318
    zotero_collections: Mapped[list[str]] = mapped_column(JSON, default=list)
    # e.g., ["PhD Research", "seL4 Papers"]
    
    # Deduplication
    file_path: Mapped[str]
    file_hash: Mapped[Optional[str]] = mapped_column(index=True)
    # SHA256 hash for detecting same PDF from different sources
    
    # Metadata FROM ZOTERO (trusted, not LLM-extracted)
    title: Mapped[Optional[str]] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    publication_date: Mapped[Optional[str]]
    journal_or_venue: Mapped[Optional[str]]
    doi: Mapped[Optional[str]]
    
    # Processing metadata
    imported_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    processed_at: Mapped[Optional[datetime]]
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        default=ProcessingStatus.PENDING
    )
    processing_model: Mapped[Optional[str]]
    
    # Document stats
    page_count: Mapped[Optional[int]]
    word_count: Mapped[Optional[int]]
    
    # Full text storage (for on-demand queries)
    full_text: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationships
    extraction: Mapped[Optional["Extraction"]] = relationship(...)
    chunks: Mapped[list["Chunk"]] = relationship(...)
```

### extractions table

```python
class Extraction(Base):
    __tablename__ = "extractions"
    
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        primary_key=True
    )
    
    # Methodology (LLM Pass 1)
    methodology_summary: Mapped[Optional[str]] = mapped_column(Text)
    methodology_detailed: Mapped[Optional[str]] = mapped_column(Text)
    study_design: Mapped[Optional[str]]
    sample_description: Mapped[Optional[str]] = mapped_column(Text)
    sample_size: Mapped[Optional[str]]
    data_collection_methods: Mapped[list[str]] = mapped_column(JSON, default=list)
    analysis_methods: Mapped[list[str]] = mapped_column(JSON, default=list)
    statistical_tests: Mapped[list[str]] = mapped_column(JSON, default=list)
    software_tools: Mapped[list[str]] = mapped_column(JSON, default=list)
    
    # Findings (LLM Pass 2)
    key_findings: Mapped[list[dict]] = mapped_column(JSON, default=list)
    quantitative_results: Mapped[Optional[dict]] = mapped_column(JSON)
    qualitative_themes: Mapped[list[dict]] = mapped_column(JSON, default=list)
    effect_sizes: Mapped[list[dict]] = mapped_column(JSON, default=list)
    
    # Critical Analysis (LLM Pass 3)
    main_arguments: Mapped[list[dict]] = mapped_column(JSON, default=list)
    theoretical_contributions: Mapped[Optional[str]] = mapped_column(Text)
    practical_implications: Mapped[Optional[str]] = mapped_column(Text)
    limitations: Mapped[list[dict]] = mapped_column(JSON, default=list)
    future_research: Mapped[list[str]] = mapped_column(JSON, default=list)
    
    # Classification (from Zotero tags + LLM)
    research_domain: Mapped[Optional[str]]
    subdomain: Mapped[Optional[str]]
    methodology_type: Mapped[Optional[str]]  # quantitative/qualitative/mixed
    paper_type: Mapped[Optional[str]]  # empirical/review/theoretical
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
```

---

## Import Logic

### Source Priority

| Source | Priority | Data Provided |
|--------|----------|---------------|
| Zotero | Primary | title, authors, abstract, DOI, year, collections |
| LLM Extraction | Secondary | methodology, findings, limitations (3 passes) |
| Manual Import | Deprecated | Not recommended, use Zotero |

### Deduplication Logic

```python
def import_paper(zotero_item) -> ImportResult:
    """
    Import logic with deduplication.
    """
    citation_key = get_citation_key(zotero_item)  # from Better BibTeX
    pdf_path = get_pdf_path(zotero_item)
    file_hash = calculate_hash(pdf_path)
    
    # Check for existing paper by hash
    existing = db.query(Paper).filter(Paper.file_hash == file_hash).first()
    
    if existing:
        if existing.paper_id == citation_key:
            # Same paper, same key → UPDATE metadata from Zotero
            update_zotero_metadata(existing, zotero_item)
            return ImportResult(status="updated", paper_id=citation_key)
        else:
            # Same PDF, different key → MERGE (add Zotero link to existing)
            existing.zotero_key = zotero_item.key
            existing.zotero_collections = get_collections(zotero_item)
            # Keep existing LLM extractions
            return ImportResult(status="merged", paper_id=existing.paper_id)
    
    # New paper → Full import
    paper = create_paper_from_zotero(zotero_item, citation_key, file_hash)
    extractions = run_llm_extraction(pdf_path)  # 3 passes
    save_paper(paper, extractions)
    index_vectors(paper)
    
    return ImportResult(status="imported", paper_id=citation_key)
```

### Re-sync Behavior

| Field | On Re-sync |
|-------|------------|
| title, authors, abstract | **Update** from Zotero |
| DOI, year, journal | **Update** from Zotero |
| zotero_collections | **Update** from Zotero |
| methodology_*, findings_* | **Keep** existing LLM extractions |
| chunks, vectors | **Keep** (expensive to regenerate) |

---

## Processing Pipeline

### What Comes From Where

```
┌────────────────────────────────────────────────────────────────────────┐
│                         PAPER DATA SOURCES                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FROM ZOTERO (free, instant)          FROM LLM (3 passes, ~50s total)  │
│  ════════════════════════════         ════════════════════════════════ │
│  ✓ Citation key (paper_id)            ✓ Methodology details            │
│  ✓ Title                              ✓ Study design analysis          │
│  ✓ Authors                            ✓ Sample description             │
│  ✓ Abstract                           ✓ Key findings extraction        │
│  ✓ Publication date                   ✓ Statistical results            │
│  ✓ Journal/venue                      ✓ Limitations identification     │
│  ✓ DOI                                ✓ Future research suggestions    │
│  ✓ Collections                        ✓ Research domain classification │
│  ✓ PDF file location                                                    │
│                                                                         │
│  SAVINGS: 2 LLM passes eliminated (metadata + classification)          │
│  OLD: 5 passes × 16s = 80s                                             │
│  NEW: 3 passes × 16s = 48s (~40% faster)                               │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### LLM Extraction Passes (3 total)

```python
class QualityExtractor:
    """Reduced to 3 passes since Zotero provides metadata."""
    
    def extract_all(self, doc_text: str) -> dict:
        results = {}
        
        # Pass 1: Methodology (detailed)
        results.update(self._extract_methodology(doc_text))
        
        # Pass 2: Findings & Statistics
        results.update(self._extract_findings(doc_text))
        
        # Pass 3: Critical Analysis
        results.update(self._extract_critical_analysis(doc_text))
        
        return results
    
    # Removed: _extract_metadata (Zotero provides this)
    # Removed: _extract_classification (Zotero tags + simpler LLM call)
```

---

## MCP Tools

### Zotero Tools

| Tool | Description |
|------|-------------|
| `zotero_list_collections` | List all Zotero collections with paper counts |
| `zotero_list_items` | List papers in a collection (with import status) |
| `zotero_import_collection` | Import all papers from a collection |
| `zotero_import_item` | Import single paper by citation key |
| `zotero_sync` | Re-sync metadata for imported papers |

### Search Tools

| Tool | Description |
|------|-------------|
| `search_papers` | Find papers by topic (vector search) |
| `search_content` | Search within papers with hybrid retrieval |
| `search_by_citation` | Find paper by `\cite{key}` reference |

### Retrieval Tools

| Tool | Description |
|------|-------------|
| `get_paper_metadata` | Get paper info (title, authors, abstract) |
| `get_methodology` | Get methodology extraction |
| `get_findings` | Get key findings extraction |
| `get_limitations` | Get limitations and critical analysis |
| `get_full_section` | Get raw text of specific section |
| `get_paper_text` | Get full raw text (for custom analysis) |

### On-Demand Analysis Tools

| Tool | Description |
|------|-------------|
| `query_paper` | Ask custom question about a paper |
| `reextract_field` | Re-extract specific field with more detail |
| `compare_papers` | Compare multiple papers on specific aspects |

### Example Usage

```
# List collections
> "Show my Zotero collections"
→ zotero_list_collections()
→ PhD Research (45 papers, 12 imported)
   seL4 Papers (23 papers, 23 imported)
   Mixed-Criticality (18 papers, 5 imported)

# Import collection
> "Import my seL4 Papers collection"
→ zotero_import_collection("seL4 Papers")
→ Importing 23 papers...
   ✓ belt_2023_modeldrivendevelopment (48.2s)
   ✓ lyons_2023_mixedcriticality (52.1s)
   - klein_2009_sel4formal (already imported)
   ...
   Done: 20 imported, 3 already existed

# Query with citation key
> "What methodology did \cite{lyons_2023_mixedcriticality} use?"
→ search_by_citation("lyons_2023_mixedcriticality")
→ get_methodology("lyons_2023_mixedcriticality")
→ The study used a microkernel-based system design...

# Custom query
> "Did \cite{belt_2023_modeldrivendevelopment} mention AADL?"
→ query_paper("belt_2023_modeldrivendevelopment", "Did they mention AADL?")
→ Yes, the paper discusses AADL in section 3.2...

# Compare papers
> "Compare methodologies of papers in seL4 collection"
→ compare_papers(collection="seL4 Papers", aspect="methodology")
```

---

## Hardware Considerations

### Tested Configuration

| Component | Model | VRAM/RAM |
|-----------|-------|----------|
| GPU | NVIDIA RTX 1000 Ada | 6GB |
| LLM | qwen2.5:3b | ~2GB |
| Embeddings | BGE-large-en-v1.5 | ~1.3GB |
| **Total VRAM** | | **~5.5GB** ✓ |

### Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Import (per paper) | ~50s | 3 LLM passes |
| Vector search | <1s | ChromaDB |
| Custom query | ~5s | Single LLM call |
| Re-extraction | ~15s | Single focused pass |

### Alternative Models

| VRAM Available | LLM Model | Embedding Model |
|----------------|-----------|-----------------|
| 4GB | qwen2.5:3b | bge-base-en-v1.5 |
| 6GB | qwen2.5:3b | bge-large-en-v1.5 |
| 8GB | qwen2.5:7b | bge-base-en-v1.5 |
| 12GB+ | qwen2.5:7b | bge-large-en-v1.5 |

---

## Directory Structure

```
academic-paper-mcp/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .env                          # Configuration
├── setup.sh                      # Setup script
├── start_server.sh               # Start with Ollama auto-start
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Pydantic settings
│   ├── server.py                 # MCP server entry
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py           # SQLAlchemy ORM models
│   │   └── vectors.py            # Pydantic + ChromaDB wrapper
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py      # PDF text extraction
│   │   ├── chunker.py            # Semantic chunking
│   │   └── extractor.py          # LLM extraction (3 passes)
│   │
│   ├── zotero/
│   │   ├── __init__.py
│   │   ├── reader.py             # Read Zotero + Better BibTeX DBs
│   │   ├── sync.py               # Import/sync logic
│   │   └── models.py             # Zotero data models
│   │
│   └── utils/
│       ├── __init__.py
│       └── ollama_utils.py       # Ollama auto-start
│
├── data/
│   ├── papers.db                 # SQLite database
│   └── chroma/                   # Vector embeddings
│
└── tests/
    ├── test_setup.py             # Validation tests
    ├── test_zotero.py            # Zotero integration tests
    └── test_extraction.py        # LLM extraction tests
```

---

## Configuration

### .env File

```bash
# Zotero paths
ZOTERO_PATH=~/Zotero
ZOTERO_STORAGE_PATH=~/Zotero/storage
BETTER_BIBTEX_PATH=~/Zotero/better-bibtex.sqlite

# Database
DATABASE_URL=sqlite:///data/papers.db
CHROMA_PERSIST_DIR=./data/chroma

# Models
LLM_MODEL=qwen2.5:3b
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
OLLAMA_HOST=http://localhost:11434

# Processing
CHUNK_SIZE=800
MAX_EXTRACTION_TOKENS=30000
```

---

## Installation

```bash
# Clone and setup
cd academic-paper-mcp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b

# Install Better BibTeX in Zotero
# Configure citation key format: auth.lower + "_" + year + "_" + shorttitle(3, 3).lower

# Initialize database
python -c "from src.models.database import Database; Database().create_tables()"

# Test setup
python test_setup.py

# Start server
./start_server.sh
```

---

## Claude Code Configuration

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

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Zotero as single source | Consistency with bibliography workflow |
| Citation key as paper_id | Direct `\cite{}` integration |
| Trust Zotero metadata | More reliable than LLM extraction |
| 3 LLM passes (not 5) | Zotero provides metadata, saves 40% time |
| File hash deduplication | Prevents duplicates from any source |
| Store full_text | Enables on-demand custom queries |
| On-demand import | User controls processing cost |
| Flatten collections | Simpler queries, papers track all collections |
