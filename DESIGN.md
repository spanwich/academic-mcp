# Academic Paper Processing MCP Server

## Design Document v3.2: Page-Based Chunking & Verbatim Extraction

**Goal:** Reduce Claude Code context consumption when analyzing academic papers by pre-processing, indexing, and retrieving relevant content on demand. Support long documents (theses, books) with reliable page-based chunking.

**Design Priorities:**
1. **Zotero as single source of truth** for bibliography
2. **Citation key = paper_id** (direct `\cite{}` integration)
3. **1 page = 1 chunk** (simple, predictable, PDF-aligned)
4. **Verbatim extraction** (copy original text, prevent hallucination)
5. **Full text always preserved** for Claude Code verification

---

## What's New in v3.2

| v3.0/v3.1 | v3.2 |
|-----------|------|
| Token-based chunking (~1500 tokens) | **Page-based chunking** (1 page = 1 chunk) |
| LLM summarizes content | **LLM extracts verbatim** (copies original text) |
| Flat extraction | **Section-aware extraction** (intro, methodology, etc.) |
| Truncated long docs (40k chars) | **Full document support** (theses, books) |
| Effect sizes sometimes fabricated | **Only extract what exists** |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 CLAUDE CODE                                      │
│                                                                                  │
│   "What methodology did \cite{lyons_2023_mixedcriticality} use?"                │
│   "Show me page 15 of the thesis"                                               │
│   "Find all results about latency measurements"                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │ MCP Protocol
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            MCP SERVER (Python)                                   │
│                                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │    Zotero     │  │    Search     │  │   Retrieve    │  │   Section     │    │
│  │    Tools      │  │    Tools      │  │   Tools       │  │   Tools       │    │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│     ZOTERO      │         │   MCP Database  │         │     Ollama      │
│  (Read Only)    │         │                 │         │   Local LLM     │
│                 │         │ • papers        │         │                 │
│ • zotero.sqlite │         │ • sections      │         │ • qwen2.5:3b    │
│ • better-bibtex │         │ • chunks        │         │ • bge-large     │
│ • storage/      │         │ • ChromaDB      │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

---

## Core Design: Page-Based Chunking

### Why Pages Instead of Tokens?

| Aspect | Token-based | Page-based |
|--------|-------------|------------|
| **Citation** | "chunk 15" (meaningless) | "page 5" (verifiable in PDF) |
| **Predictability** | Variable | 1 page = 1 chunk |
| **Split paragraphs** | Common | Never (page breaks intentional) |
| **Long documents** | Complex logic | Just more pages |
| **Claude Code reference** | "See chunk 47" | "See PDF page 12" |

### Page Character Limits vs Ollama Capacity

| Page Type | Characters | Tokens | Ollama 3b Limit |
|-----------|------------|--------|-----------------|
| Single column | 3,000-4,500 | 650-900 | ✓ 32,768 |
| Double column | 5,000-7,500 | 1,000-1,500 | ✓ 32,768 |
| Dense thesis | 2,500-4,000 | 500-800 | ✓ 32,768 |
| **Worst case** | ~9,000 | ~2,000 | ✓ 32,768 |

**Verdict:** One page per chunk is always safe with 16x headroom.

---

## Data Model

### papers table

```python
class Paper(Base):
    __tablename__ = "papers"
    
    # Primary identifier (= BibTeX citation key)
    paper_id: Mapped[str] = mapped_column(primary_key=True)
    # e.g., "lyons_2023_mixedcriticality"
    
    # Zotero linking
    zotero_key: Mapped[Optional[str]] = mapped_column(index=True)
    zotero_item_id: Mapped[Optional[int]]
    zotero_collections: Mapped[list[str]] = mapped_column(JSON, default=list)
    
    # Deduplication
    file_path: Mapped[str]
    file_hash: Mapped[Optional[str]] = mapped_column(index=True)
    
    # Metadata FROM ZOTERO (trusted)
    title: Mapped[Optional[str]] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    publication_date: Mapped[Optional[str]]
    year: Mapped[Optional[int]]
    journal_or_venue: Mapped[Optional[str]]
    doi: Mapped[Optional[str]]
    
    # Document stats
    page_count: Mapped[Optional[int]]
    word_count: Mapped[Optional[int]]
    
    # Full text storage (ALWAYS preserved for verification)
    full_text: Mapped[Optional[str]] = mapped_column(Text)
    
    # Processing metadata
    processing_status: Mapped[ProcessingStatus]
    processing_model: Mapped[Optional[str]]
    processed_at: Mapped[Optional[datetime]]
    
    # Relationships
    sections: Mapped[list["Section"]] = relationship(...)
    chunks: Mapped[list["Chunk"]] = relationship(...)
```

### sections table (NEW in v3.2)

```python
class Section(Base):
    """
    Document sections detected by LLM.
    
    Provides structure for academic papers: intro, methodology, results, etc.
    Even if detection is imperfect, chunks contain the real content.
    """
    __tablename__ = "sections"
    
    section_id: Mapped[str] = mapped_column(primary_key=True)
    # e.g., "lyons_2023_mixedcriticality_sec_methodology"
    
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"))
    
    # Section identification
    section_type: Mapped[str]
    # introduction | background | literature_review | methodology |
    # results | discussion | conclusion | references | appendix | unknown
    
    section_title: Mapped[Optional[str]]
    # Original heading text, e.g., "3.1 Experimental Setup"
    
    # Location in document
    page_start: Mapped[int]
    page_end: Mapped[int]
    char_start: Mapped[int]   # Position in full_text
    char_end: Mapped[int]
    
    # LLM-generated content
    summary: Mapped[Optional[str]] = mapped_column(Text)
    # 2-3 sentence summary (use with caution)
    
    key_points_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    # [{"text": "exact quote from paper", "page": 5}, ...]
    # TRUSTWORTHY - copied directly from paper
    
    # Detection metadata
    detection_method: Mapped[str]  # "llm" or "page_fallback"
    confidence: Mapped[float]
    
    # Relationships
    paper: Mapped["Paper"] = relationship(back_populates="sections")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="section")
```

### chunks table (Updated for page-based)

```python
class Chunk(Base):
    """
    Page-based chunks for retrieval.
    
    1 page = 1 chunk. Simple, predictable, PDF-aligned.
    """
    __tablename__ = "chunks"
    
    chunk_id: Mapped[str] = mapped_column(primary_key=True)
    # e.g., "lyons_2023_mixedcriticality_page_5"
    
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"))
    section_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sections.section_id"))
    # Nullable - may not have section assignment
    
    # Page info
    page_number: Mapped[int]
    
    # Position in full_text (for verification)
    char_start: Mapped[int]
    char_end: Mapped[int]
    
    # Content (stored for fast retrieval)
    content: Mapped[str] = mapped_column(Text)
    word_count: Mapped[int]
    
    # Relationships
    paper: Mapped["Paper"] = relationship(back_populates="chunks")
    section: Mapped[Optional["Section"]] = relationship(back_populates="chunks")
```

### extractions table (Verbatim approach)

```python
class Extraction(Base):
    """
    LLM extractions from a paper.
    
    v3.2 PHILOSOPHY: Copy original text, don't summarize.
    - *_verbatim fields: Exact text from paper (TRUSTWORTHY)
    - *_summary fields: LLM summaries (use with caution)
    """
    __tablename__ = "extractions"
    
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.paper_id"),
        primary_key=True
    )
    
    # === VERBATIM EXTRACTIONS (trustworthy) ===
    
    # Methodology - exact paragraphs copied from paper
    methodology_verbatim: Mapped[Optional[str]] = mapped_column(Text)
    evaluation_setup_verbatim: Mapped[Optional[str]] = mapped_column(Text)
    
    # Contributions - with section/page tracking
    contributions_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    # [{"text": "We present seL4, the first...", "section": "Abstract", "page": 1}]
    
    # Results - exact sentences reporting findings
    results_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    # [{"text": "The measured latency was 9400ns", "section": "Evaluation", "page": 8}]
    
    # Statistics - ONLY if they exist in paper
    statistics_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    # [{"text": "30x faster than Bitcoin Core", "section": "Results", "page": 10}]
    # EMPTY if paper has no statistics (never fabricate!)
    
    # Limitations - what AUTHORS wrote, not inferred
    limitations_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    # [{"text": "Our approach does not handle...", "section": "Discussion", "page": 12}]
    
    # Future work - author stated
    future_work_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    
    # === LLM SUMMARIES (use with caution) ===
    
    methodology_summary: Mapped[Optional[str]] = mapped_column(Text)
    # Brief summary - may miss details or misinterpret
    
    # Classification
    research_domain: Mapped[Optional[str]]
    subdomain: Mapped[Optional[str]]
    methodology_type: Mapped[Optional[str]]
    paper_type: Mapped[Optional[str]]
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
```

---

## Processing Pipeline

### Complete Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PDF PROCESSING FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

PDF File (e.g., 60k word thesis, 200 pages)
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Extract Text with Page Boundaries (PyMuPDF)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Output:                                                                        │
│   - full_text: Complete document text                                           │
│   - pages: [{"page": 1, "text": "...", "char_start": 0, "char_end": 3500}, ...] │
│   - page_count: 200                                                              │
│   - word_count: 60000                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: LLM Section Detection (Ollama)                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Input: First 1000 chars of each page (for context)                            │
│                                                                                  │
│   Prompt: "Identify section boundaries in this document..."                     │
│                                                                                  │
│   Output:                                                                        │
│   [                                                                              │
│     {"type": "introduction", "title": "1. Introduction", "pages": [1, 3]},     │
│     {"type": "background", "title": "2. Background", "pages": [4, 12]},        │
│     {"type": "methodology", "title": "3. Methodology", "pages": [13, 25]},     │
│     {"type": "results", "title": "4. Evaluation", "pages": [26, 45]},          │
│     ...                                                                          │
│   ]                                                                              │
│                                                                                  │
│   FALLBACK: If detection fails → group by 10 pages each as "unknown"            │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Per-Section Extraction (Ollama)                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│   For each section:                                                              │
│                                                                                  │
│   Pass 1 - Summary:                                                              │
│     "Summarize this section in 2-3 sentences"                                   │
│     → section.summary                                                            │
│                                                                                  │
│   Pass 2 - Key Points Verbatim:                                                 │
│     "Copy the most important sentences exactly as written"                      │
│     → section.key_points_verbatim                                               │
│                                                                                  │
│   For methodology sections, also extract:                                       │
│     → extraction.methodology_verbatim                                           │
│     → extraction.evaluation_setup_verbatim                                      │
│                                                                                  │
│   For results sections, also extract:                                           │
│     → extraction.results_verbatim                                               │
│     → extraction.statistics_verbatim                                            │
│                                                                                  │
│   For discussion/conclusion, also extract:                                      │
│     → extraction.contributions_verbatim                                         │
│     → extraction.limitations_verbatim                                           │
│     → extraction.future_work_verbatim                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Create Page Chunks                                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│   For each page:                                                                 │
│     chunk = Chunk(                                                               │
│       chunk_id = f"{paper_id}_page_{page_num}",                                 │
│       page_number = page_num,                                                    │
│       section_id = detected_section_for_page,  # may be null                    │
│       content = page_text,                                                       │
│       char_start = page_char_start,                                             │
│       char_end = page_char_end                                                  │
│     )                                                                            │
│                                                                                  │
│   200 page document → 200 chunks                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Embed & Index (BGE + ChromaDB)                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│   For each chunk:                                                                │
│     embedding = bge_model.encode(chunk.content)                                 │
│     chromadb.add(                                                                │
│       id = chunk.chunk_id,                                                       │
│       embedding = embedding,                                                     │
│       metadata = {                                                               │
│         "paper_id": paper_id,                                                   │
│         "page_number": page_num,                                                │
│         "section_type": section_type,                                           │
│         "char_start": char_start,                                               │
│         "char_end": char_end                                                    │
│       }                                                                          │
│     )                                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Store Everything                                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│   SQLite:                                                                        │
│     - papers: metadata + full_text                                              │
│     - sections: detected structure + summaries + key_points                     │
│     - chunks: page content + positions                                          │
│     - extractions: verbatim extracts organized by type                          │
│                                                                                  │
│   ChromaDB:                                                                      │
│     - Vector embeddings for semantic search                                     │
│     - Metadata for filtering (paper_id, section_type, page)                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Section Detection Strategy

```python
def detect_sections(pages: list[dict]) -> list[Section]:
    """
    LLM-based section detection with page fallback.
    """
    # Build context for LLM
    page_previews = []
    for p in pages:
        preview = p["text"][:1000]  # First 1000 chars of each page
        page_previews.append(f"=== PAGE {p['page']} ===\n{preview}")
    
    prompt = f"""Analyze this document and identify section boundaries.

Document pages:
{chr(10).join(page_previews[:30])}  # First 30 pages for context

Return JSON array:
[
  {{"type": "introduction", "title": "1. Introduction", "page_start": 1, "page_end": 3}},
  {{"type": "methodology", "title": "3. Methods", "page_start": 8, "page_end": 15}},
  ...
]

Valid section types:
- abstract
- introduction  
- background / literature_review
- methodology / methods
- results / evaluation
- discussion
- conclusion
- references
- appendix
- unknown (if unclear)
"""
    
    try:
        sections = llm.generate(prompt, format="json")
        if sections and len(sections) >= 2:
            return sections, "llm", 0.7
    except:
        pass
    
    # FALLBACK: Group pages if LLM fails
    sections = []
    page_group_size = 10
    for i in range(0, len(pages), page_group_size):
        sections.append({
            "type": "unknown",
            "title": f"Pages {i+1}-{min(i+page_group_size, len(pages))}",
            "page_start": i + 1,
            "page_end": min(i + page_group_size, len(pages))
        })
    return sections, "page_fallback", 0.3
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
| `search_papers` | Find papers by topic (searches abstracts, titles) |
| `search_content` | Semantic search within paper content (returns chunks with page numbers) |
| `search_by_citation` | Find paper by `\cite{key}` reference |

### Section Tools (NEW in v3.2)

| Tool | Description |
|------|-------------|
| `list_sections` | Get document structure (section names, page ranges) |
| `get_section_summary` | Get LLM summary of a section |
| `get_section_text` | Get full original text of a section |
| `get_section_key_points` | Get verbatim key points from a section |

### Page Tools (NEW in v3.2)

| Tool | Description |
|------|-------------|
| `get_page` | Get text of a single page |
| `get_pages` | Get text of page range (e.g., pages 5-10) |
| `get_chunk_context` | Get chunk with surrounding context |

### Retrieval Tools

| Tool | Description |
|------|-------------|
| `get_paper_metadata` | Get paper info (title, authors, abstract) |
| `get_methodology` | Get methodology extraction (verbatim + summary) |
| `get_findings` | Get results/findings (verbatim quotes) |
| `get_limitations` | Get limitations (author-stated only) |
| `get_statistics` | Get statistics (only if they exist in paper) |
| `get_paper_text` | Get full raw text |

### On-Demand Analysis Tools

| Tool | Description |
|------|-------------|
| `query_paper` | Ask custom question about a paper |
| `query_section` | Ask question about specific section |
| `reextract_field` | Re-extract with different focus |
| `compare_papers` | Compare multiple papers |

---

## Example Usage

### Basic Workflow

```
# List collections
> "Show my Zotero collections"
→ zotero_list_collections()
→ PhD Research (45 papers, 12 imported)
   seL4 Papers (23 papers, 23 imported)

# Import collection
> "Import my seL4 Papers collection"
→ zotero_import_collection("seL4 Papers")
→ Processing 23 papers...
   ✓ belt_2023_model (15 pages, 8 sections, 52s)
   ✓ lyons_2023_mixed (24 pages, 6 sections, 78s)
   ...

# Query with citation
> "What methodology did \cite{lyons_2023_mixedcriticality} use?"
→ get_methodology("lyons_2023_mixedcriticality")
→ ## Methodology (from paper)
   
   "We implemented our approach on the seL4 microkernel using
    a partition scheduler with fixed time slices..."
   — Pages 8-12, Methodology section
```

### Working with Long Documents

```
# Import thesis
> "Import my thesis"
→ zotero_import_item("smith_2024_thesis")
→ Processing: 200 pages, 60,412 words
   Detected sections:
   - Introduction (pages 1-15)
   - Literature Review (pages 16-45)
   - Methodology (pages 46-78)
   - Results (pages 79-150)
   - Discussion (pages 151-180)
   - Conclusion (pages 181-195)
   - References (pages 196-200)
   
   ✓ Imported in 312s (200 chunks indexed)

# Navigate by section
> "Show me the methodology section"
→ get_section_text("smith_2024_thesis", "methodology")
→ [Returns pages 46-78 text]

# Navigate by page
> "Show me page 52"
→ get_page("smith_2024_thesis", 52)
→ [Returns page 52 text]

# Semantic search
> "Find where I discuss validation"
→ search_content("validation methodology")
→ Found in:
   • Page 65 (methodology): "We validated our approach using..."
   • Page 142 (results): "Validation results showed..."
   • Page 168 (discussion): "The validation demonstrates..."
```

### Verification Workflow

```
# Get verbatim findings
> "What results did the paper report?"
→ get_findings("klein_2009_sel4")
→ ## Results (from paper)
   
   1. "seL4 achieves interrupt latency of 9400ns on ARM11"
      — Page 8, Evaluation section
   
   2. "The proof covers 8,700 lines of C code"
      — Page 3, Introduction section

# Verify against original
> "Show me page 8 to verify"
→ get_page("klein_2009_sel4", 8)
→ [Full page 8 text for verification]
```

---

## Linking: Vector DB ↔ SQLite

All data is cross-referenced for seamless retrieval:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA LINKING ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────────┘

ChromaDB                              SQLite
(Vector Search)                       (Full Data)
─────────────                         ───────────
                                      
┌─────────────────┐                   ┌─────────────────┐
│ Embedding       │                   │ papers          │
│                 │    paper_id       │                 │
│ id: chunk_id ───┼───────────────────┤ paper_id        │
│ metadata:       │                   │ full_text ──────┼─── Complete document
│   paper_id      │                   │ page_count      │
│   page_number   │                   └────────┬────────┘
│   section_type  │                            │
│   char_start ───┼───────────────────────────▼────────────────────────────
│   char_end      │                   ┌─────────────────┐
└─────────────────┘                   │ sections        │
                                      │                 │
        │                             │ section_id      │
        │ semantic search             │ paper_id        │
        │ returns chunk_id            │ char_start ─────┼─── Points into full_text
        │                             │ char_end        │
        ▼                             │ summary         │
                                      │ key_points      │
┌─────────────────┐                   └────────┬────────┘
│ Search Result   │                            │
│                 │                            ▼
│ chunk_id ───────┼──────────────────▶┌─────────────────┐
│ score: 0.89     │                   │ chunks          │
│                 │                   │                 │
└─────────────────┘                   │ chunk_id        │
                                      │ paper_id        │
                                      │ section_id      │
                                      │ page_number     │
                                      │ char_start ─────┼─── Points into full_text
                                      │ char_end        │
                                      │ content ────────┼─── Stored for fast access
                                      └─────────────────┘

RETRIEVAL FLOW:
1. Claude Code: search_content("verification approach")
2. ChromaDB: Returns top-k chunk_ids with scores
3. SQLite: Fetch chunk content + metadata
4. Return: Text with page numbers for citation

VERIFICATION FLOW:
1. Claude Code: "That doesn't seem right, show me the original"
2. Get chunk.char_start, chunk.char_end
3. Fetch paper.full_text[char_start:char_end]
4. Return: Exact original text (always matches chunk.content)
```

---

## Database Lock Detection

Zotero must be closed during import to avoid database locks.

```python
def _check_database_lock(self):
    """Check if Zotero database is locked."""
    # Check for lock file
    lock_file = self.zotero_db_path.parent / ".zotero-lock"
    if lock_file.exists():
        raise RuntimeError(
            "Zotero database is locked.\n"
            "Please close Zotero before importing."
        )
    
    # Try to acquire lock
    try:
        conn = sqlite3.connect(self.zotero_db_path, timeout=1)
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
        conn.close()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            raise RuntimeError(
                "Zotero database is locked.\n"
                "Please close Zotero before importing."
            )
```

**User sees:**
```
============================================================
  ERROR: Zotero Database Locked
============================================================

  Zotero database is locked.
  Please close Zotero before importing.

  Please close Zotero application and try again.
============================================================
```

---

## Storage Estimates

| Data | Per Paper (10 pages) | Per Thesis (200 pages) | 1000 Papers |
|------|---------------------|----------------------|-------------|
| full_text | ~50 KB | ~350 KB | ~50 MB |
| sections | ~5 KB | ~20 KB | ~5 MB |
| chunks | ~60 KB | ~400 KB | ~60 MB |
| extractions | ~10 KB | ~30 KB | ~10 MB |
| **SQLite Total** | ~125 KB | ~800 KB | **~125 MB** |
| ChromaDB | ~100 KB | ~2 MB | **~100 MB** |

**SQLite easily handles this.** Even 10,000 papers ≈ 1.5 GB.

---

## Hardware Requirements

### Tested Configuration

| Component | Model | VRAM |
|-----------|-------|------|
| GPU | NVIDIA RTX 1000 Ada | 6GB |
| LLM | qwen2.5:3b | ~2GB |
| Embeddings | BGE-large-en-v1.5 | ~1.3GB |
| **Peak VRAM** | | **~5.8GB** ✓ |

### Processing Time Estimates

| Document | Pages | Sections | Est. Time |
|----------|-------|----------|-----------|
| Short paper | 10 | 5 | ~60s |
| Long paper | 30 | 8 | ~120s |
| Thesis | 200 | 10 | ~400s |

### Alternative Models

| VRAM | LLM Model | Embedding Model |
|------|-----------|-----------------|
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
├── DESIGN.md                     # This document
├── .env
├── start_server.sh
├── zotero_import.py              # CLI import tool
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── server.py                 # MCP server
│   │
│   ├── models/
│   │   ├── database.py           # SQLAlchemy ORM (papers, sections, chunks)
│   │   └── vectors.py            # ChromaDB wrapper
│   │
│   ├── processing/
│   │   ├── pdf_processor.py      # PDF extraction with page boundaries
│   │   ├── section_detector.py   # LLM section detection (NEW)
│   │   ├── chunker.py            # Page-based chunking (updated)
│   │   └── extractor.py          # Verbatim extraction (updated)
│   │
│   ├── zotero/
│   │   ├── reader.py             # Read Zotero DB (with lock detection)
│   │   ├── sync.py               # Import/sync logic
│   │   └── models.py
│   │
│   └── utils/
│       └── ollama_utils.py
│
├── data/
│   ├── papers.db                 # SQLite database
│   └── chroma/                   # Vector embeddings
│
└── tests/
    └── test_setup.py
```

---

## Configuration

### .env File

```bash
# Zotero paths (auto-detected if not set)
ACADEMIC_ZOTERO_PATH=~/Zotero

# Database
ACADEMIC_DATABASE_URL=sqlite:///data/papers.db
ACADEMIC_CHROMA_PERSIST_DIR=./data/chroma

# Models
ACADEMIC_LLM_MODEL=qwen2.5:3b
ACADEMIC_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
ACADEMIC_OLLAMA_HOST=http://localhost:11434
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **1 page = 1 chunk** | Simple, predictable, PDF-aligned citations |
| **LLM section detection** | Handles creative formatting, multi-column |
| **Page fallback** | Never fails - always has some structure |
| **Verbatim extraction** | Prevents hallucination, preserves accuracy |
| **Full text storage** | Claude Code can always verify |
| **char_start/end positions** | Links chunks back to full_text |
| **Section summaries** | Quick overview (use with caution) |
| **Key points verbatim** | Trustworthy highlights |

---

## Verbatim Extraction Philosophy

### Old Approach (v3.0) — Summarization

```
LLM reads paper → LLM generates summary → May hallucinate
```

**Problem:** LLM fabricated effect sizes like "Cohen's d: 0.65" that didn't exist.

### New Approach (v3.2) — Extraction

```
LLM reads paper → LLM copies relevant text → Original preserved
```

**Result:** Only text that actually exists in the paper appears in database.

### Example Comparison

**v3.0 Output (unreliable):**
```json
{
  "effect_sizes": [
    {"measure": "Cohen's d", "value": "0.65"},
    {"measure": "eta squared", "value": "0.32"}
  ]
}
```
*Paper had no statistical analysis!*

**v3.2 Output (trustworthy):**
```json
{
  "statistics_verbatim": []
}
```
*Empty because paper has no statistics.*

Or if paper does have stats:
```json
{
  "statistics_verbatim": [
    {"text": "30x faster than the Bitcoin Core parser", "section": "Results", "page": 10}
  ]
}
```
*Exact quote from paper, verifiable by page number.*

---

## Claude Code MCP Configuration

```json
{
  "mcpServers": {
    "academic-papers": {
      "command": "/home/user/phd/academic-mcp/start_server.sh",
      "cwd": "/home/user/phd/academic-mcp"
    }
  }
}
```
