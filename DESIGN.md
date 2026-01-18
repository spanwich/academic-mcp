# Academic Paper Processing MCP Server

## Design Document v3.3: Keywords, Domains & Enhanced Search

**Goal:** Reduce Claude Code context consumption when analyzing academic papers by pre-processing, indexing, and retrieving relevant content on demand. Support long documents (theses, books) with reliable page-based chunking.

**Design Priorities:**
1. **Zotero as single source of truth** for bibliography
2. **Citation key = paper_id** (direct `\cite{}` integration)
3. **1 page = 1 chunk** (simple, predictable, PDF-aligned)
4. **Verbatim extraction** (copy original text, prevent hallucination)
5. **Full text always preserved** for Claude Code verification
6. **Research-actionable search** (venue, domain, keywords)

---

## What's New in v3.3

| v3.2 | v3.3 |
|------|------|
| Keywords in extractions (LLM only) | **Keywords from paper first**, LLM fallback |
| No keyword source tracking | **keywords_source** field ("paper" or "llm") |
| Broad classification (research_domain, subdomain) | **Single specific domain** per paper |
| Static taxonomy | **Self-organizing domain taxonomy** |
| Search: title, abstract only | **Enhanced search**: venue, domain, author, year |
| No venue/domain discovery | **list_venues**, **list_domains** tools |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 CLAUDE CODE                                      │
│                                                                                  │
│   "Find all NDSS papers"                                                        │
│   "What papers cover side-channel attacks?"                                     │
│   "Show me seL4 verification methodology"                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │ MCP Protocol
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            MCP SERVER (Python)                                   │
│                                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │    Zotero     │  │    Search     │  │   Retrieve    │  │   Discovery   │    │
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
│ • storage/      │         │ • domains       │         │                 │
└─────────────────┘         │ • ChromaDB      │         └─────────────────┘
                            └─────────────────┘
```

---

## Core Design: Keywords & Domain Classification

### Keywords: Two Sources

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           KEYWORD EXTRACTION FLOW                                │
└─────────────────────────────────────────────────────────────────────────────────┘

PDF Text
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Parse PDF for explicit keywords                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Look for patterns:                                                             │
│   • "Keywords: formal verification, operating systems, microkernel"             │
│   • "Index Terms—seL4, capability systems, IPC"                                 │
│   • "Key words: WCET, static analysis, real-time"                               │
│                                                                                  │
│   If found:                                                                      │
│     keywords = ["formal verification", "operating systems", "microkernel"]      │
│     keywords_source = "paper"  ← TRUSTWORTHY                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
    │
    │ Not found?
    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: LLM keyword extraction (fallback)                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Prompt: "Extract 3-5 keywords from this abstract..."                          │
│                                                                                  │
│   Result:                                                                        │
│     keywords = ["microkernel verification", "Isabelle/HOL", "capability"]       │
│     keywords_source = "llm"  ← USE WITH CAUTION                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Claude Code sees:**
```
Keywords: formal verification, seL4, microkernel (from paper) ← can cite directly
Keywords: capability systems, IPC (from LLM) ← verify before citing
```

### Domain: Self-Organizing Taxonomy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DOMAIN CLASSIFICATION FLOW                               │
└─────────────────────────────────────────────────────────────────────────────────┘

Paper Abstract
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ LLM Classification Prompt                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Existing domains in database:                                                  │
│   - microkernel formal verification using Isabelle                              │
│   - mixed-criticality real-time scheduling                                       │
│   - acoustic side-channel attacks                                               │
│   - hypervisor isolation mechanisms                                              │
│   - ICS/SCADA security protocols                                                │
│   ...                                                                            │
│                                                                                  │
│   Paper abstract:                                                                │
│   "We present seL4, the first operating system kernel with a formal             │
│    proof of functional correctness..."                                          │
│                                                                                  │
│   Instructions:                                                                  │
│   1. If an existing domain fits semantically, REUSE it (preferred)              │
│   2. Only create new domain if paper covers truly different area                │
│   3. Be SPECIFIC (research-actionable), not broad                               │
│   4. Return exactly ONE domain                                                   │
│                                                                                  │
│   BAD:  "security"  ← too broad, useless                                        │
│   BAD:  "formal verification" ← too broad                                       │
│   GOOD: "microkernel formal verification using Isabelle" ← actionable          │
└─────────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Taxonomy Growth Example                                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Paper #1: seL4 verification paper                                             │
│   Existing: []                                                                   │
│   Result: NEW "microkernel formal verification using Isabelle"                  │
│                                                                                  │
│   Paper #50: Another Isabelle OS verification paper                             │
│   Existing: ["microkernel formal verification using Isabelle", ...]             │
│   Result: REUSE "microkernel formal verification using Isabelle"                │
│                                                                                  │
│   Paper #100: Paper about acoustic keyboard attacks                             │
│   Existing: [20 domains...]                                                      │
│   Result: NEW "acoustic side-channel attacks on input devices"                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Why specific domains?**

| Too Broad (useless) | Research-Actionable (good) |
|---------------------|----------------------------|
| security | acoustic side-channel attacks on input devices |
| formal verification | microkernel formal verification using Isabelle |
| real-time systems | mixed-criticality scheduling with temporal isolation |
| networking | high-performance IPC in microkernels |

Specific domains let you ask: "What papers use similar methods to mine?"

---

## Data Model

### papers table (Updated)

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
    journal_or_venue: Mapped[Optional[str]] = mapped_column(index=True)  # NEW: indexed
    doi: Mapped[Optional[str]]
    
    # === NEW IN v3.3: Keywords & Domain ===
    
    # Keywords extracted from paper or LLM
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    # e.g., ["formal verification", "seL4", "microkernel"]
    
    keywords_source: Mapped[Optional[str]]
    # "paper" = extracted from PDF (trustworthy)
    # "llm" = generated by LLM (use with caution)
    
    # Single specific domain (self-organizing taxonomy)
    domain: Mapped[Optional[str]] = mapped_column(index=True)
    # e.g., "microkernel formal verification using Isabelle"
    
    # === END NEW ===
    
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

### domains table (NEW in v3.3)

```python
class Domain(Base):
    """
    Self-organizing domain taxonomy.
    
    Domains are created by LLM during import, reused when similar.
    """
    __tablename__ = "domains"
    
    domain_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Domain name (specific, research-actionable)
    name: Mapped[str] = mapped_column(unique=True, index=True)
    # e.g., "microkernel formal verification using Isabelle"
    
    # Stats
    paper_count: Mapped[int] = mapped_column(default=0)
    
    # When created
    created_at: Mapped[datetime]
    
    # Optional: broader category for grouping (not used in search)
    parent_category: Mapped[Optional[str]]
    # e.g., "formal verification" - for UI grouping only
```

### sections table (Unchanged from v3.2)

```python
class Section(Base):
    """Document sections detected by LLM."""
    __tablename__ = "sections"
    
    section_id: Mapped[str] = mapped_column(primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"))
    
    section_type: Mapped[str]
    section_title: Mapped[Optional[str]]
    
    page_start: Mapped[int]
    page_end: Mapped[int]
    char_start: Mapped[int]
    char_end: Mapped[int]
    
    summary: Mapped[Optional[str]] = mapped_column(Text)
    key_points_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    
    detection_method: Mapped[str]
    confidence: Mapped[float]
```

### chunks table (Unchanged from v3.2)

```python
class Chunk(Base):
    """Page-based chunks for retrieval. 1 page = 1 chunk."""
    __tablename__ = "chunks"
    
    chunk_id: Mapped[str] = mapped_column(primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"))
    section_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sections.section_id"))
    
    page_number: Mapped[int]
    char_start: Mapped[int]
    char_end: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    word_count: Mapped[int]
```

### extractions table (Simplified in v3.3)

```python
class Extraction(Base):
    """
    LLM extractions from a paper.
    
    v3.3: Removed redundant classification fields (now in Paper)
    """
    __tablename__ = "extractions"
    
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), primary_key=True)
    
    # === VERBATIM EXTRACTIONS (trustworthy) ===
    methodology_verbatim: Mapped[Optional[str]] = mapped_column(Text)
    evaluation_setup_verbatim: Mapped[Optional[str]] = mapped_column(Text)
    contributions_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    results_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    statistics_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    limitations_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    future_work_verbatim: Mapped[list[dict]] = mapped_column(JSON, default=list)
    
    # === LLM SUMMARIES (use with caution) ===
    methodology_summary: Mapped[Optional[str]] = mapped_column(Text)
    
    # v3.3: Removed - now in Paper table:
    # - research_domain (replaced by Paper.domain)
    # - subdomain (removed)
    # - keywords (moved to Paper.keywords)
```

---

## Processing Pipeline

### Complete Flow (Updated for v3.3)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PDF PROCESSING FLOW v3.3                            │
└─────────────────────────────────────────────────────────────────────────────────┘

PDF File
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Extract Text with Page Boundaries (PyMuPDF)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Output:                                                                        │
│   - full_text: Complete document text                                           │
│   - pages: [{"page": 1, "text": "...", "char_start": 0, "char_end": 3500}, ...] │
│   - page_count, word_count                                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Extract Keywords (NEW in v3.3)                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Phase 1: Parse PDF for explicit keywords                                       │
│     - Search for "Keywords:", "Index Terms", "Key words:"                       │
│     - If found: keywords_source = "paper" (trustworthy)                         │
│                                                                                  │
│   Phase 2: LLM fallback (if no keywords found)                                  │
│     - Prompt: "Extract 3-5 keywords from this abstract"                         │
│     - keywords_source = "llm" (use with caution)                                │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Classify Domain (NEW in v3.3)                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│   1. Fetch existing domains from database                                        │
│   2. LLM prompt with existing domains + abstract                                │
│   3. LLM either:                                                                 │
│      - REUSES existing domain (preferred)                                       │
│      - CREATES new specific domain                                              │
│   4. Update domain paper_count                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: LLM Section Detection (Unchanged)                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Detect: introduction, methodology, results, etc.                               │
│   Fallback: Group by 10 pages if detection fails                                │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Per-Section Verbatim Extraction (Unchanged)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Extract: methodology_verbatim, results_verbatim, etc.                         │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Create Page Chunks (Unchanged)                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│   1 page = 1 chunk                                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Embed & Index (Unchanged)                                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│   BGE embeddings → ChromaDB                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Keyword Extraction Strategy

```python
def extract_keywords(full_text: str, abstract: str) -> tuple[list[str], str]:
    """
    Extract keywords from paper, with LLM fallback.
    
    Returns:
        (keywords, source) where source is "paper" or "llm"
    """
    # Phase 1: Look for explicit keywords in PDF
    patterns = [
        r"Keywords?[:\s—–-]+([^\n]+)",
        r"Index Terms?[:\s—–-]+([^\n]+)", 
        r"Key words?[:\s—–-]+([^\n]+)",
    ]
    
    # Search in first few pages (where keywords usually appear)
    search_text = full_text[:15000]
    
    for pattern in patterns:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            keywords_text = match.group(1)
            # Split by comma, semicolon, or bullet
            keywords = re.split(r'[,;•]', keywords_text)
            keywords = [k.strip().lower() for k in keywords if k.strip()]
            if keywords:
                return keywords, "paper"  # TRUSTWORTHY
    
    # Phase 2: LLM fallback
    prompt = f"""Extract 3-5 keywords from this abstract.
    
Abstract:
{abstract}

Return JSON array of keywords:
["keyword1", "keyword2", "keyword3"]

Be specific. Use technical terms from the paper."""
    
    keywords = llm.generate(prompt, format="json")
    return keywords, "llm"  # USE WITH CAUTION
```

### Domain Classification Strategy

```python
def classify_domain(abstract: str, existing_domains: list[str]) -> str:
    """
    Classify paper into a specific domain.
    
    Prefers reusing existing domains for consistency.
    """
    domains_list = "\n".join(f"- {d}" for d in existing_domains) if existing_domains else "(none yet)"
    
    prompt = f"""Classify this paper into ONE specific research domain.

EXISTING DOMAINS IN DATABASE:
{domains_list}

PAPER ABSTRACT:
{abstract}

INSTRUCTIONS:
1. If an existing domain fits semantically, REUSE it (preferred)
   - Even if wording is slightly different, reuse if same concept
2. Only create NEW domain if paper covers truly different area
3. Be SPECIFIC (research-actionable), not broad

BAD examples (too broad):
- "security"
- "formal verification"
- "real-time systems"

GOOD examples (specific, actionable):
- "microkernel formal verification using Isabelle"
- "acoustic side-channel attacks on input devices"
- "mixed-criticality scheduling with temporal isolation"
- "ICS/SCADA protocol security analysis"

Return JSON:
{{"domain": "the specific domain", "is_new": true/false}}
"""
    
    result = llm.generate(prompt, format="json")
    return result["domain"]
```

---

## MCP Tools

### Zotero Tools (Unchanged)

| Tool | Description |
|------|-------------|
| `zotero_list_collections` | List all Zotero collections with paper counts |
| `zotero_list_items` | List papers in a collection (with import status) |

### Discovery Tools (NEW in v3.3)

| Tool | Description |
|------|-------------|
| `list_venues` | List all venues with paper counts |
| `list_domains` | List all domains with paper counts |
| `list_imported_papers` | List all imported papers with metadata |

#### list_venues: Discover Available Venues

Returns all unique venues in the database with paper counts. Useful for:
- Discovering what conferences/journals are in your library
- Finding the exact venue name for filtering (e.g., "NDSS" → full name)

**Example Output:**

```
Venues in database (15 total):

  • Proceedings 2025 Network and Distributed System Security Symposium (5 papers)
  • 2024 IEEE Real-Time Systems Symposium (RTSS) (12 papers)
  • USENIX Security Symposium (3 papers)
  • Communications of the ACM (4 papers)
  • ACM SIGOPS Operating Systems Review (2 papers)
  ...
```

**Workflow:** Use `list_venues` to discover venue names, then `search_papers(venue="RTSS")` to find papers.

#### list_domains: Discover Research Domains

Returns all domains in the self-organizing taxonomy with paper counts. Useful for:
- Understanding what research areas your library covers
- Finding papers in a specific research niche

**Example Output:**

```
Domains in database (23 total):

  • microkernel formal verification using Isabelle (15 papers)
  • mixed-criticality scheduling with temporal isolation (8 papers)
  • acoustic side-channel attacks on input devices (3 papers)
  • ICS/SCADA protocol security analysis (5 papers)
  • hypervisor memory isolation mechanisms (4 papers)
  ...
```

**Workflow:** Use `list_domains` to see available domains, then `search_papers(domain="side-channel")` to find papers.

### Search Tools (Enhanced in v3.3)

| Tool | Description |
|------|-------------|
| `search_papers` | **Enhanced**: Unified search with multiple filter options |
| `search_content` | Semantic search within paper content |
| `search_by_citation` | Find paper by citation key |

#### search_papers: Unified Search Tool

In v3.3, `search_papers` replaces the need for separate `search_by_venue`, `search_by_domain`, etc. tools. One tool handles all search scenarios through combinable filters:

**Available Filters:**

| Filter | Type | Description |
|--------|------|-------------|
| `query` | string | Text search in title and abstract |
| `venue` | string | Partial match on venue/conference name |
| `domain` | string | Partial match on domain |
| `author` | string | Partial match on author name |
| `year` | int | Exact year match |
| `year_from` | int | Papers from this year onwards |
| `year_to` | int | Papers up to this year |
| `keywords` | list[str] | Papers containing any of these keywords |

**Usage Examples:**

```python
# Find papers by venue
search_papers(venue="NDSS")
# Returns: All papers from NDSS (matches "Network and Distributed System Security")

# Find papers by domain
search_papers(domain="side-channel")
# Returns: Papers in domains containing "side-channel"

# Find papers by author
search_papers(author="Heiser")
# Returns: All papers with "Heiser" as author

# Find recent papers on a topic
search_papers(query="verification", year_from=2020)
# Returns: Papers about verification from 2020 onwards

# Combine multiple filters
search_papers(venue="RTSS", domain="mixed-criticality", year_from=2020)
# Returns: RTSS papers about mixed-criticality from 2020+

# Find papers by keywords
search_papers(keywords=["seL4", "microkernel"])
# Returns: Papers with either "seL4" or "microkernel" in keywords
```

**Why unified search instead of separate tools?**

| Approach | Pros | Cons |
|----------|------|------|
| Separate tools (`search_by_venue`, `search_by_domain`, etc.) | Simpler per tool | Many tools to maintain, can't combine filters |
| **Unified `search_papers`** | Flexible, combinable filters | Slightly more complex schema |

The unified approach lets Claude Code answer complex queries like "Find NDSS papers from 2024 about kernel security" with a single tool call.

**Input Schema:**

```python
Tool(
    name="search_papers",
    description="Find papers by multiple criteria. All filters are optional and combinable.",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text search in title/abstract"
            },
            "venue": {
                "type": "string",
                "description": "Filter by venue (partial match, e.g., 'NDSS' matches 'Network and Distributed System Security')"
            },
            "domain": {
                "type": "string",
                "description": "Filter by domain (partial match)"
            },
            "author": {
                "type": "string",
                "description": "Filter by author name (partial match)"
            },
            "year": {
                "type": "integer",
                "description": "Filter by exact publication year"
            },
            "year_from": {
                "type": "integer",
                "description": "Filter by year range start (inclusive)"
            },
            "year_to": {
                "type": "integer",
                "description": "Filter by year range end (inclusive)"
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by keywords (matches if paper has ANY of these)"
            }
        }
    }
)
```

### Section Tools (Unchanged)

| Tool | Description |
|------|-------------|
| `list_sections` | Get document structure |
| `get_section_summary` | Get LLM summary of a section |
| `get_section_text` | Get full original text of a section |
| `get_section_key_points` | Get verbatim key points |

### Page Tools (Unchanged)

| Tool | Description |
|------|-------------|
| `get_page` | Get text of a single page |
| `get_pages` | Get text of page range |

### Retrieval Tools (Unchanged)

| Tool | Description |
|------|-------------|
| `get_paper_metadata` | Get paper info (now includes keywords, domain) |
| `get_methodology` | Get methodology extraction |
| `get_findings` | Get results/findings |
| `get_limitations` | Get limitations |
| `get_statistics` | Get statistics |

### Analysis Tools (Unchanged)

| Tool | Description |
|------|-------------|
| `query_paper` | Ask custom question (uses local LLM) |

---

## Example Usage

### Searching by Venue

```
User: "Find all my NDSS papers"

Claude Code:
> list_venues()
> 
> Venues in database:
>   • Proceedings 2025 Network and Distributed System Security Symposium (5 papers)
>   • IEEE RTSS (12 papers)
>   • USENIX Security (3 papers)
>   ...

> search_papers(venue="NDSS")
>
> Found 5 papers:
>   • guo_2025_bulkhead - BULKHEAD: Secure, Scalable Kernel Compartmentalization
>   • kim_2025_hakcs - Preventing Kernel Hacks with HAKCs
>   ...
```

### Searching by Domain

```
User: "What papers do I have on side-channel attacks?"

Claude Code:
> list_domains()
>
> Domains in database:
>   • microkernel formal verification using Isabelle (15 papers)
>   • acoustic side-channel attacks on input devices (3 papers)
>   • mixed-criticality scheduling with temporal isolation (8 papers)
>   ...

> search_papers(domain="side-channel")
>
> Found 3 papers:
>   • zhang_2024_acoustic - Acoustic Emanations from Keyboards
>   • liu_2023_power - Power Analysis of Embedded Systems
>   ...
```

### Enhanced Metadata Display

```
User: "Tell me about the seL4 paper"

Claude Code:
> get_paper_metadata("klein_2009_sel4")
>
> ## klein_2009_sel4
> 
> **Title:** seL4: Formal Verification of an OS Kernel
> **Authors:** Klein, Elphinstone, Heiser, et al.
> **Year:** 2009
> **Venue:** ACM SIGOPS Operating Systems Review
> 
> **Domain:** microkernel formal verification using Isabelle
> **Keywords:** seL4, formal verification, microkernel, Isabelle/HOL (from paper)
>
> **Abstract:** We present seL4, the first operating system kernel...
```

### Keyword Trust Indicator

```
User: "What keywords does this paper have?"

Claude Code:
> get_paper_metadata("smith_2024_analysis")
>
> **Keywords:** machine learning, neural networks, optimization (from paper) ← trustworthy
> 
> vs.
>
> **Keywords:** deep learning, classification, accuracy (from LLM) ← verify before citing
```

---

## Storage Estimates

| Data | Per Paper | 1000 Papers |
|------|-----------|-------------|
| papers (with keywords, domain) | ~55 KB | ~55 MB |
| domains table | ~100 bytes | ~10 KB (100 domains) |
| sections | ~5 KB | ~5 MB |
| chunks | ~60 KB | ~60 MB |
| extractions | ~10 KB | ~10 MB |
| **SQLite Total** | ~130 KB | **~130 MB** |
| ChromaDB | ~100 KB | **~100 MB** |

---

## Migration from v3.2

### Database Migration

```sql
-- Add new columns to papers table
ALTER TABLE papers ADD COLUMN keywords JSON DEFAULT '[]';
ALTER TABLE papers ADD COLUMN keywords_source TEXT;
ALTER TABLE papers ADD COLUMN domain TEXT;

-- Create index for new searchable fields
CREATE INDEX ix_papers_domain ON papers(domain);
CREATE INDEX ix_papers_journal_or_venue ON papers(journal_or_venue);

-- Create domains table
CREATE TABLE domains (
    domain_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    paper_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parent_category TEXT
);
CREATE INDEX ix_domains_name ON domains(name);

-- Remove redundant columns from extractions (optional)
-- Note: SQLite doesn't support DROP COLUMN easily, may need to recreate table
```

### Re-import Strategy

Papers need re-import to populate keywords and domain:

```bash
# Re-import all papers with keyword/domain extraction
python zotero_import.py --all --force
```

Estimated time: ~60s per paper for keyword + domain extraction.

---

## Configuration

### .env File (Updated)

```bash
# Zotero paths
ACADEMIC_ZOTERO_PATH=~/Zotero

# Database
ACADEMIC_DATABASE_URL=sqlite:///data/papers.db
ACADEMIC_CHROMA_PERSIST_DIR=./data/chroma

# Models
ACADEMIC_LLM_MODEL=qwen2.5:3b
ACADEMIC_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
ACADEMIC_OLLAMA_HOST=http://localhost:11434

# v3.3: Domain classification settings
ACADEMIC_MIN_DOMAIN_REUSE_SIMILARITY=0.8  # When to reuse vs create new domain
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Keywords from paper first** | Author-provided keywords are authoritative |
| **keywords_source tracking** | Claude Code knows what to trust |
| **Single domain per paper** | Forces decisive classification, simpler search |
| **Specific domains** | Research-actionable: "What papers use similar methods?" |
| **Self-organizing taxonomy** | Grows with your research, no manual curation |
| **LLM checks existing domains** | Prevents duplicate/similar domains |
| **Venue indexed** | Fast filtering by conference/journal |

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
│   ├── server.py                 # MCP server (enhanced search tools)
│   │
│   ├── models/
│   │   ├── database.py           # Updated: keywords, domain, Domain table
│   │   └── vectors.py            # ChromaDB wrapper
│   │
│   ├── processing/
│   │   ├── pdf_processor.py      # PDF extraction
│   │   ├── section_detector.py   # LLM section detection
│   │   ├── chunker.py            # Page-based chunking
│   │   ├── extractor.py          # Verbatim extraction
│   │   ├── keyword_extractor.py  # NEW: Keyword extraction
│   │   └── domain_classifier.py  # NEW: Domain classification
│   │
│   ├── zotero/
│   │   ├── reader.py             # Read Zotero DB
│   │   ├── sync.py               # Import/sync (updated for keywords/domain)
│   │   └── models.py
│   │
│   └── utils/
│
├── data/
│   ├── papers.db                 # SQLite (updated schema)
│   └── chroma/                   # Vector embeddings
│
└── tests/
```

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
