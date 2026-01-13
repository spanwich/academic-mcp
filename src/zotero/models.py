"""
Zotero data models.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ZoteroItem:
    """Represents a Zotero library item."""
    
    # Identifiers
    item_id: int                    # Zotero internal ID
    item_key: str                   # Storage folder key (e.g., "HTJHSCCZ")
    citation_key: Optional[str]     # Better BibTeX key (e.g., "lyons_2023_mixed")
    
    # Metadata
    item_type: str                  # "journalArticle", "thesis", "conferencePaper"
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    abstract: Optional[str] = None
    date: Optional[str] = None
    year: Optional[str] = None
    
    # Publication info
    publication_title: Optional[str] = None  # Journal/conference name
    doi: Optional[str] = None
    url: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    
    # Organization
    collections: list[str] = field(default_factory=list)  # Collection names
    tags: list[str] = field(default_factory=list)
    
    # Attachments
    pdf_path: Optional[Path] = None
    
    def has_pdf(self) -> bool:
        """Check if item has an accessible PDF."""
        return self.pdf_path is not None and self.pdf_path.exists()
    
    def get_formatted_authors(self) -> str:
        """Get authors as formatted string."""
        if not self.authors:
            return "Unknown"
        if len(self.authors) == 1:
            return self.authors[0]
        if len(self.authors) == 2:
            return f"{self.authors[0]} and {self.authors[1]}"
        return f"{self.authors[0]} et al."


@dataclass
class ZoteroCollection:
    """Represents a Zotero collection."""
    
    collection_id: int
    collection_key: str
    name: str
    parent_id: Optional[int] = None
    item_count: int = 0


@dataclass 
class ImportResult:
    """Result of importing a paper."""
    
    status: str  # "imported", "updated", "merged", "skipped", "failed"
    paper_id: str
    citation_key: Optional[str] = None
    message: Optional[str] = None
    time_seconds: float = 0.0
    
    # Details for imported papers
    pages: Optional[int] = None
    chunks: Optional[int] = None
    title: Optional[str] = None
