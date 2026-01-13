"""
SQLAlchemy ORM models for academic paper storage.
v3: Zotero integration with citation key as paper_id.
"""

from datetime import datetime
from typing import Optional
from contextlib import contextmanager
from sqlalchemy import (
    create_engine, ForeignKey, Text, Index,
    UniqueConstraint
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, 
    relationship, Session, sessionmaker
)
from sqlalchemy.dialects.sqlite import JSON
from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SectionType(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    OTHER = "other"


class MethodologyType(str, Enum):
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    MIXED = "mixed"
    THEORETICAL = "theoretical"
    REVIEW = "review"


class PaperType(str, Enum):
    EMPIRICAL = "empirical"
    REVIEW = "review"
    META_ANALYSIS = "meta_analysis"
    THEORETICAL = "theoretical"
    CASE_STUDY = "case_study"
    POSITION_PAPER = "position_paper"


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Paper(Base):
    """
    Core paper entity storing metadata and processing state.
    
    paper_id = BibTeX citation key (e.g., "lyons_2023_mixedcriticality")
    This enables direct integration with LaTeX \\cite{} commands.
    """
    __tablename__ = "papers"
    
    # Primary key = citation key (e.g., "lyons_2023_mixedcriticality")
    paper_id: Mapped[str] = mapped_column(primary_key=True)
    
    # Zotero linking
    zotero_key: Mapped[Optional[str]] = mapped_column(index=True)
    # Storage folder key (e.g., "HTJHSCCZ")
    zotero_item_id: Mapped[Optional[int]]
    # Zotero internal ID
    zotero_collections: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Collection names (e.g., ["PhD Research", "seL4 Papers"])
    
    # File information
    file_path: Mapped[str] = mapped_column(nullable=False)
    file_hash: Mapped[Optional[str]] = mapped_column(index=True)
    # SHA256 hash for deduplication
    
    # Metadata FROM ZOTERO (trusted source)
    title: Mapped[Optional[str]] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    publication_date: Mapped[Optional[str]]
    year: Mapped[Optional[str]]
    journal_or_venue: Mapped[Optional[str]]
    doi: Mapped[Optional[str]] = mapped_column(index=True)
    
    # Processing metadata
    imported_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    processed_at: Mapped[Optional[datetime]]
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        default=ProcessingStatus.PENDING
    )
    processing_model: Mapped[Optional[str]]
    processing_version: Mapped[str] = mapped_column(default="3.0")
    
    # Quality metrics
    extraction_confidence: Mapped[Optional[float]]
    page_count: Mapped[Optional[int]]
    word_count: Mapped[Optional[int]]
    
    # Full text storage (for on-demand queries)
    full_text: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationships
    extraction: Mapped[Optional["Extraction"]] = relationship(
        back_populates="paper",
        uselist=False,
        cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan"
    )
    citations: Mapped[list["Citation"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan"
    )
    processing_logs: Mapped[list["ProcessingLog"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("idx_papers_status", "processing_status"),
        Index("idx_papers_zotero", "zotero_key"),
    )
    
    def __repr__(self) -> str:
        return f"Paper(id={self.paper_id!r}, title={self.title!r})"
    
    @property
    def citation_key(self) -> str:
        """Alias for paper_id (BibTeX cite key)."""
        return self.paper_id


class Extraction(Base):
    """
    Structured extractions from a paper.
    
    Contains LLM-extracted information (methodology, findings, limitations).
    Basic metadata (title, authors, abstract) comes from Zotero.
    """
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
    
    # Classification
    research_domain: Mapped[Optional[str]]
    subdomain: Mapped[Optional[str]]
    methodology_type: Mapped[Optional[str]]  # Changed from enum for flexibility
    paper_type: Mapped[Optional[str]]  # Changed from enum for flexibility
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Includes both LLM-extracted and Zotero tags
    
    paper: Mapped["Paper"] = relationship(back_populates="extraction")
    
    def __repr__(self) -> str:
        return f"Extraction(paper_id={self.paper_id!r})"


class Chunk(Base):
    """Text chunks for vector search."""
    __tablename__ = "chunks"
    
    chunk_id: Mapped[str] = mapped_column(primary_key=True)
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        index=True
    )
    
    section_type: Mapped[str]  # Changed from enum for flexibility
    section_title: Mapped[Optional[str]]
    page_numbers: Mapped[Optional[str]]
    chunk_index: Mapped[int]
    
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_clean: Mapped[Optional[str]] = mapped_column(Text)
    token_count: Mapped[int]
    
    has_equations: Mapped[bool] = mapped_column(default=False)
    has_tables: Mapped[bool] = mapped_column(default=False)
    has_figures: Mapped[bool] = mapped_column(default=False)
    citation_count: Mapped[int] = mapped_column(default=0)
    
    embedding_id: Mapped[Optional[str]]
    
    paper: Mapped["Paper"] = relationship(back_populates="chunks")
    
    __table_args__ = (
        Index("idx_chunks_section", "section_type"),
        UniqueConstraint("paper_id", "chunk_index", name="uq_chunk_order"),
    )
    
    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id!r}, section={self.section_type})"


class Citation(Base):
    """Citations extracted from papers."""
    __tablename__ = "citations"
    
    citation_id: Mapped[str] = mapped_column(primary_key=True)
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        index=True
    )
    
    cited_title: Mapped[Optional[str]] = mapped_column(Text)
    cited_authors: Mapped[Optional[str]]
    cited_year: Mapped[Optional[str]]
    cited_venue: Mapped[Optional[str]]
    
    citation_context: Mapped[Optional[str]] = mapped_column(Text)
    citation_purpose: Mapped[Optional[str]]
    in_text_marker: Mapped[Optional[str]]
    
    paper: Mapped["Paper"] = relationship(back_populates="citations")
    
    def __repr__(self) -> str:
        return f"Citation(id={self.citation_id!r})"


class ProcessingLog(Base):
    """Processing log for debugging."""
    __tablename__ = "processing_logs"
    
    log_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        index=True
    )
    
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    stage: Mapped[str]
    status: Mapped[str]
    message: Mapped[Optional[str]] = mapped_column(Text)
    duration_seconds: Mapped[Optional[float]]
    
    paper: Mapped["Paper"] = relationship(back_populates="processing_logs")
    
    def __repr__(self) -> str:
        return f"ProcessingLog(paper={self.paper_id!r}, stage={self.stage!r})"


class Database:
    """Database connection manager."""
    
    def __init__(self, db_url: str = "sqlite:///data/papers.db"):
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
    
    @contextmanager
    def get_session(self):
        """Get a database session with context manager support."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def __enter__(self) -> Session:
        self._session = self.SessionLocal()
        return self._session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._session.rollback()
        self._session.close()
