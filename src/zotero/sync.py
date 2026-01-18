"""
Zotero sync and import logic.

v3.3: Added keywords extraction and domain classification.
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models.database import (
    Database, Paper, Section, Chunk, Extraction, Domain,
    ProcessingStatus, SectionType
)
from ..models.vectors import VectorStore
from ..processing.pdf_processor import PDFProcessor, ExtractedDocument
from ..processing.section_detector import SectionDetector, DetectedSection
from ..processing.chunker import PageChunker
from ..processing.extractor import SectionExtractor
from ..processing.keyword_extractor import KeywordExtractor
from ..processing.domain_classifier import DomainClassifier
from .reader import ZoteroReader
from .models import ZoteroItem


@dataclass
class ImportResult:
    """Result of importing a paper."""
    paper_id: str
    status: str  # imported | updated | skipped | failed | merged
    message: Optional[str] = None
    time_seconds: Optional[float] = None
    page_count: Optional[int] = None
    section_count: Optional[int] = None


class ZoteroSync:
    """
    Sync papers from Zotero into MCP database.
    
    Flow:
    1. Read item from Zotero
    2. Extract PDF with page boundaries
    3. Extract keywords (from PDF or LLM)
    4. Classify domain (self-organizing taxonomy)
    5. Detect sections with LLM
    6. Extract content per section
    7. Create page-based chunks
    8. Index vectors
    9. Store everything
    """
    
    def __init__(
        self,
        reader: ZoteroReader,
        database: Database,
        vector_store: VectorStore,
        section_detector: SectionDetector,
        extractor: SectionExtractor,
        pdf_processor: PDFProcessor,
        chunker: PageChunker,
        keyword_extractor: Optional[KeywordExtractor] = None,
        domain_classifier: Optional[DomainClassifier] = None
    ):
        self.reader = reader
        self.db = database
        self.vectors = vector_store
        self.section_detector = section_detector
        self.extractor = extractor
        self.pdf_processor = pdf_processor
        self.chunker = chunker
        self.keyword_extractor = keyword_extractor
        self.domain_classifier = domain_classifier
    
    def import_item(
        self,
        zotero_item: ZoteroItem,
        force_reprocess: bool = False
    ) -> ImportResult:
        """
        Import a single Zotero item.
        
        Args:
            zotero_item: Item from Zotero reader
            force_reprocess: Reprocess even if exists
            
        Returns:
            ImportResult with status
        """
        start_time = time.time()
        
        paper_id = zotero_item.citation_key
        if not paper_id:
            return ImportResult(
                paper_id="unknown",
                status="skipped",
                message="No citation key (Better BibTeX not configured?)"
            )
        
        # Check for PDF
        if not zotero_item.has_pdf():
            return ImportResult(
                paper_id=paper_id,
                status="skipped",
                message="No PDF attachment"
            )
        
        # Calculate file hash for deduplication
        pdf_path = zotero_item.pdf_path
        file_hash = self._calculate_hash(pdf_path)
        
        # Check for existing paper
        with self.db.get_session() as session:
            # Check by hash first (catches duplicates)
            existing_by_hash = session.query(Paper).filter(
                Paper.file_hash == file_hash
            ).first()
            
            if existing_by_hash and not force_reprocess:
                if existing_by_hash.paper_id == paper_id:
                    # Same paper, update metadata from Zotero
                    self._update_metadata(session, existing_by_hash, zotero_item)
                    session.commit()
                    return ImportResult(
                        paper_id=paper_id,
                        status="updated",
                        message="Metadata updated from Zotero"
                    )
                else:
                    # Same PDF, different key - merge
                    return ImportResult(
                        paper_id=paper_id,
                        status="merged",
                        message=f"Same PDF as {existing_by_hash.paper_id}"
                    )
            
            # Check by paper_id
            existing_by_id = session.query(Paper).filter(
                Paper.paper_id == paper_id
            ).first()
            
            if existing_by_id and not force_reprocess:
                return ImportResult(
                    paper_id=paper_id,
                    status="skipped",
                    message="Already imported"
                )
            
            # Delete existing if force reprocess
            if existing_by_id and force_reprocess:
                session.delete(existing_by_id)
                session.flush()
                # Also delete from vector store
                self.vectors.delete_paper(paper_id)
        
        # Process the paper
        try:
            result = self._process_paper(zotero_item, paper_id, file_hash)
            result.time_seconds = time.time() - start_time
            return result
            
        except Exception as e:
            return ImportResult(
                paper_id=paper_id,
                status="failed",
                message=str(e),
                time_seconds=time.time() - start_time
            )
    
    def _process_paper(
        self,
        zotero_item: ZoteroItem,
        paper_id: str,
        file_hash: str
    ) -> ImportResult:
        """Process a paper through the full pipeline."""
        
        # Step 1: Extract PDF with page boundaries
        print(f"    Extracting PDF...")
        doc = self.pdf_processor.extract_with_pages(str(zotero_item.pdf_path))
        
        # Step 2: Extract keywords (NEW in v3.3)
        keywords = []
        keywords_source = None
        if self.keyword_extractor:
            print(f"    Extracting keywords...")
            keywords, keywords_source = self.keyword_extractor.extract(
                doc.full_text,
                abstract=zotero_item.abstract,
                title=zotero_item.title
            )
            # Also include Zotero tags as keywords (always trustworthy)
            if zotero_item.tags:
                keywords = list(set(keywords + [t.lower() for t in zotero_item.tags]))
                if keywords_source != "paper":
                    keywords_source = "mixed"  # Mix of LLM and Zotero tags
        
        # Step 3: Classify domain (NEW in v3.3)
        domain = None
        is_new_domain = False
        if self.domain_classifier:
            print(f"    Classifying domain...")
            # Get existing domains
            existing_domains = self._get_existing_domains()
            domain, is_new_domain = self.domain_classifier.classify(
                abstract=zotero_item.abstract or doc.full_text[:2000],
                title=zotero_item.title,
                keywords=keywords,
                existing_domains=existing_domains
            )
        
        # Step 4: Detect sections
        print(f"    Detecting sections ({doc.page_count} pages)...")
        section_result = self.section_detector.detect_sections(doc)
        detected_sections = section_result.sections
        
        # Step 5: Extract content per section
        print(f"    Extracting content ({len(detected_sections)} sections)...")
        extractions = self.extractor.extract_all(doc, detected_sections)
        
        # Step 6: Create page-based chunks
        print(f"    Creating chunks...")
        chunks = self.chunker.chunk_document(doc, paper_id, detected_sections)
        
        # Step 7: Store in database
        print(f"    Saving to database...")
        with self.db.get_session() as session:
            # Update domain count if new
            if domain:
                self._update_domain_count(session, domain, is_new_domain)
            
            # Create paper record
            paper = Paper(
                paper_id=paper_id,
                
                # Zotero linking
                zotero_key=zotero_item.item_key,
                zotero_item_id=zotero_item.item_id,
                zotero_collections=zotero_item.collections,
                
                # File info
                file_path=str(zotero_item.pdf_path),
                file_hash=file_hash,
                
                # Metadata from Zotero
                title=zotero_item.title,
                authors=zotero_item.authors,
                abstract=zotero_item.abstract,
                publication_date=zotero_item.date,
                year=int(zotero_item.year) if zotero_item.year else None,
                journal_or_venue=zotero_item.publication_title,
                doi=zotero_item.doi,
                
                # NEW in v3.3: Keywords and domain
                keywords=keywords,
                keywords_source=keywords_source,
                domain=domain,
                
                # Document stats
                page_count=doc.page_count,
                word_count=doc.word_count,
                
                # Full text
                full_text=doc.full_text,
                
                # Processing info
                processing_status=ProcessingStatus.COMPLETED,
                processing_model=self.extractor.model,
                processed_at=datetime.utcnow()
            )
            session.add(paper)
            
            # Create section records
            for i, sec in enumerate(detected_sections):
                section_id = f"{paper_id}_sec_{i}"
                section = Section(
                    section_id=section_id,
                    paper_id=paper_id,
                    section_type=sec.section_type,
                    section_title=sec.section_title,
                    section_index=i,
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                    char_start=sec.char_start,
                    char_end=sec.char_end,
                    summary=extractions["section_summaries"].get(f"sec_{i}"),
                    key_points_verbatim=extractions["section_key_points"].get(f"sec_{i}", []),
                    detection_method=section_result.detection_method,
                    confidence=section_result.confidence
                )
                session.add(section)
            
            # Create chunk records
            for chunk in chunks:
                db_chunk = Chunk(
                    chunk_id=chunk.chunk_id,
                    paper_id=paper_id,
                    section_id=chunk.section_id,
                    page_number=chunk.page_number,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    content=chunk.content,
                    word_count=chunk.word_count
                )
                session.add(db_chunk)
            
            # Create extraction record
            extraction = Extraction(
                paper_id=paper_id,
                
                # Verbatim extractions
                methodology_verbatim=extractions.get("methodology_verbatim"),
                evaluation_setup_verbatim=extractions.get("evaluation_setup_verbatim"),
                contributions_verbatim=extractions.get("contributions_verbatim", []),
                results_verbatim=extractions.get("results_verbatim", []),
                statistics_verbatim=extractions.get("statistics_verbatim", []),
                limitations_verbatim=extractions.get("limitations_verbatim", []),
                future_work_verbatim=extractions.get("future_work_verbatim", []),
                
                # Summary
                methodology_summary=extractions.get("methodology_summary"),
                
                # Classification (kept for compatibility)
                methodology_type=extractions.get("methodology_type"),
                paper_type=extractions.get("paper_type"),
                software_tools=extractions.get("software_tools", [])
            )
            session.add(extraction)
            session.commit()
        
        # Step 8: Index vectors
        print(f"    Indexing vectors...")
        for chunk in chunks:
            self.vectors.add_chunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata={
                    "paper_id": paper_id,
                    "page_number": chunk.page_number,
                    "section_type": chunk.section_type or "unknown",
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end
                }
            )
        
        return ImportResult(
            paper_id=paper_id,
            status="imported",
            page_count=doc.page_count,
            section_count=len(detected_sections)
        )
    
    def _get_existing_domains(self) -> list[str]:
        """Get list of existing domain names."""
        with self.db.get_session() as session:
            domains = session.query(Domain.name).all()
            return [d[0] for d in domains]
    
    def _update_domain_count(self, session, domain_name: str, is_new: bool):
        """Update or create domain record."""
        existing = session.query(Domain).filter(Domain.name == domain_name).first()
        
        if existing:
            existing.paper_count += 1
        else:
            new_domain = Domain(
                name=domain_name,
                paper_count=1
            )
            session.add(new_domain)
    
    def _update_metadata(
        self,
        session,
        paper: Paper,
        zotero_item: ZoteroItem
    ):
        """Update paper metadata from Zotero."""
        paper.title = zotero_item.title
        paper.authors = zotero_item.authors
        paper.abstract = zotero_item.abstract
        paper.publication_date = zotero_item.date
        paper.year = zotero_item.year
        paper.journal_or_venue = zotero_item.publication_title
        paper.doi = zotero_item.doi
        paper.zotero_collections = zotero_item.collections
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def import_collection(
        self,
        collection_name: str,
        force_reprocess: bool = False
    ) -> list[ImportResult]:
        """Import all papers from a collection."""
        items = self.reader.get_items(collection_name=collection_name)
        results = []
        
        for i, item in enumerate(items):
            print(f"  [{i+1}/{len(items)}] {item.citation_key or item.title[:50]}")
            result = self.import_item(item, force_reprocess)
            results.append(result)
            
            if result.status == "imported":
                print(f"    ✓ {result.page_count} pages, {result.section_count} sections ({result.time_seconds:.1f}s)")
            elif result.status == "failed":
                print(f"    ✗ {result.message}")
            else:
                print(f"    - {result.status}: {result.message}")
        
        return results
    
    def import_all(
        self,
        force_reprocess: bool = False
    ) -> list[ImportResult]:
        """Import all papers from Zotero."""
        items = self.reader.get_items()
        results = []
        
        for i, item in enumerate(items):
            print(f"  [{i+1}/{len(items)}] {item.citation_key or item.title[:50]}")
            result = self.import_item(item, force_reprocess)
            results.append(result)
            
            if result.status == "imported":
                print(f"    ✓ {result.page_count} pages, {result.section_count} sections ({result.time_seconds:.1f}s)")
            elif result.status == "failed":
                print(f"    ✗ {result.message}")
            else:
                print(f"    - {result.status}: {result.message}")
        
        return results
