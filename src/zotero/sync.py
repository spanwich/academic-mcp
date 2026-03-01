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


@dataclass
class CleanupResult:
    """Result of cleanup_removed() operation."""
    orphaned: list[dict]  # Papers found in MCP but not in Zotero
    removed: list[str]  # paper_ids actually removed
    domains_updated: list[str]  # Domains with decremented paper_count
    domains_removed: list[str]  # Domains deleted (paper_count reached 0)
    errors: list[str]  # Any errors encountered
    skipped_no_key: int  # Papers skipped because zotero_key is None


@dataclass
class MetadataDiff:
    """A single field difference for a paper."""
    field: str
    old_value: str
    new_value: str


@dataclass
class MetadataSyncResult:
    """Result of metadata comparison for a single paper."""
    paper_id: str
    title: str
    diffs: list[MetadataDiff]
    rekey_needed: bool = False  # Citation key changed
    new_citation_key: Optional[str] = None  # New key if rekey needed
    applied: bool = False  # Whether changes were applied


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
        section_detector: Optional[SectionDetector] = None,
        extractor: Optional[SectionExtractor] = None,
        pdf_processor: Optional[PDFProcessor] = None,
        chunker: Optional[PageChunker] = None,
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
    
    def _require_processing_infra(self):
        """Raise if processing components are not available."""
        missing = []
        if self.section_detector is None:
            missing.append("section_detector")
        if self.extractor is None:
            missing.append("extractor")
        if self.pdf_processor is None:
            missing.append("pdf_processor")
        if self.chunker is None:
            missing.append("chunker")
        if missing:
            raise RuntimeError(
                f"Import requires processing infrastructure: {', '.join(missing)}. "
                "Ensure Ollama is running and all components are initialised."
            )

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
        self._require_processing_infra()
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
                self._cleanup_paper_data(session, existing_by_id)
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
        
        # Step 3: Classify domain (v3.3.1: embedding-based classification)
        domain = None
        domain_description = None
        is_new_domain = False
        if self.domain_classifier:
            print(f"    Classifying domain...")
            # Use embedding-based classification (no existing_domains needed)
            domain, is_new_domain, domain_description = self.domain_classifier.classify(
                abstract=zotero_item.abstract or doc.full_text[:2000],
                title=zotero_item.title,
                keywords=keywords,
                vector_store=self.vectors  # Pass vector store for embedding search
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
            # Update domain count and keywords (v3.3.1: also update embeddings)
            if domain:
                self._update_domain_record(
                    session=session,
                    domain_name=domain,
                    is_new=is_new_domain,
                    description=domain_description,
                    paper_keywords=keywords
                )
            
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
    
    def _update_domain_record(
        self,
        session,
        domain_name: str,
        is_new: bool,
        description: str = None,
        paper_keywords: list[str] = None
    ):
        """
        Update or create domain record and its embedding.

        v3.3.1: Also updates aggregated_keywords and domain embedding.
        """
        existing = session.query(Domain).filter(Domain.name == domain_name).first()

        if existing:
            # Update existing domain
            existing.paper_count += 1

            # Update description if provided and not already set
            if description and not existing.description:
                existing.description = description

            # Aggregate keywords
            if paper_keywords:
                current_keywords = existing.aggregated_keywords or []
                merged_keywords = list(set(
                    [k.lower() for k in current_keywords] +
                    [k.lower() for k in paper_keywords]
                ))
                existing.aggregated_keywords = merged_keywords

                # Update domain embedding with new keywords
                self.vectors.update_domain_keywords(
                    domain_name=domain_name,
                    new_keywords=paper_keywords,
                    description=existing.description
                )
        else:
            # Create new domain
            keywords = [k.lower() for k in paper_keywords] if paper_keywords else []
            new_domain = Domain(
                name=domain_name,
                description=description,
                aggregated_keywords=keywords,
                paper_count=1
            )
            session.add(new_domain)

            # Create domain embedding
            self.vectors.add_domain_embedding(
                domain_name=domain_name,
                keywords=keywords,
                description=description
            )
    
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

    # ==================== Cleanup & Sync ====================

    def _cleanup_paper_data(self, session, paper: Paper):
        """
        Clean up domain accounting and delete paper record.

        Decrements Domain.paper_count for the paper's domain. If the domain
        reaches paper_count 0, removes the Domain record and its ChromaDB
        embedding.

        The Paper record is deleted via session.delete() which cascades to
        sections, chunks, and extractions.

        Args:
            session: Active SQLAlchemy session
            paper: Paper ORM object to remove
        """
        if paper.domain:
            domain = session.query(Domain).filter(
                Domain.name == paper.domain
            ).first()
            if domain:
                domain.paper_count = max(0, domain.paper_count - 1)
                if domain.paper_count == 0:
                    self.vectors.delete_domain(domain.name)
                    session.delete(domain)

        session.delete(paper)

    def remove_paper(self, paper_id: str) -> dict:
        """
        Remove a paper from MCP database and vector store.

        Args:
            paper_id: Citation key of the paper to remove

        Returns:
            Dict with keys: paper_id, status, domain_updated, domain_removed
        """
        result = {
            "paper_id": paper_id,
            "status": "not_found",
            "domain_updated": None,
            "domain_removed": False,
        }

        # Remove vector store chunks first
        self.vectors.delete_paper(paper_id)

        with self.db.get_session() as session:
            paper = session.query(Paper).filter(
                Paper.paper_id == paper_id
            ).first()
            if not paper:
                return result

            domain_name = paper.domain
            result["domain_updated"] = domain_name

            # Check if domain will be removed
            if domain_name:
                domain = session.query(Domain).filter(
                    Domain.name == domain_name
                ).first()
                if domain and domain.paper_count <= 1:
                    result["domain_removed"] = True

            self._cleanup_paper_data(session, paper)
            session.commit()
            result["status"] = "removed"

        return result

    def find_orphaned_papers(self) -> tuple[list[dict], int]:
        """
        Find MCP papers whose zotero_key no longer exists in Zotero.

        Returns:
            Tuple of (list of orphan dicts, count of papers with no zotero_key)
            Each orphan dict: {paper_id, title, zotero_key, domain}
        """
        zotero_keys = self.reader.get_all_item_keys()
        orphans = []
        skipped_no_key = 0

        with self.db.get_session() as session:
            papers = session.query(Paper).all()
            for paper in papers:
                if not paper.zotero_key:
                    skipped_no_key += 1
                    continue
                if paper.zotero_key not in zotero_keys:
                    orphans.append({
                        "paper_id": paper.paper_id,
                        "title": paper.title or "(no title)",
                        "zotero_key": paper.zotero_key,
                        "domain": paper.domain,
                    })

        return orphans, skipped_no_key

    def cleanup_removed(self, dry_run: bool = True) -> CleanupResult:
        """
        Find and optionally remove papers deleted from Zotero.

        Args:
            dry_run: If True, only report orphans without removing them

        Returns:
            CleanupResult with details of what was (or would be) removed
        """
        orphans, skipped_no_key = self.find_orphaned_papers()

        result = CleanupResult(
            orphaned=orphans,
            removed=[],
            domains_updated=[],
            domains_removed=[],
            errors=[],
            skipped_no_key=skipped_no_key,
        )

        if dry_run or not orphans:
            return result

        for orphan in orphans:
            try:
                removal = self.remove_paper(orphan["paper_id"])
                if removal["status"] == "removed":
                    result.removed.append(orphan["paper_id"])
                    if removal["domain_updated"]:
                        result.domains_updated.append(removal["domain_updated"])
                    if removal["domain_removed"]:
                        result.domains_removed.append(removal["domain_updated"])
                else:
                    result.errors.append(
                        f"{orphan['paper_id']}: {removal['status']}"
                    )
            except Exception as e:
                result.errors.append(f"{orphan['paper_id']}: {e}")

        return result

    def sync_metadata(self, dry_run: bool = True) -> list[MetadataSyncResult]:
        """
        Compare MCP paper metadata against live Zotero library and update.

        Compares: title, authors, abstract, year, venue, doi,
        publication_date, collections. Detects citation key renames.

        Args:
            dry_run: If True, only report diffs without applying changes

        Returns:
            List of MetadataSyncResult for papers with differences
        """
        # Load all Zotero items indexed by item_key
        zotero_items = self.reader.get_items()
        zotero_by_key: dict[str, ZoteroItem] = {}
        for item in zotero_items:
            zotero_by_key[item.item_key] = item

        results: list[MetadataSyncResult] = []

        with self.db.get_session() as session:
            papers = session.query(Paper).all()

            for paper in papers:
                if not paper.zotero_key:
                    continue

                zotero_item = zotero_by_key.get(paper.zotero_key)
                if not zotero_item:
                    # Paper not in Zotero — handled by cleanup, skip here
                    continue

                diffs: list[MetadataDiff] = []

                # Compare fields
                field_map = [
                    ("title", paper.title, zotero_item.title),
                    ("authors", paper.authors, zotero_item.authors),
                    ("abstract", paper.abstract, zotero_item.abstract),
                    ("year", str(paper.year) if paper.year else None,
                     zotero_item.year),
                    ("venue", paper.journal_or_venue,
                     zotero_item.publication_title),
                    ("doi", paper.doi, zotero_item.doi),
                    ("publication_date", paper.publication_date,
                     zotero_item.date),
                    ("collections", paper.zotero_collections,
                     zotero_item.collections),
                ]

                for field_name, old_val, new_val in field_map:
                    # Normalise for comparison
                    old_str = str(old_val) if old_val else ""
                    new_str = str(new_val) if new_val else ""
                    if old_str != new_str:
                        diffs.append(MetadataDiff(
                            field=field_name,
                            old_value=old_str,
                            new_value=new_str,
                        ))

                # Detect citation key rename
                rekey_needed = False
                new_citation_key = None
                if (zotero_item.citation_key
                        and zotero_item.citation_key != paper.paper_id):
                    rekey_needed = True
                    new_citation_key = zotero_item.citation_key

                if not diffs and not rekey_needed:
                    continue

                sync_result = MetadataSyncResult(
                    paper_id=paper.paper_id,
                    title=paper.title or "(no title)",
                    diffs=diffs,
                    rekey_needed=rekey_needed,
                    new_citation_key=new_citation_key,
                )

                # Apply changes if not dry run (and no rekey needed)
                if not dry_run and diffs and not rekey_needed:
                    for diff in diffs:
                        if diff.field == "title":
                            paper.title = zotero_item.title
                        elif diff.field == "authors":
                            paper.authors = zotero_item.authors
                        elif diff.field == "abstract":
                            paper.abstract = zotero_item.abstract
                        elif diff.field == "year":
                            paper.year = (int(zotero_item.year)
                                          if zotero_item.year else None)
                        elif diff.field == "venue":
                            paper.journal_or_venue = (
                                zotero_item.publication_title)
                        elif diff.field == "doi":
                            paper.doi = zotero_item.doi
                        elif diff.field == "publication_date":
                            paper.publication_date = zotero_item.date
                        elif diff.field == "collections":
                            paper.zotero_collections = (
                                zotero_item.collections)
                    sync_result.applied = True

                results.append(sync_result)

            if not dry_run:
                session.commit()

        return results

    def rekey_paper(self, old_paper_id: str, new_citation_key: str) -> ImportResult:
        """
        Re-import a paper whose citation key changed in Zotero.

        Removes the old paper and re-imports from Zotero with the new key.
        Requires full processing infrastructure (Ollama, LLM).

        Args:
            old_paper_id: Current paper_id in MCP database
            new_citation_key: New citation key from Zotero/BBT

        Returns:
            ImportResult from the re-import
        """
        self._require_processing_infra()

        # Look up the Zotero item by new citation key
        zotero_item = self.reader.get_item_by_citation_key(new_citation_key)
        if not zotero_item:
            return ImportResult(
                paper_id=old_paper_id,
                status="failed",
                message=f"Citation key '{new_citation_key}' not found in Zotero",
            )

        # Remove old paper
        self.remove_paper(old_paper_id)

        # Re-import with new key
        return self.import_item(zotero_item, force_reprocess=True)
