"""
Zotero sync and import logic.
Handles importing papers from Zotero into the MCP database.
"""

import hashlib
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from .reader import ZoteroReader
from .models import ZoteroItem, ImportResult


class ZoteroSync:
    """
    Sync papers from Zotero to MCP database.
    """
    
    def __init__(
        self,
        reader: ZoteroReader,
        database,  # Database instance
        vector_store,  # VectorStore instance
        extractor,  # QualityExtractor instance
        pdf_processor,  # PDFProcessor instance
        chunker,  # SemanticChunker instance
    ):
        self.reader = reader
        self.db = database
        self.vectors = vector_store
        self.extractor = extractor
        self.pdf_processor = pdf_processor
        self.chunker = chunker
    
    def import_item(
        self,
        zotero_item: ZoteroItem,
        force_reprocess: bool = False
    ) -> ImportResult:
        """
        Import a single Zotero item into MCP database.
        
        Args:
            zotero_item: The Zotero item to import
            force_reprocess: Re-run LLM extraction even if paper exists
            
        Returns:
            ImportResult with status and details
        """
        start_time = time.time()
        
        # Must have citation key
        if not zotero_item.citation_key:
            return ImportResult(
                status="skipped",
                paper_id="",
                message="No citation key (Better BibTeX not configured?)"
            )
        
        paper_id = zotero_item.citation_key
        
        # Must have PDF
        if not zotero_item.has_pdf():
            return ImportResult(
                status="skipped",
                paper_id=paper_id,
                citation_key=paper_id,
                message="No PDF attachment"
            )
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_hash(zotero_item.pdf_path)
        
        # Import models here to avoid circular imports
        from ..models.database import Paper, Extraction, Chunk, ProcessingStatus
        from ..models.vectors import ChunkDocument, PaperSummary, SectionType
        
        # Check for existing paper
        with self.db.get_session() as session:
            # Check by hash first (catches duplicates from any source)
            existing_by_hash = session.query(Paper).filter(
                Paper.file_hash == file_hash
            ).first()
            
            if existing_by_hash:
                if existing_by_hash.paper_id == paper_id:
                    # Same paper, same key - update Zotero metadata
                    self._update_from_zotero(existing_by_hash, zotero_item)
                    session.commit()
                    
                    return ImportResult(
                        status="updated",
                        paper_id=paper_id,
                        citation_key=paper_id,
                        message="Updated Zotero metadata",
                        time_seconds=time.time() - start_time
                    )
                else:
                    # Same PDF, different key - merge (add Zotero link)
                    self._merge_zotero_info(existing_by_hash, zotero_item)
                    session.commit()
                    
                    return ImportResult(
                        status="merged",
                        paper_id=existing_by_hash.paper_id,
                        citation_key=paper_id,
                        message=f"Merged with existing paper {existing_by_hash.paper_id}",
                        time_seconds=time.time() - start_time
                    )
            
            # Check by paper_id (citation key)
            existing_by_id = session.get(Paper, paper_id)
            
            if existing_by_id and not force_reprocess:
                # Already imported with this citation key
                self._update_from_zotero(existing_by_id, zotero_item)
                session.commit()
                
                return ImportResult(
                    status="updated",
                    paper_id=paper_id,
                    citation_key=paper_id,
                    message="Updated Zotero metadata",
                    time_seconds=time.time() - start_time
                )
        
        # New paper - full import
        try:
            # Step 1: Extract PDF text
            doc = self.pdf_processor.extract_text_with_structure(
                str(zotero_item.pdf_path)
            )
            
            # Step 2: Create chunks
            chunks = self.chunker.chunk_document(doc)
            
            # Step 3: LLM extraction (3 passes)
            extractions = self.extractor.extract_all(doc)
            
            # Step 4: Save to database
            with self.db.get_session() as session:
                # Delete existing if force reprocess
                if force_reprocess:
                    existing = session.get(Paper, paper_id)
                    if existing:
                        session.delete(existing)
                        session.flush()
                
                paper = Paper(
                    paper_id=paper_id,
                    
                    # Zotero linking
                    zotero_key=zotero_item.item_key,
                    zotero_item_id=zotero_item.item_id,
                    zotero_collections=zotero_item.collections,
                    
                    # File info
                    file_path=str(zotero_item.pdf_path),
                    file_hash=file_hash,
                    
                    # Metadata FROM ZOTERO
                    title=zotero_item.title,
                    authors=zotero_item.authors,
                    abstract=zotero_item.abstract,
                    publication_date=zotero_item.date,
                    year=zotero_item.year,
                    journal_or_venue=zotero_item.publication_title,
                    doi=zotero_item.doi,
                    
                    # Processing info
                    processing_status=ProcessingStatus.COMPLETED,
                    processing_model=self.extractor.model,
                    processed_at=datetime.utcnow(),
                    page_count=doc.page_count,
                    word_count=doc.word_count,
                    
                    # Full text for on-demand queries
                    full_text=doc.full_text
                )
                
                extraction = Extraction(
                    paper_id=paper_id,
                    
                    # Methodology (LLM Pass 1)
                    methodology_summary=self._ensure_string(
                        extractions.get("methodology_summary")
                    ),
                    methodology_detailed=self._ensure_string(
                        extractions.get("methodology_detailed")
                    ),
                    study_design=self._ensure_string(
                        extractions.get("study_design")
                    ),
                    sample_description=self._ensure_string(
                        extractions.get("sample_description")
                    ),
                    sample_size=self._ensure_string(
                        extractions.get("sample_size")
                    ),
                    data_collection_methods=self._ensure_list(
                        extractions.get("data_collection_methods")
                    ),
                    analysis_methods=self._ensure_list(
                        extractions.get("analysis_methods")
                    ),
                    statistical_tests=self._ensure_list(
                        extractions.get("statistical_tests")
                    ),
                    software_tools=self._ensure_list(
                        extractions.get("software_tools")
                    ),
                    
                    # Findings (LLM Pass 2)
                    key_findings=self._ensure_list(
                        extractions.get("key_findings")
                    ),
                    quantitative_results=self._ensure_dict(
                        extractions.get("quantitative_results")
                    ),
                    qualitative_themes=self._ensure_list(
                        extractions.get("qualitative_themes")
                    ),
                    effect_sizes=self._ensure_list(
                        extractions.get("effect_sizes")
                    ),
                    
                    # Critical Analysis (LLM Pass 3)
                    main_arguments=self._ensure_list(
                        extractions.get("main_arguments")
                    ),
                    theoretical_contributions=self._ensure_string(
                        extractions.get("theoretical_contributions")
                    ),
                    practical_implications=self._ensure_string(
                        extractions.get("practical_implications")
                    ),
                    limitations=self._ensure_list(
                        extractions.get("limitations")
                    ),
                    future_research=self._ensure_list(
                        extractions.get("future_research")
                    ),
                    
                    # Classification (from Zotero + LLM)
                    research_domain=self._ensure_string(
                        extractions.get("research_domain")
                    ),
                    subdomain=self._ensure_string(
                        extractions.get("subdomain")
                    ),
                    methodology_type=self._ensure_string(
                        extractions.get("methodology_type")
                    ),
                    paper_type=self._ensure_string(
                        extractions.get("paper_type")
                    ),
                    keywords=self._ensure_list(
                        extractions.get("keywords")
                    ) + zotero_item.tags  # Add Zotero tags
                )
                
                paper.extraction = extraction
                
                # Save chunks to database
                for i, chunk in enumerate(chunks):
                    db_chunk = Chunk(
                        chunk_id=f"{paper_id}_chunk_{i}",
                        paper_id=paper_id,
                        section_type=chunk.section_type,
                        section_title=chunk.section_title,
                        chunk_index=i,
                        content=chunk.content,
                        token_count=chunk.token_count,
                        has_equations=chunk.has_equations,
                        has_tables=chunk.has_tables
                    )
                    session.add(db_chunk)
                
                session.add(paper)
                session.commit()
            
            # Step 5: Index vectors
            section_map = {
                "abstract": SectionType.ABSTRACT,
                "introduction": SectionType.INTRODUCTION,
                "literature_review": SectionType.LITERATURE_REVIEW,
                "methodology": SectionType.METHODOLOGY,
                "results": SectionType.RESULTS,
                "discussion": SectionType.DISCUSSION,
                "conclusion": SectionType.CONCLUSION,
                "references": SectionType.REFERENCES,
                "other": SectionType.OTHER
            }
            
            chunk_docs = [
                ChunkDocument(
                    chunk_id=f"{paper_id}_chunk_{i}",
                    paper_id=paper_id,
                    content=chunk.content,
                    section_type=section_map.get(chunk.section_type, SectionType.OTHER),
                    section_title=chunk.section_title,
                    chunk_index=i,
                    token_count=chunk.token_count
                )
                for i, chunk in enumerate(chunks)
            ]
            self.vectors.add_chunks(chunk_docs)
            
            # Paper summary embedding
            findings_text = ""
            key_findings = self._ensure_list(extractions.get("key_findings"))
            if key_findings:
                findings_text = " ".join(
                    f.get("finding", str(f)) if isinstance(f, dict) else str(f)
                    for f in key_findings[:5]
                )
            
            summary = PaperSummary(
                paper_id=paper_id,
                title=zotero_item.title or paper_id,
                abstract=zotero_item.abstract,
                key_findings_text=findings_text,
                keywords=self._ensure_list(extractions.get("keywords")) + zotero_item.tags,
                research_domain=self._ensure_string(extractions.get("research_domain")),
                methodology_type=self._ensure_string(extractions.get("methodology_type")),
                paper_type=self._ensure_string(extractions.get("paper_type"))
            )
            self.vectors.add_paper_summary(summary)
            
            elapsed = time.time() - start_time
            
            return ImportResult(
                status="imported",
                paper_id=paper_id,
                citation_key=paper_id,
                time_seconds=elapsed,
                pages=doc.page_count,
                chunks=len(chunks),
                title=zotero_item.title
            )
            
        except Exception as e:
            # Mark as failed
            with self.db.get_session() as session:
                from ..models.database import Paper, ProcessingStatus
                
                paper = Paper(
                    paper_id=paper_id,
                    zotero_key=zotero_item.item_key,
                    file_path=str(zotero_item.pdf_path),
                    file_hash=file_hash,
                    title=zotero_item.title,
                    processing_status=ProcessingStatus.FAILED
                )
                session.merge(paper)
                session.commit()
            
            return ImportResult(
                status="failed",
                paper_id=paper_id,
                citation_key=paper_id,
                message=str(e),
                time_seconds=time.time() - start_time
            )
    
    def import_collection(
        self,
        collection_name: str,
        force_reprocess: bool = False,
        progress_callback=None
    ) -> list[ImportResult]:
        """
        Import all papers from a Zotero collection.
        
        Args:
            collection_name: Name of the collection to import
            force_reprocess: Re-run extraction even if papers exist
            progress_callback: Optional callback(current, total, item_name)
            
        Returns:
            List of ImportResult for each paper
        """
        items = self.reader.get_items(
            collection_name=collection_name,
            has_pdf=False  # Get all, we'll report which don't have PDFs
        )
        
        results = []
        
        for i, item in enumerate(items):
            if progress_callback:
                progress_callback(i + 1, len(items), item.title or item.item_key)
            
            result = self.import_item(item, force_reprocess)
            results.append(result)
        
        return results
    
    def import_all(
        self,
        force_reprocess: bool = False,
        progress_callback=None
    ) -> list[ImportResult]:
        """Import all papers from Zotero library."""
        items = self.reader.get_items(has_pdf=False)
        
        results = []
        
        for i, item in enumerate(items):
            if progress_callback:
                progress_callback(i + 1, len(items), item.title or item.item_key)
            
            result = self.import_item(item, force_reprocess)
            results.append(result)
        
        return results
    
    def sync_metadata(self) -> list[ImportResult]:
        """
        Sync metadata for all imported papers from Zotero.
        Does not re-run LLM extraction.
        """
        from ..models.database import Paper
        
        results = []
        
        with self.db.get_session() as session:
            papers = session.query(Paper).filter(
                Paper.zotero_key.isnot(None)
            ).all()
            
            for paper in papers:
                item = self.reader.get_item_by_key(paper.zotero_key)
                
                if item:
                    self._update_from_zotero(paper, item)
                    results.append(ImportResult(
                        status="updated",
                        paper_id=paper.paper_id,
                        citation_key=paper.paper_id,
                        message="Synced from Zotero"
                    ))
                else:
                    results.append(ImportResult(
                        status="skipped",
                        paper_id=paper.paper_id,
                        message="Not found in Zotero"
                    ))
            
            session.commit()
        
        return results
    
    def _update_from_zotero(self, paper, zotero_item: ZoteroItem):
        """Update paper metadata from Zotero."""
        paper.title = zotero_item.title
        paper.authors = zotero_item.authors
        paper.abstract = zotero_item.abstract
        paper.publication_date = zotero_item.date
        paper.year = zotero_item.year
        paper.journal_or_venue = zotero_item.publication_title
        paper.doi = zotero_item.doi
        paper.zotero_collections = zotero_item.collections
    
    def _merge_zotero_info(self, paper, zotero_item: ZoteroItem):
        """Merge Zotero info into existing paper."""
        paper.zotero_key = zotero_item.item_key
        paper.zotero_item_id = zotero_item.item_id
        
        # Merge collections (keep existing + add new)
        existing_collections = paper.zotero_collections or []
        new_collections = zotero_item.collections
        paper.zotero_collections = list(set(existing_collections + new_collections))
        
        # Update metadata from Zotero (more reliable)
        self._update_from_zotero(paper, zotero_item)
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]
    
    @staticmethod
    def _ensure_string(value) -> Optional[str]:
        """Convert value to string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "; ".join(str(v) for v in value)
        if isinstance(value, dict):
            import json
            return json.dumps(value)
        return str(value)
    
    @staticmethod
    def _ensure_list(value) -> list:
        """Ensure value is a list."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value] if value else []
        return [value]
    
    @staticmethod
    def _ensure_dict(value) -> Optional[dict]:
        """Ensure value is a dict or None."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            import json
            try:
                return json.loads(value)
            except:
                return {"value": value}
        return {"value": str(value)}
