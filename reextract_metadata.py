#!/usr/bin/env python3
"""
Re-extract metadata (keywords, domains) for already-imported papers.

This script operates on papers already in the database without re-processing PDFs.
Useful after upgrading extraction algorithms or domain classification.

Usage:
    # Re-extract everything for all papers
    python reextract_metadata.py --all

    # Only re-extract keywords (keep existing domains)
    python reextract_metadata.py --all --keywords-only

    # Only re-classify domains (keep existing keywords)
    python reextract_metadata.py --all --domains-only

    # Filter by current domain
    python reextract_metadata.py --domain "cyber-physical systems security"

    # Filter by venue
    python reextract_metadata.py --venue "IEEE"

    # Specific paper
    python reextract_metadata.py --paper-id "smith_2023_security"

    # Dry run (show what would change)
    python reextract_metadata.py --all --dry-run
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import or_
from src.config import get_config
from src.models.database import Database, Domain, Paper
from src.models.vectors import VectorStore
from src.processing.keyword_extractor import KeywordExtractor
from src.processing.domain_classifier import DomainClassifier

settings = get_config()


class MetadataReextractor:
    """Re-extract metadata for existing papers."""

    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        keyword_extractor: KeywordExtractor,
        domain_classifier: DomainClassifier,
        dry_run: bool = False,
        verbose: bool = True
    ):
        self.db = db
        self.vectors = vector_store
        self.keyword_extractor = keyword_extractor
        self.domain_classifier = domain_classifier
        self.dry_run = dry_run
        self.verbose = verbose

        # Stats
        self.processed = 0
        self.keywords_updated = 0
        self.domains_updated = 0
        self.errors = 0

    def reextract_paper(
        self,
        paper: Paper,
        extract_keywords: bool = True,
        extract_domains: bool = True,
        session=None
    ) -> dict:
        """
        Re-extract metadata for a single paper.

        Returns dict with changes made.
        """
        changes = {
            "paper_id": paper.paper_id,
            "keywords_changed": False,
            "domain_changed": False,
            "old_keywords": paper.keywords,
            "new_keywords": None,
            "old_domain": paper.domain,
            "new_domain": None,
            "error": None
        }

        try:
            # Re-extract keywords
            if extract_keywords and paper.full_text:
                new_keywords, source = self.keyword_extractor.extract(
                    full_text=paper.full_text,
                    abstract=paper.abstract,
                    title=paper.title
                )

                # Check if changed
                old_set = set(paper.keywords or [])
                new_set = set(new_keywords)

                if old_set != new_set:
                    changes["keywords_changed"] = True
                    changes["new_keywords"] = new_keywords
                    changes["keywords_source"] = source

                    if not self.dry_run:
                        paper.keywords = new_keywords
                        paper.keywords_source = source
                        self.keywords_updated += 1

            # Re-classify domain
            if extract_domains:
                # Use current keywords (possibly just updated)
                current_keywords = changes["new_keywords"] or paper.keywords

                new_domain, is_new, description = self.domain_classifier.classify(
                    abstract=paper.abstract or (paper.full_text[:2000] if paper.full_text else ""),
                    title=paper.title,
                    keywords=current_keywords,
                    vector_store=self.vectors
                )

                if new_domain != paper.domain:
                    changes["domain_changed"] = True
                    changes["new_domain"] = new_domain
                    changes["domain_is_new"] = is_new
                    changes["domain_description"] = description

                    if not self.dry_run:
                        # Update paper's domain
                        old_domain = paper.domain
                        paper.domain = new_domain

                        # Update domain counts
                        self._update_domain_counts(
                            session=session,
                            old_domain=old_domain,
                            new_domain=new_domain,
                            is_new=is_new,
                            description=description,
                            paper_keywords=current_keywords
                        )
                        self.domains_updated += 1

            self.processed += 1

        except Exception as e:
            changes["error"] = str(e)
            self.errors += 1
            if self.verbose:
                print(f"  ✗ Error: {e}", file=sys.stderr)

        return changes

    def _update_domain_counts(
        self,
        session,
        old_domain: str,
        new_domain: str,
        is_new: bool,
        description: str,
        paper_keywords: list[str]
    ):
        """Update domain paper counts and embeddings."""
        # Decrement old domain count
        if old_domain:
            old_domain_record = session.query(Domain).filter(
                Domain.name == old_domain
            ).first()
            if old_domain_record:
                old_domain_record.paper_count = max(0, old_domain_record.paper_count - 1)

        # Update or create new domain
        new_domain_record = session.query(Domain).filter(
            Domain.name == new_domain
        ).first()

        if new_domain_record:
            new_domain_record.paper_count += 1

            # Update description if not set
            if description and not new_domain_record.description:
                new_domain_record.description = description

            # Aggregate keywords
            if paper_keywords:
                current = new_domain_record.aggregated_keywords or []
                merged = list(set(
                    [k.lower() for k in current] +
                    [k.lower() for k in paper_keywords]
                ))
                new_domain_record.aggregated_keywords = merged

                # Update embedding
                self.vectors.update_domain_keywords(
                    domain_name=new_domain,
                    new_keywords=paper_keywords,
                    description=new_domain_record.description
                )
        else:
            # Create new domain
            keywords = [k.lower() for k in paper_keywords] if paper_keywords else []
            new_domain_record = Domain(
                name=new_domain,
                description=description,
                aggregated_keywords=keywords,
                paper_count=1
            )
            session.add(new_domain_record)

            # Create embedding
            self.vectors.add_domain_embedding(
                domain_name=new_domain,
                keywords=keywords,
                description=description
            )

    def reextract_all(
        self,
        extract_keywords: bool = True,
        extract_domains: bool = True,
        filter_domain: str = None,
        filter_venue: str = None,
        filter_paper_ids: list[str] = None,
        filter_year: int = None
    ) -> list[dict]:
        """
        Re-extract metadata for multiple papers.

        Returns list of changes.
        """
        all_changes = []

        with self.db.get_session() as session:
            # Build query
            query = session.query(Paper)

            if filter_paper_ids:
                query = query.filter(Paper.paper_id.in_(filter_paper_ids))

            if filter_domain:
                query = query.filter(Paper.domain == filter_domain)

            if filter_venue:
                query = query.filter(Paper.journal_or_venue.ilike(f"%{filter_venue}%"))

            if filter_year:
                query = query.filter(Paper.year == filter_year)

            papers = query.all()

            if not papers:
                print("No papers match the filter criteria.")
                return []

            print(f"Processing {len(papers)} papers...")
            if self.dry_run:
                print("(DRY RUN - no changes will be saved)")
            print("-" * 60)

            for i, paper in enumerate(papers):
                if self.verbose:
                    title_short = (paper.title[:50] + "...") if paper.title and len(paper.title) > 50 else paper.title
                    print(f"\n[{i+1}/{len(papers)}] {paper.paper_id}")
                    print(f"  Title: {title_short}")

                changes = self.reextract_paper(
                    paper=paper,
                    extract_keywords=extract_keywords,
                    extract_domains=extract_domains,
                    session=session
                )
                all_changes.append(changes)

                if self.verbose:
                    if changes["keywords_changed"]:
                        print(f"  Keywords: {len(changes['old_keywords'] or [])} → {len(changes['new_keywords'] or [])}")
                    if changes["domain_changed"]:
                        print(f"  Domain: {changes['old_domain']} → {changes['new_domain']}")
                    if not changes["keywords_changed"] and not changes["domain_changed"]:
                        print(f"  (no changes)")

            if not self.dry_run:
                session.commit()

        return all_changes

    def print_summary(self, changes: list[dict]):
        """Print summary of changes."""
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Papers processed: {self.processed}")
        print(f"  Keywords updated: {self.keywords_updated}")
        print(f"  Domains updated: {self.domains_updated}")
        print(f"  Errors: {self.errors}")

        if self.dry_run:
            print()
            print("(DRY RUN - no changes were saved)")

        # Show domain migration summary
        domain_changes = [c for c in changes if c["domain_changed"]]
        if domain_changes:
            print()
            print("Domain changes:")
            for c in domain_changes[:20]:  # Limit output
                print(f"  {c['paper_id']}: {c['old_domain']} → {c['new_domain']}")
            if len(domain_changes) > 20:
                print(f"  ... and {len(domain_changes) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Re-extract metadata (keywords, domains) for already-imported papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                      Re-extract everything for all papers
  %(prog)s --all --keywords-only      Only re-extract keywords
  %(prog)s --all --domains-only       Only re-classify domains
  %(prog)s --domain "security"        Filter by current domain
  %(prog)s --paper-id "smith_2023"    Specific paper
  %(prog)s --all --dry-run            Show what would change
        """
    )

    # Selection options
    selection = parser.add_argument_group("Selection")
    selection.add_argument(
        "--all", action="store_true",
        help="Process all papers in database"
    )
    selection.add_argument(
        "--paper-id", type=str, action="append", dest="paper_ids",
        help="Process specific paper(s) by ID (can be repeated)"
    )
    selection.add_argument(
        "--domain", type=str,
        help="Filter papers by current domain"
    )
    selection.add_argument(
        "--venue", type=str,
        help="Filter papers by venue (partial match)"
    )
    selection.add_argument(
        "--year", type=int,
        help="Filter papers by publication year"
    )

    # Extraction options
    extraction = parser.add_argument_group("Extraction")
    extraction.add_argument(
        "--keywords-only", action="store_true",
        help="Only re-extract keywords (keep existing domains)"
    )
    extraction.add_argument(
        "--domains-only", action="store_true",
        help="Only re-classify domains (keep existing keywords)"
    )

    # Output options
    output = parser.add_argument_group("Output")
    output.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without saving"
    )
    output.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.paper_ids and not args.domain and not args.venue and not args.year:
        parser.error("Must specify --all, --paper-id, --domain, --venue, or --year")

    if args.keywords_only and args.domains_only:
        parser.error("Cannot specify both --keywords-only and --domains-only")

    # Determine what to extract
    extract_keywords = not args.domains_only
    extract_domains = not args.keywords_only

    # Initialize components
    print("Initializing...")
    db = Database(settings.database_url)
    vector_store = VectorStore(
        persist_directory=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
        embedding_backend=settings.embedding_backend,
        ollama_host=settings.ollama_host
    )
    keyword_extractor = KeywordExtractor(
        model=settings.llm_model,
        host=settings.ollama_host
    )
    domain_classifier = DomainClassifier(
        model=settings.llm_model,
        host=settings.ollama_host
    )

    # Create reextractor
    reextractor = MetadataReextractor(
        db=db,
        vector_store=vector_store,
        keyword_extractor=keyword_extractor,
        domain_classifier=domain_classifier,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )

    # Run extraction
    changes = reextractor.reextract_all(
        extract_keywords=extract_keywords,
        extract_domains=extract_domains,
        filter_domain=args.domain,
        filter_venue=args.venue,
        filter_paper_ids=args.paper_ids,
        filter_year=args.year
    )

    # Print summary
    reextractor.print_summary(changes)


if __name__ == "__main__":
    main()
