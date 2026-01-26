"""
Read from Zotero and Better BibTeX databases.
Read-only access - safe to use while Zotero is running (via copy).
"""

import sqlite3
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from .models import ZoteroItem, ZoteroCollection


class ZoteroReader:
    """
    Read Zotero library data from SQLite databases.
    
    Uses database copies to avoid locking issues while Zotero is running.
    """
    
    # Zotero field IDs (from the fields table)
    FIELD_IDS = {
        "title": 110,
        "abstract": 90,
        "abstractNote": 90,
        "date": 14,
        "DOI": 26,
        "url": 1,
        "volume": 5,
        "issue": 6,
        "pages": 7,
        "publicationTitle": 12,
    }
    
    # Item types to include - will be populated dynamically
    WANTED_TYPES = {
        "book", "bookSection", "journalArticle", "conferencePaper",
        "thesis", "report", "document", "preprint", "presentation",
        "webpage", "blogPost", "videoRecording", "standard",
        "magazineArticle", "newspaperArticle"
    }
    
    def __init__(self, zotero_path: Optional[Path] = None):
        """
        Initialize reader.
        
        Args:
            zotero_path: Path to Zotero data directory. Auto-detected if not provided.
        """
        self.zotero_path = zotero_path or self._detect_zotero_path()
        
        if not self.zotero_path:
            raise FileNotFoundError("Could not find Zotero data directory")
        
        self.zotero_db_path = self.zotero_path / "zotero.sqlite"
        self.bbt_db_path = self.zotero_path / "better-bibtex.sqlite"
        self.storage_path = self.zotero_path / "storage"
        
        if not self.zotero_db_path.exists():
            raise FileNotFoundError(f"Zotero database not found: {self.zotero_db_path}")
        
        # Load actual item type IDs from database
        self.valid_item_types = self._load_item_types()
    
    def _detect_zotero_path(self) -> Optional[Path]:
        """Auto-detect Zotero data directory."""
        candidates = [
            Path.home() / "Zotero",
            Path.home() / ".zotero" / "zotero",
            Path.home() / "Documents" / "Zotero",
        ]
        
        for path in candidates:
            if (path / "zotero.sqlite").exists():
                return path
        
        return None
    
    def _load_item_types(self) -> dict[int, str]:
        """Load item type IDs from Zotero database."""
        types = {}
        with self._get_zotero_db() as conn:
            cursor = conn.execute(
                "SELECT itemTypeID, typeName FROM itemTypes WHERE typeName IN ({})".format(
                    ",".join(f"'{t}'" for t in self.WANTED_TYPES)
                )
            )
            for row in cursor:
                types[row["itemTypeID"]] = row["typeName"]
        return types
    
    @contextmanager
    def _get_zotero_db(self):
        """Get connection to Zotero database (using copy)."""
        # Check if database is locked by Zotero
        self._check_database_lock()
        
        # Copy to temp file to avoid locking
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            shutil.copy2(self.zotero_db_path, tmp_path)
            conn = sqlite3.connect(tmp_path)
            conn.row_factory = sqlite3.Row
            
            # Verify the copy is valid
            try:
                conn.execute("SELECT COUNT(*) FROM items").fetchone()
            except sqlite3.DatabaseError as e:
                raise RuntimeError(
                    f"Database copy is corrupted. Zotero may be running.\n"
                    f"Please close Zotero and try again.\n"
                    f"Error: {e}"
                )
            
            yield conn
            conn.close()
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def _check_database_lock(self):
        """Check if Zotero database is locked."""
        # Check for lock file
        lock_file = self.zotero_db_path.parent / ".zotero-lock"
        if lock_file.exists():
            raise RuntimeError(
                "Zotero database is locked (.zotero-lock exists).\n"
                "Please close Zotero before importing."
            )
        
        # Try to open database directly to check lock
        try:
            conn = sqlite3.connect(self.zotero_db_path, timeout=1)
            # Try to get an exclusive lock briefly
            conn.execute("BEGIN IMMEDIATE")
            conn.rollback()
            conn.close()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                raise RuntimeError(
                    "Zotero database is locked.\n"
                    "Please close Zotero before importing."
                )
            # Other errors - might be okay, continue with copy approach
    
    @contextmanager
    def _get_bbt_db(self):
        """Get connection to Better BibTeX database (using copy)."""
        if not self.bbt_db_path.exists():
            yield None
            return
        
        # Check if locked (same process as Zotero would lock it)
        try:
            conn = sqlite3.connect(self.bbt_db_path, timeout=1)
            conn.execute("BEGIN IMMEDIATE")
            conn.rollback()
            conn.close()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                raise RuntimeError(
                    "Better BibTeX database is locked.\n"
                    "Please close Zotero before importing."
                )
        
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            shutil.copy2(self.bbt_db_path, tmp_path)
            conn = sqlite3.connect(tmp_path)
            conn.row_factory = sqlite3.Row
            yield conn
            conn.close()
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def get_citation_keys(self) -> dict[int, str]:
        """
        Get all citation keys from Better BibTeX.
        
        Returns:
            Dict mapping itemID to citationKey
        """
        keys = {}
        
        with self._get_bbt_db() as conn:
            if conn is None:
                return keys
            
            cursor = conn.execute(
                "SELECT itemID, citationKey FROM citationkey"
            )
            for row in cursor:
                keys[row["itemID"]] = row["citationKey"]
        
        return keys
    
    def get_collections(self) -> list[ZoteroCollection]:
        """Get all collections (excluding deleted)."""
        collections = []
        
        with self._get_zotero_db() as conn:
            cursor = conn.execute("""
                SELECT 
                    c.collectionID,
                    c.collectionName,
                    c.parentCollectionID,
                    c.key,
                    COUNT(ci.itemID) as item_count
                FROM collections c
                LEFT JOIN collectionItems ci ON c.collectionID = ci.collectionID
                WHERE c.collectionID NOT IN (SELECT collectionID FROM deletedCollections)
                GROUP BY c.collectionID
                ORDER BY c.collectionName
            """)
            
            for row in cursor:
                collections.append(ZoteroCollection(
                    collection_id=row["collectionID"],
                    collection_key=row["key"],
                    name=row["collectionName"],
                    parent_id=row["parentCollectionID"],
                    item_count=row["item_count"]
                ))
        
        return collections
    
    def get_collection_id_by_name(self, name: str) -> Optional[int]:
        """Get collection ID by name (excluding deleted)."""
        with self._get_zotero_db() as conn:
            cursor = conn.execute(
                """SELECT collectionID FROM collections 
                WHERE collectionName = ?
                AND collectionID NOT IN (SELECT collectionID FROM deletedCollections)""",
                (name,)
            )
            row = cursor.fetchone()
            return row["collectionID"] if row else None
    
    def get_items(
        self,
        collection_name: Optional[str] = None,
        item_type: Optional[str] = None,
        has_pdf: bool = False,
        limit: Optional[int] = None
    ) -> list[ZoteroItem]:
        """
        Get items from Zotero library.
        
        Args:
            collection_name: Filter by collection name
            item_type: Filter by item type (e.g., "journalArticle")
            has_pdf: Only return items with PDF attachments
            limit: Maximum number of items to return
        """
        # Get citation keys first
        citation_keys = self.get_citation_keys()
        
        with self._get_zotero_db() as conn:
            # Build query - exclude deleted items and attachments
            query = """
                SELECT DISTINCT
                    i.itemID,
                    i.key as itemKey,
                    i.itemTypeID
                FROM items i
                WHERE i.itemTypeID IN ({})
                AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            """.format(",".join(str(t) for t in self.valid_item_types.keys()))
            
            params = []
            
            # Collection filter
            if collection_name:
                collection_id = self.get_collection_id_by_name(collection_name)
                if collection_id is None:
                    return []
                query += """
                    AND i.itemID IN (
                        SELECT itemID FROM collectionItems WHERE collectionID = ?
                    )
                """
                params.append(collection_id)
            
            # Item type filter
            if item_type:
                type_id = next(
                    (k for k, v in self.valid_item_types.items() if v == item_type),
                    None
                )
                if type_id:
                    query += " AND i.itemTypeID = ?"
                    params.append(type_id)
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            items = []
            for row in rows:
                item = self._build_item(
                    conn,
                    row["itemID"],
                    row["itemKey"],
                    row["itemTypeID"],
                    citation_keys.get(row["itemID"])
                )
                
                if has_pdf and not item.has_pdf():
                    continue
                
                items.append(item)
            
            return items
    
    def get_item_by_key(self, item_key: str) -> Optional[ZoteroItem]:
        """Get single item by Zotero key."""
        citation_keys = self.get_citation_keys()
        
        with self._get_zotero_db() as conn:
            cursor = conn.execute(
                "SELECT itemID, key, itemTypeID FROM items WHERE key = ?",
                (item_key,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._build_item(
                conn,
                row["itemID"],
                row["key"],
                row["itemTypeID"],
                citation_keys.get(row["itemID"])
            )
    
    def get_item_by_citation_key(self, citation_key: str) -> Optional[ZoteroItem]:
        """Get item by Better BibTeX citation key."""
        with self._get_bbt_db() as bbt_conn:
            if bbt_conn is None:
                return None
            
            cursor = bbt_conn.execute(
                "SELECT itemID, itemKey FROM citationkey WHERE citationKey = ?",
                (citation_key,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            item_key = row["itemKey"]
        
        return self.get_item_by_key(item_key)
    
    def _build_item(
        self,
        conn: sqlite3.Connection,
        item_id: int,
        item_key: str,
        item_type_id: int,
        citation_key: Optional[str]
    ) -> ZoteroItem:
        """Build ZoteroItem from database."""
        
        item_type = self.valid_item_types.get(item_type_id, "document")
        
        # Get field values
        fields = {}
        cursor = conn.execute("""
            SELECT f.fieldName, idv.value
            FROM itemData id
            JOIN fields f ON id.fieldID = f.fieldID
            JOIN itemDataValues idv ON id.valueID = idv.valueID
            WHERE id.itemID = ?
        """, (item_id,))
        
        for row in cursor:
            fields[row["fieldName"]] = row["value"]
        
        # Get authors/creators
        authors = []
        cursor = conn.execute("""
            SELECT c.firstName, c.lastName, ct.creatorType
            FROM itemCreators ic
            JOIN creators c ON ic.creatorID = c.creatorID
            JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
            WHERE ic.itemID = ?
            ORDER BY ic.orderIndex
        """, (item_id,))
        
        for row in cursor:
            if row["creatorType"] == "author":
                name = f"{row['firstName']} {row['lastName']}".strip()
                if name:
                    authors.append(name)
        
        # Get collections
        collections = []
        cursor = conn.execute("""
            SELECT c.collectionName
            FROM collectionItems ci
            JOIN collections c ON ci.collectionID = c.collectionID
            WHERE ci.itemID = ?
        """, (item_id,))
        
        for row in cursor:
            collections.append(row["collectionName"])
        
        # Get tags
        tags = []
        cursor = conn.execute("""
            SELECT t.name
            FROM itemTags it
            JOIN tags t ON it.tagID = t.tagID
            WHERE it.itemID = ?
        """, (item_id,))
        
        for row in cursor:
            tags.append(row["name"])
        
        # Get PDF attachment
        pdf_path = self._get_pdf_path(conn, item_id, item_key)
        
        # Extract year from date
        date = fields.get("date", "")
        year = None
        if date:
            # Try to extract 4-digit year
            import re
            match = re.search(r'\b(19|20)\d{2}\b', date)
            if match:
                year = match.group(0)
        
        # Extract venue/publication from various fields depending on item type
        # Zotero uses different field names for different item types
        # Try all possible venue fields in priority order
        venue = self._extract_venue(fields, item_type)
        
        return ZoteroItem(
            item_id=item_id,
            item_key=item_key,
            citation_key=citation_key,
            item_type=item_type,
            title=fields.get("title"),
            authors=authors,
            abstract=fields.get("abstractNote"),
            date=date,
            year=year,
            publication_title=venue,
            doi=fields.get("DOI"),
            url=fields.get("url"),
            volume=fields.get("volume"),
            issue=fields.get("issue"),
            pages=fields.get("pages"),
            collections=collections,
            tags=tags,
            pdf_path=pdf_path
        )
    
    def _extract_venue(self, fields: dict, item_type: str) -> Optional[str]:
        """
        Extract venue/publication from Zotero fields.
        
        Zotero uses different field names for different item types:
        - journalArticle: publicationTitle
        - conferencePaper: proceedingsTitle, conferenceName
        - bookSection: bookTitle
        - book: series, publisher
        - thesis: university
        - report: institution, seriesTitle
        - presentation: meetingName
        - webpage: websiteTitle
        - blogPost: blogTitle
        - podcast/audioRecording: seriesTitle, label
        - videoRecording: programTitle, network
        - newspaperArticle/magazineArticle: publicationTitle
        - patent: issuingAuthority
        - statute: code, codeNumber
        - case: court, reporter
        - encyclopediaArticle: encyclopediaTitle
        - dictionaryEntry: dictionaryTitle
        - document: publisher
        - letter: recipient (not really venue)
        - manuscript: archive
        - map: publisher
        - artwork: archive
        - film: distributor, studio
        - tvBroadcast: network, programTitle
        - radioBroadcast: network, programTitle
        - hearing: committee
        - bill: legislativeBody
        - preprint: repository
        
        We try all possible venue-related fields in priority order.
        """
        # Priority-ordered list of venue-related fields
        # More specific/common fields first
        venue_fields = [
            # Academic publications
            "publicationTitle",      # journals, magazines, newspapers
            "proceedingsTitle",      # conference papers
            "conferenceName",        # conference papers (alternative)
            "bookTitle",             # book sections, encyclopedia entries
            "encyclopediaTitle",     # encyclopedia articles
            "dictionaryTitle",       # dictionary entries
            
            # Academic institutions
            "university",            # thesis
            "institution",           # reports
            
            # Series and collections
            "seriesTitle",           # reports, podcasts
            "series",                # books
            
            # Web/digital
            "websiteTitle",          # webpages
            "blogTitle",             # blog posts
            "forumTitle",            # forum posts
            "repository",            # preprints
            
            # Media
            "programTitle",          # TV, radio, video
            "network",               # broadcasts
            "studio",                # films
            "distributor",           # films
            "label",                 # audio recordings
            
            # Legal/government
            "court",                 # legal cases
            "reporter",              # legal cases
            "code",                  # statutes
            "legislativeBody",       # bills
            "committee",             # hearings
            "issuingAuthority",      # patents
            
            # Archives/collections
            "archive",               # manuscripts, artwork
            "libraryCatalog",        # general
            
            # Events
            "meetingName",           # presentations
            "place",                 # general location (fallback)
            
            # Publisher (last resort - less specific)
            "publisher",             # books, documents, maps
        ]
        
        # Try each field in priority order
        for field in venue_fields:
            value = fields.get(field)
            if value and value.strip():
                return value.strip()
        
        return None
    
    def _get_pdf_path(
        self,
        conn: sqlite3.Connection,
        item_id: int,
        item_key: str
    ) -> Optional[Path]:
        """Get PDF file path for an item."""
        
        # Check for PDF attachment
        # linkMode: 0=imported file, 1=imported url, 2=linked file, 3=linked url
        # Prefer stored/linked files (linkMode 0 or 2) over URL links (linkMode 1 or 3)
        # Order by: has path DESC, linkMode ASC (prefer imported files)
        cursor = conn.execute("""
            SELECT ia.path, i.key as attachmentKey, ia.linkMode
            FROM itemAttachments ia
            JOIN items i ON ia.itemID = i.itemID
            WHERE ia.parentItemID = ?
            AND ia.contentType = 'application/pdf'
            ORDER BY 
                CASE WHEN ia.path IS NOT NULL AND ia.path != '' THEN 0 ELSE 1 END,
                ia.linkMode ASC
            LIMIT 1
        """, (item_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        attachment_path = row["path"]
        attachment_key = row["attachmentKey"]
        
        if attachment_path:
            # Linked file - path is stored directly
            if attachment_path.startswith("storage:"):
                # Stored in Zotero storage
                filename = attachment_path.replace("storage:", "")
                path = self.storage_path / attachment_key / filename
            else:
                # Linked file - absolute or relative path
                path = Path(attachment_path)
        else:
            # Look for PDF in storage folder
            storage_folder = self.storage_path / attachment_key
            if storage_folder.exists():
                pdfs = list(storage_folder.glob("*.pdf"))
                if pdfs:
                    path = pdfs[0]
                else:
                    return None
            else:
                return None
        
        return path if path.exists() else None
    
    def get_stats(self) -> dict:
        """Get library statistics."""
        with self._get_zotero_db() as conn:
            # Total items (excluding deleted)
            cursor = conn.execute(
                """SELECT COUNT(*) as count FROM items 
                WHERE itemTypeID IN ({})
                AND itemID NOT IN (SELECT itemID FROM deletedItems)""".format(
                    ",".join(str(t) for t in self.valid_item_types.keys())
                )
            )
            total_items = cursor.fetchone()["count"]
            
            # Collections count
            cursor = conn.execute("SELECT COUNT(*) as count FROM collections")
            total_collections = cursor.fetchone()["count"]
        
        # Citation keys
        citation_keys = self.get_citation_keys()
        
        return {
            "total_items": total_items,
            "total_collections": total_collections,
            "items_with_citation_key": len(citation_keys),
            "zotero_path": str(self.zotero_path),
            "better_bibtex_installed": self.bbt_db_path.exists()
        }
