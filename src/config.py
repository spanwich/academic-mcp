"""
Configuration management using pydantic-settings.
v3: Added Zotero integration settings.
"""

from pathlib import Path
from typing import Optional
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Zotero paths
    zotero_path: Optional[Path] = Field(
        default=None,
        description="Path to Zotero data directory (auto-detected if not set)"
    )
    
    # Database
    database_url: str = Field(
        default="sqlite:///data/papers.db",
        description="SQLAlchemy database URL"
    )
    
    # Vector store
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="ChromaDB persistence directory"
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Sentence transformer model for embeddings"
    )
    
    # LLM settings
    llm_model: str = Field(
        default="qwen2.5:3b",
        description="Ollama model for extraction"
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama API host"
    )
    
    # Processing settings
    chunk_size: int = Field(
        default=800,
        description="Target tokens per chunk"
    )
    max_extraction_tokens: int = Field(
        default=30000,
        description="Maximum characters to send to LLM"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ACADEMIC_",
        case_sensitive=False,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_directories()
        self._detect_zotero()
    
    def _setup_directories(self):
        """Create necessary directories."""
        # Database directory
        db_path = Path(self.database_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB directory
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    
    def _detect_zotero(self):
        """Auto-detect Zotero path if not set."""
        if self.zotero_path is None:
            candidates = [
                Path.home() / "Zotero",
                Path.home() / ".zotero" / "zotero",
                Path.home() / "Documents" / "Zotero",
            ]
            
            for path in candidates:
                if (path / "zotero.sqlite").exists():
                    object.__setattr__(self, 'zotero_path', path)
                    break


@lru_cache()
def get_config() -> Settings:
    """Get cached settings instance."""
    return Settings()
