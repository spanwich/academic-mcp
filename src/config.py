"""
Configuration management for Academic Paper MCP.

Uses environment variables with ACADEMIC_ prefix.
"""

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: str = Field(
        default="sqlite:///data/papers.db",
        alias="ACADEMIC_DATABASE_URL"
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        alias="ACADEMIC_CHROMA_PERSIST_DIR"
    )
    
    # Models
    llm_model: str = Field(
        default="qwen2.5:3b",
        alias="ACADEMIC_LLM_MODEL"
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        alias="ACADEMIC_EMBEDDING_MODEL"
    )
    embedding_backend: str = Field(
        default="ollama",
        alias="ACADEMIC_EMBEDDING_BACKEND"
    )
    # Options: "ollama" (recommended) or "sentence-transformers"
    ollama_host: str = Field(
        default="http://localhost:11434",
        alias="ACADEMIC_OLLAMA_HOST"
    )
    
    # Zotero
    zotero_path: Path | None = Field(
        default=None,
        alias="ACADEMIC_ZOTERO_PATH"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_config() -> Settings:
    """Get cached configuration."""
    return Settings()


def ensure_directories():
    """Ensure data directories exist."""
    config = get_config()
    
    # Database directory
    db_path = Path(config.database_url.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Chroma directory
    Path(config.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
