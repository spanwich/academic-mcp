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

    # Max output tokens for the section-detection LLM call.
    # -1 disables the cap (grammar-enforced JSON guarantees validity if the
    # model is allowed to finish). Larger models like gemma4:26b hit a hard
    # 2000-token ceiling with long papers and produce truncated JSON, which
    # falls back to an all-"unknown" page grouping.
    section_detection_num_predict: int = Field(
        default=-1,
        alias="ACADEMIC_SECTION_DETECTION_NUM_PREDICT"
    )

    # Context window for the section-detection LLM call. Ollama defaults to
    # 2048, which silently truncates the prompt and makes detection fail.
    # 16384 fits ~18 pages × 3000-char previews plus scaffold with headroom
    # for output; raise for longer documents or larger previews.
    section_detection_num_ctx: int = Field(
        default=16384,
        alias="ACADEMIC_SECTION_DETECTION_NUM_CTX"
    )

    # Per-page preview length fed to the section detector. 800 chars often
    # misses section headings (page header/footer + continuing paragraph from
    # previous page push the heading past the cutoff). 3000 comfortably
    # captures the start + first heading on a typical academic page.
    section_detection_preview_chars: int = Field(
        default=3000,
        alias="ACADEMIC_SECTION_DETECTION_PREVIEW_CHARS"
    )

    # Zotero
    zotero_path: Path | None = Field(
        default=None,
        alias="ACADEMIC_ZOTERO_PATH"
    )

    # Remote HTTP MCP deployment (used by src/server_http.py)
    mcp_token: str = Field(
        default="",
        alias="ACADEMIC_MCP_TOKEN"
    )
    mcp_http_port: int = Field(
        default=8000,
        alias="ACADEMIC_MCP_HTTP_PORT"
    )
    mcp_http_host: str = Field(
        default="127.0.0.1",
        alias="ACADEMIC_MCP_HTTP_HOST"
    )
    mcp_remote_mode: bool = Field(
        default=False,
        alias="ACADEMIC_MCP_REMOTE_MODE"
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
