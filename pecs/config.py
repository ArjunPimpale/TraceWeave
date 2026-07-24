"""
Central configuration for PECS.

All parameters are loaded from environment variables (.env file) with
sensible defaults. Import `settings` from this module to access any config value.

Usage:
    from pecs.config import settings

    print(settings.OLLAMA_BASE_URL)
    print(settings.MAX_CHUNK_CHARS)
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PecsSettings(BaseSettings):
    """
    PECS application settings loaded from environment variables.

    All values can be overridden by setting environment variables or by
    providing a .env file in the project root.
    """

    # ── Ollama ──────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "phi4-mini"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_TEMPERATURE: float = 0.05

    # ── Chunking parameters ─────────────────────────────────────────────────
    MAX_CHUNK_CHARS: int = 1500
    MIN_CHUNK_CHARS: int = 50
    OVERLAP_CHARS: int = 150
    OVERLAP_MESSAGES: int = 2

    # ── Retrieval ───────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 10
    BM25_TOP_K: int = 10

    # ── Extraction ──────────────────────────────────────────────────────────
    MAX_EXTRACTION_RETRIES: int = 3

    # ── Scoring weights (retrieval merging) ─────────────────────────────────
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3

    # ── Confidence scoring weights ───────────────────────────────────────────
    CONFIDENCE_W_R: float = 0.35   # Retrieval score weight
    CONFIDENCE_W_M: float = 0.30   # Resolution method weight
    CONFIDENCE_W_E: float = 0.20   # Evidence count weight
    CONFIDENCE_W_D: float = 0.15   # Source diversity weight

    # ── Storage paths ────────────────────────────────────────────────────────
    SQLITE_DB_PATH: str = "data/sqlite/pecs.db"
    CHROMADB_PATH: str = "data/chromadb/"
    LOG_FILE: str = "data/logs/pecs.log"

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── WhatsApp chunking ────────────────────────────────────────────────────
    WHATSAPP_TIME_GAP_MINUTES: int = 30

    # ── File upload ──────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    CHROMA_COLLECTION_NAME: str = "pecs_evidence"

    # ── Hallucination detection ───────────────────────────────────────────────
    MIN_WORD_OVERLAP_RATIO: float = 0.30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def sqlite_db_path(self) -> Path:
        """Resolved absolute path to the SQLite database file."""
        p = Path(self.SQLITE_DB_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chromadb_path(self) -> Path:
        """Resolved absolute path to the ChromaDB storage directory."""
        p = Path(self.CHROMADB_PATH)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_file_path(self) -> Path:
        """Resolved absolute path to the log file."""
        p = Path(self.LOG_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum allowed file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


# Module-level singleton — import this everywhere
settings = PecsSettings()
