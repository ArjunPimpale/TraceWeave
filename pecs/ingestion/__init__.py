"""Ingestion module — file upload dispatcher and utilities."""

from pecs.ingestion.ingestor import Ingestor, IngestionResult
from pecs.ingestion.file_utils import compute_sha256, detect_mime_type

__all__ = ["Ingestor", "IngestionResult", "compute_sha256", "detect_mime_type"]
