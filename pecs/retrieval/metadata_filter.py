"""
Metadata filter — narrows the ChromaDB search space before retrieval.

Builds ChromaDB 'where' filter dicts from user-specified filter criteria.
Pre-filtering dramatically reduces the candidate set and improves precision.
"""

from __future__ import annotations

from typing import Any


class MetadataFilter:
    """
    Builds ChromaDB-compatible 'where' filter dicts.

    ChromaDB filter operators: $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte.
    Multiple conditions are combined with $and.
    """

    def __init__(
        self,
        source_types: list[str] | None = None,
        source_document: str | None = None,
        project: str | None = None,
        date_after: str | None = None,
        date_before: str | None = None,
    ) -> None:
        """
        Args:
            source_types: Filter to specific SourceType values
                          (e.g., ["PYTHON", "GIT"]).
            source_document: Filter to a specific source filename.
            project: Filter by project metadata field.
            date_after: ISO 8601 string — only include chunks created after this date.
            date_before: ISO 8601 string — only include chunks created before this date.
        """
        self.source_types = source_types
        self.source_document = source_document
        self.project = project
        self.date_after = date_after
        self.date_before = date_before

    def build(self) -> dict[str, Any] | None:
        """
        Build a ChromaDB 'where' filter dict.

        Returns None if no filters are set (no filtering needed).
        """
        conditions: list[dict[str, Any]] = []

        if self.source_types:
            if len(self.source_types) == 1:
                conditions.append({"source_type": {"$eq": self.source_types[0]}})
            else:
                conditions.append({"source_type": {"$in": self.source_types}})

        if self.source_document:
            conditions.append({"source_document": {"$eq": self.source_document}})

        if self.project:
            conditions.append({"meta_project": {"$eq": self.project}})

        if self.date_after:
            conditions.append({"created_at": {"$gte": self.date_after}})

        if self.date_before:
            conditions.append({"created_at": {"$lte": self.date_before}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @classmethod
    def implementation_filter(cls) -> "MetadataFilter":
        """Pre-built filter for implementation evidence (Python + Git)."""
        return cls(source_types=["PYTHON", "GIT"])

    @classmethod
    def evaluation_filter(cls) -> "MetadataFilter":
        """Pre-built filter for evaluation evidence (PDF + DOCX)."""
        return cls(source_types=["PDF", "DOCX"])

    @classmethod
    def discussion_filter(cls) -> "MetadataFilter":
        """Pre-built filter for discussion evidence (WhatsApp + Email)."""
        return cls(source_types=["WHATSAPP", "EMAIL"])
