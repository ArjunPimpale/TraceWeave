"""
Git commit log parser.

Parses git log output (either from a file or live git subprocess).
Extracts commit SHA, author, date, message, and changed file summary.
Does NOT extract full diffs — only the commit message and file-change summary.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any

from pecs.logging_config import get_logger
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import BaseParser, ParseRequest, ParsedDocument

logger = get_logger(__name__)

# Git log --format separator we use to reliably split commits
_COMMIT_SEPARATOR = "<<<COMMIT_START>>>"

# Per-commit format: SHA, author name, date (ISO), subject, body
_GIT_FORMAT = f"{_COMMIT_SEPARATOR}%H%n%an%n%ai%n%s%n%b"

# Max commits to process (prevents memory exhaustion on huge repos)
_MAX_COMMITS = 500

# Regex to detect git log file format lines
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_AUTHOR_LINE = re.compile(r"^Author:\s+(.+)$")
_DATE_LINE = re.compile(r"^Date:\s+(.+)$")
_COMMIT_LINE = re.compile(r"^commit\s+([0-9a-f]{40})$")


class GitParser(BaseParser):
    """
    Parser for Git commit log exports or live repositories.

    Accepts either:
    - A plain text git log export (stdout of `git log`)
    - A path to a git repository (parses live via subprocess)

    Structural metadata: list of dicts:
        {
            "commit_sha": str,
            "author": str,
            "date": str,
            "message": str,
            "changed_files": list[str],  # up to 20 files
            "changed_files_count": int,
            "is_merge": bool,
        }
    """

    SUPPORTED_MIME_TYPES = ("text/plain",)
    SUPPORTED_EXTENSIONS = ("log", "txt")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        logger.debug(
            "Git parse started",
            extra={"context": {"filename": request.filename}},
        )

        warnings: list[str] = []
        text = self._decode(request.file_bytes)
        commits = self._parse_git_log_text(text, warnings)

        # Enforce commit limit
        if len(commits) > _MAX_COMMITS:
            warnings.append(
                f"Git log truncated to {_MAX_COMMITS} commits (total: {len(commits)})"
            )
            commits = commits[-_MAX_COMMITS:]  # Keep the most recent

        full_text_parts: list[str] = []
        for commit in commits:
            files_section = "\n".join(
                f"- {f}" for f in commit["changed_files"][:20]
            )
            chunk_text = (
                f"Commit: {commit['commit_sha']}\n"
                f"Author: {commit['author']}\n"
                f"Date: {commit['date']}\n"
                f"\n{commit['message']}"
            )
            if files_section:
                chunk_text += f"\n\nChanged files:\n{files_section}"
            full_text_parts.append(chunk_text)

        raw_text = "\n\n---\n\n".join(full_text_parts)
        global_metadata = {
            "filename": request.filename,
            "file_hash": request.file_hash,
            "source_type": SourceType.GIT.value,
            "commit_count": len(commits),
            **request.user_metadata,
        }

        logger.info(
            "Git parse completed",
            extra={"context": {
                "filename": request.filename,
                "commits": len(commits),
            }},
        )

        return ParsedDocument(
            source_type=SourceType.GIT,
            raw_text=raw_text,
            structural_metadata=commits,
            global_metadata=global_metadata,
            parse_warnings=warnings,
        )

    @staticmethod
    def _decode(data: bytes) -> str:
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def _parse_git_log_text(
        self, text: str, warnings: list[str]
    ) -> list[dict[str, Any]]:
        """
        Parse git log output text into a list of commit dicts.

        Supports two formats:
        1. The separator-based format (our custom --format)
        2. The standard `git log` format (commit/Author/Date/message lines)
        """
        # Try separator-based format first
        if _COMMIT_SEPARATOR in text:
            return self._parse_separator_format(text, warnings)
        # Fall back to standard git log format
        return self._parse_standard_format(text, warnings)

    @staticmethod
    def _parse_separator_format(
        text: str, warnings: list[str]
    ) -> list[dict[str, Any]]:
        commits: list[dict[str, Any]] = []
        segments = text.split(_COMMIT_SEPARATOR)
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            lines = seg.splitlines()
            if len(lines) < 3:
                continue
            sha = lines[0].strip()
            author = lines[1].strip()
            date = lines[2].strip()
            message_lines = lines[3:]
            message = "\n".join(message_lines).strip()
            commits.append({
                "commit_sha": sha,
                "author": author,
                "date": date,
                "message": message,
                "changed_files": [],
                "changed_files_count": 0,
                "is_merge": message.lower().startswith("merge"),
            })
        return commits

    @staticmethod
    def _parse_standard_format(
        text: str, warnings: list[str]
    ) -> list[dict[str, Any]]:
        """Parse standard `git log` format output."""
        commits: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        in_message = False

        for line in text.splitlines():
            commit_match = _COMMIT_LINE.match(line)
            author_match = _AUTHOR_LINE.match(line)
            date_match = _DATE_LINE.match(line)

            if commit_match:
                if current is not None:
                    current["message"] = current["message"].strip()
                    commits.append(current)
                sha = commit_match.group(1)
                current = {
                    "commit_sha": sha,
                    "author": "",
                    "date": "",
                    "message": "",
                    "changed_files": [],
                    "changed_files_count": 0,
                    "is_merge": False,
                }
                in_message = False
            elif current and author_match:
                current["author"] = author_match.group(1).strip()
            elif current and date_match:
                current["date"] = date_match.group(1).strip()
                in_message = True
            elif current and in_message and line.startswith("    "):
                msg_line = line[4:]  # Strip 4-space git indent
                current["message"] += msg_line + "\n"
                if msg_line.lower().startswith("merge"):
                    current["is_merge"] = True
            elif current and line.strip() == "" and in_message:
                pass  # blank lines within message body
            else:
                in_message = False

        if current is not None:
            current["message"] = current["message"].strip()
            commits.append(current)

        return commits
