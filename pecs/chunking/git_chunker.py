"""Git commit chunker — one commit per chunk."""

from __future__ import annotations

from pecs.chunking.base_chunker import BaseChunker, RawChunk
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import ParsedDocument


class GitChunker(BaseChunker):
    """
    Commit-first chunking for Git log exports.

    Each commit becomes one chunk. No overlap between commits
    — they are independent atomic units.

    Strategy:
    1. Format each commit: SHA, author, date, message, changed files.
    2. Truncate changed files list to 20 entries for very large commits.
    3. No overlap between commits.

    Metadata: commit_sha, author, date, changed_files_count, is_merge.
    """

    def chunk(self, document: ParsedDocument) -> list[RawChunk]:
        base_meta = self._base_metadata(document)
        source_doc = document.global_metadata.get("filename", "")
        source_hash = document.global_metadata.get("file_hash", "")

        chunks: list[RawChunk] = []

        for idx, commit in enumerate(document.structural_metadata):
            sha = commit.get("commit_sha", "unknown")
            author = commit.get("author", "Unknown")
            date = commit.get("date", "")
            message = commit.get("message", "").strip()
            changed_files = commit.get("changed_files", [])[:20]
            changed_files_count = commit.get("changed_files_count", len(changed_files))
            is_merge = commit.get("is_merge", False)

            chunk_text = (
                f"Commit: {sha}\n"
                f"Author: {author}\n"
                f"Date: {date}\n"
            )
            if message:
                chunk_text += f"\n{message}"

            if changed_files:
                files_section = "\n".join(f"- {f}" for f in changed_files)
                chunk_text += f"\n\nChanged files:\n{files_section}"

            if not chunk_text.strip():
                continue

            chunk_meta = {
                **base_meta,
                "commit_sha": sha,
                "author": author,
                "date": date,
                "changed_files_count": changed_files_count,
                "is_merge": is_merge,
            }

            chunks.append(RawChunk(
                chunk_text=chunk_text.strip(),
                chunk_index=idx,
                source_locator=f"commit {sha[:8]} ({date})",
                source_document=source_doc,
                source_hash=source_hash,
                source_type=SourceType.GIT,
                metadata=chunk_meta,
            ))

        # No overlap for git commits — each is an independent unit
        chunks = self._enforce_size_limits(chunks)

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks
