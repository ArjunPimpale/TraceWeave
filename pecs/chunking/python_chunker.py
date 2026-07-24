"""Python chunker — AST-element-first chunking."""

from __future__ import annotations

from pecs.chunking.base_chunker import BaseChunker, RawChunk
from pecs.models.evidence_chunk import SourceType
from pecs.parsing.base_parser import ParsedDocument


class PythonChunker(BaseChunker):
    """
    Definition-first chunking for Python source files.

    Strategy:
    1. Module docstring + imports = one "module overview" chunk.
    2. Each class = one chunk. If class > max_chunk_chars, each method = chunk.
    3. Each standalone function = one chunk.
    4. No overlap between definitions (they are independent units).
    5. Each chunk is prefixed with: "# File: filename.py"

    Metadata: element_type, element_name, line_start, line_end, file_path.
    """

    def chunk(self, document: ParsedDocument) -> list[RawChunk]:
        base_meta = self._base_metadata(document)
        source_doc = document.global_metadata.get("filename", "")
        source_hash = document.global_metadata.get("file_hash", "")

        chunks: list[RawChunk] = []
        module_parts: list[str] = []  # Collect module-level elements

        for element in document.structural_metadata:
            elem_type = element.get("element_type", "")
            elem_name = element.get("element_name", "")
            body = element.get("body", "").strip()
            line_start = element.get("line_start", 1)
            line_end = element.get("line_end", 1)

            if not body:
                continue

            chunk_meta = {
                **base_meta,
                "element_type": elem_type,
                "element_name": elem_name,
                "line_start": line_start,
                "line_end": line_end,
                "file_path": source_doc,
            }

            if elem_type in ("module", "import_block"):
                # Combine into module overview
                module_parts.append(body)
                continue

            source_locator = f"lines {line_start}-{line_end}"

            # Check if this single element needs to be split
            if len(body) > self.max_chunk_chars:
                sub_chunks = self._split_large_chunk(RawChunk(
                    chunk_text=body,
                    chunk_index=len(chunks),
                    source_locator=source_locator,
                    source_document=source_doc,
                    source_hash=source_hash,
                    source_type=SourceType.PYTHON,
                    metadata=chunk_meta,
                ))
                for sc in sub_chunks:
                    sc.metadata = chunk_meta
                    chunks.append(sc)
            else:
                chunks.append(RawChunk(
                    chunk_text=body,
                    chunk_index=len(chunks),
                    source_locator=source_locator,
                    source_document=source_doc,
                    source_hash=source_hash,
                    source_type=SourceType.PYTHON,
                    metadata=chunk_meta,
                ))

        # Prepend the module overview chunk if we collected module-level content
        if module_parts:
            module_text = "\n\n".join(module_parts)
            module_chunk = RawChunk(
                chunk_text=f"# File: {source_doc}\n\n{module_text}",
                chunk_index=0,
                source_locator="module overview",
                source_document=source_doc,
                source_hash=source_hash,
                source_type=SourceType.PYTHON,
                metadata={**base_meta, "element_type": "module", "element_name": source_doc},
            )
            chunks.insert(0, module_chunk)

        # No overlap for Python definitions — they are independent units
        chunks = self._enforce_size_limits(chunks)

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks
