# PECS — Complete Implementation Plan

> **Document version**: 0.1-PLAN · **Baseline**: TSD v0.3 (Hybrid Baseline) + finalized architecture overrides  
> **Target hardware**: RTX 3060 · 8 GB VRAM · local-only execution  
> **Primary model**: Phi-4-mini via Ollama · **Embeddings**: nomic-embed-text via Ollama  
> **Status**: DRAFT — awaiting author approval before any code is written

---

## Table of Contents

1. [Overall Project Architecture](#1-overall-project-architecture)
2. [Recommended Project Folder Structure](#2-recommended-project-folder-structure)
3. [Module Breakdown](#3-module-breakdown)
4. [Universal Internal Objects](#4-universal-internal-objects)
5. [SQLite Design](#5-sqlite-design)
6. [ChromaDB Design](#6-chromadb-design)
7. [Source-Aware Chunking](#7-source-aware-chunking)
8. [Stage 1 — Extraction](#8-stage-1--extraction)
9. [Stage 2 — Correlation & Classification](#9-stage-2--correlation--classification)
10. [Retrieval Pipeline](#10-retrieval-pipeline)
11. [Deterministic Rule Engine](#11-deterministic-rule-engine)
12. [Confidence Scoring](#12-confidence-scoring)
13. [Logging](#13-logging)
14. [Testing](#14-testing)
15. [Future Extensibility](#15-future-extensibility)

---

## 1. Overall Project Architecture

### 1.1 Design Philosophy

PECS is **not** a chatbot. It is a **traceability engine**. Every architectural decision serves one purpose: given a set of heterogeneous project documents, produce a fully-cited traceability matrix that maps each requirement to its implementation evidence and professor evaluation.

The system deliberately separates *retrieval* (finding relevant chunks) from *extraction* (pulling structured facts from those chunks) from *correlation* (linking facts across categories). This three-phase split exists because each phase has a different failure mode and a different evaluation metric, and collapsing them into one LLM call would make debugging impossible.

### 1.2 Complete Pipeline (Linear Data Flow)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EVIDENCE SOURCES                                │
│  PDFs · DOCX · Email exports · WhatsApp exports · Git logs            │
│  Markdown files · Python source code · README files                    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 1 — INGESTION                                 │
│  Streamlit file-upload → file-type detection → dispatch to adapter     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 2 — SOURCE-SPECIFIC PARSING                   │
│  Adapter per source type extracts raw text + structural metadata       │
│  (page numbers, thread IDs, commit SHAs, function boundaries, etc.)   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 3 — SOURCE-AWARE CHUNKING                     │
│  Splits text at semantically meaningful boundaries per source type     │
│  Preserves provenance metadata on every chunk                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 4 — MARKDOWN NORMALIZATION                    │
│  MarkItDown converts every chunk to clean Markdown                    │
│  Downstream components see ONE format regardless of source            │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              EvidenceChunk OBJECTS (universal internal format)          │
└──────────┬─────────────────────────────────────┬────────────────────────┘
           │                                     │
           ▼                                     ▼
┌──────────────────────────┐    ┌────────────────────────────────────────┐
│   MODULE 5 — EMBEDDINGS  │    │   (EvidenceChunk also passed to       │
│   nomic-embed-text       │    │    Stage 1 extraction later)          │
│   via Ollama             │    │                                        │
└──────────┬───────────────┘    └────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│   MODULE 6 — CHROMADB    │
│   Stores vectors +       │
│   metadata only          │
└──────────┬───────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 7 — RETRIEVAL PIPELINE                        │
│   Metadata filter → Vector Top-K → BM25 → Req-ID match → Merge       │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 8 — PHI STAGE 1 (EXTRACTION)                  │
│   Reads retrieved chunks → extracts structured JSON facts             │
│   Output: entities, claims, evaluation statements                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 9 — PYDANTIC VALIDATION                       │
│   Validates Stage 1 JSON against strict schemas                       │
│   Rejects / retries malformed output                                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 10 — SQLITE EVIDENCE STORE                    │
│   Immutable Evidence rows (one per extracted entity)                  │
│   Provenance preserved: chunk_id, source_document, author, timestamp  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 11 — DETERMINISTIC RULES                      │
│   Resolves unambiguous cases WITHOUT the LLM                          │
│   Exact requirement-ID matches, direct evaluation references          │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 12 — PHI STAGE 2 (CORRELATION)                │
│   Classifies ONLY ambiguous pairs into the 7-label taxonomy           │
│   Produces supporting evidence citations                              │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 13 — TRACEABILITY MATRIX                      │
│   Assembles final matrix: requirement → status → evidence → citations │
│   Confidence scores computed by formula, NOT by LLM                   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   MODULE 14 — STREAMLIT UI                             │
│   Project creation · file upload · traceability matrix viewer         │
│   Clickable citations · filtering · export                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Component Interaction Summary

| Producer | Consumer | Data Exchanged |
|---|---|---|
| Ingestion | Parser Adapters | Raw file bytes + file metadata |
| Parser Adapters | Chunker | Parsed text + structural metadata (pages, threads, etc.) |
| Chunker | Normalizer | Raw chunks with provenance |
| Normalizer | EvidenceChunk factory | Clean Markdown text per chunk |
| EvidenceChunk factory | Embedding module | Chunk text for vectorization |
| EvidenceChunk factory | ChromaDB | Chunk text + metadata for storage |
| ChromaDB | Retrieval pipeline | Top-K vectors + metadata |
| Retrieval pipeline | Stage 1 | Retrieved evidence chunks |
| Stage 1 | Pydantic validation | Raw JSON strings |
| Pydantic validation | SQLite | Validated ExtractionResult objects |
| SQLite | Deterministic rules | Structured evidence rows |
| Deterministic rules | Stage 2 | Only the ambiguous pairs |
| Stage 2 | Traceability generator | CorrelationResult objects |
| Traceability generator | Streamlit UI | Rendered matrix + citations |

### 1.4 Key Architectural Invariants

1. **Immutability of evidence**: Once an extracted fact is written to SQLite, it is never modified. Corrections produce new rows with provenance back to the original.
2. **Separation of stores**: ChromaDB holds embeddings for retrieval. SQLite holds structured knowledge for auditing. Original documents are never stored in either — only referenced by path.
3. **LLM as classifier, not narrator**: The LLM classifies evidence into fixed categories. It does not write free-form reports. This is the single most important design constraint.
4. **Provenance is mandatory**: Every piece of data in the system must be traceable to a specific source document, page/section, and chunk. If provenance cannot be established, the data is rejected.
5. **Deterministic-first**: Any correlation that can be resolved by exact string matching, requirement-ID matching, or rule logic must be resolved without the LLM. The LLM handles only what remains.

---

## 2. Recommended Project Folder Structure

```
LocalRAG/
├── documents/                    # TSD and project documentation (existing)
│   └── TSD_v0.3.docx
├── evidence/                     # Test evidence corpus (existing, do not touch)
│   ├── implementation_details/
│   └── initiation_discussion/
│
├── pecs/                         # ← ALL application code lives here
│   ├── __init__.py
│   ├── app.py                    # Streamlit entry point
│   ├── config.py                 # Central configuration (paths, model names, thresholds)
│   │
│   ├── ingestion/                # Module 1: File ingestion
│   │   ├── __init__.py
│   │   ├── ingestor.py           # Dispatcher: detect file type → route to adapter
│   │   └── file_utils.py         # MIME detection, path validation, deduplication
│   │
│   ├── parsing/                  # Module 2: Source-specific parsers
│   │   ├── __init__.py
│   │   ├── base_parser.py        # Abstract base class for all parsers
│   │   ├── pdf_parser.py
│   │   ├── docx_parser.py
│   │   ├── email_parser.py       # Gmail exported text
│   │   ├── whatsapp_parser.py    # WhatsApp exported chat
│   │   ├── git_parser.py         # Git commit log
│   │   ├── markdown_parser.py    # .md files
│   │   └── python_parser.py      # .py source files
│   │
│   ├── chunking/                 # Module 3: Source-aware chunking
│   │   ├── __init__.py
│   │   ├── base_chunker.py       # Abstract base class
│   │   ├── pdf_chunker.py
│   │   ├── docx_chunker.py
│   │   ├── email_chunker.py
│   │   ├── whatsapp_chunker.py
│   │   ├── git_chunker.py
│   │   ├── markdown_chunker.py
│   │   └── python_chunker.py
│   │
│   ├── normalization/            # Module 4: Markdown normalization
│   │   ├── __init__.py
│   │   └── normalizer.py         # MarkItDown wrapper + post-processing
│   │
│   ├── models/                   # Module: Universal internal objects
│   │   ├── __init__.py
│   │   ├── evidence_chunk.py     # EvidenceChunk dataclass
│   │   ├── retrieved_evidence.py # RetrievedEvidence dataclass
│   │   ├── extraction_result.py  # ExtractionResult Pydantic model
│   │   └── correlation_result.py # CorrelationResult Pydantic model
│   │
│   ├── embeddings/               # Module 5: Embedding generation
│   │   ├── __init__.py
│   │   └── embedder.py           # nomic-embed-text via Ollama
│   │
│   ├── vectorstore/              # Module 6: ChromaDB operations
│   │   ├── __init__.py
│   │   └── chroma_store.py       # Collection management, upsert, query
│   │
│   ├── store/                    # Module 10: SQLite evidence store
│   │   ├── __init__.py
│   │   ├── database.py           # Connection management, schema creation
│   │   ├── evidence_repo.py      # Evidence table CRUD (insert + read only)
│   │   └── correlation_repo.py   # Correlation table CRUD
│   │
│   ├── retrieval/                # Module 7: Retrieval pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py           # Orchestrator: filter → vector → BM25 → merge
│   │   ├── metadata_filter.py    # Pre-retrieval metadata filtering
│   │   ├── bm25.py               # Keyword/BM25 matching
│   │   └── merger.py             # Candidate merging + deduplication
│   │
│   ├── extraction/               # Module 8 + 9: Stage 1 extraction
│   │   ├── __init__.py
│   │   ├── extractor.py          # Phi-4-mini Stage 1 prompt + call
│   │   ├── prompts.py            # All Stage 1 prompt templates
│   │   └── validator.py          # Pydantic validation + retry logic
│   │
│   ├── correlation/              # Module 11 + 12: Stage 2 correlation
│   │   ├── __init__.py
│   │   ├── correlator.py         # Stage 2 orchestrator
│   │   ├── rules.py              # Deterministic rule engine
│   │   ├── classifier.py         # Phi-4-mini Stage 2 prompt + call
│   │   ├── prompts.py            # All Stage 2 prompt templates
│   │   └── confidence.py         # Confidence scoring formulas
│   │
│   ├── traceability/             # Module 13: Matrix generation
│   │   ├── __init__.py
│   │   └── matrix.py             # Assembles final traceability matrix
│   │
│   ├── ui/                       # Module 14: Streamlit UI
│   │   ├── __init__.py
│   │   ├── pages/
│   │   │   ├── 01_upload.py      # File upload + project management
│   │   │   ├── 02_evidence.py    # Evidence browser
│   │   │   ├── 03_matrix.py      # Traceability matrix viewer
│   │   │   └── 04_debug.py       # Debug / log viewer
│   │   └── components/
│   │       ├── citation.py       # Clickable citation renderer
│   │       └── matrix_table.py   # Matrix table component
│   │
│   └── logging_config.py         # Centralized logging setup
│
├── tests/                        # All tests
│   ├── __init__.py
│   ├── conftest.py               # Shared fixtures
│   ├── unit/
│   │   ├── test_parsers.py
│   │   ├── test_chunkers.py
│   │   ├── test_normalizer.py
│   │   ├── test_embedder.py
│   │   ├── test_evidence_repo.py
│   │   ├── test_validator.py
│   │   ├── test_rules.py
│   │   └── test_confidence.py
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_retrieval_pipeline.py
│   │   ├── test_extraction_pipeline.py
│   │   └── test_correlation_pipeline.py
│   └── evaluation/
│       ├── labelled_data/        # Hand-verified triples
│       │   └── golden_triples.json
│       ├── eval_retrieval.py
│       ├── eval_extraction.py
│       └── eval_correlation.py
│
├── data/                         # Runtime data (gitignored)
│   ├── chromadb/                 # ChromaDB persistent storage
│   ├── sqlite/                   # SQLite database file
│   │   └── pecs.db
│   └── logs/                     # Application logs
│       └── pecs.log
│
├── pyproject.toml                # Project metadata + dependencies
├── .env                          # Environment variables (Ollama URL, etc.)
├── .gitignore
└── README.md
```

### 2.1 Rationale for Structure

**Why `pecs/` as the application root?** Keeps application code separate from evidence, documents, tests, and runtime data. Standard Python package layout.

**Why separate `parsing/` and `chunking/`?** Parsing extracts raw text and structural metadata. Chunking decides where to split. These are conceptually different responsibilities with different failure modes. A parser bug produces garbled text; a chunker bug produces chunks that cross semantic boundaries. Keeping them separate makes debugging easier.

**Why `models/` instead of scattering dataclasses?** Every module in the pipeline produces and consumes well-defined objects. Centralizing them prevents circular imports and ensures all modules agree on the data contract.

**Why `store/` instead of `database/`?** The word "store" is intentionally neutral — it is the evidence store, not a generic database layer. This naming reinforces the architectural constraint that SQLite holds extracted evidence, not raw documents.

**Why `data/` is gitignored?** ChromaDB and SQLite files are runtime artifacts. They should never be committed. The evidence corpus in `evidence/` is the source of truth; the stores are always re-derivable from it.

---

## 3. Module Breakdown

### 3.1 Module 1 — Ingestion (`pecs/ingestion/`)

#### Why It Exists

The ingestion module is the system's front door. It accepts files from the Streamlit upload widget, validates them, detects their type, and dispatches them to the correct parser. Without it, every module downstream would need to know how to open files, detect MIME types, and handle upload errors.

#### Inputs

- Raw uploaded files from Streamlit's `file_uploader` (bytes + filename)
- User-supplied metadata: project name, evidence category (optional), description (optional)

#### Outputs

- A `ParseRequest` object dispatched to the correct parser adapter, containing:
  - File bytes or file path
  - Detected MIME type
  - Source filename
  - User-supplied metadata
  - Upload timestamp

#### Internal Processing

1. **File type detection**: Use `python-magic` (libmagic bindings) for MIME detection, not file extensions alone. File extensions lie; MIME sniffing is more reliable. Fall back to extension-based detection only if libmagic is unavailable.
2. **Deduplication check**: Compute SHA-256 hash of file contents. Query SQLite to check if this exact hash has been ingested before. If yes, warn the user and block re-ingestion (immutability constraint).
3. **Size validation**: Reject files larger than a configurable maximum (default: 50 MB). This prevents memory exhaustion during parsing.
4. **Dispatch**: Map detected MIME type to the correct parser adapter. If no adapter exists for the type, reject with a clear error message.

#### Design Decisions

- **Why SHA-256 deduplication?** Prevents the same document from being ingested twice, which would create duplicate evidence rows and pollute retrieval. SHA-256 is collision-resistant enough for this use case.
- **Why MIME detection over extension?** A `.txt` file might be a WhatsApp export. Extension alone cannot distinguish between a plain text file, a WhatsApp chat export, and an email export. MIME detection + content-sniffing heuristics (see parser adapters) are more robust.

#### Edge Cases

- **Same content, different filename**: SHA-256 catches this. Same hash → same document → reject.
- **Same filename, different content**: Different hash → different document → ingest as new.
- **Corrupted files**: Parser will fail; ingestion module should catch the exception and surface it to the user via Streamlit rather than crashing.
- **Empty files**: Reject at ingestion time with a clear error.
- **Unsupported file types**: Return a user-facing error listing supported types.

#### Module Interactions

- **Upstream**: Streamlit UI (provides files)
- **Downstream**: Parser adapters (receives `ParseRequest`)
- **Side-effect**: Writes ingestion log entry (file hash, timestamp, status)

---

### 3.2 Module 2 — Source-Specific Parsing (`pecs/parsing/`)

#### Why It Exists

Different source types have fundamentally different structures. A PDF has pages; a WhatsApp export has timestamped messages; a Python file has functions and classes. A single generic parser would lose this structural information. Each parser adapter extracts both the text content and the structural metadata that the chunker will later use to decide chunk boundaries.

#### Inputs

- `ParseRequest` object from ingestion (file bytes + metadata)

#### Outputs

- `ParsedDocument` object containing:
  - `source_type`: enum (PDF, DOCX, EMAIL, WHATSAPP, GIT, MARKDOWN, PYTHON)
  - `raw_text`: Full extracted text (as-is, before normalization)
  - `structural_metadata`: Source-specific structure (see below)
  - `global_metadata`: Author, date, filename, hash, etc.
  - `parse_warnings`: List of non-fatal issues encountered

#### Structural Metadata by Source Type

| Source Type | Structural Metadata |
|---|---|
| PDF | List of `(page_number, page_text)` tuples |
| DOCX | List of `(heading_level, heading_text, section_text)` tuples |
| Email | List of `(sender, timestamp, subject, body)` message objects |
| WhatsApp | List of `(sender, timestamp, message_text)` message objects |
| Git | List of `(commit_sha, author, date, message, diff_summary)` objects |
| Markdown | List of `(heading_level, heading_text, section_text)` tuples |
| Python | List of `(element_type, name, docstring, body, line_range)` objects |

#### Internal Processing per Parser

**PDF Parser**
- Library: `PyMuPDF` (fitz). Chosen over `pdfplumber` for speed and reliability.
- Extracts text page-by-page, preserving page numbers.
- Detects and flags image-only pages (no extractable text → logged as a warning, page skipped per the no-OCR constraint).
- Preserves table structures where possible (PyMuPDF can extract table blocks).

**DOCX Parser**
- Library: `python-docx`.
- Walks the paragraph tree, tracking heading levels to build a section hierarchy.
- Extracts tables as Markdown tables.
- Preserves lists, bold/italic emphasis where semantically meaningful.

**Email Parser**
- No external library needed for pre-exported text. Use regex-based parsing.
- Pattern: detect email headers (`From:`, `Date:`, `Subject:`, `To:`) to split into individual messages.
- Handle both single-email and thread/conversation exports.
- Preserve sender attribution and timestamp per message.

**WhatsApp Parser**
- No external library needed. WhatsApp exports follow a known format: `[DD/MM/YY, HH:MM:SS] Sender: Message`.
- Regex pattern to split by timestamp + sender.
- Handle multi-line messages (message continues until next timestamp line).
- Handle media placeholders (`<Media omitted>`) — log and skip.
- Handle system messages (`Sender added Sender2`) — log and skip.

**Git Parser**
- Library: `gitpython` or shell out to `git log --format=...`.
- Extract commit SHA, author, date, commit message, and optionally a summary of changed files.
- Do not extract full diffs (too large, too noisy). Extract only the commit message and file-change summary.

**Markdown Parser**
- No external library needed. Parse ATX headings (`#`, `##`, etc.) to build section hierarchy.
- Preserve code blocks, lists, and emphasis.
- Handle front-matter (YAML between `---` delimiters) as metadata.

**Python Parser**
- Library: Python's built-in `ast` module.
- Parse the AST to extract: module-level docstring, class definitions (name, docstring, methods), function definitions (name, docstring, body), and import statements.
- Preserve line numbers for provenance.

#### Design Decisions

- **Why one parser per source type instead of a generic parser?** MarkItDown can convert many formats to Markdown, but it loses structural metadata. We need page numbers from PDFs, timestamps from WhatsApp, commit SHAs from Git. Only source-specific parsers can extract this.
- **Why is MarkItDown not used here?** MarkItDown is used later, in the normalization step. Here, we extract raw text + structure. MarkItDown normalizes the text format. These are different responsibilities.

#### Edge Cases

- **PDF with mixed text and image pages**: Extract text pages, skip image pages with a warning.
- **DOCX with no headings**: Treat the entire document as one section. The chunker will apply fallback size-based splitting.
- **WhatsApp export in non-standard date format**: Support multiple date format patterns. Log unrecognized formats.
- **Python files with syntax errors**: `ast.parse` will fail. Fall back to treating the file as plain text.
- **Git log from a repo with 10,000+ commits**: Apply a configurable limit (e.g., last 500 commits). Log the truncation.

#### Module Interactions

- **Upstream**: Ingestion module (provides `ParseRequest`)
- **Downstream**: Chunking module (receives `ParsedDocument`)

---

### 3.3 Module 3 — Source-Aware Chunking (`pecs/chunking/`)

> Detailed chunking strategies per source type are covered in [Section 7](#7-source-aware-chunking).

#### Why It Exists

Chunking determines the granularity of retrieval. Chunk too large, and the LLM receives irrelevant context. Chunk too small, and the LLM lacks sufficient context to extract meaningful facts. Source-aware chunking exists because the optimal chunk boundary differs dramatically by source type: a WhatsApp message is a natural chunk; a PDF section is a natural chunk; a Python function is a natural chunk. Generic fixed-size chunking destroys all of these natural boundaries.

#### Inputs

- `ParsedDocument` from the parser (text + structural metadata)

#### Outputs

- List of `RawChunk` objects, each containing:
  - `chunk_text`: The raw text of the chunk (not yet normalized)
  - `chunk_index`: Position in the original document (0-indexed)
  - `source_locator`: Source-specific location (page number, timestamp, line range, etc.)
  - `metadata`: Inherited from the `ParsedDocument` global metadata
  - `char_count`: Length of the chunk text

#### Internal Processing

1. Receive `ParsedDocument`.
2. Dispatch to source-specific chunker based on `source_type`.
3. Source-specific chunker applies its strategy (see Section 7).
4. **Overlap**: Apply configurable overlap between adjacent chunks (default: 10% of chunk size, or 2 messages for conversational sources). Overlap prevents information loss at chunk boundaries.
5. **Size enforcement**: If any chunk exceeds the maximum size (default: 1500 characters), split it further using sentence-boundary detection (regex on `.`, `!`, `?` followed by whitespace + capital letter). This is the fallback — ideally, source-aware splitting already produces appropriately-sized chunks.
6. **Minimum size enforcement**: If any chunk is below the minimum size (default: 50 characters), merge it with the adjacent chunk. Very short chunks produce poor embeddings.

#### Design Decisions

- **Why character count, not token count, for size limits?** Token counting requires a tokenizer, which adds complexity and is model-specific. Character count is a sufficient proxy for an embedding model like nomic-embed-text, which has a generous context window (8192 tokens). A 1500-character chunk is approximately 300–400 tokens, well within limits.
- **Why overlap?** A fact might span a chunk boundary (e.g., "Requirement R3 was implemented" at the end of one chunk and "using a GNN-based approach" at the beginning of the next). Overlap ensures both chunks contain the complete statement.

#### Module Interactions

- **Upstream**: Parser (provides `ParsedDocument`)
- **Downstream**: Normalizer (receives list of `RawChunk`)

---

### 3.4 Module 4 — Markdown Normalization (`pecs/normalization/`)

#### Why It Exists

Downstream components (embeddings, LLM prompts) should see exactly one format. Markdown is that format. MarkItDown converts heterogeneous text (extracted from PDFs, DOCX, emails, etc.) into clean, consistent Markdown. This simplifies prompt engineering and ensures embedding quality is not degraded by formatting inconsistencies.

#### Inputs

- List of `RawChunk` objects from the chunker

#### Outputs

- List of `EvidenceChunk` objects (the universal internal object — see Section 4)

#### Internal Processing

1. For each `RawChunk`:
   a. Pass `chunk_text` through MarkItDown conversion.
   b. Post-process the Markdown output:
      - Strip excessive whitespace (more than 2 consecutive newlines → 2).
      - Normalize Unicode characters (NFKC normalization).
      - Remove MarkItDown artifacts or boilerplate if any.
   c. Generate a deterministic `chunk_id` using: `SHA-256(source_document_hash + chunk_index)`. This ensures the same document always produces the same chunk IDs, enabling deduplication.
   d. Construct an `EvidenceChunk` object with the normalized text, all inherited metadata, and the generated `chunk_id`.

#### Design Decisions

- **Why MarkItDown and not a custom normalizer?** MarkItDown handles edge cases (nested lists, tables, code blocks) that a custom regex-based normalizer would get wrong. It is the TSD-specified tool.
- **Why deterministic chunk IDs?** If the same document is re-ingested (despite the deduplication check), the same chunk IDs will be generated. This enables idempotent upserts into ChromaDB and prevents duplicate vectors.
- **Why NFKC normalization?** PDFs often contain typographic Unicode characters (em-dashes, curly quotes, ligatures) that look different but mean the same thing. NFKC normalizes these to their ASCII-compatible equivalents, improving both embedding quality and string matching.

#### Edge Cases

- **MarkItDown fails on a chunk**: Log the error, fall back to using the raw text as-is (it is already text, just not nicely formatted Markdown). Do not drop the chunk.
- **Empty chunk after normalization**: Drop it with a warning log. This can happen if the raw chunk was all whitespace or formatting.

#### Module Interactions

- **Upstream**: Chunker (provides list of `RawChunk`)
- **Downstream**: Embedding module (receives `EvidenceChunk` text), ChromaDB (receives `EvidenceChunk` for storage), Stage 1 extraction (receives `EvidenceChunk` during retrieval)

---

### 3.5 Module 5 — Embeddings (`pecs/embeddings/`)

#### Why It Exists

Semantic retrieval requires vector representations of text. This module generates embeddings for every `EvidenceChunk` using `nomic-embed-text` served by Ollama.

#### Inputs

- `EvidenceChunk` objects (specifically, their `normalized_text` field)

#### Outputs

- Embedding vectors (list of floats) associated with each `chunk_id`

#### Internal Processing

1. Batch `EvidenceChunk` texts into groups of configurable size (default: 32 chunks per batch). Batching reduces HTTP round-trips to Ollama.
2. For each batch, call Ollama's `/api/embeddings` endpoint with model `nomic-embed-text`.
3. Receive embedding vectors (768 dimensions for nomic-embed-text).
4. Associate each vector with its `chunk_id`.
5. Return the list of `(chunk_id, vector)` pairs.

#### Design Decisions

- **Why nomic-embed-text?** It is the TSD-specified model. It produces 768-dimensional embeddings, runs efficiently on local hardware, and is available through Ollama. Its context window (8192 tokens) is large enough for our chunk sizes.
- **Why batch processing?** Embedding one chunk at a time would create 100+ HTTP calls for a single document. Batching amortizes the overhead.
- **Why not embed on-the-fly during retrieval?** Queries are embedded on-the-fly (they must be, since we don't know the query in advance). But document chunks are embedded at ingestion time and stored, because re-embedding on every query would be prohibitively slow.

#### Edge Cases

- **Ollama not running**: Detect at startup. Surface a clear error in the Streamlit UI: "Ollama is not running. Please start Ollama with `ollama serve`."
- **nomic-embed-text not pulled**: Detect at startup. Surface: "Model nomic-embed-text not found. Please run `ollama pull nomic-embed-text`."
- **Embedding API timeout**: Retry up to 3 times with exponential backoff (1s, 2s, 4s). Log each retry.
- **Very long chunk text exceeding model context**: Should not happen if the chunker's size limits are enforced, but if it does, truncate the text to the model's context window and log a warning.

#### Module Interactions

- **Upstream**: Normalizer (provides `EvidenceChunk` objects)
- **Downstream**: ChromaDB (receives embedding vectors + metadata for storage)

---

### 3.6 Module 6 — ChromaDB (`pecs/vectorstore/`)

> Detailed ChromaDB design is covered in [Section 6](#6-chromadb-design).

#### Why It Exists

ChromaDB is the vector store for semantic retrieval. It stores embeddings alongside metadata so the retrieval pipeline can combine vector similarity with metadata filtering.

#### Inputs

- `EvidenceChunk` objects (metadata) + embedding vectors from the embedder

#### Outputs

- Query results: list of `(chunk_id, distance, metadata)` tuples

#### Internal Processing

- Uses ChromaDB's persistent client (storage directory: `data/chromadb/`).
- Single collection named `pecs_evidence`.
- Each document in ChromaDB stores: `id` (chunk_id), `embedding`, `document` (normalized text), `metadata` (structured fields — see Section 6).
- Upsert semantics: if a `chunk_id` already exists, update it. This supports re-ingestion.
- Query: embed the query text, then call `collection.query()` with optional `where` filters and `n_results`.

#### Module Interactions

- **Upstream**: Embedding module (provides vectors), Normalizer (provides chunk metadata)
- **Downstream**: Retrieval pipeline (queries for candidates)

---

### 3.7 Module 7 — Retrieval Pipeline (`pecs/retrieval/`)

> Detailed retrieval pipeline design is covered in [Section 10](#10-retrieval-pipeline).

#### Why It Exists

Retrieval is the bridge between the knowledge base and the LLM. Its job is to find the most relevant evidence chunks for a given requirement or query. The pipeline uses multiple retrieval strategies (vector, keyword, metadata, ID-matching) and merges the results to maximize recall without overwhelming the LLM with noise.

#### Module Interactions

- **Upstream**: ChromaDB (vector retrieval), SQLite (structured queries), BM25 index (keyword retrieval)
- **Downstream**: Stage 1 extraction (receives merged candidate evidence)

---

### 3.8 Module 8 — Stage 1 Extraction (`pecs/extraction/`)

> Detailed Stage 1 design is covered in [Section 8](#8-stage-1--extraction).

#### Why It Exists

The retrieval pipeline returns raw text chunks. Stage 1 transforms these into structured, validated facts. Without this stage, the correlation engine would need to work directly with raw text, making it impossible to apply deterministic rules or produce auditable results.

#### Inputs

- List of `RetrievedEvidence` objects (chunks + retrieval scores + metadata)

#### Outputs

- List of `ExtractionResult` objects (structured facts validated by Pydantic)

#### Module Interactions

- **Upstream**: Retrieval pipeline (provides candidate chunks)
- **Downstream**: Pydantic validation → SQLite (stores validated facts)

---

### 3.9 Module 9 — Pydantic Validation (`pecs/extraction/validator.py`)

#### Why It Exists

Phi-4-mini's JSON output is not guaranteed to be well-formed or schema-compliant. Pydantic validation enforces structural correctness and type safety before anything reaches SQLite. Without it, a single malformed LLM response could corrupt the evidence store.

#### Inputs

- Raw JSON string from Phi-4-mini Stage 1 output

#### Outputs

- Validated `ExtractionResult` object, or a validation error triggering retry

#### Internal Processing

1. Attempt to parse the JSON string.
2. If JSON parsing fails: extract JSON from the response using regex (LLMs sometimes wrap JSON in markdown code fences or add preamble text). Try again.
3. If JSON parsing succeeds: validate against the `ExtractionResult` Pydantic model.
4. If Pydantic validation fails: log the specific validation errors, increment retry counter, re-prompt the LLM with the validation errors included in the prompt ("Your previous response had the following errors: …. Please fix and respond again.").
5. Maximum retries: 3. After 3 failures, log the chunk as "extraction_failed" and skip it. Do not block the pipeline.

#### Design Decisions

- **Why include validation errors in retry prompts?** This gives the LLM specific feedback on what went wrong, significantly increasing the chance of a correct response on retry. Generic "try again" prompts are less effective.
- **Why skip after 3 failures instead of crashing?** A single problematic chunk should not block extraction for the entire document. Log it, flag it, move on. The traceability matrix will show a gap, which is honest.

#### Edge Cases

- **LLM returns valid JSON but with extra fields**: Pydantic's `model_config = ConfigDict(extra='ignore')` — silently ignore extra fields.
- **LLM returns valid JSON but with missing required fields**: Pydantic will raise `ValidationError`. Retry with error feedback.
- **LLM returns a JSON array instead of a single object**: Handle both cases — if array, validate each element independently.

#### Module Interactions

- **Upstream**: Stage 1 extractor (provides raw JSON)
- **Downstream**: SQLite evidence store (receives validated objects)

---

### 3.10 Module 10 — SQLite Evidence Store (`pecs/store/`)

> Detailed SQLite design is covered in [Section 5](#5-sqlite-design).

#### Why It Exists

ChromaDB stores embeddings for retrieval. SQLite stores extracted, structured knowledge for auditing and deterministic correlation. This separation is the core architectural decision of PECS v0.3. Without SQLite, every query would require the LLM to re-extract facts from raw chunks — expensive, non-deterministic, and non-auditable.

#### Module Interactions

- **Upstream**: Pydantic validator (provides validated `ExtractionResult` objects)
- **Downstream**: Deterministic rules engine (queries evidence rows), Correlation engine (queries evidence rows), Traceability matrix generator (queries correlation results)

---

### 3.11 Module 11 — Deterministic Rule Engine (`pecs/correlation/rules.py`)

> Detailed rule engine design is covered in [Section 11](#11-deterministic-rule-engine).

#### Why It Exists

Many correlations are unambiguous and can be resolved without the LLM. If an implementation evidence row explicitly references "Requirement R3" by ID, the correlation is deterministic. Using the LLM for these cases wastes compute and introduces unnecessary non-determinism. The rule engine resolves what it can and passes only ambiguous cases to Stage 2.

#### Module Interactions

- **Upstream**: SQLite (provides evidence rows)
- **Downstream**: Stage 2 correlator (receives only ambiguous pairs)

---

### 3.12 Module 12 — Stage 2 Correlation (`pecs/correlation/`)

> Detailed Stage 2 design is covered in [Section 9](#9-stage-2--correlation--classification).

#### Why It Exists

After deterministic rules have resolved the easy cases, ambiguous evidence pairs remain. Stage 2 uses Phi-4-mini to classify these pairs into the 7-label taxonomy. It is a focused classifier, not a free-form reasoning engine.

#### Module Interactions

- **Upstream**: Deterministic rules engine (provides ambiguous pairs + supporting context)
- **Downstream**: Confidence scorer (produces numerical confidence), Traceability matrix (receives final classifications)

---

### 3.13 Module 13 — Traceability Matrix Generation (`pecs/traceability/`)

#### Why It Exists

The traceability matrix is the primary output of PECS. It is the structured, citable answer to: "For each requirement, what evidence supports its implementation, and how was it evaluated?"

#### Inputs

- All evidence rows from SQLite (requirements, implementations, evaluations)
- All correlation results (deterministic + LLM-classified)
- Confidence scores

#### Outputs

- `TraceabilityMatrix` object: a list of `TraceabilityRow` objects, each containing:
  - `requirement_id`: The requirement identifier
  - `requirement_text`: The requirement statement
  - `status`: One of the 7 classification labels
  - `implementation_evidence`: List of supporting evidence with citations
  - `evaluation_evidence`: List of evaluation evidence with citations
  - `confidence`: Computed confidence score
  - `supporting_chunk_ids`: List of chunk IDs for provenance
  - `source_citations`: Human-readable citations (document name, page, section)

#### Internal Processing

1. Query SQLite for all requirement entities.
2. For each requirement, query correlations.
3. For each correlation, gather supporting evidence rows and their citations.
4. Compute confidence score (see Section 12).
5. Assemble into `TraceabilityRow`.
6. Sort by requirement_id.
7. Identify orphans:
   - Requirements with no linked implementation → `requirement_not_implemented`
   - Evaluation evidence with no linked requirement → `evaluation_without_requirement`

#### Design Decisions

- **Why not generate the matrix on-the-fly in the UI?** The matrix depends on joining data from SQLite, correlations, and confidence scores. This join logic is complex enough to warrant its own module. The UI should only render what the matrix module provides.

#### Module Interactions

- **Upstream**: SQLite, Correlation engine, Confidence scorer
- **Downstream**: Streamlit UI (renders the matrix)

---

### 3.14 Module 14 — Streamlit UI (`pecs/ui/`)

#### Why It Exists

Streamlit is the TSD-specified interface. It serves double duty: input (file upload, project management) and output (traceability matrix, evidence browser, debug logs).

#### Pages

**Page 1 — Upload & Project Management**
- Create a new project (project name, description).
- Upload evidence files (multi-file upload).
- See ingestion status: which files have been processed, their hash, source type, chunk count.
- Trigger re-ingestion for a specific file.

**Page 2 — Evidence Browser**
- Browse all evidence chunks in the system.
- Filter by source type, source document, entity type.
- View the normalized Markdown text of each chunk.
- See the provenance chain: source document → page/section → chunk.

**Page 3 — Traceability Matrix**
- Primary screen. This is what the user will see most of the time.
- Table view: one row per requirement.
- Columns: Requirement ID, Statement, Status (color-coded), Implementation Evidence (clickable), Evaluation Evidence (clickable), Confidence (color-coded bar).
- Filter by status, confidence threshold, source document.
- Click any evidence citation to see the exact source text in context.
- Export to CSV/JSON.

**Page 4 — Debug / Logs**
- View application logs (tail of `pecs.log`).
- View extraction failures (chunks where Stage 1 failed).
- View retrieval statistics (average recall, latency).
- View Ollama status (is it running? which models are loaded?).

#### Design Decisions

- **Why multi-page Streamlit instead of a single page?** PECS has distinct workflows (upload vs. view matrix vs. debug). Multi-page layout keeps each workflow focused and avoids a cluttered single-page UI.
- **Why no chatbot interface?** The TSD explicitly states: "The primary screen is a traceability matrix, not a chatbot." A chat interface would encourage open-ended questions, undermining the structured traceability goal.

#### Module Interactions

- **Upstream**: All modules (ingestion, traceability matrix, logs)
- **Downstream**: User (renders results)

---

### 3.15 Module 15 — Logging (`pecs/logging_config.py`)

> Detailed logging specification is covered in [Section 13](#13-logging).

#### Why It Exists

Every module in the pipeline can fail, and the failure modes are diverse (file parsing errors, embedding API timeouts, LLM hallucinations, Pydantic validation failures). Without centralized, structured logging, diagnosing issues in a multi-stage pipeline would be extremely difficult.

---

### 3.16 Module 16 — Testing (`tests/`)

> Detailed testing specification is covered in [Section 14](#14-testing).

#### Why It Exists

PECS has no external API or CI/CD pipeline to provide feedback. Tests are the only systematic way to verify that each module works correctly and that the end-to-end pipeline produces accurate results.

---

## 4. Universal Internal Objects

These are the data classes and Pydantic models that flow between modules. They are the system's internal API. Every module produces and consumes these objects; no module should pass raw dictionaries or untyped data to another.

### 4.1 `EvidenceChunk`

```
EvidenceChunk
├── chunk_id: str              # Deterministic: SHA-256(doc_hash + chunk_index)
├── source_document: str       # Original filename
├── source_hash: str           # SHA-256 of the original file
├── source_type: SourceType    # Enum: PDF, DOCX, EMAIL, WHATSAPP, GIT, MARKDOWN, PYTHON
├── chunk_index: int           # Position in the original document (0-indexed)
├── source_locator: str        # Source-specific location (e.g., "page 3", "2024-01-15 14:22", "line 45-67")
├── normalized_text: str       # Clean Markdown text after normalization
├── char_count: int            # Length of normalized_text
├── metadata: dict             # Flexible JSON metadata (author, project, etc.)
├── created_at: datetime       # Timestamp of creation
```

**Why it exists**: This is the universal currency of the pipeline. Every module from normalization onward works with `EvidenceChunk` objects. It encapsulates both the text and its full provenance, ensuring that no downstream module can accidentally lose provenance information.

**Why `source_locator` is a string, not a structured type**: Different sources have different locator formats (page numbers vs. timestamps vs. line ranges). A string is the simplest universal type. The string follows source-specific conventions that the UI can parse for display.

**Why `metadata` is a dict, not a fixed schema**: Different sources have different metadata. Email has "sender" and "subject"; Git has "commit_sha" and "author". A flexible dict accommodates this without requiring a union type of every possible metadata field.

### 4.2 `RetrievedEvidence`

```
RetrievedEvidence
├── chunk: EvidenceChunk       # The retrieved chunk
├── vector_score: float        # Cosine similarity score from ChromaDB (0.0–1.0)
├── bm25_score: float | None   # BM25 score if keyword retrieval was used
├── id_match: bool             # Whether this chunk matched by requirement-ID
├── combined_score: float      # Weighted combination (see Section 12)
├── retrieval_method: str      # Which method(s) surfaced this chunk
```

**Why it exists**: The retrieval pipeline uses multiple methods. Each method produces a different score. `RetrievedEvidence` wraps an `EvidenceChunk` with retrieval-specific scores so the downstream pipeline can make informed decisions about relevance. It also preserves which method surfaced the chunk, enabling retrieval evaluation.

**Why `bm25_score` is optional**: BM25 retrieval is supplementary. Not every chunk will have a BM25 score (e.g., if it was found only by vector search). The field is optional to reflect this.

### 4.3 `ExtractionResult`

```
ExtractionResult (Pydantic BaseModel)
├── entity_type: EntityType    # Enum: REQUIREMENT, IMPLEMENTATION, EVALUATION
├── entity_id: str             # Extracted identifier (e.g., "R3", "GNN-pipeline-step-2")
├── text: str                  # The extracted factual statement
├── linked_requirement: str | None  # If this entity references a requirement, which one?
├── source_document: str       # Inherited from the EvidenceChunk
├── chunk_id: str              # Inherited from the EvidenceChunk
├── author: str | None         # Extracted or inherited author
├── timestamp: str | None      # Extracted or inherited timestamp
├── metadata: dict             # Additional extracted metadata (flexible)
```

**Why it exists**: This is the Pydantic-validated output of Stage 1 extraction. It maps directly to a row in the SQLite Evidence table. Pydantic validation ensures type safety, required fields, and value constraints before anything touches the database.

**Why `entity_id` is a string, not an auto-incremented integer**: Entity IDs are extracted from the source text (e.g., "Requirement R3", "Feature: GNN training"). They are semantic identifiers, not database keys. The database has its own auto-incremented `id` primary key.

**Why `linked_requirement` is optional**: Not every entity explicitly references a requirement. An evaluation comment might say "The data pipeline is well-implemented" without naming a specific requirement. In this case, `linked_requirement` is null, and the correlation engine must infer the link later.

### 4.4 `CorrelationResult`

```
CorrelationResult (Pydantic BaseModel)
├── correlation_id: str           # Auto-generated UUID
├── requirement_entity_id: str    # The requirement being correlated
├── evidence_entity_id: str       # The implementation/evaluation evidence
├── status: CorrelationStatus     # One of the 7 classification labels
├── resolution_method: str        # "deterministic_rule" or "llm_stage2"
├── rule_name: str | None         # If deterministic, which rule fired
├── confidence: float             # Computed confidence score (0.0–1.0)
├── supporting_chunk_ids: list[str]  # Chunks that support this correlation
├── reasoning: str                # Brief justification (from LLM or rule description)
├── created_at: datetime          # Timestamp
```

**Why it exists**: This represents a single link in the traceability matrix. It connects a requirement to evidence, classifies the relationship, and provides the confidence score and supporting citations needed for auditing.

**Why `resolution_method` is tracked**: For auditing and debugging. If a correlation was resolved by a deterministic rule, it is more trustworthy than one resolved by the LLM. The traceability matrix can surface this distinction.

**Why `rule_name` is tracked**: If the deterministic rule engine fires, we record which rule matched. This makes the system debuggable: "This correlation was resolved by rule `exact_requirement_id_match` because the implementation evidence contains the string 'Requirement R3'."

### 4.5 `CorrelationStatus` Enum

The seven classification labels from the TSD:

| Label | Meaning |
|---|---|
| `IMPLEMENTED_AND_VALIDATED` | Requirement has implementation evidence AND positive evaluation |
| `IMPLEMENTED_BUT_NEGATIVELY_EVALUATED` | Requirement has implementation evidence AND negative evaluation |
| `IMPLEMENTED_WITHOUT_EVALUATION` | Requirement has implementation evidence but no evaluation |
| `PARTIALLY_IMPLEMENTED` | Requirement has some but not all implementation evidence |
| `CLAIMED_BUT_NO_EVIDENCE` | Requirement is claimed to be implemented but no supporting evidence exists |
| `EVALUATION_WITHOUT_REQUIREMENT` | Evaluation evidence exists but references no known requirement |
| `REQUIREMENT_NOT_IMPLEMENTED` | Requirement exists but no implementation evidence was found |

### 4.6 `SourceType` Enum

```
PDF | DOCX | EMAIL | WHATSAPP | GIT | MARKDOWN | PYTHON
```

### 4.7 `EntityType` Enum

```
REQUIREMENT | IMPLEMENTATION | EVALUATION
```

---

## 5. SQLite Design

### 5.1 Design Philosophy

The finalized architecture uses **one immutable Evidence table** and **one Correlation table**. This is a deliberate simplification from the TSD's multi-table normalized design (Requirement, ImplementationEvidence, EvaluationEvidence, Correlation). The rationale:

1. **Simplicity**: One table is easier to query, debug, and reason about.
2. **Flexibility**: The `entity_type` column distinguishes requirements from implementations from evaluations. No schema migration is needed if a new entity type is added later.
3. **Immutability**: Rows are never updated. This guarantees auditability — every fact ever extracted is preserved.

### 5.2 Evidence Table Schema

```sql
CREATE TABLE IF NOT EXISTS evidence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT NOT NULL CHECK(entity_type IN ('REQUIREMENT', 'IMPLEMENTATION', 'EVALUATION')),
    entity_id       TEXT NOT NULL,
    text            TEXT NOT NULL,
    linked_requirement TEXT,
    source_document TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,
    author          TEXT,
    timestamp       TEXT,
    metadata        TEXT,  -- JSON string
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    
    -- Provenance constraint: every row MUST have source_document and chunk_id
    CHECK(length(source_document) > 0),
    CHECK(length(chunk_id) > 0)
);
```

### 5.3 Correlation Table Schema

```sql
CREATE TABLE IF NOT EXISTS correlation (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id        TEXT NOT NULL UNIQUE,  -- UUID
    requirement_entity_id TEXT NOT NULL,
    evidence_entity_id    TEXT NOT NULL,
    status                TEXT NOT NULL CHECK(status IN (
                              'IMPLEMENTED_AND_VALIDATED',
                              'IMPLEMENTED_BUT_NEGATIVELY_EVALUATED',
                              'IMPLEMENTED_WITHOUT_EVALUATION',
                              'PARTIALLY_IMPLEMENTED',
                              'CLAIMED_BUT_NO_EVIDENCE',
                              'EVALUATION_WITHOUT_REQUIREMENT',
                              'REQUIREMENT_NOT_IMPLEMENTED'
                          )),
    resolution_method     TEXT NOT NULL CHECK(resolution_method IN ('deterministic_rule', 'llm_stage2')),
    rule_name             TEXT,
    confidence            REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    supporting_chunk_ids  TEXT,  -- JSON array of chunk_id strings
    reasoning             TEXT,
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
);
```

### 5.4 Indexes

```sql
-- Evidence table indexes
CREATE INDEX idx_evidence_entity_type ON evidence(entity_type);
CREATE INDEX idx_evidence_entity_id ON evidence(entity_id);
CREATE INDEX idx_evidence_linked_req ON evidence(linked_requirement);
CREATE INDEX idx_evidence_source_doc ON evidence(source_document);
CREATE INDEX idx_evidence_chunk_id ON evidence(chunk_id);

-- Correlation table indexes
CREATE INDEX idx_correlation_req ON correlation(requirement_entity_id);
CREATE INDEX idx_correlation_evidence ON correlation(evidence_entity_id);
CREATE INDEX idx_correlation_status ON correlation(status);
```

**Why these indexes?**
- `entity_type`: The deterministic rule engine queries all requirements, all implementations, or all evaluations separately. This index makes those queries fast.
- `entity_id`: Lookup by entity identifier (e.g., "R3") is a primary access pattern.
- `linked_requirement`: The correlation engine queries "all evidence linked to requirement X." This index supports that.
- `source_document`: Filtering by source document in the evidence browser.
- `chunk_id`: Joining evidence back to ChromaDB chunks for citation generation.
- `requirement_entity_id` / `evidence_entity_id`: The traceability matrix queries correlations by requirement.

### 5.5 Insertion Flow

1. Stage 1 extraction produces `ExtractionResult` objects.
2. Pydantic validation ensures each object is schema-compliant.
3. For each validated `ExtractionResult`:
   a. Check if an identical row already exists (same `entity_id` + `chunk_id` + `entity_type`). If yes, skip (idempotency).
   b. Insert into the Evidence table.
   c. Return the auto-generated `id`.
4. All insertions within a single document's processing are wrapped in a transaction. If any insertion fails, the entire document's evidence is rolled back.

### 5.6 Update Policy

**There is no update policy.** Evidence rows are immutable. If re-extraction produces different results for the same chunk (e.g., due to a prompt change), the old rows remain and new rows are inserted. The `created_at` timestamp distinguishes versions. The traceability matrix uses the latest extraction by default but could be configured to show all versions.

**Correlation rows** are also treated as append-only. A re-correlation run inserts new correlation rows; it does not update old ones. The UI shows the latest correlations by `created_at`.

### 5.7 Provenance Guarantees

Every evidence row has:
- `source_document`: The original filename, establishing which file the fact came from.
- `chunk_id`: The specific chunk within that file, enabling citation back to the exact location.
- `created_at`: When the extraction happened, enabling temporal auditing.

The chain is: **Evidence row → chunk_id → EvidenceChunk (in ChromaDB) → source_locator → original document + page/section/timestamp**.

### 5.8 Querying Patterns

| Query | SQL |
|---|---|
| All requirements | `SELECT * FROM evidence WHERE entity_type = 'REQUIREMENT'` |
| All evidence for requirement R3 | `SELECT * FROM evidence WHERE linked_requirement = 'R3'` |
| All evaluation evidence | `SELECT * FROM evidence WHERE entity_type = 'EVALUATION'` |
| Requirements with no linked evidence | `SELECT e.entity_id FROM evidence e WHERE e.entity_type = 'REQUIREMENT' AND NOT EXISTS (SELECT 1 FROM evidence e2 WHERE e2.linked_requirement = e.entity_id AND e2.entity_type = 'IMPLEMENTATION')` |
| Correlation results for requirement R3 | `SELECT * FROM correlation WHERE requirement_entity_id = 'R3' ORDER BY created_at DESC LIMIT 1` |

---

## 6. ChromaDB Design

### 6.1 Collection Structure

Single collection: `pecs_evidence`

Each document in ChromaDB stores:

| Field | Type | Content |
|---|---|---|
| `id` | string | `chunk_id` (deterministic, SHA-256 based) |
| `embedding` | float[] | 768-dim vector from nomic-embed-text |
| `document` | string | Normalized Markdown text of the chunk |
| `metadata` | dict | Structured metadata (see below) |

### 6.2 Metadata Schema

```json
{
    "source_document": "TSD_v0.3.docx",
    "source_type": "DOCX",
    "source_hash": "a1b2c3...",
    "chunk_index": 5,
    "source_locator": "Section 6.2 - SQLite",
    "author": "Arjun",
    "project": "AnderBahar",
    "char_count": 1247,
    "created_at": "2026-07-18T13:00:00"
}
```

**Why these metadata fields?**
- `source_document` + `source_type`: Enable metadata filtering (e.g., "retrieve only from PDF sources").
- `source_hash`: Support deduplication checks without querying SQLite.
- `chunk_index`: Enables ordered retrieval of sequential chunks from the same document.
- `source_locator`: Enables citation generation without re-parsing the original document.
- `author` + `project`: Enable project-scoped and author-scoped retrieval.
- `char_count`: Useful for debugging embedding quality (very short or very long chunks may embed poorly).
- `created_at`: Temporal filtering.

### 6.3 Embedding Pipeline

```
EvidenceChunk.normalized_text
        │
        ▼
  Ollama /api/embeddings
  model: nomic-embed-text
        │
        ▼
  768-dim float vector
        │
        ▼
  ChromaDB collection.upsert(
      ids=[chunk_id],
      embeddings=[vector],
      documents=[normalized_text],
      metadatas=[metadata_dict]
  )
```

### 6.4 Storage

- ChromaDB persistent client, stored at `data/chromadb/`.
- Uses ChromaDB's default HNSW index for approximate nearest neighbor search.
- Distance metric: **cosine similarity** (nomic-embed-text is trained with cosine similarity in mind).

### 6.5 Retrieval

```python
# Query flow (conceptual)
query_text = "implementation of GNN training pipeline"
query_embedding = ollama.embed(query_text, model="nomic-embed-text")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where={"source_type": {"$in": ["PYTHON", "MARKDOWN"]}},  # metadata filter
    include=["documents", "metadatas", "distances"]
)
```

### 6.6 Design Decisions

- **Why a single collection instead of per-project collections?** The initial version supports a single project. A single collection simplifies the code. Multi-project support (Section 15) would add per-project collections later.
- **Why store the document text in ChromaDB?** Avoids a round-trip to SQLite or the filesystem when we need the text for LLM prompts during retrieval. The text is already small (< 1500 characters per chunk).
- **Why cosine distance?** `nomic-embed-text` is optimized for cosine similarity. Using L2 or inner product would produce inferior results.
- **Why HNSW (ChromaDB default)?** It is the only index type ChromaDB supports. It provides good recall at the scale PECS operates (thousands, not millions, of chunks).

---

## 7. Source-Aware Chunking

This is the core innovation of v0.3 over v0.2. Instead of applying generic Markdown-header chunking to all sources (which fails catastrophically for WhatsApp and email), each source type has a chunker that understands its natural boundaries.

### 7.1 Universal Chunking Parameters

| Parameter | Default | Rationale |
|---|---|---|
| `max_chunk_chars` | 1500 | ~300-400 tokens. Large enough for context, small enough for focused embeddings. |
| `min_chunk_chars` | 50 | Below this, embeddings are too sparse to be meaningful. |
| `overlap_chars` | 150 | ~10% overlap. Prevents information loss at boundaries. |
| `overlap_messages` | 2 | For conversational sources (WhatsApp, Email), overlap in messages, not characters. |

### 7.2 PDF Chunking Strategy

**Natural boundary**: Page.

**Strategy**:
1. Start with page boundaries from the PDF parser.
2. Within each page, if the text exceeds `max_chunk_chars`, split further at paragraph boundaries (double newline).
3. If a paragraph itself exceeds `max_chunk_chars`, split at sentence boundaries.
4. Carry over `overlap_chars` characters from the end of each chunk to the beginning of the next.

**Metadata preserved per chunk**: `page_number`, `source_document`.

**Why page-first?** Page numbers are the universal citation unit for PDFs. Users will want to verify by opening the PDF to the cited page. Breaking across pages makes citations ambiguous.

**Edge case — tables spanning multiple pages**: If a table is detected (heuristic: lines with consistent `|` delimiters or tab-separated columns), keep the entire table as one chunk even if it crosses a page boundary. Attach it to the page where it starts.

**Edge case — headers/footers**: PyMuPDF can identify header/footer regions. Strip these before chunking to avoid noise.

### 7.3 DOCX Chunking Strategy

**Natural boundary**: Section heading.

**Strategy**:
1. Start with the heading hierarchy from the DOCX parser.
2. Each section (heading + body text until the next heading of equal or higher level) becomes one chunk.
3. If a section exceeds `max_chunk_chars`, split at paragraph boundaries within the section.
4. Nested subsections (e.g., H3 under H2) are kept together if their combined size is within `max_chunk_chars`. If not, each subsection becomes its own chunk.
5. Apply `overlap_chars` between adjacent chunks.

**Metadata preserved per chunk**: `heading_text`, `heading_level`, `section_path` (e.g., "6.2 > SQLite > Schema").

**Why heading-first?** DOCX documents are structured by headings. A section about "SQLite Design" is a natural unit of meaning. Splitting mid-section would fragment the topic.

**Edge case — DOCX with no headings**: Treat the entire document as one section. Apply paragraph-boundary splitting within it. Log a warning that the document has no heading structure.

### 7.4 Email Chunking Strategy

**Natural boundary**: Individual email message within a thread.

**Strategy**:
1. The email parser has already split the export into individual messages (sender, timestamp, subject, body).
2. Each email message becomes one chunk.
3. If a single email exceeds `max_chunk_chars`, split at paragraph boundaries within the email body.
4. Apply `overlap_messages = 2`: each chunk includes the last 2 messages from the previous chunk as context. This preserves conversational flow.
5. Prefix each chunk with a header line: `From: sender | Date: timestamp | Subject: subject`. This gives the embedding model and the LLM the context of who said what.

**Metadata preserved per chunk**: `sender`, `timestamp`, `subject`, `thread_id` (if available).

**Why message-first?** An email is a unit of communication from one person. Splitting mid-email loses the attribution. The overlap ensures that references to previous messages ("As I mentioned in my last email...") can be resolved.

**Edge case — forwarded emails**: Detect forwarded content (`---------- Forwarded message ----------`) and treat the forwarded portion as a nested message. Attribute it to the original sender.

**Edge case — attachments**: Email exports typically don't include attachment content, only attachment names. Log the attachment name as metadata, but do not attempt to retrieve or parse it.

### 7.5 WhatsApp Chunking Strategy

**Natural boundary**: Conversation window (group of temporally-close messages).

**Strategy**:
1. The WhatsApp parser has already split the export into individual messages (sender, timestamp, text).
2. Group messages into conversation windows using a time-gap heuristic: if the gap between consecutive messages exceeds **30 minutes**, start a new chunk. This groups messages that are part of the same conversation.
3. Within each window, concatenate all messages, each prefixed with `[timestamp] sender: `.
4. If a window exceeds `max_chunk_chars`, split at message boundaries (not mid-message), keeping at least 3 messages per chunk.
5. Apply `overlap_messages = 2`: each chunk includes the last 2 messages from the previous chunk.

**Metadata preserved per chunk**: `window_start_time`, `window_end_time`, `participants` (list of senders in this window), `message_count`.

**Why time-window-first instead of fixed message count?** WhatsApp conversations are bursty. A discussion about a requirement might happen in 15 rapid messages, then the chat goes silent for 2 hours before a different topic starts. Fixed message counts would arbitrarily split related messages or group unrelated ones. Time-gap detection captures topic boundaries more naturally.

**Edge case — media messages**: `<Media omitted>` messages are logged in metadata but excluded from chunk text (they have no textual content to embed).

**Edge case — system messages**: Messages like "User added User2 to the group" are excluded from chunk text but logged in metadata.

**Edge case — very long single messages**: If a single message exceeds `max_chunk_chars`, it becomes its own chunk. This is rare in WhatsApp but possible for pasted text.

### 7.6 Git Commit Chunking Strategy

**Natural boundary**: Individual commit.

**Strategy**:
1. Each commit (SHA, author, date, message, changed files list) becomes one chunk.
2. Format:
   ```
   Commit: abc123
   Author: Arjun
   Date: 2026-07-15
   
   Implemented GNN training pipeline
   
   Changed files:
   - train_gnn.py (modified)
   - helper.py (modified)
   ```
3. If the commit message is extremely long (e.g., a squash commit with 50 bullet points), truncate the changed-files list to the first 20 files.
4. No overlap between commits — they are independent units.

**Metadata preserved per chunk**: `commit_sha`, `author`, `date`, `changed_files_count`.

**Why commit-first?** A commit is the atomic unit of implementation evidence. It records what changed, who changed it, and why (commit message). Grouping commits would conflate different changes.

**Edge case — merge commits**: Include them but tag with `is_merge: true` in metadata. Merge commits typically have less informative messages.

**Edge case — empty commit messages**: Include the commit with the SHA and changed files, but flag it with a warning. The file list alone may be valuable evidence.

### 7.7 Markdown Chunking Strategy

**Natural boundary**: Section heading.

**Strategy**:
1. Parse ATX headings (`#`, `##`, etc.) to build a section hierarchy.
2. Each section (heading + body until next same-level or higher heading) becomes one chunk.
3. If a section exceeds `max_chunk_chars`, split at paragraph boundaries.
4. Code blocks (fenced with ``` or indented) are kept intact — never split mid-code-block.
5. Apply `overlap_chars` between adjacent chunks.

**Metadata preserved per chunk**: `heading_text`, `heading_level`, `section_path`.

**Why heading-first?** Same rationale as DOCX. Markdown files (especially README.md) are structured by headings. The section is the natural unit.

**Edge case — Markdown with no headings**: Treat as a single block, split at paragraph boundaries. This applies to Markdown files that are more like plain text (e.g., notes).

**Edge case — front-matter**: YAML front-matter (`---` delimited) is extracted as metadata, not included in chunk text.

### 7.8 Python File Chunking Strategy

**Natural boundary**: Top-level definitions (classes, functions, module docstring).

**Strategy**:
1. The Python parser has already extracted the AST: module docstring, classes (with methods), standalone functions, imports.
2. **Module docstring + imports**: Combined into one chunk (the "module overview" chunk).
3. **Each class**: The class definition, class docstring, and all methods become one chunk. If a class exceeds `max_chunk_chars`, each method becomes its own chunk with the class name and docstring prepended as context.
4. **Each standalone function**: One chunk per function (signature + docstring + body).
5. No overlap between definitions — they are independent units. But each chunk is prefixed with a line identifying the file: `# File: train_gnn.py`.

**Metadata preserved per chunk**: `element_type` (class/function/module), `element_name`, `line_start`, `line_end`, `file_path`.

**Why definition-first?** A function or class is a complete unit of implementation logic. Splitting mid-function would produce chunks where neither half makes sense on its own.

**Edge case — very large functions**: If a function exceeds `max_chunk_chars`, split at logical block boundaries (blank lines within the function body). This is a heuristic fallback.

**Edge case — decorators**: Include decorators with the function/class they decorate.

---

## 8. Stage 1 — Extraction

### 8.1 Purpose

Stage 1 reads retrieved evidence chunks and extracts structured factual information. It does NOT perform correlation — it only answers: "What facts are stated in this text?"

### 8.2 Prompt Design

The Stage 1 prompt must be highly constrained. It should:

1. **Define the task**: "You are an evidence extraction system. Your job is to extract factual statements from the provided text."
2. **Define entity types**: "Extract entities of type: REQUIREMENT (a stated project requirement), IMPLEMENTATION (evidence that something was built or coded), EVALUATION (a professor's evaluation, score, or ranking comment)."
3. **Demand JSON output**: "Respond ONLY with a JSON array. Do not include any text before or after the JSON."
4. **Provide the schema**: Include the exact JSON schema expected.
5. **Provide examples**: 2–3 few-shot examples showing input text and expected JSON output.
6. **Prohibit inference**: "Extract ONLY what is explicitly stated in the text. Do not infer, assume, or add information not present in the source."

#### Prompt Template (Conceptual)

```
SYSTEM:
You are a factual evidence extraction system. You extract structured entities
from project documentation. You NEVER infer, assume, or speculate. You extract
ONLY what the text explicitly states.

Entity types:
- REQUIREMENT: A stated project requirement or goal
- IMPLEMENTATION: Evidence that something was built, coded, or delivered
- EVALUATION: A professor's or evaluator's comment, score, ranking, or feedback

Respond ONLY with a JSON array matching this schema:
[
  {
    "entity_type": "REQUIREMENT" | "IMPLEMENTATION" | "EVALUATION",
    "entity_id": "<descriptive identifier>",
    "text": "<exact factual statement from the source>",
    "linked_requirement": "<requirement ID if explicitly referenced, else null>",
    "author": "<author if identifiable, else null>",
    "timestamp": "<date/time if identifiable, else null>"
  }
]

If no entities can be extracted, respond with an empty array: []

EXAMPLES:
[few-shot examples here]

USER:
Source document: {source_document}
Source location: {source_locator}
Text:
---
{chunk_text}
---

Extract all entities from the above text.
```

### 8.3 Expected JSON Output

```json
[
  {
    "entity_type": "REQUIREMENT",
    "entity_id": "R3-gnn-training",
    "text": "The system must implement a GNN-based training pipeline for anomaly detection",
    "linked_requirement": null,
    "author": "Arjun",
    "timestamp": "2026-01-15"
  },
  {
    "entity_type": "IMPLEMENTATION",
    "entity_id": "impl-gnn-trainer",
    "text": "train_gnn.py implements the GNN training loop using PyTorch Geometric with edge-level supervision",
    "linked_requirement": "R3-gnn-training",
    "author": null,
    "timestamp": null
  }
]
```

### 8.4 Validation

1. **JSON parsing**: Attempt `json.loads()`. If it fails, try to extract JSON from markdown code fences: match `` ```json ... ``` `` or `` ``` ... ``` `` patterns.
2. **Schema validation**: Pass each object through the `ExtractionResult` Pydantic model.
3. **Semantic validation**: Additional checks beyond schema:
   - `entity_type` must be one of the three valid types.
   - `text` must not be empty.
   - `entity_id` must not be empty.
   - `text` should be a substring or close paraphrase of the source chunk (heuristic: at least 30% word overlap with the source chunk). If not, flag as "potentially hallucinated" in logs.

### 8.5 Retry Strategy

```
Attempt 1: Standard prompt
    ↓ (if validation fails)
Attempt 2: Standard prompt + "Your previous response had the following errors: {errors}. Please fix and respond again."
    ↓ (if validation fails)
Attempt 3: Simplified prompt + "Respond with ONLY valid JSON. No explanation."
    ↓ (if validation fails)
Log as extraction_failed, skip this chunk, continue pipeline
```

**Why 3 retries?** Empirically, most LLM formatting errors are resolved in 1–2 retries. 3 retries provides a safety margin without burning excessive compute. After 3 failures, the chunk is likely genuinely problematic (e.g., the text is too ambiguous for extraction).

### 8.6 Batch vs. Individual Processing

**Recommendation: Process chunks individually, not in batches.**

Rationale:
- Phi-4-mini has a limited context window. Cramming multiple chunks into one prompt risks truncation and cross-contamination (the model attributes facts from chunk A to chunk B).
- Individual processing makes retry logic cleaner — you retry only the failed chunk, not the entire batch.
- The overhead of multiple Ollama calls is acceptable for a local-only system.

### 8.7 Design Decisions

- **Why temperature ≈ 0?** Extraction should be deterministic. The same text should always produce the same facts. Temperature > 0 introduces variation, which is undesirable for evidence extraction.
- **Why few-shot examples?** They dramatically improve compliance with the output schema. Without examples, Phi-4-mini is more likely to add preamble text, use a different JSON structure, or narrate instead of extracting.
- **Why "extract only" instead of "extract and link"?** Linking (correlation) is Stage 2's job. Stage 1 extracts raw facts. If the text explicitly says "Requirement R3 is implemented by train_gnn.py," Stage 1 captures that in `linked_requirement`. But Stage 1 does not infer links that are not explicitly stated. This separation makes each stage independently testable.

---

## 9. Stage 2 — Correlation & Classification

### 9.1 Purpose

Stage 2 takes the ambiguous evidence pairs that the deterministic rule engine could not resolve and classifies each pair into one of the 7 status labels. It also produces a brief reasoning justification for each classification.

### 9.2 Candidate Generation

Before Stage 2 runs, the system must generate candidate pairs to classify. The process:

1. **Query all requirements** from the Evidence table: `SELECT * FROM evidence WHERE entity_type = 'REQUIREMENT'`.
2. **For each requirement**, gather candidate evidence:
   a. **Direct links**: Query `SELECT * FROM evidence WHERE linked_requirement = ?` — evidence that Stage 1 already linked.
   b. **Retrieval-based candidates**: Use the retrieval pipeline (Section 10) with the requirement text as the query. Retrieve top-K (default K=10) chunks.
   c. **Entity-ID match**: Search for evidence rows whose `entity_id` contains tokens from the requirement's `entity_id` (fuzzy matching).
3. **Merge** all candidates, deduplicate by `chunk_id`.
4. **Pass through deterministic rules** (Section 11). Rules resolve unambiguous pairs.
5. **Remaining ambiguous pairs** → Stage 2 LLM.

### 9.3 Prompt Design for Ambiguous Cases

The Stage 2 prompt must be even more constrained than Stage 1. It is a classification prompt, not a reasoning prompt.

#### Prompt Template (Conceptual)

```
SYSTEM:
You are an evidence correlation classifier. You classify the relationship 
between a project requirement and a piece of evidence into exactly one of 
these categories:

- IMPLEMENTED_AND_VALIDATED: Evidence shows the requirement is implemented 
  AND a positive evaluation exists
- IMPLEMENTED_BUT_NEGATIVELY_EVALUATED: Evidence shows implementation exists 
  BUT evaluation is negative
- IMPLEMENTED_WITHOUT_EVALUATION: Evidence shows implementation but no 
  evaluation was found
- PARTIALLY_IMPLEMENTED: Evidence shows some but not all aspects of the 
  requirement are implemented
- CLAIMED_BUT_NO_EVIDENCE: The text claims implementation but provides no 
  concrete evidence (no code, no test results, no demonstration)
- EVALUATION_WITHOUT_REQUIREMENT: Evaluation evidence exists but does not 
  reference any known requirement
- REQUIREMENT_NOT_IMPLEMENTED: No evidence of implementation was found

Respond ONLY with JSON matching this schema:
{
  "status": "<one of the 7 labels above>",
  "reasoning": "<1-2 sentence justification citing specific text from the evidence>"
}

USER:
REQUIREMENT:
  ID: {requirement_entity_id}
  Text: {requirement_text}

EVIDENCE:
  Type: {evidence_entity_type}
  ID: {evidence_entity_id}
  Text: {evidence_text}
  Source: {evidence_source_document}

EVALUATION (if available):
  Text: {evaluation_text}
  Source: {evaluation_source_document}

Classify the relationship between the requirement and the evidence.
```

### 9.4 Handling Ambiguous Cases

Ambiguity arises when:

1. **Partial keyword overlap**: The requirement mentions "GNN" and the evidence mentions "graph neural network" but not "GNN" explicitly. Deterministic rules miss this. The LLM can recognize the semantic equivalence.
2. **Implicit references**: The evidence says "The training pipeline was completed" without naming a specific requirement. The LLM must decide which requirement this refers to (if any).
3. **Conflicting evidence**: One document says "Feature X is implemented" and another says "Feature X was not completed." The LLM must weigh the evidence.
4. **Evaluation without explicit linkage**: A professor comment says "The data preprocessing is excellent" without naming a requirement. The LLM must match this to the correct requirement.

For cases 3 (conflicting evidence), the classification should be `PARTIALLY_IMPLEMENTED` or `CLAIMED_BUT_NO_EVIDENCE`, depending on which evidence is more concrete. The `reasoning` field should cite both conflicting pieces.

### 9.5 Classification Taxonomy Details

| Status | Trigger Conditions |
|---|---|
| `IMPLEMENTED_AND_VALIDATED` | Implementation evidence exists AND evaluation evidence is positive |
| `IMPLEMENTED_BUT_NEGATIVELY_EVALUATED` | Implementation evidence exists AND evaluation evidence is negative (criticism, low score) |
| `IMPLEMENTED_WITHOUT_EVALUATION` | Implementation evidence exists AND no evaluation evidence was found |
| `PARTIALLY_IMPLEMENTED` | Some but not all aspects of the requirement have evidence, OR conflicting evidence |
| `CLAIMED_BUT_NO_EVIDENCE` | Text claims implementation (e.g., "We implemented X") but no code, tests, or concrete output is referenced |
| `EVALUATION_WITHOUT_REQUIREMENT` | Evaluation/professor comment exists but cannot be linked to any known requirement |
| `REQUIREMENT_NOT_IMPLEMENTED` | Requirement exists but zero implementation evidence was found |

### 9.6 Design Decisions

- **Why classify per-pair instead of holistically?** Per-pair classification is simpler, more testable, and produces finer-grained results. Holistic classification ("classify all requirements at once") would overwhelm the context window and make errors harder to trace.
- **Why require the LLM to cite text in its reasoning?** This makes the output auditable. If the reasoning doesn't reference specific text from the evidence, the classification is suspect.
- **Why not let the LLM assign confidence?** Confidence must be computed by a defined function (see Section 12), not generated by the LLM. LLM-generated confidence is unreliable and often miscalibrated.

---

## 10. Retrieval Pipeline

### 10.1 Pipeline Overview

The retrieval pipeline is a multi-strategy system that combines four methods to maximize recall:

```
Query (requirement text or user query)
        │
        ▼
┌─────────────────────────────────────┐
│   Step 1: METADATA FILTERING        │
│   Narrow the search space           │
│   before any retrieval runs         │
└────────────────┬────────────────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
     ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────────┐
│ Step 2  │ │ Step 3  │ │ Step 4      │
│ VECTOR  │ │ BM25    │ │ REQ-ID      │
│ Top-K   │ │ Keyword │ │ MATCHING    │
└────┬────┘ └────┬────┘ └──────┬──────┘
     │           │              │
     └───────────┼──────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Step 5: CANDIDATE MERGING         │
│   Deduplicate, combine scores,      │
│   rank by combined relevance        │
└─────────────────────────────────────┘
```

### 10.2 Step 1 — Metadata Filtering

**What it does**: Narrows the search space before any vector or keyword retrieval runs.

**Filters available**:
- `source_type`: Filter by source type (e.g., only Python files, only PDFs).
- `source_document`: Filter by specific document.
- `project`: Filter by project name (single-project for now, future-proofing for multi-project).
- `date_range`: Filter by `created_at` or `timestamp` metadata.

**Implementation**: Applied as ChromaDB `where` filters. For BM25, applied as a post-filter.

**Why pre-filter instead of post-filter?** Reducing the candidate set before retrieval is dramatically faster than retrieving everything and filtering afterward. It also improves precision by excluding irrelevant source types (e.g., when looking for implementation evidence, you probably want Python files and commit logs, not WhatsApp messages).

### 10.3 Step 2 — Vector Top-K Retrieval

**What it does**: Embeds the query text using nomic-embed-text and retrieves the top-K (default K=10) most similar chunks from ChromaDB.

**Process**:
1. Embed the query text: `query_vector = ollama.embed(query, model="nomic-embed-text")`
2. Query ChromaDB: `collection.query(query_embeddings=[query_vector], n_results=K, where=metadata_filters)`
3. Convert results to `RetrievedEvidence` objects with `vector_score` populated.

**Why K=10?** A balance between recall and noise. K=5 risks missing relevant chunks; K=20 includes too much noise for the LLM to process. K=10 is the TSD-specified default. It can be tuned based on evaluation results.

### 10.4 Step 3 — BM25 Keyword Matching

**What it does**: Performs traditional keyword-based retrieval using BM25 scoring. Catches chunks that are lexically relevant but semantically dissimilar (a common failure mode for embedding-only retrieval).

**Implementation**:
- Use `rank_bm25` Python library.
- Build a BM25 index over all `EvidenceChunk.normalized_text` values at ingestion time.
- At query time, tokenize the query and retrieve the top-K matches.
- If metadata filters are active, post-filter the BM25 results.

**Why BM25 in addition to vector search?** Vector search excels at semantic similarity but can miss exact keyword matches. If the requirement says "GNN" and the code has a comment "# GNN training loop," vector search might rank a semantically similar but lexically different chunk higher. BM25 catches exact keyword matches that vector search misses.

**Trade-off: BM25 adds complexity.** If the initial evaluation (Section 14) shows that vector-only retrieval achieves > 90% recall, BM25 can be deprioritized. But it should be built and evaluated.

### 10.5 Step 4 — Requirement-ID Matching

**What it does**: Searches for chunks that explicitly mention a requirement identifier (e.g., "R3", "Requirement 3", "req-3").

**Implementation**:
- Extract all requirement identifiers from the Evidence table.
- For each identifier, search ChromaDB metadata and document text for exact string matches (case-insensitive).
- Also search for common variants: "R3", "Requirement 3", "Req 3", "requirement_3", "req-3".

**Why a separate step?** If an implementation document explicitly says "This implements Requirement R3," that is the strongest possible evidence of correlation. Vector similarity might rank this chunk highly, but it might not. ID matching guarantees it is included in the candidate set.

### 10.6 Step 5 — Candidate Merging

**What it does**: Combines results from all retrieval methods, deduplicates by `chunk_id`, and computes a combined relevance score.

**Merging strategy**:
1. Collect all `RetrievedEvidence` objects from Steps 2–4.
2. Group by `chunk_id`.
3. For each unique chunk, compute `combined_score`:
   - If only vector: `combined_score = vector_score`
   - If only BM25: `combined_score = normalized_bm25_score` (normalize BM25 to 0–1 range using min-max normalization across the current result set)
   - If both vector and BM25: `combined_score = 0.7 * vector_score + 0.3 * normalized_bm25_score`
   - If ID match: `combined_score = max(combined_score, 0.95)` (ID match is a very strong signal, boosted to near-maximum)
4. Sort by `combined_score` descending.
5. Return top-K merged results (default K=10, but expanded to K=15 if ID matches are present to avoid displacing them).
6. Record `retrieval_method` for each result: `"vector"`, `"bm25"`, `"id_match"`, or combinations.

**Why 0.7/0.3 weighting?** Semantic similarity (vector) is generally more informative than keyword overlap (BM25) for this domain. The 0.7/0.3 split reflects this prior. These weights are configurable and should be tuned based on evaluation results.

---

## 11. Deterministic Rule Engine

### 11.1 Purpose

The rule engine runs **before** Stage 2 (the LLM). Its job is to resolve correlations that are unambiguous — cases where the evidence is clear enough that no LLM judgment is needed. This reduces LLM compute, increases determinism, and produces more trustworthy results.

### 11.2 Rules

#### Rule 1: `exact_requirement_id_match`

**Trigger**: An implementation or evaluation evidence row has `linked_requirement` set, and that value exactly matches a requirement's `entity_id`.

**Action**: Create a correlation between the requirement and the evidence. Status depends on entity type:
- If implementation evidence: `IMPLEMENTED_WITHOUT_EVALUATION` (unless evaluation evidence is also linked — see Rule 3).
- If evaluation evidence: `EVALUATION_WITHOUT_REQUIREMENT` should NOT apply here (since the requirement exists). Instead, note the evaluation for the requirement.

**Confidence**: 0.95 (very high — explicit ID match is strong evidence).

#### Rule 2: `requirement_with_no_evidence`

**Trigger**: A requirement exists in the Evidence table, but no evidence row (IMPLEMENTATION or EVALUATION) has `linked_requirement` matching this requirement's `entity_id`, AND no retrieval-based candidate was found above a minimum vector score threshold (default: 0.3).

**Action**: Create a correlation with status `REQUIREMENT_NOT_IMPLEMENTED`.

**Confidence**: 0.85 (high — absence of evidence is strong signal when both explicit links and semantic search found nothing).

#### Rule 3: `implementation_plus_positive_evaluation`

**Trigger**: A requirement has at least one linked implementation evidence AND at least one linked evaluation evidence where the evaluation text contains positive sentiment markers (configurable list: "excellent", "well-implemented", "good", "satisfactory", "meets expectations", numerical score ≥ threshold).

**Action**: Create a correlation with status `IMPLEMENTED_AND_VALIDATED`.

**Confidence**: 0.90 (high — both implementation and positive evaluation are present).

#### Rule 4: `implementation_plus_negative_evaluation`

**Trigger**: Same as Rule 3, but evaluation text contains negative sentiment markers ("poor", "not implemented", "missing", "incomplete", "unsatisfactory", numerical score below threshold).

**Action**: Create a correlation with status `IMPLEMENTED_BUT_NEGATIVELY_EVALUATED`.

**Confidence**: 0.90.

#### Rule 5: `claim_without_evidence`

**Trigger**: An evidence row of type IMPLEMENTATION contains claim-language markers ("we implemented", "we built", "was completed") but:
- No Python source file or Git commit is linked or found via retrieval for the same requirement.
- The claim is from a discussion/chat source (WhatsApp, Email) rather than a technical document.

**Action**: Flag for LLM review with a bias toward `CLAIMED_BUT_NO_EVIDENCE`. The rule does not resolve this; it annotates the pair and passes it to Stage 2 with the annotation.

**Confidence**: N/A (deferred to Stage 2).

#### Rule 6: `orphan_evaluation`

**Trigger**: An evaluation evidence row has `linked_requirement = null` AND no requirement in the Evidence table can be matched by entity_id or keyword overlap.

**Action**: Create a correlation with status `EVALUATION_WITHOUT_REQUIREMENT`.

**Confidence**: 0.80 (moderately high — the evaluation genuinely may not reference any extracted requirement).

### 11.3 Rule Execution Order

```
1. exact_requirement_id_match     (resolves the easiest cases)
2. requirement_with_no_evidence   (resolves orphan requirements)
3. implementation_plus_positive_evaluation   (resolves complete positive correlations)
4. implementation_plus_negative_evaluation   (resolves complete negative correlations)
5. orphan_evaluation              (resolves orphan evaluations)
6. claim_without_evidence         (annotates but does NOT resolve — passes to LLM)
```

Rules are applied in order. Once a requirement-evidence pair is resolved by a rule, it is removed from the candidate set. Later rules do not re-process resolved pairs.

### 11.4 Design Decisions

- **Why sentiment markers instead of a sentiment classifier?** A keyword-based approach is simpler, faster, and more transparent than a sentiment classifier. For a prototype, a curated list of positive/negative markers is sufficient. A sentiment classifier can be added later if evaluation accuracy is poor.
- **Why not resolve `claim_without_evidence` deterministically?** Claims are inherently ambiguous. "We implemented X" might be true (the code exists but wasn't linked) or false (it was discussed but never built). Only the LLM, with access to the full evidence context, can make this judgment.

---

## 12. Confidence Scoring

### 12.1 Design Principle

> "Confidence values must be derived from a defined function, not produced as free-floating LLM output." — TSD v0.3, Section 11

The LLM never generates confidence scores. Confidence is computed deterministically from observable signals.

### 12.2 Confidence Components

| Component | Symbol | Range | Source |
|---|---|---|---|
| Retrieval score | `R` | 0.0–1.0 | `combined_score` from the retrieval pipeline |
| Resolution method | `M` | 0.0–1.0 | 1.0 for deterministic rules, 0.7 for LLM classification |
| Evidence count | `E` | 0.0–1.0 | Number of supporting evidence chunks, normalized |
| Source diversity | `D` | 0.0–1.0 | Number of distinct source types that provide evidence |

### 12.3 Confidence Formula

```
confidence = w_R * R + w_M * M + w_E * E_norm + w_D * D_norm
```

Where:
- `w_R = 0.35` (retrieval quality is the strongest signal)
- `w_M = 0.30` (deterministic resolution is more trustworthy than LLM)
- `w_E = 0.20` (more evidence chunks = higher confidence)
- `w_D = 0.15` (evidence from multiple source types is stronger)

And:
- `E_norm = min(evidence_count / 5, 1.0)` — 5 or more supporting chunks → full score. This caps the benefit of additional evidence to prevent inflation.
- `D_norm = min(distinct_source_types / 3, 1.0)` — evidence from 3 or more source types → full score.
- `R` is the `combined_score` of the best-matching chunk.
- `M` is 1.0 if the correlation was resolved by a deterministic rule, 0.7 if resolved by the LLM.

### 12.4 Example Calculations

**Case 1: Strong correlation**
- Requirement R3 has 4 implementation evidence chunks (from Python files and Git commits), 1 evaluation evidence (from a PDF).
- Best retrieval score: 0.85.
- Resolved by deterministic rule `exact_requirement_id_match`.
- Source types: PYTHON, GIT, PDF → 3 distinct types.
- `confidence = 0.35 * 0.85 + 0.30 * 1.0 + 0.20 * min(5/5, 1.0) + 0.15 * min(3/3, 1.0)`
- `confidence = 0.2975 + 0.30 + 0.20 + 0.15 = 0.9475`

**Case 2: Weak correlation**
- Requirement R7 has 1 implementation evidence chunk (from a WhatsApp message), no evaluation.
- Best retrieval score: 0.42.
- Resolved by LLM Stage 2 (ambiguous).
- Source types: WHATSAPP → 1 distinct type.
- `confidence = 0.35 * 0.42 + 0.30 * 0.7 + 0.20 * min(1/5, 1.0) + 0.15 * min(1/3, 1.0)`
- `confidence = 0.147 + 0.21 + 0.04 + 0.05 = 0.447`

### 12.5 Design Decisions

- **Why not use the LLM's own confidence?** LLMs are notoriously poor at self-calibration. A model might say "I am 90% confident" when its accuracy is 60%. Formula-based confidence is reproducible and auditable.
- **Why include source diversity?** Cross-source corroboration is a hallmark of reliable evidence. If both the code (Python), the commit log (Git), and the evaluation report (PDF) all support a correlation, that's much stronger than a single WhatsApp message.
- **Why are the weights configurable?** These weights are initial estimates. After evaluation on the labelled test set, they should be tuned to maximize calibration (predicted confidence should match actual accuracy).

---

## 13. Logging

### 13.1 Logging Infrastructure

- **Library**: Python's built-in `logging` module. No third-party logging libraries.
- **Format**: Structured JSON logging for machine readability. Each log entry contains:
  ```json
  {
    "timestamp": "2026-07-18T13:00:00",
    "level": "INFO",
    "module": "ingestion",
    "message": "File ingested successfully",
    "context": {"filename": "TSD_v0.3.docx", "hash": "a1b2c3...", "chunks": 15}
  }
  ```
- **Output**: Dual output — file (`data/logs/pecs.log`) + Streamlit debug page.
- **Log rotation**: Rotate when file exceeds 10 MB. Keep last 5 files.
- **Log levels**: DEBUG, INFO, WARNING, ERROR. Default level: INFO.

### 13.2 What Each Module Should Log

#### Ingestion Module
| Event | Level | Context |
|---|---|---|
| File uploaded | INFO | filename, file_size, mime_type |
| Duplicate detected | WARNING | filename, hash, original_ingest_date |
| File type unsupported | WARNING | filename, detected_mime_type |
| File too large | WARNING | filename, file_size, max_allowed |
| Ingestion successful | INFO | filename, hash, source_type, parse_time_ms |
| Ingestion failed | ERROR | filename, error_message, traceback |

#### Parsing Module
| Event | Level | Context |
|---|---|---|
| Parse started | DEBUG | filename, parser_type |
| Parse completed | INFO | filename, parser_type, text_length, structural_elements_count, parse_time_ms |
| Image-only PDF page skipped | WARNING | filename, page_number |
| Parse failed | ERROR | filename, parser_type, error_message |
| WhatsApp format unrecognized | WARNING | filename, line_number, raw_line |

#### Chunking Module
| Event | Level | Context |
|---|---|---|
| Chunking completed | INFO | filename, chunk_count, avg_chunk_size, min_chunk_size, max_chunk_size |
| Chunk below minimum size (merged) | DEBUG | filename, chunk_index, original_size |
| Chunk above maximum size (split) | DEBUG | filename, chunk_index, original_size, split_count |
| Overlap applied | DEBUG | filename, chunk_index, overlap_chars |

#### Embedding Module
| Event | Level | Context |
|---|---|---|
| Embedding batch started | DEBUG | batch_size, batch_number |
| Embedding batch completed | INFO | batch_size, batch_number, latency_ms |
| Ollama connection failed | ERROR | endpoint, error_message |
| Embedding retry | WARNING | batch_number, attempt, error_message |
| Model not found | ERROR | model_name |

#### ChromaDB Module
| Event | Level | Context |
|---|---|---|
| Upsert completed | INFO | chunk_count, collection_name |
| Query executed | DEBUG | query_text_preview (first 50 chars), n_results, filter_applied, latency_ms |
| Collection created | INFO | collection_name |

#### Retrieval Pipeline
| Event | Level | Context |
|---|---|---|
| Retrieval started | INFO | query_text_preview, filters_applied |
| Vector results | DEBUG | result_count, top_score, bottom_score |
| BM25 results | DEBUG | result_count, top_score |
| ID match results | DEBUG | result_count, matched_ids |
| Merged results | INFO | total_candidates, retrieval_latency_ms |

#### Stage 1 Extraction
| Event | Level | Context |
|---|---|---|
| Extraction started | INFO | chunk_id, source_document |
| LLM call | DEBUG | prompt_length_chars, model_name |
| LLM response received | DEBUG | response_length_chars, latency_ms |
| Validation passed | INFO | chunk_id, entity_count |
| Validation failed | WARNING | chunk_id, validation_errors, attempt_number |
| Extraction failed (max retries) | ERROR | chunk_id, all_errors |
| Hallucination suspected | WARNING | chunk_id, entity_id, word_overlap_ratio |

#### Stage 2 Correlation
| Event | Level | Context |
|---|---|---|
| Correlation started | INFO | requirement_id, candidate_count |
| Deterministic rule fired | INFO | requirement_id, evidence_id, rule_name, status |
| LLM classification | INFO | requirement_id, evidence_id, status, latency_ms |
| Classification validation failed | WARNING | requirement_id, raw_response |
| Ambiguous pair count | INFO | total_pairs, deterministic_resolved, llm_resolved |

#### Confidence Scoring
| Event | Level | Context |
|---|---|---|
| Confidence computed | DEBUG | correlation_id, R, M, E_norm, D_norm, final_confidence |
| Low confidence alert | WARNING | correlation_id, confidence, threshold |

#### Traceability Matrix
| Event | Level | Context |
|---|---|---|
| Matrix generated | INFO | requirement_count, correlation_count, avg_confidence, generation_time_ms |
| Orphan requirement | WARNING | requirement_id (no implementation evidence) |
| Orphan evaluation | WARNING | evaluation_entity_id (no linked requirement) |

---

## 14. Testing

### 14.1 Testing Philosophy

PECS is a pipeline system. Each stage transforms data for the next stage. Testing must validate both **individual stage correctness** (unit tests) and **end-to-end pipeline behavior** (integration tests). Additionally, because the system involves an LLM, testing must include **evaluation** against a labelled dataset.

### 14.2 Unit Testing

Framework: `pytest` (TSD-specified).

#### Parser Unit Tests (`tests/unit/test_parsers.py`)
- **What to test**: Each parser adapter, given known input, produces the expected `ParsedDocument`.
- **Fixtures**: Create small, representative test files for each source type (a 2-page PDF, a 3-message WhatsApp export, a Python file with 2 functions, etc.). Store in `tests/fixtures/`.
- **Assertions**:
  - PDF parser extracts the correct number of pages.
  - PDF parser preserves page numbers.
  - WhatsApp parser correctly splits messages by timestamp.
  - WhatsApp parser handles multi-line messages.
  - Email parser correctly identifies sender, subject, date.
  - Python parser correctly identifies functions and classes.
  - Git parser correctly extracts commit SHA, author, date.
  - DOCX parser correctly identifies heading hierarchy.

#### Chunker Unit Tests (`tests/unit/test_chunkers.py`)
- **What to test**: Each chunker, given a known `ParsedDocument`, produces chunks with the expected boundaries and sizes.
- **Assertions**:
  - Chunk count is correct.
  - No chunk exceeds `max_chunk_chars`.
  - No chunk is below `min_chunk_chars` (except the last chunk).
  - Overlap is correctly applied.
  - Source locators are correct (page numbers, timestamps, line ranges).

#### Normalizer Unit Tests (`tests/unit/test_normalizer.py`)
- **What to test**: MarkItDown normalization produces clean Markdown.
- **Assertions**:
  - Excessive whitespace is stripped.
  - Unicode is NFKC-normalized.
  - Chunk IDs are deterministic (same input → same output).

#### Validator Unit Tests (`tests/unit/test_validator.py`)
- **What to test**: Pydantic validation accepts valid JSON and rejects invalid JSON.
- **Assertions**:
  - Valid JSON with all required fields → passes.
  - Missing required field → `ValidationError`.
  - Extra fields → silently ignored.
  - Wrong type (e.g., entity_type = 123 instead of string) → `ValidationError`.
  - JSON wrapped in code fences → extracted and validated.

#### Rule Engine Unit Tests (`tests/unit/test_rules.py`)
- **What to test**: Each deterministic rule, given known evidence rows, produces the expected correlation.
- **Assertions**:
  - `exact_requirement_id_match`: Given evidence with `linked_requirement = "R3"` and a requirement with `entity_id = "R3"`, produces a correlation.
  - `requirement_with_no_evidence`: Given a requirement with no linked evidence and no retrieval candidates, produces `REQUIREMENT_NOT_IMPLEMENTED`.
  - Rules are applied in order.
  - Resolved pairs are excluded from later rules.

#### Confidence Scorer Unit Tests (`tests/unit/test_confidence.py`)
- **What to test**: The confidence formula produces expected values for known inputs.
- **Assertions**: Reproduce the example calculations from Section 12.4 exactly.

### 14.3 Integration Testing

#### Ingestion Pipeline (`tests/integration/test_ingestion_pipeline.py`)
- **What to test**: File upload → parse → chunk → normalize → embed → ChromaDB upsert, end-to-end.
- **Method**: Ingest a small test file. Verify that:
  - ChromaDB contains the expected number of documents.
  - Each document has the expected metadata fields.
  - Embeddings are 768-dimensional.
  - Chunk IDs are deterministic (re-ingesting produces the same IDs).

#### Retrieval Pipeline (`tests/integration/test_retrieval_pipeline.py`)
- **What to test**: Given ingested documents, retrieval returns relevant chunks.
- **Method**: Ingest a small test corpus. Query with known queries. Verify that expected chunks appear in the top-K results.

#### Extraction Pipeline (`tests/integration/test_extraction_pipeline.py`)
- **What to test**: Stage 1 extraction produces valid, sensible ExtractionResults.
- **Method**: Provide known text chunks to Stage 1. Verify that the extracted entities match expectations. This test requires Ollama running with Phi-4-mini.

#### Correlation Pipeline (`tests/integration/test_correlation_pipeline.py`)
- **What to test**: End-to-end from evidence rows to traceability matrix.
- **Method**: Populate SQLite with known evidence rows. Run the correlation pipeline. Verify that the matrix contains the expected statuses.

### 14.4 Retrieval Evaluation (`tests/evaluation/eval_retrieval.py`)

**Purpose**: Measure retrieval recall and precision to determine if a reranker is needed.

**Method**:
1. Create a labelled evaluation set: 10–20 queries with known relevant chunks (ground truth).
2. For each query, run the retrieval pipeline and record the retrieved chunks.
3. Compute:
   - **Recall@K**: What fraction of ground-truth chunks appear in the top-K results?
   - **Precision@K**: What fraction of top-K results are ground-truth relevant chunks?
   - **MRR (Mean Reciprocal Rank)**: Where does the first relevant chunk appear?

**Decision threshold**: If Recall@10 < 0.80, consider adding a reranker.

### 14.5 Extraction Evaluation (`tests/evaluation/eval_extraction.py`)

**Purpose**: Measure Stage 1 extraction accuracy.

**Method**:
1. Create a labelled evaluation set: 10–20 chunks with known expected extractions (ground truth entities).
2. For each chunk, run Stage 1 extraction and compare output to ground truth.
3. Compute:
   - **Entity precision**: What fraction of extracted entities match ground truth?
   - **Entity recall**: What fraction of ground truth entities were extracted?
   - **Schema compliance**: What fraction of responses are valid JSON on the first attempt?

### 14.6 Correlation Evaluation (`tests/evaluation/eval_correlation.py`)

**Purpose**: Measure Stage 2 classification accuracy against the 7-label taxonomy.

**Method**:
1. Create a labelled evaluation set: 10–20 requirement-evidence pairs with known correct status labels (ground truth).
2. For each pair, run the full correlation pipeline (deterministic rules + Stage 2) and compare to ground truth.
3. Compute:
   - **Classification accuracy**: What fraction of pairs are correctly labelled?
   - **Confusion matrix**: Which labels are confused with which?
   - **Deterministic resolution rate**: What fraction of pairs were resolved without the LLM?

### 14.7 Golden Triples Dataset

The file `tests/evaluation/labelled_data/golden_triples.json` should contain hand-verified triples:

```json
[
  {
    "requirement": {
      "entity_id": "R3-gnn-training",
      "text": "Implement GNN training pipeline"
    },
    "implementation_evidence": [
      {
        "entity_id": "impl-train-gnn",
        "text": "train_gnn.py implements the GNN training loop",
        "source_document": "train_gnn.py"
      }
    ],
    "evaluation_evidence": [
      {
        "entity_id": "eval-gnn-positive",
        "text": "GNN implementation is well-structured",
        "source_document": "disc_evi3.pdf"
      }
    ],
    "expected_status": "IMPLEMENTED_AND_VALIDATED",
    "expected_confidence_range": [0.8, 1.0]
  }
]
```

This dataset is small (10–20 triples) and hand-curated from the actual evidence corpus. It is the ground truth for all evaluation metrics.

---

## 15. Future Extensibility

### 15.1 OCR Support

**Current limitation**: Image-only PDF pages are skipped.

**Extension path**:
1. Add an OCR adapter using `pytesseract` or `easyocr`.
2. In the PDF parser, when a page has no extractable text and has images, route it to the OCR adapter.
3. OCR output is treated as raw text and fed into the existing chunking → normalization pipeline.
4. Metadata: add `ocr_processed: true` flag to distinguish OCR-derived text from native text. OCR text is inherently less reliable and may warrant a lower confidence weight.

**Why deferred?** OCR adds complexity (image processing dependencies, accuracy issues) and is not needed for the initial evidence corpus (which is all text-based).

### 15.2 Reranker Integration

**Current state**: Retrieval uses vector + BM25 + ID matching. No reranker.

**Extension path**:
1. After Step 5 (candidate merging) in the retrieval pipeline, add a Step 5.5: Reranker.
2. Use a cross-encoder (e.g., BGE-M3, Jina Reranker, or Cohere Reranker) to re-score the top-K candidates.
3. The reranker takes `(query, chunk_text)` pairs and produces a relevance score.
4. Replace or supplement the `combined_score` with the reranker score.

**When to add**: Only if Recall@10 < 0.80 in the retrieval evaluation (Section 14.4).

**VRAM consideration**: A reranker model needs to be loaded alongside Phi-4-mini and nomic-embed-text. On an 8 GB RTX 3060, this may require swapping models (unload the reranker before loading Phi-4-mini). Alternatively, use a small reranker (< 500 MB) that can coexist in VRAM.

### 15.3 Multi-Project Support

**Current state**: Single project.

**Extension path**:
1. Add a `project_id` field to every `EvidenceChunk`, Evidence row, and Correlation row.
2. Create a per-project ChromaDB collection (e.g., `pecs_evidence_project1`, `pecs_evidence_project2`).
3. Add a project selector in the Streamlit UI.
4. All retrieval and correlation operations are scoped by `project_id`.

**Schema migration**: Add `project_id TEXT NOT NULL DEFAULT 'default'` to the Evidence and Correlation tables. Existing rows get `project_id = 'default'`.

### 15.4 Additional Evidence Sources

**Current supported**: PDF, DOCX, Email, WhatsApp, Git, Markdown, Python.

**Extension path for new source types**:
1. Create a new parser adapter in `pecs/parsing/` (e.g., `slack_parser.py`).
2. Create a new chunker in `pecs/chunking/` (e.g., `slack_chunker.py`).
3. Add the new `SourceType` enum value.
4. Register the new adapter in the ingestion module's dispatch table.

The rest of the pipeline (normalization, embedding, retrieval, extraction, correlation) requires no changes — they work with `EvidenceChunk` objects regardless of source type. This is the key benefit of the universal internal object design.

**Likely future sources**:
- **Slack/Discord**: Similar to WhatsApp. Message-based chunking with time-gap windows.
- **Jupyter Notebooks**: Extract Markdown cells and code cells separately. Treat Markdown cells like Markdown files and code cells like Python files.
- **Jira/Trello**: Ticket-based chunking (one ticket = one chunk). Extract title, description, status, assignee.
- **Spreadsheets**: Row-based or sheet-based chunking, depending on structure.

### 15.5 Incremental Knowledge Base Updates

**Current state**: Full re-ingestion on every upload.

**Extension path**:
1. Track ingested file hashes in a metadata table.
2. On upload, check if the hash already exists. If yes, skip re-ingestion.
3. If a file is modified (same name, different hash), mark old chunks as superseded (add `superseded_by: new_chunk_id` metadata) and ingest the new version.
4. Periodically garbage-collect superseded chunks from ChromaDB.

### 15.6 Automatic Evidence Synchronization

**Current state**: Manual file upload only.

**Extension path**:
- **Git sync**: Use `gitpython` to pull latest commits periodically or on-demand.
- **Gmail sync**: Use Gmail API with OAuth to fetch new emails matching a project label.
- **File system watcher**: Use `watchdog` to monitor a directory for new/modified files.

**Why deferred?** API integrations add authentication complexity, error handling, and security considerations. The manual upload workflow is sufficient for the prototype.

---

## Appendix A: Dependency List

| Package | Purpose | Version Constraint |
|---|---|---|
| `streamlit` | UI framework | ≥ 1.30 |
| `chromadb` | Vector store | ≥ 0.4 |
| `ollama` | LLM and embedding API client | latest |
| `pydantic` | Data validation | ≥ 2.0 |
| `pymupdf` (fitz) | PDF parsing | ≥ 1.23 |
| `python-docx` | DOCX parsing | ≥ 1.0 |
| `python-magic` | MIME detection | ≥ 0.4 |
| `markitdown` | Markdown normalization | latest |
| `rank-bm25` | BM25 keyword retrieval | ≥ 0.2 |
| `pytest` | Testing | ≥ 7.0 |

## Appendix B: Configuration Parameters

All parameters are defined in `pecs/config.py` and overridable via `.env`:

| Parameter | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `LLM_MODEL` | `phi4-mini` | Model for Stage 1 and Stage 2 |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings |
| `LLM_TEMPERATURE` | `0.05` | Near-zero for deterministic output |
| `MAX_CHUNK_CHARS` | `1500` | Maximum chunk size |
| `MIN_CHUNK_CHARS` | `50` | Minimum chunk size |
| `OVERLAP_CHARS` | `150` | Character overlap between chunks |
| `OVERLAP_MESSAGES` | `2` | Message overlap for conversational sources |
| `RETRIEVAL_TOP_K` | `10` | Top-K results from vector retrieval |
| `BM25_TOP_K` | `10` | Top-K results from BM25 |
| `MAX_EXTRACTION_RETRIES` | `3` | Max Stage 1 retry attempts |
| `VECTOR_WEIGHT` | `0.7` | Weight for vector score in combined scoring |
| `BM25_WEIGHT` | `0.3` | Weight for BM25 score in combined scoring |
| `CONFIDENCE_W_R` | `0.35` | Confidence weight: retrieval score |
| `CONFIDENCE_W_M` | `0.30` | Confidence weight: resolution method |
| `CONFIDENCE_W_E` | `0.20` | Confidence weight: evidence count |
| `CONFIDENCE_W_D` | `0.15` | Confidence weight: source diversity |
| `SQLITE_DB_PATH` | `data/sqlite/pecs.db` | SQLite database path |
| `CHROMADB_PATH` | `data/chromadb/` | ChromaDB storage path |
| `LOG_FILE` | `data/logs/pecs.log` | Log file path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `WHATSAPP_TIME_GAP_MINUTES` | `30` | Time gap for WhatsApp conversation windowing |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size |

---

## Appendix C: Recommended Build Order

Phase 1 — Foundation (no LLM needed):
1. `pecs/models/` — Define all universal internal objects
2. `pecs/config.py` — Central configuration
3. `pecs/logging_config.py` — Logging setup
4. `pecs/store/database.py` — SQLite schema creation
5. `pecs/store/evidence_repo.py` — Evidence CRUD

Phase 2 — Ingestion pipeline:
6. `pecs/parsing/` — All parser adapters (testable independently)
7. `pecs/chunking/` — All chunkers (testable independently)
8. `pecs/normalization/` — MarkItDown wrapper
9. `pecs/ingestion/` — File upload dispatcher

Phase 3 — Embedding + storage:
10. `pecs/embeddings/` — nomic-embed-text via Ollama
11. `pecs/vectorstore/` — ChromaDB operations

Phase 4 — Retrieval:
12. `pecs/retrieval/` — Full retrieval pipeline

Phase 5 — Extraction + Correlation:
13. `pecs/extraction/` — Stage 1 + Pydantic validation
14. `pecs/correlation/rules.py` — Deterministic rule engine
15. `pecs/correlation/confidence.py` — Confidence scoring
16. `pecs/correlation/classifier.py` — Stage 2 LLM correlation

Phase 6 — Output + UI:
17. `pecs/traceability/` — Matrix generation
18. `pecs/ui/` — Streamlit pages

Phase 7 — Evaluation:
19. `tests/` — All tests and evaluations

Each phase should be completed and tested before the next begins. Phases 1–2 require no GPU and no Ollama. Phase 3 requires Ollama with nomic-embed-text. Phase 5 requires Ollama with Phi-4-mini.
