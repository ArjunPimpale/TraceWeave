# PECS Internals: A Comprehensive Engineering Guide

Welcome to this deep dive into **PECS** (Project Evidence Correlation System). You are reading this because you want to master the architecture, design decisions, and data flow of this local-first, privacy-preserving traceability engine. 

By the end of this guide, you will understand not just *what* code exists, but *why* it was written this way, the tradeoffs considered, and how the data moves from raw bytes to a fully correlated traceability matrix. We will go step-by-step, explaining the underlying concepts before diving into the implementation.

---

## Chapter 1: The Core Philosophy & Architecture

Before looking at a single line of code, we need to understand the problem PECS solves.

### What Problem Does PECS Solve?

In software engineering, a **Traceability Matrix** proves that a system actually does what it was asked to do. For every requirement (e.g., "The system must encrypt passwords"), you need evidence that it was implemented (e.g., a specific Python function, a Git commit, or an email discussing the design) and an evaluation (e.g., a professor grading it "Complete").

Traditionally, this is a nightmare of manual data entry in Excel. PECS automates this by ingesting your entire project history—PDFs, code, WhatsApp chats, emails—and using AI to figure out which pieces of data serve as evidence for which requirements.

### Key Architectural Principles

PECS is built on a few non-negotiable principles (found in the project's README):

1. **Immutability of Evidence:** Once we extract a fact from a document, we never mutate it. We create a permanent record.
2. **Deterministic-First:** LLMs hallucinate. To prevent this, PECS uses deterministic rules (exact keyword matching, regex) whenever possible, falling back to the LLM only for fuzzy semantic matching.
3. **Provenance is Mandatory:** If the system says "Requirement 3 is met," it must point to the exact document, chunk, and character location.
4. **Separation of Storage:** We use **ChromaDB** (a vector database) for *searching* by meaning, but **SQLite** (a relational database) for *storing* the final, structured truth.

### The Data Currency: `EvidenceChunk`

When building a system that handles heterogeneous data (WhatsApp messages vs. Python code), the most dangerous thing you can do is pass raw strings around. The downstream systems won't know where the string came from.

To solve this, PECS introduces a universal internal currency called the `EvidenceChunk`.

Let's look at `pecs/models/evidence_chunk.py`:

```python
@dataclass
class EvidenceChunk:
    chunk_id: str
    source_document: str
    source_hash: str
    source_type: SourceType
    chunk_index: int
    source_locator: str
    normalized_text: str
    char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
```

**Why this design?**
- `chunk_id`: Notice that this is not a random UUID. It is generated via `make_chunk_id()` using `SHA-256(source_hash + chunk_index)`. 
  - **Why?** Idempotency. If a user uploads the same PDF twice, the system generates the exact same IDs. When it writes to ChromaDB, it silently overwrites rather than duplicating data.
- `source_locator`: A human-readable string. Why a string and not a structured object like `{page: 3, line: 10}`? Because a Git commit locator looks like `commit abc1234`, while a WhatsApp locator looks like `[2023-10-01 10:00 AM]`. A string provides maximum flexibility for provenance UI without forcing unrelated data into an awkward schema.
- `normalized_text`: Regardless of whether the source was a messy PDF or a DOCX file, the text here has been converted to clean Markdown. This standardizes the input for the LLM.

**Data Flow Check:**
Whenever data enters the system, the goal of the first pipeline stage is to convert *everything* into a list of `EvidenceChunk` objects. 

---

## Chapter 2: The Ingestion Pipeline (The Front Door)

If 10,000 users upload PDFs simultaneously, the entry point of your system needs to be robust, secure, and smart enough to know what to do with them.

In PECS, this is handled by `Ingestor` in `pecs/ingestion/ingestor.py`.

### Concept: The Pipeline Pattern

The `Ingestor` class uses a classic **Pipeline Pattern**. A pipeline takes an input, passes it through a sequence of discrete stages (filters/transformers), and outputs a result. 

**Why use a pipeline?** 
If parsing, chunking, and embedding were all mixed in one giant function, you could never test them individually, and adding a new file type (like `.csv`) would require rewriting the whole function.

### Walkthrough: `ingestor.py`

Let's trace exactly what happens when a user uploads a file.

#### Step 1 & 2: Validation and Deduplication

```python
    def ingest_file(self, file_bytes: bytes, filename: str, user_metadata: dict) -> IngestionResult:
        # 1. Validation
        validate_not_empty(file_bytes, filename)
        validate_file_size(file_bytes, filename)

        # 2. Deduplication check
        file_hash = compute_sha256(file_bytes)
        if self._repo.is_file_ingested(file_hash):
            return IngestionResult(..., skipped=True)
```

**Why hash the file?** 
Users often rename files (e.g., `design_v1.pdf` -> `design_final.pdf`). If we relied on filenames to detect duplicates, we'd ingest identical content twice, polluting the vector space and wasting expensive LLM tokens. Hashing the raw bytes ensures we catch identical contents regardless of the name. It checks SQLite (`self._repo`) to see if this hash exists.

#### Step 3: MIME Type Detection & Content Sniffing

```python
        mime_type = detect_mime_type(file_bytes, filename)
        source_type = self._resolve_source_type(mime_type, filename, file_bytes)
```

**How `_resolve_source_type` works internally:**
MIME types (like `application/pdf`) are standard, but what about a `.txt` file? It could be a WhatsApp export, an email log, or a Git diff. All of them register as `text/plain`.
If the file is `text/plain`, the system uses **content sniffing** (heuristics). For example, if every line starts with `[DD/MM/YY, HH:MM:SS]`, the system sniffs it as `WHATSAPP`. 

#### Step 4 & 5: Dispatching (The Registry Pattern)

```python
        parser = self._parsers[source_type]
        parsed_doc = parser.parse(parse_request)

        chunker = self._chunkers[source_type]
        raw_chunks = chunker.chunk(parsed_doc)
```

**Why this implementation?**
Notice that `Ingestor` doesn't have a giant `if/elif` block (e.g., `if source_type == PDF: parse_pdf()`). 
Instead, it uses a **Registry Pattern**:
```python
        self._parsers = {
            SourceType.PDF: PDFParser(),
            SourceType.GIT: GitParser(),
            # ...
        }
```
This is a manifestation of the **Open/Closed Principle** (SOLID). If you want to add support for audio files later, you don't touch the `Ingestor` logic; you just add `SourceType.AUDIO: AudioParser()` to the dictionary.

#### Step 6: Normalization

```python
        evidence_chunks = self._normalizer.normalize_batch(raw_chunks)
```

**What is Normalization?**
A PDF parser might extract text like `T H I S  I S  A  T I T L E`. An email parser might leave in HTML tags. The `Normalizer` uses Microsoft's `MarkItDown` library to convert all raw chunks into clean, standard Markdown. 
**Why Markdown?** LLMs are overwhelmingly trained on Markdown (from GitHub, Reddit, StackOverflow). An LLM will perform significantly better at extracting data if the input text is formatted as Markdown with `## Headers` rather than raw, disjointed strings.

### Data Flow Summary (Ingestion)

1. **Input:** Raw Bytes (`b'\x25\x50\x44\x46...'`) + Filename (`"spec.pdf"`)
2. **Validation:** Checks size limit (e.g., 50MB) via `config.py`
3. **MIME/Type:** Detected as `application/pdf` -> mapped to `SourceType.PDF`
4. **Parser:** Extracts raw text blocks and metadata from the binary.
5. **Chunker:** Splits text into ~1500 character chunks with overlap.
6. **Normalizer:** Cleans chunks to Markdown.
7. **Output:** `list[EvidenceChunk]`

---

## Chapter 3: The Storage Strategy (The Dual Database Approach)

If you have used basic RAG tutorials before, you likely saw a system that parses a PDF, dumps the vectors into a database (like Pinecone, FAISS, or ChromaDB), and queries them directly. 

PECS does **not** do this. PECS uses a deliberate **Dual Database Strategy**:
1. **ChromaDB**: Used *only* for semantic search (finding chunks that have similar meaning to a query).
2. **SQLite**: Used as the source of ground-truth, storing immutable evidence and correlations.

### Why use two databases?

Vector databases are fantastic at calculating distance (semantic similarity). They are *terrible* at being relational databases. 

In a traceability system, you need to be able to say: *"Show me all evidence for Requirement 3 that was authored by John Smith, sorted by timestamp."* 
If you try to do that in a pure vector database, you have to do massive metadata filtering and scanning, which is inefficient and brittle. 

Therefore, PECS separates **search** from **structured knowledge**.

### SQLite: The Immutable Record

Let's look at `pecs/store/database.py` and `pecs/store/evidence_repo.py`.

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
    metadata        TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
);
```

**Design Decisions in SQLite:**
1. **Append-Only (No UPDATEs)**: Notice in `EvidenceRepo` there are `insert()` and `insert_batch()` methods, but **no** `update()` method. This enforces the architectural principle of **Immutability of Evidence**. An audit trail must not be editable.
2. **WAL Mode**: In `database.py`, the connection executes `PRAGMA journal_mode=WAL;`. Write-Ahead Logging allows multiple readers to read from the database simultaneously while a writer is inserting data. In a UI like Streamlit where multiple widgets might query the DB while ingestion is happening in the background, WAL prevents database locking errors.
3. **Idempotency**: The `EvidenceRepo` checks `_row_exists()` before inserting. If the exact same extraction is yielded twice, it is silently dropped. 

### ChromaDB: The Semantic Index

While SQLite holds the final structured facts, **ChromaDB** holds the unstructured text chunks so we can search them.

Look at `pecs/vectorstore/chroma_store.py`:

```python
    def upsert_chunks(self, chunks: list[EvidenceChunk], embeddings: list[list[float]]) -> None:
        ids = [c.chunk_id for c in chunks]
        documents = [c.normalized_text for c in chunks]
        metadatas = [c.to_chroma_metadata() for c in chunks]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
```

**How data flows here:**
1. A chunk arrives.
2. `chunk.to_chroma_metadata()` flattens complex data structures. ChromaDB only accepts strings, integers, floats, and booleans. If you pass a Python `datetime` object to Chroma, it crashes. 
3. The system uses `upsert` (Update or Insert) using the deterministic `chunk_id`. If you ingest the same file twice, ChromaDB just overwrites the old vectors, preventing duplicate search results.

---

## Chapter 4: The Hybrid Retrieval Pipeline

When a user asks: "How is the password encryption requirement implemented?", the system must find the right chunk out of thousands. 

If we only used Vector Search, we might fail. 
**Why?** Vector models understand *meaning*, but they are notoriously bad at *exact keyword matching*. If a user searches for "Requirement 3" or "R3", a vector search might return chunks talking about "Requirement 4" because they are semantically identical (they both talk about requirements).

To fix this, PECS uses a **Multi-Strategy Hybrid Retrieval Pipeline**. 

### The Components

1. **Embedder (`pecs/embeddings/embedder.py`)**: 
   Converts text into a 768-dimensional array of floats using the `nomic-embed-text` model via Ollama. It processes data in batches (e.g., 32 chunks at a time) to minimize HTTP overhead and implements exponential backoff retries if the Ollama server blips.
2. **Vector Store (`ChromaStore`)**:
   Calculates the **Cosine Distance** between the query vector and the chunk vectors.
3. **BM25 Index (`pecs/retrieval/bm25.py`)**:
   A classic keyword-frequency search engine (the same algorithm Elasticsearch uses). It excels at finding exact acronyms, names, and IDs.
4. **ID Matcher (`pecs/retrieval/pipeline.py`)**:
   Uses regex to find exact requirement identifiers (e.g., "R3", "req-3") to guarantee they are included in the context.

### The Pipeline Flow

Let's look at how these are orchestrated in `pecs/retrieval/pipeline.py`:

```python
    def retrieve(self, query_text: str, requirement_ids: list[str]) -> list[RetrievedEvidence]:
        # 1. Vector retrieval (ChromaDB)
        vector_results = self._vector_retrieve(query_text, where_filter, top_k)

        # 2. BM25 keyword retrieval
        bm25_results_raw = self._bm25.query(query_text, top_k)
        bm25_results_normalized = self._bm25.normalize_scores(bm25_results_raw)

        # 3. Requirement-ID matching
        id_match_results = []
        if requirement_ids:
            id_match_results = self._id_match_retrieve(requirement_ids, where_filter)

        # 4. Merge candidates
        merged = self._merger.merge(
            vector_results=vector_results,
            bm25_results=bm25_results_normalized,
            id_match_results=id_match_results,
            top_k=top_k,
        )
        return merged
```

### The Merger Logic (Section 10.6 Formula)

The most complex part of Hybrid Search is combining the scores. A vector score from ChromaDB is bounded between 0 and 1. A BM25 score is unbounded (it could be 12.5 or 145.2). You can't just add them together.

First, BM25 scores are **normalized** (min-max scaling relative to the top score in that specific query).

Then, `CandidateMerger` (`pecs/retrieval/merger.py`) iterates through all candidate chunks and calculates a `combined_score`:

```python
        if has_vector and has_bm25:
            # Alpha weighting: 0.7 Vector + 0.3 Keyword
            combined = self.vector_weight * vector_score + self.bm25_weight * bm25_score
        elif has_vector:
            combined = vector_score
        elif has_bm25:
            combined = bm25_score

        if id_match:
            # If the chunk explicitly names "R3", guarantee a high score
            combined = max(combined, 0.95)
```

**Why this works in production:**
If a chunk contains the exact phrase the user typed, BM25 catches it. If a chunk uses synonyms (e.g., query="auth", chunk="login"), Vector Search catches it. If a chunk explicitly references a tracking ID, the ID Matcher forces it to the top. This guarantees the LLM receives the absolute best context possible in the next step.

---

## Chapter 5: LLM Extraction & Validation

The hardest part of building AI systems is getting structured, predictable data out of unstructured, unpredictable text. 

A naive approach would be to dump the entire codebase into an LLM and ask: *"Generate a traceability matrix."* This fails catastrophically because context windows get overloaded, models hallucinate, and you lose provenance (where the fact came from).

Instead, PECS processes documents one chunk at a time, asking the LLM to extract specific entities (Requirements, Implementations, Evaluations) and return them as strict JSON.

### The Extractor (`pecs/extraction/extractor.py`)

The Stage 1 Extractor uses the **Phi-4-mini** model via Ollama. 

**Design Decisions:**
1. **Temperature = 0.05**: We want extraction to be as deterministic as possible. Creativity is the enemy of data extraction.
2. **Individual Processing (No Batching)**: While embeddings are batched for network efficiency, the LLM processes one chunk per prompt. If you ask an LLM to extract data from 5 chunks at once, it often accidentally attributes data from Chunk 2 to Chunk 4.

### The Retry Strategy (The Escalation Pattern)

LLMs frequently output malformed JSON. PECS implements a 3-tier escalation strategy:

```python
    def _build_prompt(self, chunk, attempt, last_errors):
        if attempt == 1:
            # Standard prompt with system + examples + user
            return [...]
        elif attempt == 2:
            # Retry with error feedback
            error_text = self._validator.format_errors_for_retry(last_errors)
            # "You failed because: Field 'entity_id' is missing. Try again."
            return [...]
        else:
            # Simplified prompt for final attempt
            return [{"role": "user", "content": STAGE1_SIMPLIFIED_TEMPLATE}]
```

**Why this works:** 
Often, an LLM knows *what* to output but messed up the syntax. By feeding the Pydantic validation errors back into the prompt on Attempt 2, the LLM can self-correct. If that still fails, Attempt 3 strips away complex instructions and asks for a bare-minimum JSON object.

### The Validator (`pecs/extraction/validator.py`)

Even with strict prompts, LLMs will wrap JSON in markdown code fences (```json ... ```) or add conversational prefix text ("Here is the JSON you requested:"). 

The `ExtractionValidator` uses regex to aggressively hunt for the JSON object within the raw string:
```python
        # Attempt 1: Direct parse
        # Attempt 2: Extract from code fences
        # Attempt 3: Find bare JSON array/object using regex
```

Once it finds the JSON, it parses it into Pydantic models. 

#### Hallucination Detection (Word Overlap)
A major risk in RAG is the LLM inventing facts (e.g., claiming a function encrypts data when the text doesn't mention encryption).

PECS uses a simple but highly effective deterministic check: **Word Overlap**.
```python
    def _word_overlap(extracted_text: str, source_text: str) -> float:
        # Returns the fraction of words in extracted_text that also appear in source_text.
```
If the LLM extracts an "Implementation" description, but less than 30% of the words in that description actually exist in the source chunk, PECS flags it as `hallucination_risk = True`. It doesn't delete it (immutability), but it warns the user.

---

## Chapter 6: Correlation & The Traceability Matrix

At this point, the system has:
1. A database of extracted Requirements, Implementations, and Evaluations (SQLite).
2. A semantic index of all text (ChromaDB).

Now it must correlate them. Did the team actually build Requirement 3? 

PECS uses a two-stage approach: **Deterministic Rules** first, **LLM Classification** second.

### Stage 1: The Deterministic Rule Engine (`pecs/correlation/rules.py`)

LLMs are expensive and non-deterministic. If we can prove a correlation using code logic, we should. The `RuleEngine` applies a cascade of rules to every requirement:

1. **Exact ID Match:** If an extracted Implementation row explicitly says `linked_requirement: "REQ-3"`, it's an automatic pass. The engine then scans linked Evaluations. If the evaluation text contains negative words ("bug", "failed"), the status becomes `IMPLEMENTED_BUT_NEGATIVELY_EVALUATED`.
2. **Strong Semantic Match:** The engine queries ChromaDB for the requirement. If the top retrieved chunk has a similarity score `>= 0.75` (a very high semantic match), the system accepts it as evidence deterministically, *even if the LLM didn't explicitly extract an implementation row for it*.
3. **No Evidence Floor:** If there is no explicit link AND the vector search returns a top score of `< 0.30` (garbage results), the system deterministically fails the requirement as `REQUIREMENT_NOT_IMPLEMENTED`.

Any requirement resolved by these rules is removed from the queue.

### Stage 2: The Fallback LLM Classifier (`pecs/correlation/stage2.py`)

What happens to the requirements in the "gray zone" (e.g., retrieval score is 0.50)? 

They are passed to the `Stage2Classifier`. 

**Critical Design Choice:**
The LLM is NOT asked to "find" the evidence. The Python code finds the top 5 candidates using Hybrid Search, formats them into a single string, and asks the LLM a highly constrained multiple-choice question:

*Prompt excerpt:*
```text
Given this requirement and these 5 retrieved chunks, classify their relationship into EXACTLY ONE of these 7 labels:
- IMPLEMENTED_AND_VALIDATED
- PARTIALLY_IMPLEMENTED
- CLAIMED_BUT_NO_EVIDENCE
...
Respond ONLY with a JSON object.
```

By doing this, the LLM is relegated to the role of a **reading comprehension classifier**, a task modern LLMs are exceptionally good at.

### What would happen if Stage 2 didn't exist?
If we only used rules, we would have to guess what a 0.50 semantic score means. Is it a partial implementation? Is it someone talking about the requirement in an email but not actually writing code (`CLAIMED_BUT_NO_EVIDENCE`)? Vector distance cannot distinguish between a developer saying "I will build the login system" and the actual Python file containing `def login():`. The Stage 2 LLM bridges that semantic gap.
