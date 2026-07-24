# PECS — Project Evidence Correlation System

A traceability engine that maps project requirements to implementation evidence and professor evaluations.

## Overview

PECS (Project Evidence Correlation System) is a local-only, privacy-preserving traceability engine that ingests heterogeneous project documents (PDFs, DOCX, emails, WhatsApp exports, Git logs, Markdown, Python source code) and produces a fully-cited traceability matrix mapping each requirement to its implementation evidence and evaluation.

## Architecture

```
Evidence Sources → Ingestion → Parsing → Chunking → Normalization → Embeddings → ChromaDB
                                                                                      ↓
                                                                               Retrieval Pipeline
                                                                                      ↓
                                                                            Stage 1 Extraction (Phi-4-mini)
                                                                                      ↓
                                                                              SQLite Evidence Store
                                                                                      ↓
                                                                          Deterministic Rules + Stage 2 LLM
                                                                                      ↓
                                                                           Traceability Matrix → Streamlit UI
```

## Requirements

- **Hardware**: RTX 3060 · 8 GB VRAM
- **Ollama**: Must be running with `phi4-mini` and `nomic-embed-text` models pulled
- **Python**: 3.13+

## Setup

1. Install dependencies:
   ```bash
   uv pip install -e .
   ```

2. Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```

3. Start Ollama:
   ```bash
   ollama serve
   ollama pull phi4-mini
   ollama pull nomic-embed-text
   ```

4. Run the application:
   ```bash
   streamlit run pecs/app.py
   ```

## Project Structure

```
pecs/
├── app.py                    # Streamlit entry point
├── config.py                 # Central configuration
├── logging_config.py         # Centralized logging
├── ingestion/                # File ingestion & dispatch
├── parsing/                  # Source-specific parsers
├── chunking/                 # Source-aware chunkers
├── normalization/            # Markdown normalization
├── models/                   # Universal internal objects
├── embeddings/               # nomic-embed-text via Ollama
├── vectorstore/              # ChromaDB operations
├── retrieval/                # Retrieval pipeline
├── extraction/               # Stage 1 extraction (Phi-4-mini)
├── correlation/              # Stage 2 correlation & rules
├── traceability/             # Matrix generation
└── ui/                       # Streamlit pages & components
tests/                        # Unit, integration, evaluation tests
data/                         # Runtime data (gitignored)
```

## Key Design Principles

1. **Immutability of evidence**: Once extracted, evidence rows are never modified.
2. **Separation of stores**: ChromaDB for retrieval, SQLite for structured knowledge.
3. **LLM as classifier, not narrator**: The LLM classifies evidence into fixed categories only.
4. **Provenance is mandatory**: Every fact is traceable to a specific source document and chunk.
5. **Deterministic-first**: Exact matches are resolved by rules; only ambiguous cases go to the LLM.
