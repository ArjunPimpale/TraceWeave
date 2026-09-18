# PECS

Standalone Streamlit traceability application. Run commands from the repository
root; `.env` and default `data/` paths are relative to the working directory.

## Commands

- Install and sync: `uv sync --locked`
- Run the UI: `uv run --locked streamlit run pecs/app.py`
- Run unit tests: `uv run --locked pytest tests/unit -q`
- Run the default suite: `uv run --locked pytest -q`
- Verify the lockfile: `uv lock --check`

Python 3.13 is selected by `.python-version`. Declare dependencies in
`pyproject.toml` and commit `uv.lock`; test dependencies belong in the `dev`
dependency group.

## Repository contracts

- `parsing/` preserves source structure, `chunking/` selects boundaries, and
  `normalization/` produces `EvidenceChunk` objects.
- ChromaDB stores searchable chunks. SQLite stores extracted evidence,
  correlations, and ingestion status.
- Evidence and correlation writes are append-only. Ingestion status may be
  replaced on retry; Settings provides the explicit data-reset operation.
- Preserve chunk IDs, provenance fields, prompts, seven correlation statuses,
  scoring rules, and deterministic-before-Stage-2 processing.
- Extraction calls Ollama; progress callbacks and SQLite writes stay on the
  calling thread. The graph is a derived SQLite traceability view and adds no inference.
- `ui/pages/` owns Streamlit widgets and session state.

## Local services and tests

Ollama defaults to `phi4-mini` and `nomic-embed-text`.
Graph and Matrix read saved SQLite traceability runs. The graph frontend is
built locally from `pecs/ui/components/trace_graph_frontend`; see `docs/testing.md`.

`data/` is local state and `evidence/` is local input, not a test-fixture
directory. Historical proposals are under `docs/archive/` and `docs/design/`.

## Documentation invariants

- Changes to public APIs, architecture, module boundaries, configuration, or user-visible behavior must update the relevant documentation in the same change.
- `docs/architecture.md` is the canonical architecture reference.
- Do not update docs for purely internal refactors that do not change documented behavior or structure.


## Testing

- Tests use pytest and live under `tests/`.
- Run the smallest relevant test subset while developing.
- Before completing a broad refactor, run the full suite with `uv run pytest`.
- Bug fixes should include a regression test when practical.
- Changes to public API behavior require corresponding test updates.
- Do not delete failing tests merely to make the suite pass.
