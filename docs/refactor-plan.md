# PECS Refactoring Plan

## Observed facts

Audit baseline: commit 2031175b0e5dfc46fa05080198d7fc580e6bc272 (Backup for refactoring), with a clean worktree. The repository has 94 tracked files, including 84 Python files; every tracked Python file passed static parsing. No repository or ancestor AGENTS.md exists. No tracked file matches .gitignore. There are no tracked requirements files, Poetry, Pipenv, Conda, or competing environment configurations.

The project is a standalone Streamlit PECS application. The pecs package is organized into parsing, chunking, normalization, models, ingestion, embeddings, Chroma vector storage, retrieval, extraction, SQLite stores, correlation, traceability, graph, and UI packages. Its current flow is upload, validate/type-detect/parse/chunk/normalize, SQLite ingestion log, separate UI embedding and Chroma write, Chroma-backed Stage 1 extraction, then retrieval/rules/optional Stage 2/scoring/matrix construction. Neo4j is a derived SQLite graph with no new inference. These package boundaries are reasonable and should remain.

Environment findings: pyproject.toml uses Hatchling and Python 3.13+, .python-version is 3.13, and .venv is uv-created CPython 3.13.9. uv.lock exists but is ignored. README uses uv pip install -e and then an unqualified streamlit command. pytest and pytest-mock are runtime dependencies. pydantic-settings, pandas, and PyYAML are imported directly but only available transitively; installed and locked versions are 2.14.2, 3.0.5, and 6.0.3. MarkItDown is installed but its conversion API is not called. .env.example has a Python-style ENV_FILE_TEMPLATE wrapper that causes dotenv parsing warnings and a stray key.

Documentation conflicts must be recorded, not silently resolved: README places retrieval before Stage 1; TSD describes separate entity tables while code has evidence, correlation, and ingestion_log; TSD conflicts on retrieval/extraction and rule/LLM order; normalizer docs claim MarkItDown conversion; rule docs claim literal ID matching and 0.70/0.15 although code does semantic matching at 0.75/0.30; graph docs claim atomic rollback although deletion is committed before rebuild; extraction UI claims only unextracted chunks are processed; BM25 is described as active but has no production index-rebuild caller; graph docs conflict on auth; graph comments mention nonexistent tooltip injection; worker defaults differ between example, config, and docs; and integration documentation uses an unregistered marker.

Preserve known limitations: timestamp ties may yield more than one latest correlation before matrix selects highest confidence, and graph show_orphans is passed but unused.

## Likely stale/junk artifacts

- main.py is a greeting-only scaffold. Remove only after confirming no external launcher uses it.
- implementation_plan.md is historical design/proposal material. Move to docs/archive with a historical-status note.
- documents/graph_interpretability_layer.md is historical implementation planning with machine-specific links. Move to docs/design and annotate implementation differences.
- documents/pecs_architecture_tutorial.md is useful but stale. Retain and update as current architecture documentation.
- pytest cache, bytecode, and .venv are generated/local and already ignored. Keep them during validation.
- data/sqlite and data/chromadb are persistent local application state; data/logs is runtime output. Preserve and ignore.
- evidence is an intentional local input corpus, not tests; preserve and ignore.
- AIandUser is empty and untracked; no action.
- uv.lock is a reproducibility artifact and should be tracked.
- golden_triples.json is a useful but unused evaluation seed. Keep it; its verification is unconfirmed.

No tracked debug scripts, generated graph HTML, screenshots, databases, or temporary experiments were found. Existing tests are real automated tests.

## Uncertainties requiring human confirmation

Confirm external use of main.py or the installed pecs command before removal. Treat golden triples as samples unless they are confirmed hand-reviewed. Retain MarkItDown dependency and behavior unless conversion intent is confirmed. Preserve six workers in the example and four in config. Exclude BM25 activation, incremental extraction, graph atomicity, orphan filtering, and timestamp tie-breaking from cleanup. Preserve UI error/setup text under the no-UI-change requirement.

## Recommended changes

Keep the flat pecs package. Do not migrate to src, add a service framework or generic repository superclass, or redesign parsing/chunking.

Target documentation structure:

    docs/
      refactor-plan.md
      architecture.md
      testing.md
      specifications/TSD_v0.3.docx
      design/graph-interpretability-layer.md
      archive/implementation-plan.md

Move tutorial to architecture.md, graph plan to design, implementation plan to archive, and Word specification to specifications. Keep source, tests, evidence, and runtime databases. Fix graph-plan absolute file links when moving it; preserve DOCX bytes.

For uv: stop ignoring and track uv.lock; move pytest packages to a dev dependency group; declare pydantic-settings, pandas, and PyYAML directly while retaining current locked versions; retain Python 3.13, Hatchling, MarkItDown, and python-dotenv; remove only the invalid wrapper in .env.example. Canonical commands are uv sync --locked, uv run --locked streamlit run pecs/app.py, and uv run --locked pytest tests/unit -q. Run uv lock without upgrades only after intentional metadata changes, then uv lock --check. Do not add requirements files, another environment manager, activation instructions, or upgrades.

For .gitignore: remove the uv.lock rule; include .env, .env.*, and !.env.example; add .coverage.*, .ruff_cache, .mypy_cache, and .streamlit/secrets.toml; replace blanket .streamlit ignoring so a shared config may be tracked; prefer root-anchored runtime paths. Do not ignore broad file extensions or test/debug name patterns.

Low-level refactoring should be limited to private extractions:
- split the 1,024-line graph page into graph network/tooltip and graph detail components while retaining page controls and composition;
- split RuleEngine.apply_rules into private methods while retaining six-pass ordering;
- extract small ingestion failure/logging/parse/chunk helpers;
- deduplicate repository SQL and parameter construction without changing commit boundaries;
- provide a public Chroma raw-record read and shared reconstruction only if existing metadata differences remain intact;
- extract candidate-map initialization, matrix decode/assembly helpers, and only justified graph payload helpers;
- retain Extractor and GraphBuilder class boundaries.

Remove only verified duplicate or unused items: duplicate graph icon constant, unused retrieval ID-pattern constant, unused Rule 1 score local, unused ingestion page local, and verified unused imports. Preserve chunk ID generation, metadata behavior, rule thresholds and ordering, prompts, retries, callback/main-thread writes, graph IDs/styles/tooltips/widget/session keys, logger/database lifecycle, and existing value-or-default semantics. Atomic sync, BM25 activation, incremental filtering, metadata hydration, and latest-selection changes are separate work.

## Testing and validation strategy

The suite contains 130 unit-test functions and 10 integration-test functions before parametrization. Keep them. Add characterization tests for configuration, ingestion, repositories, retrieval, extraction retry/concurrency, Stage 2, rule boundaries, matrix assembly, and graph UI helpers. Add missing PDF/DOCX parser coverage, email/DOCX chunker coverage, and an executable evaluation runner.

Current graph integration tests contact local Neo4j, target its default database, delete all nodes, bypass GraphSync.sync_graph, and clear before their idempotency rerun. Register the integration marker, require explicit opt-in and disposable URI, skip before connectivity without opt-in, use a separate instance such as port 17687, test the orchestrator and failures, and test repeated MERGE without clearing. Proposed command:

    uv run --locked pytest tests/integration -m integration \
      --run-neo4j --neo4j-test-uri bolt://localhost:17687

Use fake Ollama clients, temporary SQLite/Chroma paths, and synthetic fixtures. Do not use evidence as implicit test input. Redirect logging and storage before imports. For graph UI checks compare structure and attributes, not physics positions.

## Proposed AGENTS.md

    # PECS

    Standalone Streamlit traceability application. Run from repository root.
    .env and default data paths are relative to the working directory.

    Commands:
    - uv sync --locked
    - uv run --locked streamlit run pecs/app.py
    - uv run --locked pytest tests/unit -q
    - uv run --locked pytest -q
    - uv lock --check

    Python 3.13 is selected by .python-version. Dependencies belong in
    pyproject.toml; commit uv.lock. Test dependencies use the dev group.

    Parsing preserves structure, chunking chooses boundaries, and normalization
    produces EvidenceChunk. ChromaDB stores searchable chunks; SQLite stores
    evidence, correlations, and ingestion status. Evidence/correlation writes are
    append-only; ingestion status may be replaced on retry. Preserve chunk IDs,
    provenance, prompts, statuses, scoring, and deterministic-before-Stage-2 flow.
    Extraction invokes Ollama; callbacks and SQLite writes stay on the calling
    thread. Neo4j is a derived SQLite view. UI pages own widgets and session state.

    Ollama defaults are phi4-mini and nomic-embed-text. NEO4J_ENABLED=false
    disables graph features. Neo4j integration tests require explicit disposable
    endpoint opt-in because they delete its graph. data is local state and
    evidence is local input, not test fixtures.

## Ordered implementation phases

### Phase 0: preserve baseline

Only this plan is affected. Record commit/status, lock and installed versions, a no-sync unit baseline, failures, and disposable-state UI behavior. Do not run integration tests. Validate baseline output and git status. Preserve .venv, lock, and persistent data; handle SQLite WAL consistently.

### Phase 1: make tests safe

Affect tests/conftest.py, graph integration tests, pytest settings, docs/testing.md, and new graph-sync unit tests. Add opt-in, isolation, marker registration, and real orchestrator tests. Validate unit, collection, and default test runs; they must not contact Neo4j. Keep external tests disabled if isolation is uncertain.

### Phase 2: canonical uv

Affect .gitignore, pyproject.toml, uv.lock, .env.example, README, and config tests. Track lock, move dev dependencies, declare direct imports, repair example and commands. Validate with uv lock, uv lock --check, uv sync --locked, unit tests, and a fresh temporary uv environment. Stop on unexplained version changes.

### Phase 3: documentation and ignores

Affect README, .gitignore, documentation moves, AGENTS.md, docs/testing.md, and main.py only if confirmed. Validate diff formatting, tracked ignored files, check-ignore behavior, links, and unchanged DOCX bytes. Use moves, not destructive recreation.

### Phase 4: small duplication/private access

Affect repositories, Chroma store, optional codec, retrieval pipeline/merger, extraction page, and focused tests. Add characterization tests first. Preserve schemas, transactions, imports, and persisted-data compatibility.

### Phase 5: local orchestration cleanup

Affect ingestor, rules, matrix, extractor comments/types, and focused tests. Use private helpers only. Validate deterministic before/after results. Preserve rule ordering and matrix selection.

### Phase 6: graph presentation extraction

Affect graph page, graph network/details components, and focused tests. Move functions/constants only; remove duplicate icon/stale comment; preserve orphan-toggle behavior. Validate unit tests and disposable-state Streamlit UI, including graph modes, controls, tooltips, details, sync feedback, disabled states, and other pages. Widget/session execution order is the rollback risk.

### Phase 7: final verification

Run uv lock --check, uv sync --locked, default tests, diff check, and status. Run Neo4j tests only against disposable instance. Completion requires one documented uv workflow and tracked lock; safe default tests; correct ignores; compatible public imports; unchanged persistence, identities, prompts, statuses, thresholds, scoring, and UI; documented historical conflicts; classified removals; and no unexplained upgrades or migrations.

