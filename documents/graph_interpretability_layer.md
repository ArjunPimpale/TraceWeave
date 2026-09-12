# Neo4j Graph Interpretability Layer for PECS

## Background

PECS currently produces a traceability matrix by extracting entities (Requirements, Implementations, Evaluations) into SQLite via Stage 1 LLM extraction, then correlating them through a deterministic rule engine and Stage 2 LLM classifier. The result is a flat list of `CorrelationResult` rows linking requirements to evidence.

The goal is to add a **graph-based exploration layer** that deterministically transforms these existing validated entities and correlations into an interactive Neo4j graph, without introducing any new reasoning or LLM inference.

### Central Principle

> **PECS determines what is correlated. The graph layer deterministically visualizes those existing correlations and allows users to explore the connected evidence.**

---

## User Review Required

> [!IMPORTANT]
> **Neo4j deployment model.** ✅ **Confirmed:** Local Neo4j Community Edition (Docker or direct install). No cloud or shared instance.

> [!IMPORTANT]
> **Visualization library choice.** ✅ **Confirmed:** pyvis (renders to HTML via `st.components.v1.html`).

> [!WARNING]
> **Graph page scope.** ✅ **Confirmed:** New "🕸️ Evidence Graph" sidebar page complements the existing Matrix page. Matrix page is not replaced.

---

## Open Questions

1. **Neo4j authentication.** ✅ **Decided:** No auth for this build. Local Neo4j runs without authentication. No username/password in config.

2. **Graph page entry point.** ✅ **Decided:** Sidebar only. No "View in Graph" button on the Matrix page.

3. **Historical correlation runs.** ✅ **Decided:** Latest-only for V1. Code will include comments marking extension points for future snapshot support.

---

## Architecture Review Notes

### ARN-1: `entity_id` Is Not a Stable Natural Key

> **Concern:** The `entity_id` field in `ExtractionResult` is LLM-generated (e.g., `"R3-gnn-training"`, `"impl-train-gnn"`). The TSD confirms it is "descriptive" and "not a DB key." Two extraction runs on the same document can produce different `entity_id` values for the same real-world entity (e.g., `"R3-gnn-training"` vs `"R3-gnn_training_pipeline"`).
>
> **Why it matters:** If Neo4j nodes use `entity_id` as their identity key, re-running extraction + correlation will create **duplicate nodes** for the same conceptual entity. The graph becomes polluted with near-identical nodes that aren't linked to each other.
>
> **Recommendation:** For V1, accept this limitation and use `entity_id` as the graph identity key — it is what both `CorrelationResult.requirement_entity_id` and `CorrelationResult.evidence_entity_id` reference, so it is the only joinable key available without a major refactor. Mitigate by:
> 1. Making graph construction idempotent within a single correlation run (MERGE on `entity_id`).
> 2. Adding a "Clear & Rebuild Graph" action that wipes Neo4j and reconstructs from the latest SQLite snapshot.
> 3. Documenting that entity ID instability is a known limitation to address if PECS adds entity deduplication in the future.

### ARN-2: Source Document as a Graph Node — Implicit Relationship

> **Concern:** Your proposed schema includes "Code Files" and "Commits" as node types. However, PECS does not extract these as first-class entities. The `ExtractionResult` model has `source_document` (a filename string) and `SourceType` (PDF, GIT, PYTHON, etc.), but there is no separate `CodeFile` or `Commit` entity in the evidence table. A "code file" is just a source document with `source_type = PYTHON`, and a "commit" is a source document with `source_type = GIT`.
>
> **Why it matters:** Creating separate `CodeFile` and `Commit` node types would require inventing entity boundaries that PECS's extraction pipeline doesn't produce. This would violate the "no additional reasoning" constraint.
>
> **Recommendation:** Model source documents as `SourceDocument` nodes with a `source_type` property (PYTHON, GIT, PDF, etc.). The UI can render these with type-specific icons (📄 for PDF, 🐍 for Python, etc.), achieving the same visual effect without fabricating entity types. If you later add structured Git commit parsing (extracting commit SHA, author, message), those can become first-class `Commit` nodes at that time.

### ARN-3: Evaluation Nodes Should Not Be Independent Top-Level Entities

> **Concern:** The proposed graph has Evaluations as standalone nodes at the same level as Requirements. However, in the PECS data model, an Evaluation only has meaning in relation to a Requirement (via `linked_requirement`) or as an orphan (`EVALUATION_WITHOUT_REQUIREMENT`). Rendering orphan evaluations as prominent standalone graph nodes may confuse users.
>
> **Why it matters:** An orphan evaluation floating in the graph with no connections adds visual noise without interpretability value.
>
> **Recommendation:** Evaluations are modeled as nodes, but orphan evaluations (status = `EVALUATION_WITHOUT_REQUIREMENT`) are hidden by default in the graph view. Users can toggle "Show orphan evaluations" to surface them. Connected evaluations are always shown.

---

## Proposed Changes

### Data Mapping: PECS → Neo4j

The graph schema is derived directly from the three data structures that exist in the codebase:

| PECS Data Source | Neo4j Node Type | Identity Key | Source |
|---|---|---|---|
| `evidence` table rows where `entity_type = 'REQUIREMENT'` | `:Requirement` | `entity_id` | [evidence_repo.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/store/evidence_repo.py#L173-L175) |
| `evidence` table rows where `entity_type = 'IMPLEMENTATION'` | `:Implementation` | `entity_id` | [evidence_repo.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/store/evidence_repo.py#L177-L179) |
| `evidence` table rows where `entity_type = 'EVALUATION'` | `:Evaluation` | `entity_id` | [evidence_repo.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/store/evidence_repo.py#L181-L183) |
| Distinct `source_document` values from `evidence` table | `:SourceDocument` | `name` (filename) | [ExtractionResult.source_document](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/models/extraction_result.py#L59) |
| `correlation` table rows (latest per requirement) | Relationship: `CORRELATES_TO` | `correlation_id` | [correlation_repo.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/store/correlation_repo.py#L153-L173) |

---

### Graph Schema Design

#### Node Types

```
(:Requirement {
    entity_id: String,        -- Primary key (MERGE target)
    text: String,             -- Requirement description
    source_document: String,  -- Originating document
    chunk_id: String,         -- Provenance back to EvidenceChunk
    author: String?,          -- Extracted author if available
    timestamp: String?,       -- Extracted timestamp if available
    created_at: String        -- Extraction timestamp
})

(:Implementation {
    entity_id: String,        -- Primary key
    text: String,             -- Implementation description
    linked_requirement: String?,  -- Explicit link from Stage 1 (if any)
    source_document: String,
    chunk_id: String,
    author: String?,
    timestamp: String?,
    created_at: String
})

(:Evaluation {
    entity_id: String,        -- Primary key
    text: String,             -- Evaluation/feedback text
    linked_requirement: String?,
    source_document: String,
    chunk_id: String,
    author: String?,
    timestamp: String?,
    created_at: String
})

(:SourceDocument {
    name: String,             -- Primary key (filename)
    source_type: String       -- SourceType enum value (PDF, GIT, PYTHON, etc.)
})
```

#### Relationship Types

```
-- Core correlation relationship (from CorrelationResult)
(req:Requirement)-[:CORRELATES_TO {
    correlation_id: String,
    status: String,              -- CorrelationStatus enum value
    confidence: Float,           -- 0.0–1.0
    resolution_method: String,   -- "deterministic_rule" or "llm_stage2"
    rule_name: String?,          -- e.g., "exact_requirement_id_match"
    reasoning: String,           -- Justification text
    created_at: String
}]->(impl:Implementation)

-- Explicit link from Stage 1 extraction (linked_requirement field)
(impl:Implementation)-[:EXPLICITLY_LINKS {
    source: "stage1_extraction"
}]->(req:Requirement)

-- Evaluation linkage
(eval:Evaluation)-[:EVALUATES {
    source: "stage1_extraction"
}]->(req:Requirement)

-- Provenance: entity originated from this source document
(entity)-[:EXTRACTED_FROM]->(doc:SourceDocument)

-- Supporting evidence (from supporting_chunk_ids in CorrelationResult)
(req:Requirement)-[:SUPPORTED_BY {
    correlation_id: String
}]->(chunk_owner)  -- The entity that owns the supporting chunk_id
```

#### Visual Representation

```
                    ┌──────────────────────┐
                    │   SourceDocument      │
                    │   (spec.pdf / GIT)    │
                    └─────────┬────────────┘
                              │ EXTRACTED_FROM
                              ▼
┌──────────────┐   CORRELATES_TO    ┌──────────────────┐
│ Requirement  │──────────────────→│  Implementation   │
│ (R3-gnn...)  │  status, conf.    │  (impl-train...)  │
└──────┬───────┘                   └────────┬──────────┘
       │                                     │
       │ EVALUATES                           │ EXTRACTED_FROM
       │                                     ▼
┌──────▼───────┐                   ┌──────────────────┐
│  Evaluation  │                   │  SourceDocument   │
│ (eval-prof..)│                   │  (train.py / PY)  │
└──────────────┘                   └──────────────────┘
```

---

### Graph Construction Pipeline

#### Module: `pecs/graph/` (new package)

```
pecs/graph/
├── __init__.py
├── neo4j_client.py       -- Connection management, health check
├── graph_builder.py      -- Deterministic construction from SQLite
├── graph_queries.py      -- Cypher query templates for the UI
└── graph_sync.py         -- Sync orchestration, idempotency
```

---

#### [NEW] `pecs/graph/neo4j_client.py`

Manages the Neo4j driver lifecycle using the official `neo4j` Python driver.

**Responsibilities:**
- Connection initialization from config (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`)
- Health check (`RETURN 1`)
- Session factory (read/write sessions)
- Graceful shutdown (driver close)
- Module-level singleton pattern matching `pecs/store/database.py`

**Connection pattern:**
```
neo4j.GraphDatabase.driver(uri, auth=(user, password))
```

Uses `neo4j://localhost:7687` by default (Bolt protocol).

---

#### [NEW] `pecs/graph/graph_builder.py`

The core deterministic transformation. Reads from SQLite, writes to Neo4j.

**Construction Algorithm (Pseudocode):**

```
def build_graph():
    # 1. Read all entities from SQLite
    requirements = evidence_repo.get_all_requirements()
    implementations = evidence_repo.get_all_implementations()
    evaluations = evidence_repo.get_all_evaluations()
    correlations = correlation_repo.get_all_latest()

    # 2. Build SourceDocument nodes from distinct source_document values
    all_entities = requirements + implementations + evaluations
    source_docs = {(e["source_document"], e.get("source_type_from_metadata"))
                   for e in all_entities}

    # 3. MERGE nodes (idempotent upsert)
    for req in requirements:
        MERGE (:Requirement {entity_id: req.entity_id})
        SET properties...

    for impl in implementations:
        MERGE (:Implementation {entity_id: impl.entity_id})
        SET properties...

    for eval in evaluations:
        MERGE (:Evaluation {entity_id: eval.entity_id})
        SET properties...

    for (name, type) in source_docs:
        MERGE (:SourceDocument {name: name})
        SET source_type = type

    # 4. Build EXTRACTED_FROM relationships
    for entity in all_entities:
        MERGE (entity)-[:EXTRACTED_FROM]->(SourceDocument {name: entity.source_document})

    # 5. Build CORRELATES_TO relationships from correlation table
    for corr in correlations:
        # Skip "(none)" evidence IDs (REQUIREMENT_NOT_IMPLEMENTED)
        if corr.evidence_entity_id == "(none)":
            continue
        MERGE (Requirement {entity_id: corr.requirement_entity_id})
              -[:CORRELATES_TO {correlation_id: corr.correlation_id, ...}]->
              (target node matching corr.evidence_entity_id)

    # 6. Build EXPLICITLY_LINKS from linked_requirement field
    for impl in implementations:
        if impl.linked_requirement:
            MERGE (impl)-[:EXPLICITLY_LINKS]->(Requirement {entity_id: impl.linked_requirement})

    # 7. Build EVALUATES from linked_requirement on evaluations
    for eval in evaluations:
        if eval.linked_requirement:
            MERGE (eval)-[:EVALUATES]->(Requirement {entity_id: eval.linked_requirement})
```

**Key design decisions:**

1. **MERGE, not CREATE.** Every node and relationship write uses Cypher `MERGE` to ensure idempotency. Running the builder twice produces the same graph.

2. **Batch writes via Cypher `UNWIND`.** Instead of one-by-one MERGE calls, nodes are batched:
   ```cypher
   UNWIND $batch AS row
   MERGE (r:Requirement {entity_id: row.entity_id})
   SET r.text = row.text, r.source_document = row.source_document, ...
   ```
   This reduces network round trips from O(n) to O(1) per node type.

3. **Handling `evidence_entity_id` pointing to chunk_ids.** In `CorrelationResult`, when resolved by `strong_semantic_match` (Rule 2), `evidence_entity_id` is set to `best.chunk.chunk_id` — a chunk hash, not an entity_id. The graph builder must detect this case and resolve the chunk_id back to its owning entity via `evidence_repo.get_by_chunk_id()`. If no evidence row owns the chunk (it's a raw EvidenceChunk not yet extracted), create a lightweight "UnextractedChunk" node or skip the edge.

   > [!WARNING]
   > **This is a data model inconsistency in the current codebase.** `CorrelationResult.evidence_entity_id` sometimes holds an `entity_id` (from Rule 1) and sometimes holds a `chunk_id` (from Rule 2 and Stage 2). The graph builder must handle both. Long-term, consider standardizing `evidence_entity_id` to always be an `entity_id` by resolving chunk→entity at correlation time.

4. **SourceDocument `source_type` inference.** The `evidence` table does not store `source_type` directly (it's in the EvidenceChunk/ChromaDB metadata, not the evidence table). The builder will infer source_type from the file extension (`.py` → PYTHON, `.pdf` → PDF) or from ChromaDB metadata if available. This is a minor gap.

---

#### [NEW] `pecs/graph/graph_queries.py`

Pre-defined Cypher queries used by the UI. Not a query engine — just a library of parameterized templates.

**Core queries:**

| Query Name | Purpose | Parameters |
|---|---|---|
| `get_requirement_subgraph` | All nodes within 2 hops of a requirement | `entity_id` |
| `get_all_requirements_overview` | All requirements with their direct connections (count only) | — |
| `get_requirement_evidence_network` | Full evidence chain for a requirement | `entity_id` |
| `get_shared_artifacts` | Artifacts connected to 2+ requirements | — |
| `get_unlinked_entities` | Entities with no relationships | — |
| `get_source_document_coverage` | All entities extracted from a source document | `doc_name` |
| `get_graph_stats` | Node/relationship counts by type | — |

**Example Cypher for `get_requirement_subgraph`:**

```cypher
MATCH path = (r:Requirement {entity_id: $entity_id})-[*1..2]-(connected)
RETURN path
```

**Example Cypher for `get_shared_artifacts`:**

```cypher
MATCH (r1:Requirement)-[:CORRELATES_TO]->(shared)<-[:CORRELATES_TO]-(r2:Requirement)
WHERE r1.entity_id < r2.entity_id
RETURN r1, shared, r2
```

---

#### [NEW] `pecs/graph/graph_sync.py`

Orchestrates the sync between SQLite and Neo4j.

**Sync strategy:**

1. **Full rebuild (V1 default).** Drop all nodes/relationships, rebuild from SQLite. Simple, correct, and fast enough for the expected data volume (tens to low hundreds of entities).

2. **Why full rebuild over incremental sync:** The correlation table is append-only and `get_all_latest()` already handles deduplication. An incremental sync would need to track which correlations have been synced, detect deletions when a re-run produces different results, and handle the `entity_id` instability problem (ARN-1). This complexity is not justified for the expected data scale.

**Sync trigger options:**
- **Manual:** "🔄 Sync Graph" button on the Graph page
- **Automatic:** After the correlation pipeline completes (hook into `_run_correlation` in [correlate_page.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/ui/pages/correlate_page.py#L63))

**Sync operation (atomic):**

```python
def sync_graph(self) -> GraphSyncResult:
    """Full rebuild: clear Neo4j, reconstruct from SQLite."""
    with self._neo4j_client.write_session() as session:
        # 1. Clear existing graph
        session.run("MATCH (n) DETACH DELETE n")

        # 2. Create constraints (idempotent)
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Requirement) REQUIRE r.entity_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Implementation) REQUIRE i.entity_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Evaluation) REQUIRE e.entity_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:SourceDocument) REQUIRE d.name IS UNIQUE")

        # 3. Build nodes and relationships
        builder = GraphBuilder(session, evidence_repo, correlation_repo)
        stats = builder.build()

    return GraphSyncResult(
        nodes_created=stats.node_count,
        relationships_created=stats.rel_count,
        sync_timestamp=datetime.utcnow(),
    )
```

---

### Configuration

#### [MODIFY] [config.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/config.py)

Add Neo4j configuration fields to `PecsSettings`:

```python
# ── Neo4j ────────────────────────────────────────────────────────────
NEO4J_URI: str = "neo4j://localhost:7687"
NEO4J_USER: str = "neo4j"
NEO4J_PASSWORD: str = "neo4j"
NEO4J_DATABASE: str = "neo4j"         # Default database name
NEO4J_ENABLED: bool = True            # Feature flag — graph features disabled if False
```

The `NEO4J_ENABLED` flag allows PECS to function without Neo4j installed (graceful degradation). When disabled, the Graph page shows a setup guide instead of erroring.

#### [MODIFY] `.env.example`

Add Neo4j entries:

```
# Neo4j graph database (optional — graph features)
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j
NEO4J_DATABASE=neo4j
NEO4J_ENABLED=true
```

---

### Dependencies

#### [MODIFY] [pyproject.toml](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pyproject.toml)

Add:
```toml
# Graph database
"neo4j>=5.0",
# Graph visualization
"pyvis>=0.3",
```

`neo4j` is the official Python driver. `pyvis` generates interactive HTML network graphs using vis.js under the hood.

---

### Streamlit UI Integration

#### [NEW] `pecs/ui/pages/graph_page.py`

New page: **🕸️ Evidence Graph**

**Page layout:**

```
┌──────────────────────────────────────────────────────┐
│  🕸️ Evidence Graph                                    │
│                                                      │
│  ┌─ Sidebar Controls ─┐  ┌─ Graph Canvas ──────────┐ │
│  │ Select Requirement ▼│  │                         │ │
│  │ ☐ Show Evaluations  │  │   [Interactive pyvis    │ │
│  │ ☐ Show Source Docs  │  │    network graph]       │ │
│  │ ☐ Show Orphans      │  │                         │ │
│  │ Status Filter    ▼  │  │                         │ │
│  │ Min Confidence ──── │  │                         │ │
│  │ [🔄 Sync Graph]     │  │                         │ │
│  └─────────────────────┘  └─────────────────────────┘ │
│                                                      │
│  ┌─ Node Detail Panel ──────────────────────────────┐ │
│  │ (Appears when a node is clicked)                  │ │
│  │ Entity ID: R3-gnn-training                        │ │
│  │ Text: "The system must implement GNN training..." │ │
│  │ Source: spec.pdf | Page 12                        │ │
│  │ Status: IMPLEMENTED_AND_VALIDATED (0.92)          │ │
│  │ Resolution: deterministic_rule                    │ │
│  │ Connected: 3 implementations, 1 evaluation        │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Interaction model:**

1. **Default view:** Requirement-centric overview. Each requirement is a node, sized by confidence score. Connected artifacts are shown as smaller peripheral nodes. This prevents the "giant unreadable graph" problem.

2. **Requirement selection:** Dropdown or click. When a requirement is selected, the graph zooms to its **ego network** (1–2 hops), showing:
   - The requirement node (center)
   - Connected implementations (with correlation edge properties)
   - Connected evaluations
   - Source documents
   - Other requirements that share the same evidence (shared artifact discovery)

3. **Node click → detail panel:** Clicking a node populates the detail panel below the graph with:
   - Full entity text
   - Source document + locator (provenance)
   - Correlation metadata (status, confidence, reasoning)
   - List of chunk_ids for tracing back to ChromaDB

4. **Edge click → relationship detail:** Clicking an edge shows:
   - Correlation status and confidence
   - Resolution method (deterministic vs LLM)
   - Rule name if deterministic
   - Reasoning text

**Node visual encoding:**

| Node Type | Color | Shape | Size |
|---|---|---|---|
| Requirement | `#3b82f6` (blue) | Circle | Proportional to # connections |
| Implementation | `#22c55e` (green) | Square | Fixed |
| Evaluation | `#f97316` (orange) | Triangle | Fixed |
| SourceDocument | `#6b7280` (gray) | Diamond | Fixed |

**Edge visual encoding:**

| Relationship | Color | Style | Label |
|---|---|---|---|
| CORRELATES_TO (IMPLEMENTED_AND_VALIDATED) | `#22c55e` green | Solid | Status |
| CORRELATES_TO (PARTIALLY_IMPLEMENTED) | `#eab308` yellow | Dashed | Status |
| CORRELATES_TO (NOT_IMPLEMENTED) | `#ef4444` red | Dotted | Status |
| EXPLICITLY_LINKS | `#8b5cf6` purple | Solid | — |
| EVALUATES | `#f97316` orange | Solid | — |
| EXTRACTED_FROM | `#9ca3af` gray | Dotted, thin | — |

**pyvis integration:**

```python
from pyvis.network import Network
import streamlit.components.v1 as components

net = Network(height="600px", width="100%", directed=True, notebook=False)
net.set_options("""{
    "physics": {
        "barnesHut": {"gravitationalConstant": -3000, "springLength": 150}
    },
    "interaction": {"hover": true, "tooltipDelay": 100}
}""")

# Add nodes and edges from Neo4j query results
for node in nodes:
    net.add_node(node.id, label=node.label, color=node.color, ...)

for edge in edges:
    net.add_edge(edge.source, edge.target, title=edge.tooltip, ...)

# Render in Streamlit
html = net.generate_html()
components.html(html, height=620, scrolling=True)
```

**Node click handling:**

pyvis supports `selectNode` events via JavaScript callbacks. The plan is to:
1. Register a JS callback that writes the selected node ID to a hidden HTML element
2. Use Streamlit's `components.html` with bidirectional communication to capture the selection
3. Store the selected node in `st.session_state.selected_graph_node`
4. Re-render the detail panel based on the selection

> [!NOTE]
> Bidirectional communication between pyvis and Streamlit is limited. For V1, an alternative is to use a Streamlit selectbox synchronized with the graph: the user can either click a node (which updates the selectbox via JS) or select from the dropdown (which highlights the node). If the JS bridge proves unreliable, fallback to selectbox-only node inspection.

---

#### [MODIFY] [sidebar.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/ui/components/sidebar.py)

Add the graph page to the navigation and add Neo4j status to the system status section.

Navigation entry (inserted after "📊 Traceability Matrix"):
```python
("🕸️ Evidence Graph", "graph"),
```

Status indicator:
```python
# Neo4j status
if settings.NEO4J_ENABLED:
    try:
        from pecs.graph.neo4j_client import get_neo4j_client
        client = get_neo4j_client()
        if client.check_health():
            stats = client.get_node_count()
            st.success(f"🕸️ Graph: {stats} nodes")
        else:
            st.warning("🕸️ Graph: Offline")
    except Exception:
        st.warning("🕸️ Graph: Error")
```

#### [MODIFY] [app.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/app.py)

Add graph page import and routing:
```python
from pecs.ui.pages import graph_page

# In the routing block:
elif page == "graph":
    graph_page.render()
```

Add session state default:
```python
"selected_graph_node": None,
"graph_synced": False,
```

#### [MODIFY] [correlate_page.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/ui/pages/correlate_page.py)

After correlation completes successfully (line ~132), trigger graph sync if Neo4j is enabled:

```python
# Auto-sync graph after correlation
if settings.NEO4J_ENABLED:
    try:
        from pecs.graph.graph_sync import GraphSync
        sync = GraphSync()
        sync_result = sync.sync_graph()
        st.success(f"🕸️ Graph synced: {sync_result.nodes_created} nodes, "
                   f"{sync_result.relationships_created} relationships")
        st.session_state.graph_synced = True
    except Exception as exc:
        st.warning(f"Graph sync failed (non-blocking): {exc}")
```

#### matrix_page.py — No changes

✅ **Decided:** No "View in Graph" button on the Matrix page. The graph is accessible only from the sidebar navigation.

---

### Identity, Deduplication, and Idempotency

**Node identity rules:**

| Node Type | MERGE Key | Deduplication Guarantee |
|---|---|---|
| Requirement | `entity_id` | Unique per extraction run (see ARN-1 caveat) |
| Implementation | `entity_id` | Unique per extraction run |
| Evaluation | `entity_id` | Unique per extraction run |
| SourceDocument | `name` (filename) | Stable across runs |

**Relationship identity:**

| Relationship | MERGE Key | Notes |
|---|---|---|
| CORRELATES_TO | `(req.entity_id, impl.entity_id)` pair + `correlation_id` | correlation_id is a UUID, unique per run |
| EXPLICITLY_LINKS | `(impl.entity_id, req.entity_id)` pair | Idempotent — same link from same impl |
| EVALUATES | `(eval.entity_id, req.entity_id)` pair | Idempotent |
| EXTRACTED_FROM | `(entity.entity_id, doc.name)` pair | Idempotent |

**Full rebuild strategy:**

Since V1 uses full rebuild (DETACH DELETE + reconstruct), deduplication within a single build is the only concern. The MERGE operations handle this.

**Cross-run deduplication:**

Not attempted in V1. The full rebuild wipes the previous graph entirely. This is acceptable because:
1. The graph is a derived view, not a source of truth.
2. SQLite remains authoritative.
3. Expected data volume is small (undergraduate project with < 100 entities).

---

### Provenance

Every graph node carries full provenance metadata:

- `chunk_id` → traces back to the specific EvidenceChunk in ChromaDB
- `source_document` → the original file
- `created_at` → extraction timestamp
- Relationship `EXTRACTED_FROM` → links entity to its source document node

The detail panel in the UI displays all provenance fields. Users can trace:

```
Graph Node → chunk_id → ChromaDB → normalized_text → source_document + source_locator
```

This chain is complete and auditable without any additional data.

---

### Confidence and Correlation Metadata

Correlation metadata flows directly from `CorrelationResult` properties onto `CORRELATES_TO` relationship properties:

- `status` → one of the 7 `CorrelationStatus` labels
- `confidence` → float 0.0–1.0 from [confidence.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/correlation/confidence.py)
- `resolution_method` → `"deterministic_rule"` or `"llm_stage2"`
- `rule_name` → e.g., `"exact_requirement_id_match"` (null for LLM-resolved)
- `reasoning` → justification text

**Visual encoding:**
- Edge thickness proportional to confidence score
- Edge color determined by status (green/yellow/red palette matching [STATUS_COLORS](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/traceability/matrix.py#L36-L44))
- Tooltip shows full metadata on hover

---

### Error Handling and Consistency

**Neo4j unavailability:**

The graph layer is non-critical. If Neo4j is down:
1. `NEO4J_ENABLED = false` disables all graph features → no errors
2. If enabled but unreachable, the graph page shows a connection error with retry button
3. The correlation pipeline never blocks on graph sync failure — sync errors are logged and reported as warnings, not exceptions
4. The sidebar status indicator shows "🕸️ Graph: Offline"

**Data consistency:**

SQLite is the source of truth. The graph is always reconstructible from SQLite. If the graph becomes inconsistent (e.g., partial sync failure), the user can click "🔄 Rebuild Graph" to do a full resync.

**Transaction handling:**

The full rebuild runs inside a single Neo4j transaction. If any MERGE fails, the entire rebuild is rolled back, and the previous graph state is preserved (since DETACH DELETE is inside the same transaction).

> [!WARNING]
> **Transaction size limit.** Neo4j has a default transaction memory limit (e.g., `dbms.memory.transaction.max_size`). For a typical PECS project with < 500 nodes and < 2000 relationships, this is not a concern. If PECS scales to thousands of entities, the rebuild should be batched into multiple transactions.

---

### Logging

All graph operations use the existing PECS logging pattern ([logging_config.py](file:///home/arjun/Arjun/AnderBahar/LocalRAG/pecs/logging_config.py)):

```python
from pecs.logging_config import get_logger
logger = get_logger(__name__)
```

**Log events:**

| Event | Level | Context Fields |
|---|---|---|
| Neo4j connection established | INFO | `uri`, `database` |
| Neo4j connection failed | ERROR | `uri`, `error` |
| Graph sync started | INFO | `trigger` ("manual" / "auto") |
| Graph sync completed | INFO | `nodes_created`, `relationships_created`, `duration_ms` |
| Graph sync failed | ERROR | `error`, `stage` |
| Node MERGE batch completed | DEBUG | `node_type`, `count` |
| Relationship MERGE batch completed | DEBUG | `rel_type`, `count` |
| Graph query executed | DEBUG | `query_name`, `params`, `result_count`, `duration_ms` |
| `evidence_entity_id` resolved as chunk_id | WARNING | `correlation_id`, `chunk_id` |

---

### Testing

#### [NEW] `tests/unit/test_graph_builder.py`

Tests the deterministic transformation logic without a real Neo4j instance.

**Strategy:** Mock the Neo4j session. Assert that the correct Cypher statements are generated with the correct parameters.

**Test cases:**

1. **`test_requirements_become_requirement_nodes`** — Given 3 requirement rows from SQLite, assert 3 MERGE statements with `:Requirement` label.

2. **`test_correlations_become_edges`** — Given a CorrelationResult linking req A to impl B, assert a `CORRELATES_TO` relationship is created with correct properties.

3. **`test_none_evidence_id_skipped`** — Given a CorrelationResult with `evidence_entity_id = "(none)"`, assert no CORRELATES_TO edge is created.

4. **`test_linked_requirement_creates_explicit_link`** — Given an Implementation with `linked_requirement = "R3"`, assert an `EXPLICITLY_LINKS` relationship is created.

5. **`test_orphan_evaluation_handling`** — Given an Evaluation with no `linked_requirement`, assert no `EVALUATES` edge is created.

6. **`test_source_documents_deduplicated`** — Given 5 entities from 2 source documents, assert only 2 SourceDocument nodes are created.

7. **`test_chunk_id_as_evidence_entity_id`** — Given a CorrelationResult where `evidence_entity_id` is a chunk hash (64 hex chars), assert the builder resolves it to the owning entity.

8. **`test_idempotent_rebuild`** — Run the builder twice with the same data, assert the Cypher MERGE calls are identical.

#### [NEW] `tests/unit/test_graph_queries.py`

Tests that query templates are valid Cypher and produce expected parameters.

#### [NEW] `tests/integration/test_graph_sync.py`

Integration test requiring a running Neo4j instance (skipped in CI if unavailable).

**Test:**
1. Insert sample data into SQLite (in-memory DB)
2. Run graph sync
3. Query Neo4j and assert node/relationship counts match expectations
4. Run sync again — assert counts are unchanged (idempotency)

Mark with `@pytest.mark.integration` and skip if Neo4j is unavailable:
```python
@pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not running")
```

---

### Performance Considerations

| Concern | Expected Scale | Mitigation |
|---|---|---|
| Node count | < 200 entities for a typical undergraduate project | No concern |
| Relationship count | < 500 edges | No concern |
| Graph build time | < 2 seconds for full rebuild | Acceptable for manual trigger |
| pyvis render time | < 1 second for < 200 nodes | No concern |
| Neo4j memory | < 50 MB for this data volume | Default config sufficient |
| Cypher query time | < 100ms for ego network queries | Indexes on `entity_id` and `name` (via UNIQUE constraints) |

**If scale increases beyond expectations:**
1. Batch MERGE operations using `UNWIND` (already planned)
2. Add pagination to graph queries
3. Switch from full rebuild to incremental sync
4. Use Neo4j APOC library for bulk imports

**pyvis DOM performance:**
For > 500 nodes, pyvis/vis.js can become sluggish. The ego-network approach (showing only 1–2 hops from a selected requirement) naturally limits the rendered node count to < 30 per view, avoiding this issue entirely.

---

## Verification Plan

### Automated Tests

```bash
# Unit tests (no Neo4j required)
pytest tests/unit/test_graph_builder.py tests/unit/test_graph_queries.py -v

# Integration tests (requires Neo4j running)
pytest tests/integration/test_graph_sync.py -v -m integration
```

### Manual Verification

1. **Start Neo4j** (Docker: `docker run -p 7474:7474 -p 7687:7687 neo4j:community`)
2. **Run the PECS pipeline** end-to-end: Ingest → Extract → Correlate
3. **Navigate to 🕸️ Evidence Graph** page
4. **Verify graph renders** with correct node types and colors
5. **Click a requirement node** → verify detail panel shows correct text, source, and correlation info
6. **Click a relationship edge** → verify tooltip shows status, confidence, reasoning
7. **Select a different requirement** from the dropdown → verify graph updates to that requirement's ego network
8. **Click "View in Graph"** from the Matrix page → verify navigation to graph with correct node selected
9. **Stop Neo4j** → verify the rest of PECS still works, graph page shows graceful error
10. **Restart Neo4j and click "Rebuild Graph"** → verify full resync succeeds

---

## Recommended Implementation Sequence

### Phase 1: Foundation (Graph backend)

| Step | Files | Description |
|---|---|---|
| 1.1 | `pyproject.toml` | Add `neo4j` and `pyvis` dependencies |
| 1.2 | `pecs/config.py`, `.env.example` | Add Neo4j config fields |
| 1.3 | `pecs/graph/__init__.py` | Create package |
| 1.4 | `pecs/graph/neo4j_client.py` | Connection management + health check |
| 1.5 | `pecs/graph/graph_builder.py` | Deterministic construction algorithm |
| 1.6 | `pecs/graph/graph_sync.py` | Full rebuild orchestration |
| 1.7 | `tests/unit/test_graph_builder.py` | Unit tests for builder logic |

### Phase 2: Query layer

| Step | Files | Description |
|---|---|---|
| 2.1 | `pecs/graph/graph_queries.py` | Cypher query templates |
| 2.2 | `tests/unit/test_graph_queries.py` | Query template tests |

### Phase 3: UI

| Step | Files | Description |
|---|---|---|
| 3.1 | `pecs/ui/pages/graph_page.py` | New graph page with pyvis rendering |
| 3.2 | `pecs/ui/components/sidebar.py` | Add graph nav + Neo4j status |
| 3.3 | `pecs/app.py` | Add graph page routing |
| 3.4 | `pecs/ui/pages/correlate_page.py` | Auto-sync after correlation |
| 3.5 | `pecs/ui/pages/matrix_page.py` | "View in Graph" button |

### Phase 4: Integration testing

| Step | Files | Description |
|---|---|---|
| 4.1 | `tests/integration/test_graph_sync.py` | End-to-end sync test |
| 4.2 | Manual testing | Full pipeline walkthrough |

---

## V1 Scope vs Future Extensions

### V1 Scope (This Plan)

- [x] Neo4j connection management with feature flag
- [x] Full graph rebuild from SQLite
- [x] 4 node types: Requirement, Implementation, Evaluation, SourceDocument
- [x] 4 relationship types: CORRELATES_TO, EXPLICITLY_LINKS, EVALUATES, EXTRACTED_FROM
- [x] Interactive pyvis visualization in Streamlit
- [x] Ego-network view per requirement
- [x] Node/edge click → detail panel
- [x] Auto-sync after correlation
- [x] "Rebuild Graph" manual trigger
- [x] Graceful degradation when Neo4j is unavailable
- [x] Unit tests for builder logic
- [x] Integration test for sync

### Future Extensions (Not in V1)

| Extension | Description | Prerequisite |
|---|---|---|
| **Incremental sync** | Only push deltas instead of full rebuild | Sync tracking table in SQLite |
| **Temporal graph** | Navigate historical correlation runs | Versioned graph snapshots or Neo4j temporal properties |
| **Structured Git nodes** | First-class `Commit` nodes with SHA, author, message | Git-specific entity extraction in Stage 1 |
| **Cross-requirement path analysis** | "What connects Requirement A to Requirement B?" | Cypher path queries |
| **Graph-based gap detection** | "Which requirements have no evaluation?" as a graph query | Already partially possible via `MATCH (r:Requirement) WHERE NOT (r)<-[:EVALUATES]-() RETURN r` |
| **Export graph to PNG/SVG** | Downloadable graph images | pyvis `write_html` + headless browser screenshot |
| **Neovis.js frontend** | Replace pyvis with direct browser↔Neo4j visualization | Expose Bolt port, handle auth in frontend |
| **Entity deduplication** | Merge near-identical entity_ids (e.g., "R3-gnn-training" and "R3_gnn_training") | Fuzzy matching or embedding-based dedup |
| **Confidence-filtered graph views** | Only show edges above a confidence threshold | Cypher `WHERE r.confidence >= $threshold` (trivial to add) |
