# Testing

Run the Python suite from the repository root:

```bash
uv run --locked pytest -q
```

Graph tests use temporary SQLite data and synthetic source chunks. They do not
need Ollama, Chroma, or a graph server. The old Neo4j tests and opt-in flags
were removed with the Neo4j adapter.

```bash
cd pecs/ui/components/trace_graph_frontend
npm ci
npm test
npm run build
```

The graph frontend is locally bundled. Application users need no Node runtime
or remote script. After changing packaging, verify the built wheel contains
`pecs/ui/components/trace_graph_frontend/dist/`.
The pinned frontend packages are Cytoscape.js 3.30.4 (MIT),
streamlit-component-lib 2.0.0 (Apache-2.0), and Vite 6.3.5 (MIT; build only).
