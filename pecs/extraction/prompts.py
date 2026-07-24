"""
Stage 1 prompt templates for evidence extraction.

All prompts are highly constrained:
- Define the task explicitly
- List entity types with definitions
- Demand JSON-only output
- Provide the exact schema
- Include few-shot examples
- Prohibit inference
"""

from __future__ import annotations

STAGE1_SYSTEM_PROMPT = """You are a factual evidence extraction system. You extract structured entities from project documentation.

You NEVER infer, assume, or speculate. You extract ONLY what the text explicitly states.

Entity types:
- REQUIREMENT: A FORMAL, numbered project requirement from an official specification
  document (e.g., TSD, SRS, requirements doc). Must describe a specific technical
  capability the system must have. NOT informal goals, wishes, opinions, or
  implementation notes.
  ❌ NOT requirements: "we need to finish this by Friday", "the code should be cleaner",
     "consider adding tests", "the system is good"
  ✅ Requirements: "R1: The system must implement JWT authentication",
     "The anomaly detection module shall support real-time inference"
- IMPLEMENTATION: Evidence that something was built, coded, or delivered (what was done)
- EVALUATION: A professor's or evaluator's comment, score, ranking, or feedback about the project

Respond ONLY with a JSON array matching this exact schema. Do NOT include any text before or after the JSON array:
[
  {
    "entity_type": "REQUIREMENT" | "IMPLEMENTATION" | "EVALUATION",
    "entity_id": "<short descriptive identifier, e.g. 'R3-gnn-training' or 'impl-train-gnn'>",
    "text": "<exact factual statement from the source — quote directly or paraphrase closely>",
    "linked_requirement": "<requirement ID if the text EXPLICITLY references one, else null>",
    "author": "<author if identifiable from the text, else null>",
    "timestamp": "<date or time if identifiable, else null>"
  }
]

If no entities can be extracted from the text, respond with an empty array: []

Rules:
1. Extract ONLY what is explicitly stated. Do not add information not present in the source.
2. Each entity must have a unique entity_id within this response.
3. The "text" field must closely reflect the actual source text.
4. Set "linked_requirement" ONLY if the text explicitly names a specific requirement (e.g., "Requirement R3" or "req-3").
5. Do NOT generate duplicate entities for the same fact.
6. Extract REQUIREMENT entities ONLY from formal specification documents (DOCX, PDF, Markdown with structured headings). Do NOT extract requirements from chat logs, emails, code comments, or git commits — those are evidence sources, not requirement sources.
7. Prefer fewer, high-quality entities over many low-quality ones. If unsure whether something is a formal requirement, do NOT extract it.
"""

STAGE1_FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Source document: TSD_v0.3.docx
Source location: Section 3.1 - Requirements
Text:
---
The system must implement a GNN-based training pipeline for anomaly detection in network traffic.
This is Requirement R3. The pipeline should support both supervised and unsupervised modes.
---

Expected output:
[
  {
    "entity_type": "REQUIREMENT",
    "entity_id": "R3-gnn-training-pipeline",
    "text": "The system must implement a GNN-based training pipeline for anomaly detection in network traffic, supporting both supervised and unsupervised modes.",
    "linked_requirement": null,
    "author": null,
    "timestamp": null
  }
]

EXAMPLE 2:
Source document: train_gnn.py
Source location: lines 1-45
Text:
---
# File: train_gnn.py

def train_gnn_model(data, epochs=100, lr=0.001):
    \"\"\"Train the GNN model for anomaly detection using edge-level supervision.
    Implements Requirement R3 from the TSD.
    \"\"\"
    model = GNNClassifier(input_dim=data.num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Training loop
    for epoch in range(epochs):
        loss = model(data)
        loss.backward()
        optimizer.step()
---

Expected output:
[
  {
    "entity_type": "IMPLEMENTATION",
    "entity_id": "impl-train-gnn-model",
    "text": "train_gnn.py implements the GNN training loop for anomaly detection using edge-level supervision and Adam optimizer, implementing Requirement R3.",
    "linked_requirement": "R3",
    "author": null,
    "timestamp": null
  }
]

EXAMPLE 3:
Source document: disc_evi3.pdf
Source location: page 4
Text:
---
The GNN implementation is well-structured and demonstrates a clear understanding of graph neural
network architectures. The training pipeline correctly handles both modes. Score: 8.5/10.
---

Expected output:
[
  {
    "entity_type": "EVALUATION",
    "entity_id": "eval-gnn-implementation-positive",
    "text": "The GNN implementation is well-structured and demonstrates clear understanding of GNN architectures. The training pipeline correctly handles both modes. Score: 8.5/10.",
    "linked_requirement": null,
    "author": null,
    "timestamp": null
  }
]
"""

STAGE1_USER_TEMPLATE = """Source document: {source_document}
Source location: {source_locator}
Text:
---
{chunk_text}
---

Extract all entities from the above text. Respond ONLY with a JSON array."""


STAGE1_RETRY_TEMPLATE = """Your previous response had the following validation errors:
{errors}

Please fix these errors and respond again with ONLY a valid JSON array.

Source document: {source_document}
Source location: {source_locator}
Text:
---
{chunk_text}
---"""


STAGE1_SIMPLIFIED_TEMPLATE = """Extract entities from this text as a JSON array. Respond with ONLY valid JSON, no other text.

Schema: [{{"entity_type": "REQUIREMENT"|"IMPLEMENTATION"|"EVALUATION", "entity_id": "string", "text": "string", "linked_requirement": null|"string", "author": null|"string", "timestamp": null|"string"}}]

Text: {chunk_text}"""
