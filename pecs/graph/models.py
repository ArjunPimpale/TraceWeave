"""Serializable canvas objects; IDs are namespaced record keys."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class GraphNode:
    id: str
    kind: str
    title: str
    subtitle: str = ""
    status: str | None = None
    x: float = 0
    y: float = 0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphEdge:
    id: str
    source: str
    target: str
    kind: str
    label: str
    assessment_ids: list[str] = field(default_factory=list)
    reference_ids: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
