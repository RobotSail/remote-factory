"""Intermediate representation — static workflow graph structure."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass(slots=True)
class NodeSpec:
    id: str
    spec: str
    theta_prior: str
    tools: list[str] = field(default_factory=list)
    effect: Literal["pure", "effectful"] = "pure"
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)


@dataclass(slots=True)
class EdgeSpec:
    source: str
    target: str
    condition: str | None = None


@dataclass(slots=True)
class WorkflowSpec:
    name: str
    nodes: list[NodeSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    entry: str = ""

    def validate(self) -> list[str]:
        node_ids = {n.id for n in self.nodes}
        issues: list[str] = []
        if self.entry and self.entry not in node_ids:
            issues.append(f"entry '{self.entry}' not in nodes")
        for e in self.edges:
            if e.source not in node_ids:
                issues.append(f"edge source '{e.source}' not in nodes")
            if e.target not in node_ids:
                issues.append(f"edge target '{e.target}' not in nodes")
        return issues

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> WorkflowSpec:
        d = json.loads(s)
        nodes = [NodeSpec(**n) for n in d.get("nodes", [])]
        edges = [EdgeSpec(**e) for e in d.get("edges", [])]
        return cls(
            name=d["name"],
            nodes=nodes,
            edges=edges,
            entry=d.get("entry", ""),
        )
