"""Runtime execution state — particles, beliefs, trace tree."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(slots=True)
class Particle:
    brief: str
    weight: float = 1.0
    evidence: str = ""


@dataclass(slots=True)
class Belief:
    particles: list[Particle] = field(default_factory=list)

    def normalize(self) -> None:
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total

    def ess(self) -> float:
        self.normalize()
        sum_sq = sum(p.weight ** 2 for p in self.particles)
        if sum_sq == 0:
            return 0.0
        return 1.0 / sum_sq

    def resample(self, n: int | None = None, rng: random.Random | None = None) -> None:
        """Systematic resampling — pure Python, no numpy."""
        rng = rng or random.Random()
        if not self.particles:
            return
        self.normalize()
        m = n if n is not None else len(self.particles)
        weights = [p.weight for p in self.particles]
        cumulative = []
        acc = 0.0
        for w in weights:
            acc += w
            cumulative.append(acc)

        u0 = rng.random() / m
        indices: list[int] = []
        i = 0
        for j in range(m):
            threshold = u0 + j / m
            while i < len(cumulative) - 1 and cumulative[i] < threshold:
                i += 1
            indices.append(i)

        old = self.particles
        self.particles = [
            Particle(brief=old[idx].brief, weight=1.0 / m, evidence=old[idx].evidence)
            for idx in indices
        ]


@dataclass(slots=True)
class TraceNode:
    node_id: str
    checkpoint_id: str = ""
    alive: bool = True
    children: list[TraceNode] = field(default_factory=list)
    summary: str = ""

    def mark_dead(self, target_id: str) -> bool:
        if self.node_id == target_id:
            self.alive = False
            return True
        for child in self.children:
            if child.mark_dead(target_id):
                return True
        return False

    def collect_summaries(self) -> list[str]:
        result: list[str] = []
        if self.summary:
            result.append(self.summary)
        for child in self.children:
            result.extend(child.collect_summaries())
        return result


@dataclass(slots=True)
class TraceTree:
    root: TraceNode

    def mark_dead(self, node_id: str) -> bool:
        return self.root.mark_dead(node_id)

    def summarize(self) -> str:
        summaries = self.root.collect_summaries()
        return "; ".join(summaries) if summaries else ""

    def add_step(self, node_id: str, checkpoint_id: str = "") -> TraceNode:
        node = TraceNode(node_id=node_id, checkpoint_id=checkpoint_id)
        self._find_leaf(self.root).children.append(node)
        return node

    def _find_leaf(self, node: TraceNode) -> TraceNode:
        if not node.children:
            return node
        for child in reversed(node.children):
            if child.alive:
                return self._find_leaf(child)
        return node


@dataclass(slots=True)
class ExecutionState:
    pointer: str
    belief: Belief
    trace: TraceTree
    step: int = 0
    budget_remaining: int = 50
    user_input: str = ""
    node_outputs: dict[str, str] = field(default_factory=dict)
