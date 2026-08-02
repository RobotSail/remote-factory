"""DAG execution loop — walks the workflow graph with fork triggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pfexec.ir import WorkflowSpec
from pfexec.llm import LLMBackend
from pfexec.primitives import fork, init, observe, sample
from pfexec.state import Belief, ExecutionState


@dataclass(slots=True)
class EngineConfig:
    n_particles: int = 5
    tau: float = 0.3
    max_steps: int = 50
    max_forks: int = 3
    rewind_steps: int = 2


@dataclass(slots=True)
class EngineResult:
    final_state: ExecutionState
    output: str
    steps_taken: int
    forks_triggered: int
    terminated_by: Literal["complete", "budget", "max_forks"]
    all_outputs: list[str] = field(default_factory=list)


def run(
    workflow: WorkflowSpec,
    user_input: str,
    backend: LLMBackend,
    config: EngineConfig | None = None,
) -> EngineResult:
    cfg = config or EngineConfig()
    state = init(workflow, user_input, cfg.n_particles, backend)
    state.budget_remaining = cfg.max_steps

    node_map = {n.id: n for n in workflow.nodes}
    outputs: list[str] = []
    forks_triggered = 0
    visited: set[str] = set()

    current = state.pointer
    while current and state.budget_remaining > 0:
        if current in visited and current not in _has_incoming_from_unvisited(workflow, visited):
            break
        visited.add(current)

        node = node_map[current]
        state, output = sample(state, node, backend)
        outputs.append(output)
        state = observe(state, output, backend)

        score = _suffix_score(state.belief)
        if node.effect == "effectful" and score < cfg.tau and forks_triggered < cfg.max_forks:
            state = fork(state, cfg.rewind_steps, backend)
            forks_triggered += 1
            current = state.pointer
            visited.discard(current)
            continue

        if node.effect == "effectful" and forks_triggered >= cfg.max_forks and score < cfg.tau:
            terminal = _terminal_nodes(workflow)
            terminal_output = ""
            for tid in terminal:
                if tid in state.node_outputs:
                    terminal_output = state.node_outputs[tid]
                    break
            if not terminal_output and outputs:
                terminal_output = outputs[-1]
            return EngineResult(
                final_state=state,
                output=terminal_output,
                steps_taken=cfg.max_steps - state.budget_remaining,
                forks_triggered=forks_triggered,
                terminated_by="max_forks",
                all_outputs=outputs,
            )

        successors = _topological_successors(workflow, current)
        current = successors[0] if successors else None

    terminated_by: Literal["complete", "budget", "max_forks"]
    if state.budget_remaining <= 0:
        terminated_by = "budget"
    else:
        terminated_by = "complete"

    terminal = _terminal_nodes(workflow)
    terminal_output = ""
    for tid in terminal:
        if tid in state.node_outputs:
            terminal_output = state.node_outputs[tid]
            break
    if not terminal_output and outputs:
        terminal_output = outputs[-1]

    return EngineResult(
        final_state=state,
        output=terminal_output,
        steps_taken=cfg.max_steps - state.budget_remaining,
        forks_triggered=forks_triggered,
        terminated_by=terminated_by,
        all_outputs=outputs,
    )


def _topological_successors(workflow: WorkflowSpec, node_id: str) -> list[str]:
    return [e.target for e in workflow.edges if e.source == node_id]


def _suffix_score(belief: Belief, k: int = 3) -> float:
    if not belief.particles:
        return 0.0
    belief.normalize()
    weights = sorted((p.weight for p in belief.particles), reverse=True)
    top_k = weights[:k]
    return sum(top_k) / len(top_k) if top_k else 0.0


def _has_incoming_from_unvisited(workflow: WorkflowSpec, visited: set[str]) -> set[str]:
    result: set[str] = set()
    for e in workflow.edges:
        if e.source not in visited:
            result.add(e.target)
    return result


def _terminal_nodes(workflow: WorkflowSpec) -> list[str]:
    sources = {e.source for e in workflow.edges}
    return [n.id for n in workflow.nodes if n.id not in sources]
