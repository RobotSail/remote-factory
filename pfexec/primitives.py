"""Core inference primitives — init, sample, observe, fork."""

from __future__ import annotations

import json
import random
from dataclasses import replace

from pfexec.ir import NodeSpec, WorkflowSpec
from pfexec.llm import LLMBackend
from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


def init(
    workflow: WorkflowSpec,
    user_input: str,
    n_particles: int,
    backend: LLMBackend,
    rng: random.Random | None = None,
) -> ExecutionState:
    rng = rng or random.Random()
    prompt = (
        f"You are generating diverse execution strategies for a workflow.\n"
        f"Workflow: {workflow.name}\n"
        f"Input: {user_input}\n"
        f"Generate {n_particles} diverse, concise execution plan briefs "
        f"as a JSON array of strings."
    )
    raw = backend.call(prompt)
    try:
        briefs = json.loads(raw)
        if not isinstance(briefs, list):
            briefs = [raw]
    except (json.JSONDecodeError, TypeError):
        briefs = [raw]

    while len(briefs) < n_particles:
        briefs.append(f"plan-{len(briefs)}")
    briefs = briefs[:n_particles]

    particles = [Particle(brief=b, weight=1.0 / n_particles) for b in briefs]
    belief = Belief(particles=particles)
    trace = TraceTree(root=TraceNode(node_id=workflow.entry, checkpoint_id="init"))
    return ExecutionState(
        pointer=workflow.entry,
        belief=belief,
        trace=trace,
        step=0,
        budget_remaining=50,
    )


def sample(
    state: ExecutionState,
    node: NodeSpec,
    backend: LLMBackend,
    rng: random.Random | None = None,
) -> tuple[ExecutionState, str]:
    rng = rng or random.Random()
    state.belief.normalize()
    weights = [p.weight for p in state.belief.particles]
    chosen = rng.choices(state.belief.particles, weights=weights, k=1)[0]

    prompt = node.theta_prior.replace("{input}", chosen.brief)
    if node.effect == "effectful":
        prompt = f"[EFFECTFUL] {prompt}"

    output = backend.call(prompt, system=node.spec)

    new_state = replace(
        state,
        step=state.step + 1,
        budget_remaining=state.budget_remaining - 1,
    )
    new_state.trace.add_step(node.id, checkpoint_id=f"step-{new_state.step}")
    return new_state, output


def observe(
    state: ExecutionState,
    observation: str,
    backend: LLMBackend,
) -> ExecutionState:
    particles = state.belief.particles
    n = len(particles)
    if n < 2:
        return state

    wins = [0.0] * n
    total_comparisons = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            prompt = (
                f"Compare two execution plans against this observation.\n"
                f"Observation: {observation}\n"
                f"Plan A: {particles[i].brief}\n"
                f"Plan B: {particles[j].brief}\n"
                f"Which plan better explains the observation? Reply 'A' or 'B'."
            )
            result = backend.call(prompt)
            if "A" in result.upper().split()[0] if result.strip() else False:
                wins[i] += 1.0
            else:
                wins[j] += 1.0
            total_comparisons[i] += 1
            total_comparisons[j] += 1

    for i in range(n):
        if total_comparisons[i] > 0:
            win_rate = wins[i] / total_comparisons[i]
            particles[i].weight *= (0.5 + win_rate)
        particles[i].evidence += f" | {observation}"

    state.belief.normalize()

    if state.belief.ess() < n / 2:
        state.belief.resample()

    return state


def fork(
    state: ExecutionState,
    k: int,
    backend: LLMBackend,
    rng: random.Random | None = None,
) -> ExecutionState:
    rng = rng or random.Random()
    state.trace.mark_dead(state.pointer)
    dead_summary = state.trace.summarize()

    summary_prompt = (
        f"Summarize what went wrong in this execution branch.\n"
        f"Trace: {dead_summary}\n"
        f"Provide a concise lesson learned."
    )
    lesson = backend.call(summary_prompt)

    ancestors = _trace_ancestors(state.trace.root, state.pointer)
    rewind_target = state.pointer
    if len(ancestors) > k:
        rewind_target = ancestors[-(k + 1)]
    elif ancestors:
        rewind_target = ancestors[0]

    n = len(state.belief.particles)
    rejuv_prompt = (
        f"Generate {n} fresh execution plan briefs.\n"
        f"Lesson from failed branch: {lesson}\n"
        f"Avoid the same mistakes. Return a JSON array of strings."
    )
    raw = backend.call(rejuv_prompt)
    try:
        briefs = json.loads(raw)
        if not isinstance(briefs, list):
            briefs = [raw]
    except (json.JSONDecodeError, TypeError):
        briefs = [raw]

    while len(briefs) < n:
        briefs.append(f"rejuv-{len(briefs)}")
    briefs = briefs[:n]

    new_particles = [Particle(brief=b, weight=1.0 / n) for b in briefs]
    state.belief.particles = new_particles
    state.pointer = rewind_target
    return state


def _trace_ancestors(node: TraceNode, target_id: str) -> list[str]:
    if node.node_id == target_id:
        return [node.node_id]
    for child in node.children:
        path = _trace_ancestors(child, target_id)
        if path:
            return [node.node_id] + path
    return []
