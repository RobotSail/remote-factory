"""Wrapped B2 runner — claude --bare with engine logic in the wrapper.

Claude never sees pfexec machinery. The SKILL.md is identical to the
factory baseline. All belief tracking, observe, and fork decisions
happen in the wrapper between (potentially multiple) --bare calls.
"""

from __future__ import annotations

import subprocess

from pfexec.dist.cc.factory_baseline import (
    _extract_final_answer,
    generate_skill_md,
    parse_skill_output,
)
from pfexec.dist.cc.skill_gen import _terminal_nodes, _topo_order
from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import WorkflowSpec
from pfexec.llm import DeterministicBackend, get_backend
from pfexec.primitives import fork, init as pfexec_init, observe, observe_sequential, observe_rewind, observe_lightweight
from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


def _call_claude_bare(skill_md: str, user_prompt: str, timeout: int = 600) -> str:
    """Single claude --bare call. Returns stdout."""
    result = subprocess.run(
        ["claude", "--bare",
         "--system-prompt", skill_md,
         "-p", user_prompt],
        capture_output=True, text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude failed (exit {result.returncode}): {result.stderr}")
    return result.stdout.strip()


def _suffix_score(belief: Belief, k: int = 3) -> float:
    """Top-k average weight — same scoring as engine.py."""
    if not belief.particles:
        return 0.0
    belief.normalize()
    weights = sorted((p.weight for p in belief.particles), reverse=True)
    top_k = weights[:k]
    return sum(top_k) / len(top_k) if top_k else 0.0


def _mock_output(order: list[str]) -> str:
    """Generate structured mock output with ### Output: markers."""
    sections = [f"### Output: {nid}\nmock output" for nid in order]
    sections.append("### Final Answer\nmock output")
    return "\n".join(sections)


def _extract_lesson(state: ExecutionState, config: EngineConfig, failed_output: str) -> str:
    if config.observe_mode == 'none':
        return failed_output[:300] if failed_output else 'Try a different approach.'
    elif config.observe_mode == 'sequential':
        if state.evidence_seq:
            last = state.evidence_seq[-1]
            return last.get('output', '')[:300] or 'Try a different approach.'
        return failed_output[:300] if failed_output else 'Try a different approach.'
    elif config.observe_mode == 'rewind':
        if state.belief.particles:
            brief = state.belief.particles[0].brief
            if brief:
                return brief
        return 'Try a different approach.'
    elif config.observe_mode == 'lightweight':
        best = max(state.belief.particles, key=lambda p: p.weight) if state.belief.particles else None
        if best and best.evidence:
            return best.evidence[-300:]
        return 'Try a different approach.'
    else:  # full
        best = max(state.belief.particles, key=lambda p: p.weight) if state.belief.particles else None
        if best and best.brief:
            return best.brief
        return 'Try a different approach.'


def run(workflow: WorkflowSpec, user_input: str, config: EngineConfig,
        backend_mode: str = "claude") -> EngineResult:
    """Run workflow as claude --bare with engine logic in the wrapper."""
    backend = get_backend(backend_mode)

    # 1. Init particles
    state = pfexec_init(workflow, user_input, config.n_particles, backend)
    state.budget_remaining = config.max_steps

    order = _topo_order(workflow)
    node_map = {n.id: n for n in workflow.nodes}
    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]

    # 2. Generate SKILL.md — IDENTICAL to factory baseline
    skill_md = generate_skill_md(workflow)

    # 3. First claude --bare call
    user_prompt = f"Execute the workflow for the following input:\n\n{user_input}"
    if backend_mode == "mock":
        raw_output = _mock_output(order)
    else:
        raw_output = _call_claude_bare(skill_md, user_prompt)

    # 4. Parse output — extract per-node sections
    node_outputs = parse_skill_output(raw_output, workflow)
    final_answer = _extract_final_answer(raw_output)

    # 5. For each parsed node: run observe + fork-check
    forks_triggered = 0
    fork_at_phase = -1

    for i, nid in enumerate(order):
        if nid not in node_outputs:
            continue

        output_text = node_outputs[nid]
        state.node_outputs[nid] = output_text

        if config.observe_mode == 'none':
            pass
        elif config.observe_mode == 'sequential':
            state = observe_sequential(state, output_text, nid)
        elif config.observe_mode == 'rewind':
            state = observe_rewind(state, output_text, backend)
        elif config.observe_mode == 'lightweight':
            state = observe_lightweight(state, output_text)
        else:  # 'full' — default
            if config.n_particles > 1:
                state = observe(state, output_text, backend)

        state.step += 1
        state.budget_remaining -= 1

        node = node_map[nid]
        if config.observe_mode == 'none':
            pass
        elif (node.effect == "effectful"
                and forks_triggered < config.max_forks
                and _suffix_score(state.belief) < config.tau):
            state = fork(state, config.rewind_steps, backend)
            forks_triggered += 1
            fork_at_phase = i
            break

    # 6. If fork triggered: second claude --bare call with context
    if fork_at_phase >= 0:
        prior_context_parts = []
        for j, nid in enumerate(order):
            if j >= fork_at_phase:
                break
            if nid in node_outputs:
                prior_context_parts.append(f"### Output: {nid}\n{node_outputs[nid]}")

        prior_context = "\n\n".join(prior_context_parts)

        failed_nid = order[fork_at_phase]
        failed_output = node_outputs.get(failed_nid, '')
        lesson = _extract_lesson(state, config, failed_output)
        resume_prompt = (
            f"Execute the workflow for the following input:\n\n{user_input}\n\n"
            f"--- Prior attempt (phases completed so far) ---\n\n"
            f"{prior_context}\n\n"
            f"--- Revision needed ---\n\n"
            f"Phase {fork_at_phase + 1} ({failed_nid}) produced a low-confidence result. "
            f"Lesson from analysis: {lesson}\n"
            f"Re-execute from phase {fork_at_phase + 1} ({failed_nid}) onward, "
            f"incorporating the lesson above. "
            f"Keep all prior phase outputs unchanged."
        )

        if backend_mode == "mock":
            raw_output_2 = _mock_output(order)
        else:
            raw_output_2 = _call_claude_bare(skill_md, resume_prompt)

        node_outputs_2 = parse_skill_output(raw_output_2, workflow)
        final_answer_2 = _extract_final_answer(raw_output_2)

        for nid in order[fork_at_phase:]:
            if nid in node_outputs_2:
                node_outputs[nid] = node_outputs_2[nid]
                state.node_outputs[nid] = node_outputs_2[nid]

        if final_answer_2:
            final_answer = final_answer_2

    # 7. Determine final answer
    if not final_answer:
        for nid in reversed(order):
            if nid in node_outputs:
                final_answer = node_outputs[nid]
                break
    if not final_answer and raw_output:
        final_answer = raw_output.split("\n")[-1].strip()

    all_outputs = [node_outputs[nid] for nid in order if nid in node_outputs]
    steps_taken = len(node_outputs)

    state.step = steps_taken
    state.budget_remaining = config.max_steps - steps_taken

    return EngineResult(
        final_state=state,
        output=final_answer,
        steps_taken=steps_taken,
        forks_triggered=forks_triggered,
        terminated_by="complete",
        all_outputs=all_outputs,
    )
