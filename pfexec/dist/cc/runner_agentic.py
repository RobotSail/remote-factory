"""B2 Agentic runner — single Claude --bare call with engine-computed hints.

The pfexec engine pre-computes strategy hints from the particle filter
and embeds them in the system prompt. Claude reasons in a single --bare
call (no tools, pure reasoning) like the factory baseline.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from pfexec.dist.cc.belief_io import read_state, write_state
from pfexec.dist.cc.skill_gen import _terminal_nodes, _topo_order
from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import WorkflowSpec
from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


def _format_initial_hints(state: ExecutionState) -> dict[str, str]:
    """Generate initial hint strings from the particle briefs."""
    state.belief.normalize()
    particles = sorted(state.belief.particles, key=lambda p: p.weight, reverse=True)

    briefs = [p.brief for p in particles
              if p.brief and not p.brief.startswith(("plan-", "rejuv-"))]
    if not briefs:
        return {}

    top = particles[0]
    confidence = top.weight * 100
    hint = f'[pfexec hint: consider strategy "{top.brief}" (confidence: {confidence:.0f}%)'

    alternatives = [p.brief for p in particles[1:3]
                    if p.brief and not p.brief.startswith(("plan-", "rejuv-"))]
    if alternatives:
        alt_str = ", ".join(f'"{a}"' for a in alternatives)
        hint += f"; alternatives: {alt_str}"
    hint += "]"

    return {"default": hint}


def generate_hinted_skill_md(workflow: WorkflowSpec, state: ExecutionState) -> str:
    """Generate factory-baseline-style SKILL.md with embedded engine hints."""
    node_map = {n.id: n for n in workflow.nodes}
    order = _topo_order(workflow)

    hints = _format_initial_hints(state)
    default_hint = hints.get("default", "")

    lines = [
        f"# {workflow.name} — pfexec Workflow",
        "",
        "You are executing a multi-step reasoning workflow with probabilistic guidance.",
        "Follow each phase in order. For each phase, use the output of the previous",
        "phase as context.",
        "",
        "Strategy hints from the pfexec engine appear in [pfexec: ...] brackets.",
        "These are advisory — use them as context for your reasoning, not as commands.",
        "",
        "**Output format:** After completing each phase, write your result",
        "under a `### Output: <node_id>` header.",
        "",
    ]

    for i, nid in enumerate(order, 1):
        node = node_map[nid]
        lines.append(f"## Phase {i}: {nid}")
        if default_hint:
            lines.append(default_hint)
        lines.append("")
        lines.append(f"**Role:** {node.spec}")
        lines.append("")
        lines.append("**Task:**")
        lines.append(node.theta_prior)
        lines.append("")
        if i == 1:
            lines.append(
                "The `{input}` above will be provided in the user message."
            )
        else:
            prev_nid = order[i - 2]
            lines.append(
                f"Use the output from Phase {i - 1} (`{prev_nid}`) as "
                f"the `{{input}}` for this phase."
            )
        lines.append("")
        lines.append(
            f"Write your result under `### Output: {nid}`"
        )
        lines.append("")

    lines.append("## Completion")
    lines.append("")
    lines.append(
        f"After completing all {len(order)} phases, provide your final "
        f"consolidated answer under `### Final Answer`."
    )
    lines.append("")

    return "\n".join(lines)


def _parse_output(raw_output: str, workflow: WorkflowSpec) -> tuple[dict[str, str], str]:
    """Parse ### Output: markers and ### Final Answer from Claude's output."""
    node_ids = {n.id for n in workflow.nodes}
    node_outputs: dict[str, str] = {}

    pattern = re.compile(r"###\s+Output:\s*(\S+)")
    matches = list(pattern.finditer(raw_output))

    for i, match in enumerate(matches):
        node_id = match.group(1)
        if node_id not in node_ids:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_output)
        section = raw_output[start:end]
        final_marker = section.find("### Final Answer")
        if final_marker != -1:
            section = section[:final_marker]
        node_outputs[node_id] = section.strip()

    final = ""
    marker = "### Final Answer"
    idx = raw_output.rfind(marker)
    if idx != -1:
        final = raw_output[idx + len(marker):].strip().lstrip(":").strip()

    return node_outputs, final


def run(workflow: WorkflowSpec, user_input: str, config: EngineConfig,
        backend_mode: str = "claude") -> EngineResult:
    """Run a workflow as a single Claude --bare call with engine-computed hints."""
    from pfexec.llm import DeterministicBackend, get_backend
    from pfexec.primitives import init as pfexec_init

    backend = get_backend(backend_mode)
    state = pfexec_init(workflow, user_input, config.n_particles, backend)

    session_dir = Path(tempfile.mkdtemp(prefix="pfexec-agentic-"))
    (session_dir / "workflow.json").write_text(workflow.to_json())
    (session_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    write_state(session_dir / "state.json", state)

    skill_md = generate_hinted_skill_md(workflow, state)

    order = _topo_order(workflow)
    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]

    if backend_mode == "mock":
        mock = DeterministicBackend(default="mock output")
        mock_sections = []
        for nid in order:
            mock_sections.append(f"### Output: {nid}\n{mock.call(f'Execute {nid}')}")
        mock_sections.append("### Final Answer\nmock output")
        raw_output = "\n".join(mock_sections)
    else:
        result = subprocess.run(
            ["claude", "--bare",
             "--system-prompt", skill_md,
             "-p", f"Execute the workflow for: {user_input}"],
            capture_output=True, text=True,
            timeout=600,
        )
        raw_output = result.stdout.strip()

    parsed_outputs, parsed_final = _parse_output(raw_output, workflow)

    node_outputs = parsed_outputs
    steps_taken = len(node_outputs)
    all_outputs = [node_outputs[nid] for nid in order if nid in node_outputs]

    final_answer = parsed_final
    if not final_answer:
        for nid in reversed(order):
            if nid in node_outputs:
                final_answer = node_outputs[nid]
                break
    if not final_answer and raw_output:
        final_answer = raw_output.split("\n")[-1].strip()

    final_state_path = session_dir / "state.json"
    if final_state_path.exists():
        final_state = read_state(final_state_path)
        final_state.node_outputs = node_outputs
    else:
        belief = Belief(particles=[Particle(brief="", weight=1.0)])
        trace = TraceTree(root=TraceNode(node_id="root"))
        final_state = ExecutionState(
            pointer=terminal_id,
            belief=belief,
            trace=trace,
            step=steps_taken,
            budget_remaining=config.max_steps - steps_taken,
            user_input=user_input,
            node_outputs=node_outputs,
        )

    return EngineResult(
        final_state=final_state,
        output=final_answer,
        steps_taken=steps_taken,
        forks_triggered=0,
        terminated_by="complete",
        all_outputs=all_outputs,
    )
