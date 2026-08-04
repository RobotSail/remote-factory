"""B2 Agentic runner — single Claude session with engine-computed hints.

The pfexec engine runs alongside Claude, injecting strategy hints
computed from the particle filter via PostToolUse hooks. Claude reasons
freely in one session (like the factory baseline) while receiving
dynamic guidance from the engine.
"""

from __future__ import annotations

import json
import re
import stat
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


def generate_hinted_skill_md(workflow: WorkflowSpec, state: ExecutionState,
                              session_dir: Path) -> str:
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
        "Updated hints will appear automatically after you complete each phase.",
        "",
        "**Output format:** After completing each phase:",
        "1. Write your result under a `### Output: <node_id>` header",
        f"2. Save it to `{session_dir}/node_outputs/<node_id>.txt`",
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
            f"Write your result under `### Output: {nid}` and save to "
            f"`node_outputs/{nid}.txt`"
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


def _generate_hint_hook(session_dir: Path, config: EngineConfig,
                         backend_mode: str) -> Path:
    """Generate the PostToolUse hook that runs observe + prints hints."""
    hooks_dir = session_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    hook_path = hooks_dir / "write_observer.sh"
    hook_path.write_text(
        "#!/bin/bash\n"
        f'SESSION_DIR="{session_dir}"\n'
        'for f in "$SESSION_DIR/node_outputs/"*.txt; do\n'
        '    [ -f "$f" ] || continue\n'
        '    NODE_ID=$(basename "$f" .txt)\n'
        '    MARKER="$SESSION_DIR/hooks/.observed_${NODE_ID}"\n'
        '    if [ ! -f "$MARKER" ]; then\n'
        f"        python3 -m pfexec.dist.cc.belief_io observe"
        f' --session "$SESSION_DIR" --node "$NODE_ID"'
        f" --backend {backend_mode} 2>/dev/null\n"
        f"        python3 -m pfexec.dist.cc.belief_io fork-check"
        f' --session "$SESSION_DIR" --node "$NODE_ID"'
        f" --tau {config.tau} --max-forks {config.max_forks}"
        f" --backend {backend_mode}"
        f' > "$SESSION_DIR/hooks/fork_status.txt" 2>/dev/null\n'
        f"        python3 -m pfexec.dist.cc.belief_io hint"
        f' --session "$SESSION_DIR" --node "$NODE_ID"\n'
        '        FORK_STATUS=$(cat "$SESSION_DIR/hooks/fork_status.txt")\n'
        '        if [ "$FORK_STATUS" = "FORK" ]; then\n'
        '            echo "[pfexec replan: low confidence — revised strategies generated.'
        ' Consider revisiting earlier reasoning.]"\n'
        "        fi\n"
        '        touch "$MARKER"\n'
        "    fi\n"
        "done\n"
    )
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return hook_path


def _generate_settings(session_dir: Path, hook_path: Path) -> Path:
    """Generate .claude/settings.json with PostToolUse hook."""
    claude_dir = session_dir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    settings = {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Write",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"bash {hook_path}",
                        }
                    ],
                }
            ]
        }
    }
    settings_path = claude_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))
    return settings_path


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
    """Run a workflow as a single Claude session with engine-computed hints."""
    from pfexec.llm import DeterministicBackend, get_backend
    from pfexec.primitives import init as pfexec_init

    backend = get_backend(backend_mode)
    state = pfexec_init(workflow, user_input, config.n_particles, backend)

    session_dir = Path(tempfile.mkdtemp(prefix="pfexec-agentic-"))
    (session_dir / "node_outputs").mkdir()
    (session_dir / "hooks").mkdir()

    (session_dir / "workflow.json").write_text(workflow.to_json())
    (session_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    write_state(session_dir / "state.json", state)

    skill_md = generate_hinted_skill_md(workflow, state, session_dir)
    skill_path = session_dir / "SKILL.md"
    skill_path.write_text(skill_md)

    hook_path = _generate_hint_hook(session_dir, config, backend_mode)
    settings_path = _generate_settings(session_dir, hook_path)

    if backend_mode == "mock":
        mock = DeterministicBackend(default="mock output")
        order = _topo_order(workflow)
        for nid in order:
            (session_dir / "node_outputs" / f"{nid}.txt").write_text(
                mock.call(f"Execute {nid}")
            )
        raw_output = ""
    else:
        result = subprocess.run(
            ["claude",
             "--settings", str(settings_path),
             "--system-prompt-file", str(skill_path),
             "--allowedTools", "Write",
             "--dangerously-skip-permissions",
             "-p", f"Execute the workflow for: {user_input}"],
            capture_output=True, text=True,
            timeout=config.max_steps * 120,
            cwd=str(session_dir),
        )
        raw_output = result.stdout.strip()

    order = _topo_order(workflow)
    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]

    file_outputs: dict[str, str] = {}
    all_outputs: list[str] = []
    for nid in order:
        out_file = session_dir / "node_outputs" / f"{nid}.txt"
        if out_file.exists():
            text = out_file.read_text().strip()
            if text:
                file_outputs[nid] = text
                all_outputs.append(text)

    parsed_outputs, parsed_final = _parse_output(raw_output, workflow)

    node_outputs = {**parsed_outputs, **file_outputs}
    steps_taken = len(node_outputs)

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
