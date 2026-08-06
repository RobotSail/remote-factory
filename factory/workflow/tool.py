"""Tool-based workflow execution — step-by-step cursor over the DAG."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import structlog

from factory.workflow.primitives import (
    AgentConfig,
    AgentNode,
    DEFAULT_AGENT_POOL,
    FnNode,
    ForkNode,
    GateNode,
    JoinNode,
    Study,
    VerdictType,
    Workflow,
)
from factory.workflow.registry import WorkflowRegistry
from factory.workflow.skill_export import _topological_sort

log = structlog.get_logger()


def _load_state(project_path: Path) -> dict:
    state_path = project_path / ".factory" / "tool_session" / "state.json"
    return json.loads(state_path.read_text())


def _save_state(project_path: Path, state: dict) -> None:
    state_path = project_path / ".factory" / "tool_session" / "state.json"
    state_path.write_text(json.dumps(state, indent=2))


def _get_workflow(state: dict, project_path: Path) -> Workflow:
    wf = WorkflowRegistry.get_workflow(state["workflow_name"], project_path)
    if not wf:
        raise ValueError(f"Workflow not found: {state['workflow_name']}")
    return wf


def tool_init(workflow_name: str, project_path: Path) -> str:
    """Initialize a tool session. Returns session dir path."""
    wf = WorkflowRegistry.get_workflow(workflow_name, project_path)
    if not wf:
        raise ValueError(f"Unknown workflow: {workflow_name}")

    session_dir = project_path / ".factory" / "tool_session"
    session_dir.mkdir(parents=True, exist_ok=True)

    order = _topological_sort(wf)

    order = [nid for nid in order if not isinstance(wf.nodes.get(nid), JoinNode)]

    state = {
        "workflow_name": workflow_name,
        "session_id": uuid.uuid4().hex[:12],
        "topo_order": order,
        "pointer_idx": 0,
        "completed": {},
        "gate_results": {},
        "iteration_counts": {},
        "status": "active",
    }

    (session_dir / "state.json").write_text(json.dumps(state, indent=2))
    return str(session_dir)


def tool_next(project_path: Path) -> str:
    """Get the next node to execute.

    Auto-submits any pending node whose artifacts exist:
    - AgentNode: .factory/reviews/<role>-latest.md or <role>-<tag>-latest.md
    - Study: .factory/strategy/observations.md
    - FnNode: declared writes exist
    - ForkNode: skip (handled by sequential ordering)

    The CEO never calls submit for agent/fn nodes — just next repeatedly.
    Submit is only needed for gate verdicts.
    """
    state = _load_state(project_path)

    if state["status"] != "active":
        return f"DONE\nWorkflow {state['workflow_name']} completed."

    wf = _get_workflow(state, project_path)
    order = state["topo_order"]
    idx = state["pointer_idx"]

    while idx < len(order):
        nid = order[idx]

        if nid in state["completed"]:
            idx += 1
            continue

        node = wf.nodes[nid]
        artifact = _detect_artifact(nid, node, project_path)

        if artifact is not None:
            state["completed"][nid] = artifact
            if isinstance(node, AgentNode) and node.writes:
                for wp in node.writes:
                    out = project_path / wp
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if not out.exists():
                        out.write_text(artifact)
            log.info("tool.auto_submit", node=nid)
            idx += 1
            state["pointer_idx"] = idx

            if idx < len(order):
                next_nid = order[idx]
                next_node = wf.nodes.get(next_nid)
                if (
                    isinstance(next_node, GateNode)
                    and next_node.evaluator_type == "fn"
                    and next_node.evaluator_command
                ):
                    gate_result = _auto_evaluate_fn_gate(
                        next_node, project_path, state, wf, order, idx,
                    )
                    if gate_result:
                        return gate_result
                    idx = state["pointer_idx"]

            _save_state(project_path, state)
            continue

        break

    state["pointer_idx"] = idx
    _save_state(project_path, state)

    if idx >= len(order):
        state["status"] = "completed"
        _save_state(project_path, state)
        return "DONE\nAll nodes completed."

    nid = order[idx]
    node = wf.nodes[nid]

    if isinstance(node, GateNode) and node.evaluator_type == "agent":
        return f"GATE\n{_format_gate_task(nid, node, state, project_path)}"

    if isinstance(node, GateNode) and node.evaluator_type == "user":
        return f"APPROVAL_NEEDED\n{node.gate_prompt}"

    return _format_node_task(nid, node, wf, state, project_path)


def tool_submit(project_path: Path, node_id: str, output: str) -> str:
    """Submit output for a node (primarily used for gate verdicts)."""
    state = _load_state(project_path)
    wf = _get_workflow(state, project_path)

    state["completed"][node_id] = output

    node = wf.nodes.get(node_id)
    if isinstance(node, AgentNode) and node.writes:
        for write_path in node.writes:
            out_file = project_path / write_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(output)

    order = state["topo_order"]
    idx = state["pointer_idx"]

    if idx < len(order) and order[idx] == node_id:
        idx += 1

    state["pointer_idx"] = idx

    if idx >= len(order):
        state["status"] = "completed"
        _save_state(project_path, state)
        return "DONE"

    next_nid = order[idx]
    next_node = wf.nodes.get(next_nid)
    if (
        isinstance(next_node, GateNode)
        and next_node.evaluator_type == "fn"
        and next_node.evaluator_command
    ):
        gate_result = _auto_evaluate_fn_gate(
            next_node, project_path, state, wf, order, idx,
        )
        if gate_result:
            return gate_result

    _save_state(project_path, state)
    return "CONTINUE"


def tool_status(project_path: Path) -> str:
    """Get current session status."""
    state_path = project_path / ".factory" / "tool_session" / "state.json"
    if not state_path.exists():
        return "No active session. Run: factory tool init <workflow> <project_path>"

    state = json.loads(state_path.read_text())
    order = state["topo_order"]
    idx = state["pointer_idx"]
    current = order[idx] if idx < len(order) else "DONE"
    completed_count = len(state["completed"])
    total = len(order)

    lines = [
        f"Workflow: {state['workflow_name']}",
        f"Session:  {state['session_id']}",
        f"Status:   {state['status']}",
        f"Progress: {completed_count}/{total} nodes",
        f"Current:  {current}",
    ]

    if state["gate_results"]:
        lines.append(f"Gates:    {json.dumps(state['gate_results'])}")

    if state["completed"]:
        lines.append("")
        lines.append("Completed nodes:")
        for nid in order:
            if nid in state["completed"]:
                preview = state["completed"][nid][:80].replace("\n", " ")
                lines.append(f"  [{nid}] {preview}")

    return "\n".join(lines)


# ── helpers ─────────────────────────────────────────────────────


def _format_node_task(
    nid: str, node: object, wf: Workflow, state: dict, project_path: Path,
) -> str:
    """Format a node as a human-readable task description."""
    lines = [f"Node: {nid}"]

    if isinstance(node, AgentNode):
        role = node.role.value
        pool_cfg: AgentConfig | None = DEFAULT_AGENT_POOL.get(role)
        model = node.model or (pool_cfg.model if pool_cfg else "opus")
        timeout = node.timeout or (pool_cfg.timeout if pool_cfg else 600)

        lines.append(f"Type: Agent ({role})")
        lines.append(f"Model: {model}")
        lines.append(f"Timeout: {timeout}s")

        if node.prompt_template:
            task = node.prompt_template.replace("{project_path}", str(project_path))
            lines.append(f"Task: {task}")

        if node.reads:
            lines.append(f"Reads: {', '.join(sorted(node.reads))}")
        if node.writes:
            lines.append(f"Writes: {', '.join(sorted(node.writes))}")

    elif isinstance(node, GateNode):
        lines.append(f"Type: Gate ({node.evaluator_type})")
        if node.gate_prompt:
            lines.append(f"Evaluate: {node.gate_prompt}")
        if node.evaluator_command:
            cmd = node.evaluator_command.replace("{project_path}", str(project_path))
            lines.append(f"Command: {cmd}")
        if node.reads:
            lines.append(f"Reads: {', '.join(sorted(node.reads))}")

    elif isinstance(node, Study):
        cmd = node.command.replace("{project_path}", str(project_path))
        lines.append("Type: Study")
        lines.append(f"Command: {cmd}")

    elif isinstance(node, FnNode):
        cmd = node.command.replace("{project_path}", str(project_path))
        lines.append("Type: Function")
        lines.append(f"Command: {cmd}")
        if node.notes:
            lines.append(f"Notes: {node.notes}")

    elif isinstance(node, ForkNode):
        lines.append("Type: Fork")
        lines.append(f"Targets: {', '.join(node.targets)}")
        lines.append("Execute all targets (listed as subsequent nodes).")

    return "\n".join(lines)


def _format_gate_task(
    nid: str, gate_node: GateNode, state: dict, project_path: Path,
) -> str:
    """Format a gate node as a review task."""
    prompt = gate_node.gate_prompt or "Review the output of the preceding step."
    prompt = prompt.replace("{project_path}", str(project_path))

    reads = ", ".join(sorted(gate_node.reads)) if gate_node.reads else "none"

    reloop_targets: list[str] = []
    wf = _get_workflow(state, project_path)
    for edge in wf.edges:
        if edge.source == nid and edge.condition == VerdictType.RELOOP:
            reloop_targets.append(edge.target)

    lines = [
        f"Gate: {nid}",
        f"Review: {prompt}",
        f"Read: {reads}",
        f"Reloop targets: {reloop_targets if reloop_targets else 'none'}",
        "",
        "Respond with one of:",
        "  PROCEED",
        '  RETRY target=<node_id> feedback="<feedback>"',
        '  HALT reason="<reason>"',
    ]
    return "\n".join(lines)


def _detect_artifact(nid: str, node: object, project_path: Path) -> str | None:
    """Check if a node's output artifact exists. Returns content or None."""
    reviews_dir = project_path / ".factory" / "reviews"

    if isinstance(node, AgentNode):
        role = node.role.value
        tag = nid.replace(f"{role}_", "").replace(role, "")
        if tag and tag != nid:
            tagged_file = reviews_dir / f"{role}-{tag}-latest.md"
            if tagged_file.exists():
                content = tagged_file.read_text().strip()
                if content:
                    return content
        review_file = reviews_dir / f"{role}-latest.md"
        if review_file.exists():
            content = review_file.read_text().strip()
            if content:
                return content
        if node.writes:
            for wp in node.writes:
                f = project_path / wp
                if f.exists():
                    content = f.read_text().strip()
                    if content:
                        return content
        return None

    elif isinstance(node, Study):
        obs_file = project_path / ".factory" / "strategy" / "observations.md"
        if obs_file.exists():
            content = obs_file.read_text().strip()
            if content and len(content) > 50:
                return content
        return None

    elif isinstance(node, FnNode):
        if node.writes:
            all_exist = all((project_path / wp).exists() for wp in node.writes)
            if all_exist:
                parts = []
                for wp in node.writes:
                    parts.append((project_path / wp).read_text().strip()[:500])
                return "; ".join(parts) if parts else None
        return None

    elif isinstance(node, ForkNode):
        return f"Fork targets: {', '.join(node.targets)}"

    elif isinstance(node, GateNode):
        return None

    return None


def _auto_evaluate_fn_gate(
    gate_node: GateNode,
    project_path: Path,
    state: dict,
    wf: Workflow,
    order: list[str],
    idx: int,
) -> str | None:
    """Auto-evaluate a fn gate. Returns RETRY/HALT string or None if passed."""
    nid = order[idx]
    assert gate_node.evaluator_command is not None
    cmd = gate_node.evaluator_command.replace("{project_path}", str(project_path))
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60,
        )
        gate_output = result.stdout.strip()
        gate_passed = result.returncode == 0 and "FAIL" not in gate_output
    except subprocess.TimeoutExpired:
        gate_output = "Gate command timed out"
        gate_passed = False

    state["gate_results"][nid] = "PROCEED" if gate_passed else "HALT"
    state["completed"][nid] = gate_output

    if not gate_passed:
        reloop_target = _find_reloop_target(wf, nid)
        if reloop_target:
            iter_key = f"{nid}->{reloop_target}"
            count = state["iteration_counts"].get(iter_key, 0) + 1
            state["iteration_counts"][iter_key] = count

            if count <= 3:
                if reloop_target in order:
                    state["pointer_idx"] = order.index(reloop_target)
                _save_state(project_path, state)
                return (
                    f"RETRY\nGate {nid} failed: {gate_output}\n"
                    f"Retry from: {reloop_target} (attempt {count}/3)"
                )

        state["status"] = "halted"
        state["pointer_idx"] = idx + 1
        _save_state(project_path, state)
        return f"HALT\nGate {nid} failed: {gate_output}"

    state["pointer_idx"] = idx + 1
    _save_state(project_path, state)
    return None


def _find_reloop_target(wf: Workflow, gate_id: str) -> str | None:
    """Find the RELOOP target for a gate node."""
    for edge in wf.edges:
        if edge.source == gate_id and edge.condition == VerdictType.RELOOP:
            return edge.target
    return None
