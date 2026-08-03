"""Disk-based state management and CLI for pfexec belief tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pfexec.ir import WorkflowSpec
from pfexec.llm import ClaudeBackend, DeterministicBackend, LLMBackend
from pfexec.primitives import fork, init, observe, sample
from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


def _trace_node_to_dict(node: TraceNode) -> dict:
    return {
        "node_id": node.node_id,
        "checkpoint_id": node.checkpoint_id,
        "alive": node.alive,
        "summary": node.summary,
        "children": [_trace_node_to_dict(c) for c in node.children],
    }


def _trace_node_from_dict(d: dict) -> TraceNode:
    return TraceNode(
        node_id=d["node_id"],
        checkpoint_id=d.get("checkpoint_id", ""),
        alive=d.get("alive", True),
        summary=d.get("summary", ""),
        children=[_trace_node_from_dict(c) for c in d.get("children", [])],
    )


def state_to_dict(state: ExecutionState) -> dict:
    return {
        "pointer": state.pointer,
        "step": state.step,
        "budget_remaining": state.budget_remaining,
        "user_input": state.user_input,
        "node_outputs": state.node_outputs,
        "belief": {
            "particles": [
                {"brief": p.brief, "weight": p.weight, "evidence": p.evidence}
                for p in state.belief.particles
            ],
        },
        "trace": _trace_node_to_dict(state.trace.root),
    }


def state_from_dict(d: dict) -> ExecutionState:
    particles = [
        Particle(
            brief=p["brief"],
            weight=p.get("weight", 1.0),
            evidence=p.get("evidence", ""),
        )
        for p in d["belief"]["particles"]
    ]
    belief = Belief(particles=particles)
    trace = TraceTree(root=_trace_node_from_dict(d["trace"]))
    return ExecutionState(
        pointer=d["pointer"],
        belief=belief,
        trace=trace,
        step=d.get("step", 0),
        budget_remaining=d.get("budget_remaining", 50),
        user_input=d.get("user_input", ""),
        node_outputs=d.get("node_outputs", {}),
    )


def write_state(path: Path, state: ExecutionState) -> None:
    path.write_text(json.dumps(state_to_dict(state), indent=2))


def read_state(path: Path) -> ExecutionState:
    return state_from_dict(json.loads(path.read_text()))


def write_belief(path: Path, belief: Belief) -> None:
    data = {
        "particles": [
            {"brief": p.brief, "weight": p.weight, "evidence": p.evidence}
            for p in belief.particles
        ],
    }
    path.write_text(json.dumps(data, indent=2))


def read_belief(path: Path) -> Belief:
    data = json.loads(path.read_text())
    return Belief(
        particles=[
            Particle(
                brief=p["brief"],
                weight=p.get("weight", 1.0),
                evidence=p.get("evidence", ""),
            )
            for p in data["particles"]
        ]
    )


def _get_backend(mode: str) -> LLMBackend:
    if mode == "mock":
        return DeterministicBackend(default="ok")
    return ClaudeBackend()


def _state_path(session_dir: Path) -> Path:
    return session_dir / "state.json"


def cmd_init(session_dir: Path, workflow_path: Path, user_input: str, n_particles: int,
             backend_mode: str) -> None:
    workflow = WorkflowSpec.from_json(workflow_path.read_text())
    backend = _get_backend(backend_mode)
    state = init(workflow, user_input, n_particles, backend)

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "trace").mkdir(exist_ok=True)
    (session_dir / "node_outputs").mkdir(exist_ok=True)
    (session_dir / "hooks").mkdir(exist_ok=True)

    write_state(_state_path(session_dir), state)
    write_belief(session_dir / "belief.json", state.belief)

    trace_data = _trace_node_to_dict(state.trace.root)
    (session_dir / "trace" / "root.json").write_text(json.dumps(trace_data, indent=2))


def cmd_sample(session_dir: Path, node_id: str, backend_mode: str) -> None:
    state = read_state(_state_path(session_dir))
    workflow = WorkflowSpec.from_json((session_dir / "workflow.json").read_text())
    backend = _get_backend(backend_mode)

    node_map = {n.id: n for n in workflow.nodes}
    node = node_map[node_id]

    state, _output = sample(state, node, backend)

    hint = ""
    n = len(state.belief.particles)
    if n > 1:
        state.belief.normalize()
        best = max(state.belief.particles, key=lambda p: p.weight)
        uniform = 1.0 / n
        if best.brief and not best.brief.startswith("plan-") and best.weight > uniform * 1.2:
            hint = best.brief

    hooks_dir = session_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    (hooks_dir / "hint.txt").write_text(hint)

    if not state.node_outputs:
        data_input = state.user_input
    else:
        last_key = list(state.node_outputs.keys())[-1]
        if last_key != node_id:
            data_input = state.node_outputs[last_key]
        else:
            prior_keys = [k for k in state.node_outputs if k != node_id]
            data_input = state.node_outputs[prior_keys[-1]] if prior_keys else state.user_input

    prompt = node.theta_prior.replace("{input}", data_input)
    if hint:
        prompt = f"[Strategy hint: {hint}]\n\n{prompt}"

    (hooks_dir / "prompt.txt").write_text(prompt)
    write_state(_state_path(session_dir), state)


def cmd_observe(session_dir: Path, node_id: str, backend_mode: str) -> None:
    state = read_state(_state_path(session_dir))
    backend = _get_backend(backend_mode)

    output_file = session_dir / "node_outputs" / f"{node_id}.txt"
    observation = output_file.read_text() if output_file.exists() else ""

    state.node_outputs[node_id] = observation
    state = observe(state, observation, backend)
    write_state(_state_path(session_dir), state)


def cmd_fork_check(session_dir: Path, node_id: str, tau: float, max_forks: int,
                   backend_mode: str) -> None:
    state = read_state(_state_path(session_dir))
    backend = _get_backend(backend_mode)

    workflow = WorkflowSpec.from_json((session_dir / "workflow.json").read_text())
    node_map = {n.id: n for n in workflow.nodes}
    node = node_map[node_id]

    if node.effect != "effectful":
        print("CONTINUE")
        return

    state.belief.normalize()
    weights = sorted((p.weight for p in state.belief.particles), reverse=True)
    top_k = weights[:3]
    score = sum(top_k) / len(top_k) if top_k else 0.0

    if score < tau and max_forks > 0:
        state = fork(state, 2, backend)
        write_state(_state_path(session_dir), state)
        print("FORK")
    else:
        print("CONTINUE")


def main() -> None:
    parser = argparse.ArgumentParser(prog="pfexec.dist.cc.belief_io")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--session", required=True, type=Path)
    p_init.add_argument("--workflow", required=True, type=Path)
    p_init.add_argument("--input", required=True)
    p_init.add_argument("--particles", type=int, default=3)
    p_init.add_argument("--backend", default="mock", choices=["mock", "claude"])

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--session", required=True, type=Path)
    p_sample.add_argument("--node", required=True)
    p_sample.add_argument("--backend", default="mock", choices=["mock", "claude"])

    p_observe = sub.add_parser("observe")
    p_observe.add_argument("--session", required=True, type=Path)
    p_observe.add_argument("--node", required=True)
    p_observe.add_argument("--backend", default="mock", choices=["mock", "claude"])

    p_fork = sub.add_parser("fork-check")
    p_fork.add_argument("--session", required=True, type=Path)
    p_fork.add_argument("--node", required=True)
    p_fork.add_argument("--tau", type=float, default=0.3)
    p_fork.add_argument("--max-forks", type=int, default=3)
    p_fork.add_argument("--backend", default="mock", choices=["mock", "claude"])

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args.session, args.workflow, args.input, args.particles, args.backend)
    elif args.command == "sample":
        cmd_sample(args.session, args.node, args.backend)
    elif args.command == "observe":
        cmd_observe(args.session, args.node, args.backend)
    elif args.command == "fork-check":
        cmd_fork_check(args.session, args.node, args.tau, args.max_forks, args.backend)


if __name__ == "__main__":
    main()
