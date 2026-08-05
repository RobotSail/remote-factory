"""pfexec CLI tool — Claude interacts with the SSM engine via this interface.

Subcommands: init, next, submit, status.
All state persists in a session directory.

Usage:
    python -m pfexec.tool init --workflow workflow.json --input 'Fix the bug' --particles 5
    python -m pfexec.tool next --session /tmp/pfexec-tool-xxx
    python -m pfexec.tool submit --session /tmp/pfexec-tool-xxx --node reason_sub1 <<< 'answer'
    python -m pfexec.tool status --session /tmp/pfexec-tool-xxx
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from pfexec.dist.cc.belief_io import read_state, write_state
from pfexec.dist.cc.skill_gen import _terminal_nodes, _topo_order
from pfexec.dist.cc.runner_wrapped import _suffix_score, _extract_lesson
from pfexec.engine import EngineConfig
from pfexec.ir import WorkflowSpec
from pfexec.llm import get_backend
from pfexec.primitives import (
    init as pfexec_init,
    observe,
    observe_lightweight,
    observe_rewind,
    observe_sequential,
    fork,
)


def _load_workflow(session_dir: Path) -> WorkflowSpec:
    return WorkflowSpec.from_json((session_dir / "workflow.json").read_text())


def _load_config(session_dir: Path) -> dict:
    return json.loads((session_dir / "config.json").read_text())


def cmd_init(args: argparse.Namespace) -> None:
    workflow = WorkflowSpec.from_json(Path(args.workflow).read_text())
    backend = get_backend(args.backend)

    state = pfexec_init(workflow, args.input, args.particles, backend)
    state.budget_remaining = 50

    session_dir = Path(tempfile.mkdtemp(prefix="pfexec-tool-"))
    write_state(session_dir / "state.json", state)
    (session_dir / "workflow.json").write_text(workflow.to_json())

    order = _topo_order(workflow)
    config = {
        "n_particles": args.particles,
        "tau": args.tau,
        "max_forks": args.max_forks,
        "observe_mode": args.observe_mode,
        "backend": args.backend,
        "pointer_idx": 0,
        "order": order,
    }
    (session_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(str(session_dir))


def cmd_next(args: argparse.Namespace) -> None:
    session_dir = Path(args.session)
    state = read_state(session_dir / "state.json")
    workflow = _load_workflow(session_dir)
    config = _load_config(session_dir)

    order = config["order"]
    pointer_idx = config["pointer_idx"]
    node_map = {n.id: n for n in workflow.nodes}
    terminal = set(_terminal_nodes(workflow))

    if pointer_idx >= len(order):
        terminal_id = order[-1]
        output = state.node_outputs.get(terminal_id, "")
        print(f"DONE\n{output}")
        return

    nid = order[pointer_idx]
    node = node_map[nid]
    phase_num = pointer_idx + 1

    if not state.node_outputs:
        data_input = state.user_input
        task = node.theta_prior.replace("{input}", data_input)
    else:
        prev_keys = [k for k in order[:pointer_idx] if k in state.node_outputs]
        data_input = state.node_outputs[prev_keys[-1]] if prev_keys else state.user_input
        truncated = data_input[:500] + '...' if len(data_input) > 500 else data_input
        task = node.theta_prior.replace("{input}", truncated)
        task += '\n\nUse your reasoning from prior steps as additional context.'

    print(f"Phase {phase_num}: {nid}")
    print(f"Role: {node.spec}")
    print(f"Task: {task}")

    n = len(state.belief.particles)
    if n > 1:
        state.belief.normalize()
        best = max(state.belief.particles, key=lambda p: p.weight)
        uniform = 1.0 / n
        if (best.brief
                and not best.brief.startswith(("plan-", "rejuv-"))
                and best.weight > uniform * 1.5):
            confidence = best.weight * 100
            print(f'Hint: strategy "{best.brief}" leads (confidence: {confidence:.0f}%)')


def cmd_submit(args: argparse.Namespace) -> None:
    session_dir = Path(args.session)
    state = read_state(session_dir / "state.json")
    workflow = _load_workflow(session_dir)
    config = _load_config(session_dir)

    output_text = sys.stdin.read().strip()
    node_id = args.node

    state.node_outputs[node_id] = output_text

    observe_mode = config.get("observe_mode", "full")
    backend_mode = args.backend or config.get("backend", "mock")
    backend = get_backend(backend_mode)

    if observe_mode == "none":
        pass
    elif observe_mode == "sequential":
        state = observe_sequential(state, output_text, node_id)
    elif observe_mode == "rewind":
        state = observe_rewind(state, output_text, backend)
    elif observe_mode == "lightweight":
        state = observe_lightweight(state, output_text)
    else:
        if config.get("n_particles", 1) > 1:
            state = observe(state, output_text, backend)

    state.step += 1
    state.budget_remaining -= 1

    order = config["order"]
    pointer_idx = config["pointer_idx"]
    node_map = {n.id: n for n in workflow.nodes}
    node = node_map[node_id]

    tau = config.get("tau", 0.3)
    max_forks = config.get("max_forks", 2)

    fork_triggered = False
    if observe_mode != "none" and node.effect == "effectful" and max_forks > 0:
        if _suffix_score(state.belief) < tau:
            eng_config = EngineConfig(
                n_particles=config.get("n_particles", 5),
                tau=tau,
                max_forks=max_forks,
                observe_mode=observe_mode,
            )
            lesson = _extract_lesson(state, eng_config, output_text)
            state = fork(state, 2, backend)
            fork_triggered = True

            rewind_nid = state.pointer
            rewind_idx = order.index(rewind_nid) if rewind_nid in order else 0
            config["pointer_idx"] = rewind_idx
            config["max_forks"] = max_forks - 1
            (session_dir / "config.json").write_text(json.dumps(config, indent=2))
            write_state(session_dir / "state.json", state)
            print("FORK")
            print(f"Lesson: {lesson}")
            print(f"Restart from: {rewind_nid}")
            return

    new_idx = pointer_idx + 1
    config["pointer_idx"] = new_idx
    (session_dir / "config.json").write_text(json.dumps(config, indent=2))
    write_state(session_dir / "state.json", state)

    if new_idx >= len(order):
        print("DONE")
    else:
        print("CONTINUE")


def cmd_status(args: argparse.Namespace) -> None:
    session_dir = Path(args.session)
    state = read_state(session_dir / "state.json")
    config = _load_config(session_dir)

    order = config["order"]
    pointer_idx = config["pointer_idx"]

    current = order[pointer_idx] if pointer_idx < len(order) else "DONE"
    print(f"Current node: {current}")
    print(f"Step: {state.step}")
    print(f"Pointer index: {pointer_idx}/{len(order)}")

    print("\nParticles:")
    state.belief.normalize()
    for i, p in enumerate(state.belief.particles):
        print(f"  [{i}] weight={p.weight:.3f} brief={p.brief[:60]}")

    if state.node_outputs:
        print("\nNode outputs:")
        for nid in order:
            if nid in state.node_outputs:
                preview = state.node_outputs[nid][:80].replace("\n", " ")
                print(f"  {nid}: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="pfexec.tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--workflow", required=True)
    p_init.add_argument("--input", required=True)
    p_init.add_argument("--particles", type=int, default=5)
    p_init.add_argument("--tau", type=float, default=0.4)
    p_init.add_argument("--max-forks", type=int, default=2)
    p_init.add_argument("--observe-mode", default="full",
                        choices=["full", "sequential", "rewind", "lightweight", "none"])
    p_init.add_argument("--backend", default="mock", choices=["claude", "mock"])

    p_next = sub.add_parser("next")
    p_next.add_argument("--session", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--session", required=True)
    p_submit.add_argument("--node", required=True)
    p_submit.add_argument("--backend", default=None, choices=["claude", "mock"])

    p_status = sub.add_parser("status")
    p_status.add_argument("--session", required=True)

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "next":
        cmd_next(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
