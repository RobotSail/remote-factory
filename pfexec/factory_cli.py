"""CLI for running factory workflows via pfexec.

Usage:
    python -m pfexec.factory_cli list
    python -m pfexec.factory_cli compile improve
    python -m pfexec.factory_cli run improve --project /path/to/project
    python -m pfexec.factory_cli run improve --project /path --mode tool
"""

from __future__ import annotations

import argparse

from pfexec.factory_bridge import compile_workflow, get_workflow, list_workflows


def cmd_list(args: argparse.Namespace) -> None:
    for name in list_workflows():
        print(f"  {name}")


def cmd_compile(args: argparse.Namespace) -> None:
    factory_wf = get_workflow(args.workflow)
    pfexec_wf = compile_workflow(factory_wf)
    print(f"Compiled {args.workflow}: {len(pfexec_wf.nodes)} nodes, {len(pfexec_wf.edges)} edges")
    print(f"Entry: {pfexec_wf.entry}")
    print("\nNodes:")
    for n in pfexec_wf.nodes:
        effect_tag = " [effectful]" if n.effect == "effectful" else ""
        print(f"  {n.id}{effect_tag}: {n.spec[:80]}")
    print("\nEdges:")
    for e in pfexec_wf.edges:
        print(f"  {e.source} -> {e.target}")
    if args.json:
        print("\nJSON:")
        print(pfexec_wf.to_json())


def cmd_run(args: argparse.Namespace) -> None:
    from pfexec.engine import EngineConfig

    factory_wf = get_workflow(args.workflow)
    pfexec_wf = compile_workflow(factory_wf)

    config = EngineConfig(
        n_particles=args.particles,
        tau=0.4,
        max_forks=2,
        max_steps=50,
        observe_mode=args.observe_mode,
    )

    project_path = args.project

    for node in pfexec_wf.nodes:
        node.theta_prior = node.theta_prior.replace("{project_path}", project_path)

    if args.mode == "tool":
        from pfexec.dist.cc.runner_tool import run
    elif args.mode == "wrapped":
        from pfexec.dist.cc.runner_wrapped import run
    else:
        from pfexec.dist.cc.runner_session_baseline import run

    result = run(pfexec_wf, project_path, config, backend_mode="claude")

    print("\n=== Result ===")
    print(f"Steps: {result.steps_taken}/{len(pfexec_wf.nodes)}")
    print(f"Forks: {result.forks_triggered}")
    print(f"Terminated: {result.terminated_by}")
    print("\nNode outputs:")
    for nid, out in result.final_state.node_outputs.items():
        print(f"  [{nid}] {out[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pfexec.factory_cli",
        description="Run factory workflows via pfexec",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available factory workflows")

    p_compile = sub.add_parser("compile", help="Compile a factory workflow to pfexec IR")
    p_compile.add_argument("workflow", help="Workflow name (e.g. improve, build, research)")
    p_compile.add_argument("--json", action="store_true", help="Output as JSON")

    p_run = sub.add_parser("run", help="Run a factory workflow via pfexec")
    p_run.add_argument("workflow", help="Workflow name")
    p_run.add_argument("--project", required=True, help="Project path")
    p_run.add_argument(
        "--mode",
        default="tool",
        choices=["tool", "wrapped", "session"],
        help="Execution mode",
    )
    p_run.add_argument("--particles", type=int, default=1)
    p_run.add_argument(
        "--observe-mode",
        default="none",
        choices=["full", "sequential", "rewind", "lightweight", "none"],
    )

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "compile":
        cmd_compile(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
