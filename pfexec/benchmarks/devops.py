"""DevOps Dockerfile benchmark — tests fork recovery on effectful workflows.

10 scenarios with planted failures. The build/verify nodes are effectful
and run simulation scripts that check the Dockerfile for known issues.

Usage:
    python -m pfexec.benchmarks.devops --tool --limit 5
    python -m pfexec.benchmarks.devops --session-baseline --limit 5
    python -m pfexec.benchmarks.devops --dry-run
"""

from __future__ import annotations

import argparse
import json
import stat
import subprocess
import tempfile
from pathlib import Path

from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec


def build_workflow(project_dir: str) -> WorkflowSpec:
    """Build the devops workflow with project_dir baked into theta_prior."""
    return WorkflowSpec(
        name="devops_dockerize",
        nodes=[
            NodeSpec(
                id="detect_stack",
                spec="Analyze project files to detect the technology stack",
                theta_prior=(
                    f"Read the project files in {project_dir} and identify:\n"
                    "- Programming language and version\n"
                    "- Framework or runtime\n"
                    "- Key dependencies from package manifests\n"
                    "List your findings concisely."
                ),
            ),
            NodeSpec(
                id="select_image",
                spec="Select the best Docker base image for this stack",
                theta_prior=(
                    "Based on the detected stack:\n{input}\n\n"
                    "Select the best Docker base image.\n"
                    "Output ONLY the image name:tag (e.g. python:3.11-slim)."
                ),
            ),
            NodeSpec(
                id="write_dockerfile",
                spec="Write a production Dockerfile",
                theta_prior=(
                    f"Write a production Dockerfile for the project at {project_dir}.\n"
                    "Base image from prior step: {input}\n\n"
                    "Requirements:\n"
                    "- Install ALL system dependencies needed by pip/npm packages\n"
                    "- COPY source files\n"
                    "- Install application dependencies\n"
                    "- Set correct EXPOSE port (check the source code for the actual port)\n"
                    "- Set appropriate CMD/ENTRYPOINT\n\n"
                    f"Save the Dockerfile to {project_dir}/Dockerfile\n"
                    "Output the Dockerfile content."
                ),
            ),
            NodeSpec(
                id="build",
                spec="Build the Docker image (simulated)",
                theta_prior=(
                    f"Run the build simulation to check your Dockerfile:\n"
                    f"  bash {project_dir}/check.sh build\n\n"
                    "Report the EXACT output. Do not interpret or modify it."
                ),
                effect="effectful",
            ),
            NodeSpec(
                id="verify",
                spec="Verify the container works (simulated)",
                theta_prior=(
                    f"Run the verification to check your Dockerfile:\n"
                    f"  bash {project_dir}/check.sh verify\n\n"
                    "Report the EXACT output. Do not interpret or modify it."
                ),
                effect="effectful",
            ),
        ],
        edges=[
            EdgeSpec(source="detect_stack", target="select_image"),
            EdgeSpec(source="select_image", target="write_dockerfile"),
            EdgeSpec(source="write_dockerfile", target="build"),
            EdgeSpec(source="build", target="verify"),
        ],
        entry="detect_stack",
    )


def load_scenarios(limit: int | None = None, start: int = 0) -> list[dict]:
    data_path = Path(__file__).parent / "data" / "devops_10.json"
    with open(data_path) as f:
        scenarios = json.load(f)
    scenarios = scenarios[start:]
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios


def setup_scenario(scenario: dict) -> str:
    """Create a temp project dir with the scenario's files and check script."""
    project_dir = tempfile.mkdtemp(prefix=f'devops-{scenario["id"]}-')

    for filename, content in scenario["files"].items():
        filepath = Path(project_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)

    check_script = Path(project_dir) / "check.sh"
    check_script.write_text(scenario["check_script"])
    check_script.chmod(
        check_script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )

    return project_dir


def run_benchmark(
    runner,
    config: EngineConfig,
    limit: int | None = None,
    start: int = 0,
) -> list[dict]:
    scenarios = load_scenarios(limit, start)
    results = []

    for i, scenario in enumerate(scenarios):
        project_dir = setup_scenario(scenario)
        workflow = build_workflow(project_dir)

        try:
            result: EngineResult = runner(workflow, project_dir, config)

            check_path = Path(project_dir) / "check.sh"
            dockerfile_path = Path(project_dir) / "Dockerfile"

            build_pass = False
            verify_pass = False
            if dockerfile_path.exists():
                build_result = subprocess.run(
                    ["bash", str(check_path), "build"],
                    capture_output=True, text=True, cwd=project_dir,
                )
                build_pass = "PASS" in build_result.stdout
                if build_pass:
                    verify_result = subprocess.run(
                        ["bash", str(check_path), "verify"],
                        capture_output=True, text=True, cwd=project_dir,
                    )
                    verify_pass = "PASS" in verify_result.stdout

            passed = build_pass and verify_pass
            results.append({
                "id": scenario["id"],
                "name": scenario["name"],
                "passed": passed,
                "build_pass": build_pass,
                "verify_pass": verify_pass,
                "forks": result.forks_triggered,
                "steps": result.steps_taken,
            })

            marker = "+" if passed else "-"
            print(
                f"  [{marker}] {i + 1:2d}  {scenario['id']}: "
                f'build={"PASS" if build_pass else "FAIL"} '
                f'verify={"PASS" if verify_pass else "FAIL"} '
                f"forks={result.forks_triggered}"
            )
        except Exception as e:
            results.append({
                "id": scenario["id"],
                "name": scenario["name"],
                "passed": False,
                "build_pass": False,
                "verify_pass": False,
                "forks": 0,
                "steps": 0,
                "error": str(e),
            })
            print(f"  [-] {i + 1:2d}  {scenario['id']}: ERROR: {e}")

    return results


def print_summary(results: list[dict], mode: str) -> None:
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    total_forks = sum(r["forks"] for r in results)

    print(f'\n{"=" * 60}')
    print(f"DevOps Benchmark — {mode}")
    print(f'{"=" * 60}')
    print(f"  Pass rate:    {passed}/{total} ({passed / total:.0%})")
    print(f"  Total forks:  {total_forks}")
    print(f'{"=" * 60}')


def main():
    parser = argparse.ArgumentParser(description="DevOps Dockerfile benchmark")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--tool", action="store_true",
                            help="Tool-based with engine fork")
    mode_group.add_argument("--session-baseline", action="store_true",
                            help="Session baseline, no engine")
    mode_group.add_argument("--dry-run", action="store_true",
                            help="Dry run with mock backend")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--observe-mode", default="sequential",
                        choices=["full", "sequential", "rewind", "lightweight", "none"])
    parser.add_argument("--particles", type=int, default=3)
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run — skipping (no mock runner for devops)")
        return

    if args.tool:
        from pfexec.dist.cc.runner_tool import run as run_tool
        config = EngineConfig(
            n_particles=args.particles, tau=0.4, max_forks=2,
            rewind_steps=2, max_steps=30, observe_mode=args.observe_mode,
        )

        def runner(workflow, user_input, config):
            return run_tool(workflow, user_input, config, backend_mode="claude")

        mode = "tool"
    elif args.session_baseline:
        from pfexec.dist.cc.runner_session_baseline import run as run_sb
        config = EngineConfig(n_particles=1, tau=0.0, max_steps=30)

        def runner(workflow, user_input, config):
            return run_sb(workflow, user_input, config, backend_mode="claude")

        mode = "session-baseline"

    print(f"Running DevOps benchmark ({mode})...")
    results = run_benchmark(runner, config, args.limit, args.start)
    print_summary(results, mode)


if __name__ == "__main__":
    main()
