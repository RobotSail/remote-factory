"""Code generation benchmark — tests fork recovery via real test execution.

Each scenario provides a function spec with edge cases. The workflow:
  analyze → implement → test (effectful) → report

The test step runs real pytest. Fork triggers on test failure and provides
the test output as a lesson for the retry.

Usage:
    python -m pfexec.benchmarks.codegen --tool --limit 5
    python -m pfexec.benchmarks.codegen --session-baseline --limit 5
    python -m pfexec.benchmarks.codegen --wrapped --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec


def build_workflow(project_dir: str) -> WorkflowSpec:
    """Build the codegen workflow with project_dir baked into theta_prior."""
    return WorkflowSpec(
        name="codegen",
        nodes=[
            NodeSpec(
                id="analyze",
                spec="Analyze the function specification and identify edge cases",
                theta_prior=(
                    f"Read the specification at {project_dir}/spec.md\n"
                    "Identify:\n"
                    "- Input/output types\n"
                    "- Edge cases that could cause bugs\n"
                    "- Tricky test cases to watch for\n"
                    "List your analysis concisely."
                ),
            ),
            NodeSpec(
                id="implement",
                spec="Write the function implementation",
                theta_prior=(
                    f"Based on your analysis:\n{{input}}\n\n"
                    f"Write the implementation to {project_dir}/solution.py\n"
                    "The file must define the function specified in spec.md.\n"
                    "Handle ALL edge cases identified in your analysis.\n"
                    "Output the code."
                ),
            ),
            NodeSpec(
                id="test",
                spec="Run tests to verify the implementation",
                theta_prior=(
                    f"Run the tests:\n"
                    f"  cd {project_dir} && python -m pytest test_solution.py -v 2>&1\n\n"
                    "Report the EXACT output. Do not modify or interpret it."
                ),
                effect="effectful",
            ),
            NodeSpec(
                id="report",
                spec="Report the results",
                theta_prior=(
                    "Based on the test results:\n{input}\n\n"
                    "Report: PASS (all tests passed) or FAIL (some tests failed).\n"
                    "Output ONLY: PASS or FAIL"
                ),
            ),
        ],
        edges=[
            EdgeSpec(source="analyze", target="implement"),
            EdgeSpec(source="implement", target="test"),
            EdgeSpec(source="test", target="report"),
        ],
        entry="analyze",
    )


def load_scenarios(limit: int | None = None, start: int = 0) -> list[dict]:
    data_path = Path(__file__).parent / "data" / "codegen_10.json"
    with open(data_path) as f:
        scenarios = json.load(f)
    scenarios = scenarios[start:]
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios


def setup_scenario(scenario: dict) -> str:
    """Create a temp project dir with spec.md, test_solution.py, and empty solution.py."""
    project_dir = tempfile.mkdtemp(prefix=f'codegen-{scenario["id"]}-')

    spec_path = Path(project_dir) / "spec.md"
    spec_path.write_text(scenario["spec_md"])

    test_path = Path(project_dir) / "test_solution.py"
    test_path.write_text(scenario["test_code"])

    solution_path = Path(project_dir) / "solution.py"
    solution_path.write_text(scenario["solution_template"])

    return project_dir


def _parse_pytest_results(output: str) -> tuple[int, int]:
    """Parse pytest output to extract passed/total counts."""
    match = re.search(r"(\d+) passed", output)
    passed = int(match.group(1)) if match else 0

    failed_match = re.search(r"(\d+) failed", output)
    failed = int(failed_match.group(1)) if failed_match else 0

    error_match = re.search(r"(\d+) error", output)
    errors = int(error_match.group(1)) if error_match else 0

    total = passed + failed + errors
    return passed, total


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

            pytest_result = subprocess.run(
                ["python", "-m", "pytest", "test_solution.py", "-v"],
                capture_output=True, text=True, cwd=project_dir,
            )
            output = pytest_result.stdout + pytest_result.stderr
            passed, total = _parse_pytest_results(output)

            full_pass = pytest_result.returncode == 0 and total > 0
            pass_rate = passed / total if total > 0 else 0.0

            results.append({
                "id": scenario["id"],
                "name": scenario["name"],
                "passed": passed,
                "total": total,
                "pass_rate": pass_rate,
                "full_pass": full_pass,
                "forks": result.forks_triggered,
                "steps": result.steps_taken,
            })

            marker = "+" if full_pass else ("~" if pass_rate > 0.5 else "-")
            print(
                f"  [{marker}] {i + 1:2d}  {scenario['id']}: "
                f"{passed}/{total} tests "
                f"({pass_rate:.0%}) "
                f"forks={result.forks_triggered}"
            )
        except Exception as e:
            results.append({
                "id": scenario["id"],
                "name": scenario["name"],
                "passed": 0,
                "total": 0,
                "pass_rate": 0.0,
                "full_pass": False,
                "forks": 0,
                "steps": 0,
                "error": str(e),
            })
            print(f"  [-] {i + 1:2d}  {scenario['id']}: ERROR: {e}")

    return results


def print_summary(results: list[dict], mode: str) -> None:
    full_passes = sum(1 for r in results if r["full_pass"])
    total = len(results)
    avg_pass_rate = (
        sum(r["pass_rate"] for r in results) / total if total else 0.0
    )
    total_forks = sum(r["forks"] for r in results)

    print(f'\n{"=" * 60}')
    print(f"Codegen Benchmark — {mode}")
    print(f'{"=" * 60}')
    print(f"  Full pass rate: {full_passes}/{total} ({full_passes / total:.0%})" if total else "  No scenarios run")
    print(f"  Avg test pass:  {avg_pass_rate:.0%}")
    print(f"  Total forks:    {total_forks}")
    print(f'{"=" * 60}')


def main():
    parser = argparse.ArgumentParser(description="Code generation benchmark")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--tool", action="store_true",
                            help="Tool-based with engine fork")
    mode_group.add_argument("--session-baseline", action="store_true",
                            help="Session baseline, no engine")
    mode_group.add_argument("--wrapped", action="store_true",
                            help="Wrapped runner with engine fork")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--observe-mode", default="sequential",
                        choices=["full", "sequential", "rewind", "lightweight", "none"])
    parser.add_argument("--particles", type=int, default=3)
    args = parser.parse_args()

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
    elif args.wrapped:
        from pfexec.dist.cc.runner_wrapped import run as run_wrapped
        config = EngineConfig(
            n_particles=args.particles, tau=0.4, max_forks=2,
            rewind_steps=2, max_steps=30, observe_mode=args.observe_mode,
        )

        def runner(workflow, user_input, config):
            return run_wrapped(workflow, user_input, config, backend_mode="claude")

        mode = "wrapped"

    if args.particles != 3:
        config = EngineConfig(
            n_particles=args.particles,
            tau=config.tau,
            max_steps=config.max_steps,
            max_forks=config.max_forks,
            rewind_steps=config.rewind_steps,
            observe_mode=config.observe_mode,
        )

    print(f"Running Codegen benchmark ({mode})...")
    results = run_benchmark(runner, config, args.limit, args.start)
    print_summary(results, mode)


if __name__ == "__main__":
    main()
