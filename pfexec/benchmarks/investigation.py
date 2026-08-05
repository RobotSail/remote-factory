"""Investigation benchmark — tests multi-step fact extraction from data files.

Each scenario provides data files (logs, configs, CSVs) and a question.
The workflow has 7 nodes, each extracting one specific fact. The final
answer requires combining ALL facts — skipping or rushing any step
produces a wrong answer.

Scores both intermediate facts AND the final answer.

Usage:
    python -m pfexec.benchmarks.investigation --tool --limit 5
    python -m pfexec.benchmarks.investigation --session-baseline --limit 5
    python -m pfexec.benchmarks.investigation --wrapped --limit 5
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec


def build_workflow(project_dir: str) -> WorkflowSpec:
    """Build the investigation workflow with project_dir baked into theta_prior."""
    return WorkflowSpec(
        name="investigation",
        nodes=[
            NodeSpec(
                id="step1_extract",
                spec="Extract the first key fact from the data",
                theta_prior=(
                    f"Read the investigation brief at {project_dir}/brief.md\n"
                    f"Read the data file referenced for Step 1.\n"
                    "Extract the specific fact requested. Output ONLY the fact value, nothing else."
                ),
            ),
            NodeSpec(
                id="step2_extract",
                spec="Extract the second key fact",
                theta_prior=(
                    f"Prior findings: {{input}}\n\n"
                    f"Read the data file referenced for Step 2 in {project_dir}/brief.md\n"
                    "Extract the specific fact requested. Output ONLY the fact value."
                ),
            ),
            NodeSpec(
                id="step3_extract",
                spec="Extract the third key fact",
                theta_prior=(
                    f"Prior findings: {{input}}\n\n"
                    f"Read the data file referenced for Step 3 in {project_dir}/brief.md\n"
                    "Extract the specific fact requested. Output ONLY the fact value."
                ),
            ),
            NodeSpec(
                id="step4_correlate",
                spec="Correlate facts from steps 1-3",
                theta_prior=(
                    f"Prior findings: {{input}}\n\n"
                    f"Read {project_dir}/brief.md Step 4 instructions.\n"
                    "Correlate the extracted facts. Output ONLY the correlation result."
                ),
            ),
            NodeSpec(
                id="step5_verify",
                spec="Verify the correlation against additional data",
                theta_prior=(
                    f"Correlation result: {{input}}\n\n"
                    f"Read the verification data referenced in Step 5 of {project_dir}/brief.md\n"
                    "Does the data support the correlation? "
                    "Output: CONFIRMED or CONTRADICTED, with the key evidence."
                ),
                effect="effectful",
            ),
            NodeSpec(
                id="step6_conclude",
                spec="Draw the conclusion",
                theta_prior=(
                    f"Verified findings: {{input}}\n\n"
                    f"Read Step 6 instructions in {project_dir}/brief.md\n"
                    "State the root cause or conclusion. Output in 1-2 sentences."
                ),
            ),
            NodeSpec(
                id="step7_answer",
                spec="Produce the final answer",
                theta_prior=(
                    f"Conclusion: {{input}}\n\n"
                    f"Read the question in {project_dir}/brief.md\n"
                    "Output ONLY the final answer in the exact format requested, nothing else."
                ),
            ),
        ],
        edges=[
            EdgeSpec(source="step1_extract", target="step2_extract"),
            EdgeSpec(source="step2_extract", target="step3_extract"),
            EdgeSpec(source="step3_extract", target="step4_correlate"),
            EdgeSpec(source="step4_correlate", target="step5_verify"),
            EdgeSpec(source="step5_verify", target="step6_conclude"),
            EdgeSpec(source="step6_conclude", target="step7_answer"),
        ],
        entry="step1_extract",
    )


def load_scenarios(limit: int | None = None, start: int = 0) -> list[dict]:
    data_path = Path(__file__).parent / "data" / "investigation_10.json"
    with open(data_path) as f:
        scenarios = json.load(f)
    scenarios = scenarios[start:]
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios


def setup_scenario(scenario: dict) -> str:
    """Create a temp project dir with brief.md and all data files."""
    project_dir = tempfile.mkdtemp(prefix=f'investigation-{scenario["id"]}-')

    brief_path = Path(project_dir) / "brief.md"
    brief_path.write_text(scenario["brief_md"])

    for filename, content in scenario["files"].items():
        filepath = Path(project_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)

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

            facts_correct = 0
            facts_total = len(scenario["expected_facts"])
            fact_details: list[dict] = []
            for step_id, expected in scenario["expected_facts"].items():
                actual = result.final_state.node_outputs.get(step_id, "")
                match = expected.lower() in actual.lower()
                if match:
                    facts_correct += 1
                fact_details.append({
                    "step": step_id,
                    "expected": expected,
                    "actual": actual[:80],
                    "match": match,
                })

            final_correct = scenario["expected_answer"].lower() in result.output.lower()

            results.append({
                "id": scenario["id"],
                "name": scenario["name"],
                "facts_score": facts_correct / facts_total if facts_total else 0,
                "facts_correct": facts_correct,
                "facts_total": facts_total,
                "final_correct": final_correct,
                "steps_completed": result.steps_taken,
                "total_steps": len(workflow.nodes),
                "forks": result.forks_triggered,
            })

            marker = "+" if final_correct else ("~" if facts_correct > facts_total // 2 else "-")
            print(
                f"  [{marker}] {i + 1:2d}  {scenario['id']}: "
                f"facts={facts_correct}/{facts_total} "
                f'final={"PASS" if final_correct else "FAIL"} '
                f"steps={result.steps_taken}/7 forks={result.forks_triggered}"
            )
        except Exception as e:
            results.append({
                "id": scenario["id"],
                "name": scenario["name"],
                "facts_score": 0.0,
                "facts_correct": 0,
                "facts_total": len(scenario.get("expected_facts", {})),
                "final_correct": False,
                "steps_completed": 0,
                "total_steps": 7,
                "forks": 0,
                "error": str(e),
            })
            print(f"  [-] {i + 1:2d}  {scenario['id']}: ERROR: {e}")

    return results


def print_summary(results: list[dict], mode: str) -> None:
    total = len(results)
    if not total:
        print("  No scenarios run")
        return

    avg_facts = sum(r["facts_score"] for r in results) / total
    final_passes = sum(1 for r in results if r["final_correct"])
    avg_steps = sum(r["steps_completed"] for r in results) / total
    total_forks = sum(r["forks"] for r in results)

    print(f'\n{"=" * 60}')
    print(f"Investigation Benchmark — {mode}")
    print(f'{"=" * 60}')
    print(f"  Avg facts score:  {avg_facts:.0%}")
    print(f"  Final answer:     {final_passes}/{total} ({final_passes / total:.0%})")
    print(f"  Avg steps:        {avg_steps:.1f}/7")
    print(f"  Total forks:      {total_forks}")
    print(f'{"=" * 60}')


def main():
    parser = argparse.ArgumentParser(description="Investigation benchmark")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--tool", action="store_true",
                            help="Tool-based with engine fork")
    mode_group.add_argument("--session-baseline", action="store_true",
                            help="Session baseline, no engine")
    mode_group.add_argument("--wrapped", action="store_true",
                            help="Wrapped runner with engine fork")
    mode_group.add_argument("--factory-baseline", action="store_true",
                            help="Factory SKILL.md single-prompt baseline")
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
    elif args.factory_baseline:
        from pfexec.dist.cc.factory_baseline import run_factory_baseline
        config = EngineConfig(n_particles=1, tau=0.0, max_steps=30)

        def runner(workflow, user_input, config):
            return run_factory_baseline(workflow, user_input, config)

        mode = "factory-baseline"

    if args.particles != 3:
        config = EngineConfig(
            n_particles=args.particles,
            tau=config.tau,
            max_steps=config.max_steps,
            max_forks=config.max_forks,
            rewind_steps=config.rewind_steps,
            observe_mode=config.observe_mode,
        )

    print(f"Running Investigation benchmark ({mode})...")
    results = run_benchmark(runner, config, args.limit, args.start)
    print_summary(results, mode)


if __name__ == "__main__":
    main()
