"""Forensic analysis benchmark — tests deep multi-step reasoning over long workflows.

15 nodes per scenario (vs 7 in investigation), each requiring computation
(counting, filtering, aggregating). Data files are 50-100 lines. Later nodes
require recalling earlier facts — tests context retention over long workflows.

Key metric: facts score at nodes 10+ — do later nodes still get correct answers?
Session mode may degrade on nodes 12-15 while tool mode stays consistent.

Usage:
    python -m pfexec.benchmarks.forensics --tool --limit 3
    python -m pfexec.benchmarks.forensics --session-baseline --limit 3
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec


def build_workflow(project_dir: str) -> WorkflowSpec:
    """Build the 15-node forensic analysis workflow."""
    nodes = [
        NodeSpec(
            id="n01_scan_access",
            spec="Scan access logs",
            theta_prior=(
                f"Read {project_dir}/access.log. "
                "Count the total number of unique source IPs. "
                "Output ONLY the count."
            ),
        ),
        NodeSpec(
            id="n02_filter_heavy",
            spec="Filter heavy hitters",
            theta_prior=(
                f"Read {project_dir}/access.log. "
                "List IPs with more than 10 requests. "
                "Prior count: {{input}}. "
                "Output: IP,count pairs, one per line."
            ),
        ),
        NodeSpec(
            id="n03_check_allowlist",
            spec="Check against allowlist",
            theta_prior=(
                f"Read {project_dir}/allowlist.txt. "
                "Compare against heavy hitters from prior step: {{input}}. "
                "List IPs NOT in the allowlist. "
                "Output: suspicious IPs, one per line."
            ),
        ),
        NodeSpec(
            id="n04_extract_paths",
            spec="Extract request paths",
            theta_prior=(
                f"Read {project_dir}/access.log. "
                "For the suspicious IPs from prior step: {{input}}. "
                "List the most common request paths for each suspicious IP. "
                "Output: IP -> top path, one per line."
            ),
        ),
        NodeSpec(
            id="n05_identify_target",
            spec="Identify targeted endpoint",
            theta_prior=(
                "From the path analysis: {input}. "
                "Which endpoint was most targeted across all suspicious IPs? "
                "Output ONLY the endpoint path."
            ),
        ),
        NodeSpec(
            id="n06_check_ratelimit",
            spec="Check rate limiting config",
            theta_prior=(
                f"Read {project_dir}/config.json. "
                "Is rate limiting enabled for the endpoint: {{input}}? "
                "Output: enabled/disabled and the limit value if any."
            ),
        ),
        NodeSpec(
            id="n07_check_status",
            spec="Check rate limiter status",
            theta_prior=(
                f"Read {project_dir}/status.log. "
                "Was the rate limiter actually active during the incident? "
                "Search for rate_limit events. Prior config: {{input}}. "
                "Output: active/inactive with evidence."
            ),
        ),
        NodeSpec(
            id="n08_find_window",
            spec="Find attack time window",
            theta_prior=(
                f"Read {project_dir}/access.log. "
                "Using the suspicious IPs from step 3 and the target endpoint "
                "from step 5, find the time window (start and end) of "
                "concentrated malicious activity. "
                "Output: start_time - end_time."
            ),
        ),
        NodeSpec(
            id="n09_concurrent_events",
            spec="Check concurrent events",
            theta_prior=(
                f"Read {project_dir}/events.log. "
                "What other system events occurred during the time window: "
                "{{input}}? List events with timestamps."
            ),
        ),
        NodeSpec(
            id="n10_check_deploys",
            spec="Check deployments",
            theta_prior=(
                f"Read {project_dir}/deploys.log. "
                "Were there any deployments during or just before the attack "
                "window? Prior events: {{input}}. "
                'Output: deploy details or "none".'
            ),
        ),
        NodeSpec(
            id="n11_diff_changes",
            spec="Identify changes",
            theta_prior=(
                f"Based on deployment info: {{input}}. "
                f"Read {project_dir}/changelog.txt. "
                "What specific code changes were in that deploy? "
                "Output the relevant change description."
            ),
        ),
        NodeSpec(
            id="n12_find_vuln",
            spec="Identify vulnerability",
            effect="effectful",
            theta_prior=(
                "Based on the targeted endpoint (step 5) and the code changes "
                "(step 11): {input}. What vulnerability was likely introduced? "
                "Output: vulnerability description in 1-2 sentences."
            ),
        ),
        NodeSpec(
            id="n13_assess_data",
            spec="Assess data exposure",
            theta_prior=(
                f"Read {project_dir}/schema.json. "
                "Given the vulnerability: {{input}}. "
                "What data could have been accessed? "
                "Output: list of affected data fields."
            ),
        ),
        NodeSpec(
            id="n14_count_affected",
            spec="Count affected records",
            theta_prior=(
                f"Read {project_dir}/access.log. "
                "Count the number of successful (status 200) requests from "
                "suspicious IPs to the target endpoint during the attack "
                "window. Data exposure context: {{input}}. "
                "Output ONLY the count."
            ),
        ),
        NodeSpec(
            id="n15_report",
            spec="Produce incident report",
            theta_prior=(
                "Compile findings from all prior steps: {input}. "
                "Output a one-line incident summary in the format: "
                '"INCIDENT: [vulnerability] via [endpoint] from [IP count] '
                'IPs, [record count] records exposed, root cause: '
                '[deploy/change]."'
            ),
        ),
    ]

    edges = [
        EdgeSpec(source=nodes[i].id, target=nodes[i + 1].id)
        for i in range(len(nodes) - 1)
    ]

    return WorkflowSpec(
        name="forensics", nodes=nodes, edges=edges, entry="n01_scan_access"
    )


def load_scenarios(limit: int | None = None, start: int = 0) -> list[dict]:
    data_path = Path(__file__).parent / "data" / "forensics_5.json"
    with open(data_path) as f:
        scenarios = json.load(f)
    scenarios = scenarios[start:]
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios


def setup_scenario(scenario: dict) -> str:
    """Create a temp project dir with all data files for the scenario."""
    project_dir = tempfile.mkdtemp(prefix=f'forensics-{scenario["id"]}-')
    for filename, content in scenario["files"].items():
        filepath = Path(project_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
    return project_dir


def _normalize(val: str | list[str]) -> list[str]:
    return [val] if isinstance(val, str) else val


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
            facts_total = 0
            early_correct = 0
            early_total = 0
            mid_correct = 0
            mid_total = 0
            late_correct = 0
            late_total = 0

            for step_id, expected in scenario["expected_facts"].items():
                expected_list = _normalize(expected)
                actual = result.final_state.node_outputs.get(step_id, "")
                match = all(
                    exp.lower() in actual.lower() for exp in expected_list
                )
                facts_total += 1
                if match:
                    facts_correct += 1

                node_num = int(step_id.split("_")[0][1:])
                if node_num <= 5:
                    early_total += 1
                    if match:
                        early_correct += 1
                elif node_num <= 10:
                    mid_total += 1
                    if match:
                        mid_correct += 1
                else:
                    late_total += 1
                    if match:
                        late_correct += 1

            expected_answer = _normalize(scenario["expected_answer"])
            final_correct = all(
                exp.lower() in result.output.lower()
                for exp in expected_answer
            )

            results.append({
                "id": scenario["id"],
                "name": scenario["name"],
                "facts_score": (
                    facts_correct / facts_total if facts_total else 0
                ),
                "facts_correct": facts_correct,
                "facts_total": facts_total,
                "early_score": (
                    early_correct / early_total if early_total else 0
                ),
                "mid_score": (
                    mid_correct / mid_total if mid_total else 0
                ),
                "late_score": (
                    late_correct / late_total if late_total else 0
                ),
                "final_correct": final_correct,
                "steps_completed": result.steps_taken,
                "total_steps": len(workflow.nodes),
                "forks": result.forks_triggered,
            })

            marker = "+" if final_correct else (
                "~" if facts_correct > facts_total // 2 else "-"
            )
            print(
                f"  [{marker}] {i + 1:2d}  {scenario['id']}: "
                f"facts={facts_correct}/{facts_total} "
                f"early={early_correct}/{early_total} "
                f"mid={mid_correct}/{mid_total} "
                f"late={late_correct}/{late_total} "
                f'final={"PASS" if final_correct else "FAIL"} '
                f"steps={result.steps_taken}/15 "
                f"forks={result.forks_triggered}"
            )
        except Exception as e:
            results.append({
                "id": scenario["id"],
                "name": scenario["name"],
                "facts_score": 0.0,
                "facts_correct": 0,
                "facts_total": len(scenario.get("expected_facts", {})),
                "early_score": 0.0,
                "mid_score": 0.0,
                "late_score": 0.0,
                "final_correct": False,
                "steps_completed": 0,
                "total_steps": 15,
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
    avg_early = sum(r["early_score"] for r in results) / total
    avg_mid = sum(r["mid_score"] for r in results) / total
    avg_late = sum(r["late_score"] for r in results) / total
    final_passes = sum(1 for r in results if r["final_correct"])
    avg_steps = sum(r["steps_completed"] for r in results) / total
    total_forks = sum(r["forks"] for r in results)

    print(f'\n{"=" * 60}')
    print(f"Forensic Analysis Benchmark — {mode}")
    print(f'{"=" * 60}')
    print(f"  Avg facts score:     {avg_facts:.0%}")
    print(f"    Early (n01-n05):   {avg_early:.0%}")
    print(f"    Middle (n06-n10):  {avg_mid:.0%}")
    print(f"    Late (n11-n15):    {avg_late:.0%}")
    print(f"  Final answer:        {final_passes}/{total}"
          f" ({final_passes / total:.0%})")
    print(f"  Avg steps:           {avg_steps:.1f}/15")
    print(f"  Total forks:         {total_forks}")
    print(f'{"=" * 60}')


def main():
    parser = argparse.ArgumentParser(
        description="Forensic analysis benchmark"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--tool", action="store_true",
        help="Tool-based with engine fork",
    )
    mode_group.add_argument(
        "--session-baseline", action="store_true",
        help="Session baseline, no engine",
    )
    mode_group.add_argument(
        "--wrapped", action="store_true",
        help="Wrapped runner with engine fork",
    )
    mode_group.add_argument(
        "--factory-baseline", action="store_true",
        help="Factory SKILL.md single-prompt baseline",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--observe-mode", default="sequential",
        choices=["full", "sequential", "rewind", "lightweight", "none"],
    )
    parser.add_argument("--particles", type=int, default=3)
    args = parser.parse_args()

    if args.tool:
        from pfexec.dist.cc.runner_tool import run as run_tool

        config = EngineConfig(
            n_particles=args.particles, tau=0.4, max_forks=2,
            rewind_steps=2, max_steps=50, observe_mode=args.observe_mode,
        )

        def runner(workflow, user_input, config):
            return run_tool(
                workflow, user_input, config, backend_mode="claude"
            )

        mode = "tool"
    elif args.session_baseline:
        from pfexec.dist.cc.runner_session_baseline import run as run_sb

        config = EngineConfig(n_particles=1, tau=0.0, max_steps=50)

        def runner(workflow, user_input, config):
            return run_sb(
                workflow, user_input, config, backend_mode="claude"
            )

        mode = "session-baseline"
    elif args.wrapped:
        from pfexec.dist.cc.runner_wrapped import run as run_wrapped

        config = EngineConfig(
            n_particles=args.particles, tau=0.4, max_forks=2,
            rewind_steps=2, max_steps=50, observe_mode=args.observe_mode,
        )

        def runner(workflow, user_input, config):
            return run_wrapped(
                workflow, user_input, config, backend_mode="claude"
            )

        mode = "wrapped"
    elif args.factory_baseline:
        from pfexec.dist.cc.factory_baseline import run_factory_baseline

        config = EngineConfig(n_particles=1, tau=0.0, max_steps=50)

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

    print(f"Running Forensic Analysis benchmark ({mode})...")
    results = run_benchmark(runner, config, args.limit, args.start)
    print_summary(results, mode)


if __name__ == "__main__":
    main()
