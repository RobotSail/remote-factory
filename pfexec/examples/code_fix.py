"""Code bug localization and fix — demonstrates fork trigger.

Workflow: localize -> patch -> test (effectful)
Latent variable: which module has the bug.
When test fails and suffix score drops, fork back to localize.

Usage:
    python -m pfexec.examples.code_fix "Fix the off-by-one error in utils.py"
    python -m pfexec.examples.code_fix "..." --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pfexec.engine import EngineConfig
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec
from pfexec.langgraph import compile, run_compiled
from pfexec.llm import ClaudeBackend, DeterministicBackend


def build_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        name="code_fix",
        nodes=[
            NodeSpec(
                id="localize",
                spec="Localize the bug in the codebase",
                theta_prior="Analyze the codebase to find the bug: {input}",
            ),
            NodeSpec(
                id="patch",
                spec="Generate a code patch to fix the bug",
                theta_prior="Write a fix for the localized bug: {input}",
            ),
            NodeSpec(
                id="test",
                spec="Run tests to verify the fix",
                theta_prior="Run the test suite to verify: {input}",
                effect="effectful",
            ),
        ],
        edges=[
            EdgeSpec(source="localize", target="patch"),
            EdgeSpec(source="patch", target="test"),
        ],
        entry="localize",
    )


def load_fixtures() -> dict[str, str]:
    fixture_path = Path(__file__).parent / "fixtures" / "code_fix.json"
    with open(fixture_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Code fix with pfexec")
    parser.add_argument("task", help="Description of the bug to fix")
    parser.add_argument("--dry-run", action="store_true", help="Use canned responses")
    parser.add_argument("--particles", type=int, default=3, help="Number of particles")
    args = parser.parse_args()

    if args.dry_run:
        fixtures = load_fixtures()
        backend = DeterministicBackend(responses=fixtures, default=fixtures.get("default", "ok"))
    else:
        backend = ClaudeBackend()

    workflow = build_workflow()
    config = EngineConfig(
        n_particles=args.particles,
        tau=0.4,
        max_forks=2,
        rewind_steps=2,
        max_steps=30,
    )
    graph = compile(workflow, backend, config)
    result = run_compiled(graph, workflow, args.task, backend, config)

    print("=== Code Fix ===")
    print(f"Task: {args.task}")
    print(f"Steps taken: {result.steps_taken}")
    print(f"Forks triggered: {result.forks_triggered}")
    print(f"Terminated by: {result.terminated_by}")
    print("\n--- Particles ---")
    for i, p in enumerate(result.final_state.belief.particles):
        print(f"  [{i}] weight={p.weight:.3f}  brief={p.brief[:60]}")
    print("\n--- Output ---")
    print(result.output)


if __name__ == "__main__":
    main()
