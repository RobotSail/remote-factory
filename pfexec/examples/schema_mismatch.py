"""Schema format discovery — demonstrates mid-run belief shift and resample.

Workflow: parse -> transform -> validate
Planted format mismatch discovered at validate step.

Usage:
    python -m pfexec.examples.schema_mismatch "Convert the customer records"
    python -m pfexec.examples.schema_mismatch "..." --dry-run
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
        name="schema_mismatch",
        nodes=[
            NodeSpec(
                id="parse",
                spec="Parse input records and detect schema",
                theta_prior="Parse the input data and identify the schema: {input}",
            ),
            NodeSpec(
                id="transform",
                spec="Transform records to target format",
                theta_prior="Transform the parsed records to the target schema: {input}",
            ),
            NodeSpec(
                id="validate",
                spec="Validate transformed records against target schema",
                theta_prior="Validate all transformed records: {input}",
                effect="effectful",
            ),
        ],
        edges=[
            EdgeSpec(source="parse", target="transform"),
            EdgeSpec(source="transform", target="validate"),
        ],
        entry="parse",
    )


def load_fixtures() -> dict[str, str]:
    fixture_path = Path(__file__).parent / "fixtures" / "schema_mismatch.json"
    with open(fixture_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Schema mismatch recovery with pfexec")
    parser.add_argument("task", help="Description of the conversion task")
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

    print("=== Schema Mismatch Recovery ===")
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
