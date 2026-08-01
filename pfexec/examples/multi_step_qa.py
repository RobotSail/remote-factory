"""Multi-step QA pipeline — demonstrates belief tracking across steps.

Workflow: decompose -> retrieve -> answer
Latent variable: question decomposition strategy (bridge vs comparison).

Usage:
    python -m pfexec.examples.multi_step_qa "What is the capital of the largest country in Europe?"
    python -m pfexec.examples.multi_step_qa "..." --dry-run
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
        name="multi_step_qa",
        nodes=[
            NodeSpec(
                id="decompose",
                spec="Decompose a complex question into sub-questions",
                theta_prior="Decompose this question into simpler parts: {input}",
            ),
            NodeSpec(
                id="retrieve",
                spec="Retrieve information to answer sub-questions",
                theta_prior="Find answers to these sub-questions: {input}",
            ),
            NodeSpec(
                id="answer",
                spec="Synthesize a final answer from retrieved information",
                theta_prior="Given the retrieved facts, answer the original question: {input}",
            ),
        ],
        edges=[
            EdgeSpec(source="decompose", target="retrieve"),
            EdgeSpec(source="retrieve", target="answer"),
        ],
        entry="decompose",
    )


def load_fixtures() -> dict[str, str]:
    fixture_path = Path(__file__).parent / "fixtures" / "multi_step_qa.json"
    with open(fixture_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Multi-step QA with pfexec")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--dry-run", action="store_true", help="Use canned responses")
    parser.add_argument("--particles", type=int, default=3, help="Number of particles")
    args = parser.parse_args()

    if args.dry_run:
        fixtures = load_fixtures()
        backend = DeterministicBackend(responses=fixtures, default=fixtures.get("default", "ok"))
    else:
        backend = ClaudeBackend()

    workflow = build_workflow()
    config = EngineConfig(n_particles=args.particles, tau=0.0, max_steps=20)
    graph = compile(workflow, backend, config)
    result = run_compiled(graph, workflow, args.question, backend, config)

    print(f"=== Multi-Step QA ===")
    print(f"Question: {args.question}")
    print(f"Steps taken: {result.steps_taken}")
    print(f"Forks: {result.forks_triggered}")
    print(f"Terminated by: {result.terminated_by}")
    print(f"\n--- Particles ---")
    for i, p in enumerate(result.final_state.belief.particles):
        print(f"  [{i}] weight={p.weight:.3f}  brief={p.brief[:60]}")
    print(f"\n--- Output ---")
    print(result.output)


if __name__ == "__main__":
    main()
