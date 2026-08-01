"""AFlow-style HotpotQA benchmark — 5-node multi-hop QA workflow.

Workflow: decompose -> reason_sub1 -> reason_sub2 -> ensemble -> synthesize

Usage:
    python -m pfexec.benchmarks.hotpotqa --dry-run
    python -m pfexec.benchmarks.hotpotqa --deterministic
    python -m pfexec.benchmarks.hotpotqa --pfexec
    python -m pfexec.benchmarks.hotpotqa --pfexec --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pfexec.benchmarks.eval_utils import run_eval
from pfexec.engine import EngineConfig, EngineResult, run
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec
from pfexec.llm import ClaudeBackend, DeterministicBackend, LLMBackend


def build_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        name="hotpotqa_aflow",
        nodes=[
            NodeSpec(
                id="decompose",
                spec="Decompose a multi-hop question into sub-questions",
                theta_prior=(
                    "Decompose this multi-hop question into two simpler "
                    "sub-questions that can be answered independently.\n"
                    "Question: {input}\n"
                    "List the sub-questions:"
                ),
            ),
            NodeSpec(
                id="reason_sub1",
                spec="Answer the first sub-question with chain-of-thought",
                theta_prior=(
                    "Answer the following question step by step with "
                    "chain-of-thought reasoning.\n"
                    "Question: {input}\n"
                    "Let's think step by step:"
                ),
            ),
            NodeSpec(
                id="reason_sub2",
                spec="Answer the second sub-question with chain-of-thought",
                theta_prior=(
                    "Answer the following question step by step with "
                    "chain-of-thought reasoning.\n"
                    "Question: {input}\n"
                    "Let's think step by step:"
                ),
            ),
            NodeSpec(
                id="ensemble",
                spec="ScEnsemble-style self-consistency majority voting",
                theta_prior=(
                    "Given multiple candidate answers below, determine the "
                    "most consistent answer through majority voting. "
                    "Candidates:\n{input}\n"
                    "The most consistent answer is:"
                ),
            ),
            NodeSpec(
                id="synthesize",
                spec="Combine sub-answers into a final answer",
                theta_prior=(
                    "Combine the sub-answers below into a single, concise "
                    "final answer to the original question.\n"
                    "Sub-answers: {input}\n"
                    "Final answer:"
                ),
            ),
        ],
        edges=[
            EdgeSpec(source="decompose", target="reason_sub1"),
            EdgeSpec(source="reason_sub1", target="reason_sub2"),
            EdgeSpec(source="reason_sub2", target="ensemble"),
            EdgeSpec(source="ensemble", target="synthesize"),
        ],
        entry="decompose",
    )


def load_fixtures() -> dict[str, str]:
    fixture_path = Path(__file__).parent / "fixtures" / "hotpotqa.json"
    with open(fixture_path) as f:
        return json.load(f)


def load_data(limit: int | None = None) -> list[dict]:
    data_path = Path(__file__).parent / "data" / "hotpotqa_20.json"
    with open(data_path) as f:
        questions = json.load(f)
    if limit is not None:
        questions = questions[:limit]
    return questions


def run_benchmark(
    backend: LLMBackend,
    config: EngineConfig,
    limit: int | None = None,
) -> dict:
    workflow = build_workflow()
    questions = load_data(limit)
    results: list[tuple[str, str]] = []

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item["answer"]
        result: EngineResult = run(workflow, question, backend, config)
        prediction = result.output.split("\n")[-1].strip()
        results.append((prediction, ground_truth))
        print(f"  [{i + 1}/{len(questions)}] Q: {question[:60]}...")
        print(f"           Pred: {prediction[:60]}")
        print(f"           Gold: {ground_truth}")

    return run_eval(results)


def print_summary(eval_result: dict, mode: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"HotpotQA Benchmark — {mode}")
    print(f"{'=' * 60}")
    print(f"  Avg F1:          {eval_result['avg_f1']:.4f}")
    print(f"  Avg EM:          {eval_result['avg_em']:.4f}")
    print(f"  Questions:       {len(eval_result['per_question'])}")
    print(f"{'=' * 60}")
    for i, q in enumerate(eval_result["per_question"]):
        marker = "+" if q["em"] == 1.0 else ("~" if q["f1"] > 0.5 else "-")
        print(f"  [{marker}] {i + 1:2d}  F1={q['f1']:.3f}  EM={q['em']:.0f}  "
              f"pred={q['prediction'][:40]}")


def main():
    parser = argparse.ArgumentParser(description="HotpotQA benchmark with pfexec")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true",
                            help="Use canned fixture responses")
    mode_group.add_argument("--deterministic", action="store_true",
                            help="Single-path LLM, no particles/fork")
    mode_group.add_argument("--pfexec", action="store_true",
                            help="Full probabilistic engine")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only first N questions")
    args = parser.parse_args()

    if args.dry_run:
        fixtures = load_fixtures()
        backend: LLMBackend = DeterministicBackend(
            responses=fixtures, default=fixtures.get("default", "ok"),
        )
        config = EngineConfig(n_particles=3, tau=0.0, max_steps=30)
        mode = "dry-run"
    elif args.deterministic:
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=1, tau=0.0, max_steps=30)
        mode = "deterministic"
    else:
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=5, tau=0.3, max_steps=50)
        mode = "pfexec"

    print(f"Running HotpotQA benchmark ({mode})...")
    eval_result = run_benchmark(backend, config, args.limit)
    print_summary(eval_result, mode)


if __name__ == "__main__":
    main()
