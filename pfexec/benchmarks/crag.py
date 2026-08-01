"""Corrective RAG benchmark — 4-node retrieval-augmented generation workflow.

Workflow: retrieve -> grade -> web_search -> generate
The grade node output determines whether web_search does real work or passes through.

Usage:
    python -m pfexec.benchmarks.crag --dry-run
    python -m pfexec.benchmarks.crag --deterministic
    python -m pfexec.benchmarks.crag --pfexec
    python -m pfexec.benchmarks.crag --pfexec --limit 5
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
        name="crag",
        nodes=[
            NodeSpec(
                id="retrieve",
                spec="Retrieve relevant documents for the query",
                theta_prior=(
                    "Retrieve relevant documents for this question. "
                    "Return the most relevant passage.\n"
                    "Question: {input}\n"
                    "Retrieved document:"
                ),
            ),
            NodeSpec(
                id="grade",
                spec="Assess relevance of retrieved documents",
                theta_prior=(
                    "Assess the relevance of the retrieved document to the "
                    "question. Reply RELEVANT if it answers the question, "
                    "or NOT_RELEVANT if it does not.\n"
                    "Document: {input}\n"
                    "Relevance:"
                ),
            ),
            NodeSpec(
                id="web_search",
                spec="Fallback web search when retrieval quality is poor",
                theta_prior=(
                    "Search the web for an answer to this question. If the "
                    "previous grading was RELEVANT, simply pass through the "
                    "existing answer. Otherwise, provide a web search result.\n"
                    "Context: {input}\n"
                    "Web search result:"
                ),
            ),
            NodeSpec(
                id="generate",
                spec="Generate final answer from best available documents",
                theta_prior=(
                    "Generate a comprehensive answer to the original "
                    "question based on the available documents and search "
                    "results.\n"
                    "Documents: {input}\n"
                    "Answer:"
                ),
            ),
        ],
        edges=[
            EdgeSpec(source="retrieve", target="grade"),
            EdgeSpec(source="grade", target="web_search"),
            EdgeSpec(source="web_search", target="generate"),
        ],
        entry="retrieve",
    )


def load_fixtures() -> dict[str, str]:
    fixture_path = Path(__file__).parent / "fixtures" / "crag.json"
    with open(fixture_path) as f:
        return json.load(f)


def load_data(limit: int | None = None) -> list[dict]:
    data_path = Path(__file__).parent / "data" / "crag_15.json"
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
    print(f"CRAG Benchmark — {mode}")
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
    parser = argparse.ArgumentParser(description="CRAG benchmark with pfexec")
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
        config = EngineConfig(n_particles=3, tau=0.0, max_steps=25)
        mode = "dry-run"
    elif args.deterministic:
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=1, tau=0.0, max_steps=25)
        mode = "deterministic"
    else:
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=5, tau=0.3, max_steps=40)
        mode = "pfexec"

    print(f"Running CRAG benchmark ({mode})...")
    eval_result = run_benchmark(backend, config, args.limit)
    print_summary(eval_result, mode)


if __name__ == "__main__":
    main()
