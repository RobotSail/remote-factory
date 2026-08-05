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
from collections.abc import Callable
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
                    "If the question asks whether/if something is true, answer yes or no.\n"
                    "Sub-answers: {input}\n"
                    "Final answer:\n"
                    "Output ONLY the answer in 1-5 words, no explanation."
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


def load_data(limit: int | None = None, start: int = 0) -> list[dict]:
    data_path = Path(__file__).parent / "data" / "hotpotqa_20.json"
    with open(data_path) as f:
        questions = json.load(f)
    questions = questions[start:]
    if limit is not None:
        questions = questions[:limit]
    return questions


def run_benchmark(
    backend: LLMBackend,
    config: EngineConfig,
    limit: int | None = None,
    runner: Callable[[WorkflowSpec, str, EngineConfig], EngineResult] | None = None,
    start: int = 0,
) -> dict:
    workflow = build_workflow()
    questions = load_data(limit, start=start)
    total_nodes = len(workflow.nodes)
    results: list[tuple[str, str]] = []
    completion_rates: list[float] = []

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item["answer"]
        if runner is not None:
            result: EngineResult = runner(workflow, question, config)
        else:
            result = run(workflow, question, backend, config)
        prediction = result.output.split("\n")[-1].strip()
        results.append((prediction, ground_truth))
        node_rate = result.steps_taken / total_nodes if total_nodes else 0.0
        completion_rates.append(node_rate)
        print(f"  [{i + 1}/{len(questions)}] Q: {question[:60]}...")
        print(f"           Pred: {prediction[:60]}")
        print(f"           Gold: {ground_truth}")
        print(f"           Nodes: {result.steps_taken}/{total_nodes} ({node_rate:.0%})")

    eval_result = run_eval(results)
    avg_completion = sum(completion_rates) / len(completion_rates) if completion_rates else 0.0
    eval_result["avg_node_completion"] = avg_completion
    return eval_result


def print_summary(eval_result: dict, mode: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"HotpotQA Benchmark — {mode}")
    print(f"{'=' * 60}")
    print(f"  Avg F1:          {eval_result['avg_f1']:.4f}")
    print(f"  Avg EM:          {eval_result['avg_em']:.4f}")
    if "avg_node_completion" in eval_result:
        print(f"  Node Completion: {eval_result['avg_node_completion']:.1%}")
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
    mode_group.add_argument("--factory-baseline", action="store_true",
                            help="Factory SKILL.md single-prompt baseline")
    mode_group.add_argument("--agentic", action="store_true",
                            help="Agentic mode with PostToolUse hooks")
    mode_group.add_argument("--agentic-v3", action="store_true",
                            help="Agentic mode with engine-computed hints via hooks")
    mode_group.add_argument("--wrapped", action="store_true",
                            help="Wrapped mode: claude --bare with engine in wrapper")
    mode_group.add_argument("--tool", action="store_true",
                            help="Tool-based mode: Claude drives loop via pfexec CLI")
    mode_group.add_argument("--session-baseline", action="store_true",
                            help="Session baseline: SKILL.md + tools, no engine")
    parser.add_argument("--observe-mode", type=str, default="full",
                        choices=["full", "sequential", "rewind", "lightweight", "none"],
                        help="Observe mode for belief updates")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only first N questions")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip first N questions")
    parser.add_argument("--particles", type=int, default=None,
                        help="Number of particles (overrides mode default)")
    args = parser.parse_args()

    runner: Callable[[WorkflowSpec, str, EngineConfig], EngineResult] | None = None

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
    elif args.factory_baseline:
        from pfexec.dist.cc.factory_baseline import run_factory_baseline
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=1, tau=0.0, max_steps=30)
        runner = run_factory_baseline
        mode = "factory-baseline"
    elif args.agentic:
        from pfexec.dist.cc.runner import _run_agentic
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=5, tau=0.3, max_steps=50, observe_mode=args.observe_mode)

        def agentic_runner(workflow, user_input, config):
            return _run_agentic(workflow, user_input, config, backend_mode="claude")

        runner = agentic_runner
        mode = "agentic"
    elif args.agentic_v3:
        from pfexec.dist.cc.runner_agentic import run as run_agentic_v3
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=5, tau=0.3, max_steps=50)

        def agentic_v3_runner(workflow, user_input, config):
            return run_agentic_v3(workflow, user_input, config, backend_mode="claude")

        runner = agentic_v3_runner
        mode = "agentic-v3"
    elif args.wrapped:
        from pfexec.dist.cc.runner_wrapped import run as run_wrapped
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=5, tau=0.3, max_steps=50, observe_mode=args.observe_mode)

        def wrapped_runner(workflow, user_input, config):
            return run_wrapped(workflow, user_input, config, backend_mode="claude")

        runner = wrapped_runner
        mode = "wrapped"
    elif args.tool:
        from pfexec.dist.cc.runner_tool import run as run_tool
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=5, tau=0.3, max_steps=50, observe_mode=args.observe_mode)

        def tool_runner(workflow, user_input, config):
            return run_tool(workflow, user_input, config, backend_mode="claude")

        runner = tool_runner
        mode = "tool"
    elif args.session_baseline:
        from pfexec.dist.cc.runner_session_baseline import run as run_session_baseline
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=1, tau=0.0, max_steps=30)

        def session_baseline_runner(workflow, user_input, config):
            return run_session_baseline(workflow, user_input, config, backend_mode='claude')

        runner = session_baseline_runner
        mode = "session-baseline"
    else:
        backend = ClaudeBackend()
        config = EngineConfig(n_particles=5, tau=0.3, max_steps=50, observe_mode=args.observe_mode)
        mode = "pfexec" if args.observe_mode == "full" else f"pfexec (observe={args.observe_mode})"

    if args.particles is not None:
        config = EngineConfig(
            n_particles=args.particles,
            tau=config.tau,
            max_steps=config.max_steps,
            max_forks=config.max_forks,
            rewind_steps=config.rewind_steps,
            observe_mode=config.observe_mode,
        )

    print(f"Running HotpotQA benchmark ({mode})...")
    eval_result = run_benchmark(backend, config, args.limit, runner=runner, start=args.start)
    print_summary(eval_result, mode)


if __name__ == "__main__":
    main()
