"""Tests for pfexec.benchmarks — all run in dry-run mode."""

import json
from pathlib import Path

from pfexec.benchmarks.eval_utils import exact_match, f1_score, normalize_answer, run_eval
from pfexec.benchmarks.hotpotqa import (
    build_workflow as build_hotpotqa,
    load_fixtures as hotpotqa_fixtures,
)
from pfexec.benchmarks.crag import (
    build_workflow as build_crag,
    load_fixtures as crag_fixtures,
)
from pfexec.engine import EngineConfig, EngineResult, run
from pfexec.llm import DeterministicBackend


def _run_benchmark(build_workflow, fixtures: dict[str, str], config: EngineConfig) -> EngineResult:
    backend = DeterministicBackend(responses=fixtures, default=fixtures.get("default", "ok"))
    workflow = build_workflow()
    return run(workflow, "test input", backend, config)


# --- HotpotQA tests ---


def test_hotpotqa_dry_run():
    fixtures = hotpotqa_fixtures()
    config = EngineConfig(n_particles=3, tau=0.0, max_steps=30)
    result = _run_benchmark(build_hotpotqa, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.terminated_by == "complete"
    assert result.steps_taken == 5
    assert result.output


def test_hotpotqa_eval_f1():
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("the answer is yes", "yes") > 0.0
    assert f1_score("completely wrong answer", "yes") == 0.0


# --- CRAG tests ---


def test_crag_dry_run():
    fixtures = crag_fixtures()
    config = EngineConfig(n_particles=3, tau=0.0, max_steps=25)
    result = _run_benchmark(build_crag, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.terminated_by == "complete"
    assert result.steps_taken == 4
    assert result.output


def test_crag_routing():
    """Verify the grade node output influences web_search behavior."""
    fixtures = crag_fixtures()
    config = EngineConfig(n_particles=3, tau=0.0, max_steps=25)
    backend = DeterministicBackend(responses=fixtures, default=fixtures.get("default", "ok"))
    workflow = build_crag()
    result = run(workflow, "What is the capital of France?", backend, config)
    assert "RELEVANT" in result.output or "Paris" in result.output


# --- eval_utils tests ---


def test_normalize_answer():
    assert normalize_answer("The Quick Brown Fox") == "quick brown fox"
    assert normalize_answer("  a  an  the  ") == ""
    assert normalize_answer("Hello, World!") == "hello world"
    assert normalize_answer("U.S.A.") == "usa"
    assert normalize_answer("  multiple   spaces  ") == "multiple spaces"


def test_f1_score():
    assert f1_score("paris", "paris") == 1.0
    assert f1_score("the capital is paris", "paris") > 0.0
    assert f1_score("london", "paris") == 0.0
    assert f1_score("", "") == 1.0
    assert f1_score("", "paris") == 0.0
    assert f1_score("paris", "") == 0.0

    f1 = f1_score("john hopfield and geoffrey hinton", "john hopfield and geoffrey hinton")
    assert f1 == 1.0

    f1_partial = f1_score("john hopfield", "john hopfield and geoffrey hinton")
    assert 0.0 < f1_partial < 1.0


def test_exact_match():
    assert exact_match("Paris", "paris") == 1.0
    assert exact_match("The Paris", "paris") == 1.0
    assert exact_match("London", "Paris") == 0.0


def test_run_eval():
    results = [("paris", "paris"), ("london", "paris"), ("yes", "yes")]
    eval_result = run_eval(results)
    assert "avg_f1" in eval_result
    assert "avg_em" in eval_result
    assert "per_question" in eval_result
    assert len(eval_result["per_question"]) == 3
    assert eval_result["per_question"][0]["f1"] == 1.0
    assert eval_result["per_question"][1]["f1"] == 0.0


def test_eval_data_valid_json():
    data_dir = Path(__file__).parent.parent / "benchmarks" / "data"
    for f in data_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) > 0
        for item in data:
            assert "question" in item
            assert "answer" in item
