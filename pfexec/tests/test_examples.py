"""Tests for pfexec.examples — all run in dry-run mode."""

import json
from pathlib import Path

from pfexec.engine import EngineConfig, EngineResult, run
from pfexec.examples.multi_step_qa import build_workflow as build_qa, load_fixtures as qa_fixtures
from pfexec.examples.code_fix import build_workflow as build_fix, load_fixtures as fix_fixtures
from pfexec.examples.schema_mismatch import (
    build_workflow as build_schema,
    load_fixtures as schema_fixtures,
)
from pfexec.langgraph import compile, run_compiled
from pfexec.llm import DeterministicBackend


def _run_example(build_workflow, fixtures: dict[str, str], config: EngineConfig) -> EngineResult:
    backend = DeterministicBackend(responses=fixtures, default=fixtures.get("default", "ok"))
    workflow = build_workflow()
    graph = compile(workflow, backend, config)
    return run_compiled(graph, workflow, "test input", backend, config)


def test_multi_step_qa_dry_run():
    fixtures = qa_fixtures()
    config = EngineConfig(n_particles=3, tau=0.0, max_steps=20)
    result = _run_example(build_qa, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.terminated_by == "complete"
    assert result.steps_taken == 3
    assert result.output


def test_multi_step_qa_produces_valid_result():
    fixtures = qa_fixtures()
    config = EngineConfig(n_particles=2, tau=0.0, max_steps=20)
    result = _run_example(build_qa, fixtures, config)
    assert result.final_state is not None
    assert len(result.final_state.belief.particles) > 0


def test_code_fix_dry_run():
    fixtures = fix_fixtures()
    config = EngineConfig(n_particles=3, tau=0.4, max_forks=2, rewind_steps=2, max_steps=30)
    result = _run_example(build_fix, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.output


def test_code_fix_triggers_fork():
    fixtures = fix_fixtures()
    config = EngineConfig(n_particles=3, tau=0.99, max_forks=2, rewind_steps=2, max_steps=30)
    result = _run_example(build_fix, fixtures, config)
    assert result.forks_triggered >= 1


def test_schema_mismatch_dry_run():
    fixtures = schema_fixtures()
    config = EngineConfig(n_particles=3, tau=0.4, max_forks=2, rewind_steps=2, max_steps=30)
    result = _run_example(build_schema, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.output


def test_schema_mismatch_triggers_resample():
    fixtures = schema_fixtures()
    config = EngineConfig(n_particles=3, tau=0.99, max_forks=2, rewind_steps=1, max_steps=30)
    result = _run_example(build_schema, fixtures, config)
    assert result.final_state is not None
    assert len(result.final_state.belief.particles) == 3


def test_all_examples_produce_valid_engine_result():
    for build_fn, fixture_fn in [
        (build_qa, qa_fixtures),
        (build_fix, fix_fixtures),
        (build_schema, schema_fixtures),
    ]:
        fixtures = fixture_fn()
        config = EngineConfig(n_particles=2, tau=0.0, max_steps=20)
        result = _run_example(build_fn, fixtures, config)
        assert isinstance(result, EngineResult)
        assert result.final_state is not None
        assert result.steps_taken > 0
        assert result.output


def test_fixtures_are_valid_json():
    fixture_dir = Path(__file__).parent.parent / "examples" / "fixtures"
    for f in fixture_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        assert isinstance(data, dict)
        assert "Generate" in data


def _run_example_engine(build_workflow, fixtures: dict[str, str], config: EngineConfig) -> EngineResult:
    backend = DeterministicBackend(responses=fixtures, default=fixtures.get("default", "ok"))
    workflow = build_workflow()
    return run(workflow, "test input", backend, config)


def test_sequential_mode_qa():
    fixtures = qa_fixtures()
    config = EngineConfig(n_particles=1, tau=0.0, max_steps=20, observe_mode="sequential")
    result = _run_example_engine(build_qa, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.terminated_by == "complete"
    assert result.output


def test_rewind_mode_qa():
    fixtures = qa_fixtures()
    config = EngineConfig(n_particles=1, tau=0.0, max_steps=20, observe_mode="rewind")
    result = _run_example_engine(build_qa, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.terminated_by == "complete"
    assert result.output


def test_lightweight_mode_qa():
    fixtures = qa_fixtures()
    config = EngineConfig(n_particles=3, tau=0.0, max_steps=20, observe_mode="lightweight")
    result = _run_example_engine(build_qa, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.terminated_by == "complete"
    assert result.output


def test_sequential_mode_code_fix():
    fixtures = fix_fixtures()
    config = EngineConfig(n_particles=1, tau=0.0, max_steps=30, observe_mode="sequential")
    result = _run_example_engine(build_fix, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.output


def test_rewind_mode_code_fix():
    fixtures = fix_fixtures()
    config = EngineConfig(n_particles=1, tau=0.0, max_steps=30, observe_mode="rewind")
    result = _run_example_engine(build_fix, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.output


def test_lightweight_mode_code_fix():
    fixtures = fix_fixtures()
    config = EngineConfig(n_particles=3, tau=0.0, max_steps=30, observe_mode="lightweight")
    result = _run_example_engine(build_fix, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.output


def test_sequential_mode_schema():
    fixtures = schema_fixtures()
    config = EngineConfig(n_particles=1, tau=0.0, max_steps=30, observe_mode="sequential")
    result = _run_example_engine(build_schema, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.output


def test_rewind_mode_schema():
    fixtures = schema_fixtures()
    config = EngineConfig(n_particles=1, tau=0.0, max_steps=30, observe_mode="rewind")
    result = _run_example_engine(build_schema, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.output


def test_lightweight_mode_schema():
    fixtures = schema_fixtures()
    config = EngineConfig(n_particles=3, tau=0.0, max_steps=30, observe_mode="lightweight")
    result = _run_example_engine(build_schema, fixtures, config)
    assert isinstance(result, EngineResult)
    assert result.output
