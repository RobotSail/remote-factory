"""Tests for pfexec.engine — DAG execution loop."""

import json

from pfexec.engine import EngineConfig, EngineResult, run, _suffix_score, _topological_successors
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec
from pfexec.llm import DeterministicBackend
from pfexec.state import Belief, Particle


def _backend(default: str = "ok") -> DeterministicBackend:
    return DeterministicBackend(
        responses={"Generate": json.dumps(["p1", "p2", "p3"])},
        default=default,
    )


def test_linear_workflow_completes(linear_workflow: WorkflowSpec):
    result = run(linear_workflow, "test", _backend(), EngineConfig(n_particles=3, tau=0.0))
    assert result.terminated_by == "complete"
    assert result.steps_taken == 3
    assert result.forks_triggered == 0


def test_budget_exhaustion():
    wf = WorkflowSpec(
        name="long",
        nodes=[
            NodeSpec(id=f"n{i}", spec=f"step {i}", theta_prior="Do: {input}")
            for i in range(10)
        ],
        edges=[
            EdgeSpec(source=f"n{i}", target=f"n{i+1}")
            for i in range(9)
        ],
        entry="n0",
    )
    result = run(wf, "test", _backend(), EngineConfig(n_particles=2, max_steps=3, tau=0.0))
    assert result.terminated_by == "budget"
    assert result.steps_taken == 3


def test_fork_triggers_on_low_suffix_score():
    wf = WorkflowSpec(
        name="forkable",
        nodes=[
            NodeSpec(id="a", spec="step A", theta_prior="Do A: {input}"),
            NodeSpec(id="b", spec="step B", theta_prior="Do B: {input}"),
            NodeSpec(id="c", spec="step C", theta_prior="Do C: {input}"),
        ],
        edges=[
            EdgeSpec(source="a", target="b"),
            EdgeSpec(source="b", target="c"),
        ],
        entry="a",
    )
    backend = DeterministicBackend(
        responses={
            "Generate": json.dumps(["p1", "p2", "p3"]),
            "Compare": "B",
            "Summarize": "lesson",
        },
        default=json.dumps(["fresh-1", "fresh-2", "fresh-3"]),
    )
    result = run(wf, "test", backend, EngineConfig(n_particles=3, tau=0.99, max_forks=1, max_steps=20))
    assert result.forks_triggered >= 1


def test_max_forks_limit():
    wf = WorkflowSpec(
        name="fork-limit",
        nodes=[
            NodeSpec(id="a", spec="A", theta_prior="{input}"),
            NodeSpec(id="b", spec="B", theta_prior="{input}"),
        ],
        edges=[EdgeSpec(source="a", target="b")],
        entry="a",
    )
    backend = DeterministicBackend(
        responses={
            "Generate": json.dumps(["p1", "p2"]),
            "Summarize": "lesson",
        },
        default=json.dumps(["r1", "r2"]),
    )
    result = run(wf, "test", backend, EngineConfig(
        n_particles=2, tau=0.99, max_forks=2, max_steps=30,
    ))
    assert result.forks_triggered <= 2


def test_branching_dag_follows_edges(branching_workflow: WorkflowSpec):
    result = run(branching_workflow, "test", _backend(), EngineConfig(n_particles=2, tau=0.0))
    assert result.terminated_by == "complete"
    assert result.steps_taken >= 2


def test_topological_successors():
    wf = WorkflowSpec(
        name="test",
        nodes=[
            NodeSpec(id="a", spec="A", theta_prior="p"),
            NodeSpec(id="b", spec="B", theta_prior="p"),
            NodeSpec(id="c", spec="C", theta_prior="p"),
        ],
        edges=[
            EdgeSpec(source="a", target="b"),
            EdgeSpec(source="a", target="c"),
        ],
        entry="a",
    )
    succs = _topological_successors(wf, "a")
    assert set(succs) == {"b", "c"}
    assert _topological_successors(wf, "b") == []


def test_suffix_score_uniform():
    b = Belief(particles=[Particle(brief=f"p{i}", weight=1.0) for i in range(5)])
    score = _suffix_score(b, k=3)
    assert abs(score - 0.2) < 1e-9


def test_suffix_score_degenerate():
    b = Belief(particles=[
        Particle(brief="winner", weight=1.0),
        Particle(brief="loser", weight=0.0),
    ])
    score = _suffix_score(b, k=1)
    assert abs(score - 1.0) < 1e-9


def test_suffix_score_empty():
    b = Belief(particles=[])
    assert _suffix_score(b) == 0.0


def test_engine_result_structure(linear_workflow: WorkflowSpec):
    result = run(linear_workflow, "test", _backend(), EngineConfig(n_particles=2, tau=0.0))
    assert isinstance(result, EngineResult)
    assert result.final_state is not None
    assert isinstance(result.output, str)
    assert result.steps_taken > 0
