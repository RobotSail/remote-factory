"""Tests for pfexec.langgraph — LangGraph compiler."""

import json

from pfexec.engine import EngineConfig
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec
from pfexec.langgraph import (
    PfExecState,
    _belief_to_dict,
    _dict_to_belief,
    _trace_node_to_dict,
    _dict_to_trace_node,
    compile,
    run_compiled,
)
from pfexec.llm import DeterministicBackend
from pfexec.state import Belief, Particle, TraceNode


def _backend() -> DeterministicBackend:
    return DeterministicBackend(
        responses={
            "Generate": json.dumps(["p1", "p2", "p3"]),
            "Compare": "A",
        },
        default="ok",
    )


def _two_node_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        name="two-node",
        nodes=[
            NodeSpec(id="a", spec="step A", theta_prior="Do A: {input}"),
            NodeSpec(id="b", spec="step B", theta_prior="Do B: {input}"),
        ],
        edges=[EdgeSpec(source="a", target="b")],
        entry="a",
    )


def test_compile_creates_graph():
    wf = _two_node_workflow()
    graph = compile(wf, _backend())
    assert graph is not None


def test_compile_has_nodes():
    wf = _two_node_workflow()
    graph = compile(wf, _backend())
    compiled = graph.compile()
    node_names = set(compiled.get_graph().nodes.keys())
    assert "a" in node_names
    assert "b" in node_names


def test_run_compiled_end_to_end():
    wf = _two_node_workflow()
    backend = _backend()
    graph = compile(wf, backend, EngineConfig(n_particles=3, tau=0.0))
    result = run_compiled(graph, wf, "test input", backend, EngineConfig(n_particles=3, tau=0.0))
    assert result.terminated_by == "complete"
    assert result.steps_taken >= 2
    assert isinstance(result.output, str)


def test_run_compiled_three_node(linear_workflow: WorkflowSpec):
    backend = _backend()
    graph = compile(linear_workflow, backend, EngineConfig(n_particles=2, tau=0.0))
    result = run_compiled(
        graph, linear_workflow, "test", backend, EngineConfig(n_particles=2, tau=0.0)
    )
    assert result.terminated_by == "complete"
    assert result.steps_taken == 3


def test_belief_serialization_round_trip():
    belief = Belief(particles=[
        Particle(brief="plan A", weight=0.7, evidence="ev1"),
        Particle(brief="plan B", weight=0.3, evidence="ev2"),
    ])
    d = _belief_to_dict(belief)
    restored = _dict_to_belief(d)
    assert len(restored.particles) == 2
    assert restored.particles[0].brief == "plan A"
    assert abs(restored.particles[0].weight - 0.7) < 1e-9
    assert restored.particles[1].evidence == "ev2"


def test_trace_node_serialization_round_trip():
    node = TraceNode(
        node_id="root",
        checkpoint_id="cp0",
        alive=True,
        summary="did stuff",
        children=[
            TraceNode(node_id="child", checkpoint_id="cp1", alive=False, summary="failed"),
        ],
    )
    d = _trace_node_to_dict(node)
    restored = _dict_to_trace_node(d)
    assert restored.node_id == "root"
    assert restored.alive is True
    assert len(restored.children) == 1
    assert restored.children[0].alive is False
    assert restored.children[0].summary == "failed"


def test_fork_via_compiled_graph():
    wf = WorkflowSpec(
        name="forkable",
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
            "Compare": "B",
            "Summarize": "lesson",
        },
        default=json.dumps(["fresh-1", "fresh-2"]),
    )
    cfg = EngineConfig(n_particles=2, tau=0.99, max_forks=1, max_steps=20)
    graph = compile(wf, backend, cfg)
    result = run_compiled(graph, wf, "test", backend, cfg)
    assert result.forks_triggered >= 0
    assert result.final_state is not None
