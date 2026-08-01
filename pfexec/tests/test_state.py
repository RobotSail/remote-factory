"""Tests for pfexec.state — runtime execution state."""

import random

from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


def test_particle_defaults():
    p = Particle(brief="plan A")
    assert p.weight == 1.0
    assert p.evidence == ""


def test_belief_normalize():
    b = Belief(particles=[Particle(brief="a", weight=2.0), Particle(brief="b", weight=8.0)])
    b.normalize()
    assert abs(b.particles[0].weight - 0.2) < 1e-9
    assert abs(b.particles[1].weight - 0.8) < 1e-9


def test_belief_normalize_zero():
    b = Belief(particles=[Particle(brief="a", weight=0.0), Particle(brief="b", weight=0.0)])
    b.normalize()
    assert b.particles[0].weight == 0.0


def test_belief_ess_uniform():
    n = 5
    b = Belief(particles=[Particle(brief=f"p{i}", weight=1.0) for i in range(n)])
    assert abs(b.ess() - n) < 1e-9


def test_belief_ess_degenerate():
    b = Belief(particles=[
        Particle(brief="a", weight=1.0),
        Particle(brief="b", weight=0.0),
        Particle(brief="c", weight=0.0),
    ])
    assert abs(b.ess() - 1.0) < 1e-9


def test_belief_ess_empty():
    b = Belief(particles=[])
    assert b.ess() == 0.0


def test_belief_resample_preserves_count():
    b = Belief(particles=[
        Particle(brief="a", weight=0.9),
        Particle(brief="b", weight=0.05),
        Particle(brief="c", weight=0.05),
    ])
    b.resample(rng=random.Random(42))
    assert len(b.particles) == 3


def test_belief_resample_favors_high_weight():
    b = Belief(particles=[
        Particle(brief="dominant", weight=0.99),
        Particle(brief="rare", weight=0.01),
    ])
    b.resample(n=10, rng=random.Random(42))
    assert len(b.particles) == 10
    dominant_count = sum(1 for p in b.particles if p.brief == "dominant")
    assert dominant_count >= 8


def test_belief_resample_uniform_weights():
    b = Belief(particles=[Particle(brief=f"p{i}", weight=0.5) for i in range(4)])
    b.resample(rng=random.Random(42))
    for p in b.particles:
        assert abs(p.weight - 0.25) < 1e-9


def test_trace_node_mark_dead():
    root = TraceNode(node_id="a")
    child = TraceNode(node_id="b")
    root.children.append(child)
    assert child.alive
    root.mark_dead("b")
    assert not child.alive


def test_trace_node_collect_summaries():
    root = TraceNode(node_id="a", summary="did A")
    child = TraceNode(node_id="b", summary="did B")
    root.children.append(child)
    assert root.collect_summaries() == ["did A", "did B"]


def test_trace_tree_summarize():
    tree = TraceTree(root=TraceNode(node_id="root", summary="started"))
    tree.add_step("step1", "cp1")
    tree.root.children[0].summary = "completed step1"
    assert "started" in tree.summarize()
    assert "completed step1" in tree.summarize()


def test_trace_tree_add_step():
    tree = TraceTree(root=TraceNode(node_id="root"))
    tree.add_step("a")
    tree.add_step("b")
    assert len(tree.root.children) == 1
    assert tree.root.children[0].node_id == "a"
    assert tree.root.children[0].children[0].node_id == "b"


def test_trace_tree_mark_dead():
    tree = TraceTree(root=TraceNode(node_id="root"))
    tree.add_step("a")
    tree.add_step("b")
    tree.mark_dead("b")
    leaf = tree.root.children[0].children[0]
    assert not leaf.alive


def test_execution_state():
    belief = Belief(particles=[Particle(brief="test")])
    trace = TraceTree(root=TraceNode(node_id="start"))
    state = ExecutionState(pointer="start", belief=belief, trace=trace)
    assert state.step == 0
    assert state.budget_remaining == 50
    assert state.pointer == "start"
