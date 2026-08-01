"""Tests for pfexec.primitives — core inference primitives."""

import json
import random

from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec
from pfexec.llm import DeterministicBackend
from pfexec.primitives import _extract_json, fork, init, observe, sample


def _make_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        name="test",
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


def test_init_produces_n_particles():
    wf = _make_workflow()
    backend = DeterministicBackend(
        responses={"Generate": json.dumps(["plan-A", "plan-B", "plan-C"])}
    )
    state = init(wf, "test input", n_particles=3, backend=backend)
    assert len(state.belief.particles) == 3
    assert state.pointer == "a"
    assert state.step == 0


def test_init_uniform_weights():
    wf = _make_workflow()
    backend = DeterministicBackend(
        responses={"Generate": json.dumps(["a", "b", "c", "d"])}
    )
    state = init(wf, "test", n_particles=4, backend=backend)
    for p in state.belief.particles:
        assert abs(p.weight - 0.25) < 1e-9


def test_init_pads_when_few_briefs():
    wf = _make_workflow()
    backend = DeterministicBackend(responses={"Generate": json.dumps(["only-one"])})
    state = init(wf, "test", n_particles=3, backend=backend)
    assert len(state.belief.particles) == 3


def test_init_handles_non_json():
    wf = _make_workflow()
    backend = DeterministicBackend(default="not json at all")
    state = init(wf, "test", n_particles=2, backend=backend)
    assert len(state.belief.particles) == 2


def test_sample_produces_output():
    wf = _make_workflow()
    backend = DeterministicBackend(
        responses={"Generate": json.dumps(["brief-1", "brief-2"])},
        default="sample output",
    )
    state = init(wf, "test", n_particles=2, backend=backend)
    node = wf.nodes[0]
    new_state, output = sample(state, node, backend, rng=random.Random(42))
    assert output == "sample output"
    assert new_state.step == 1
    assert new_state.budget_remaining == 49


def test_sample_effectful_node():
    wf = _make_workflow()
    node = NodeSpec(id="eff", spec="run tests", theta_prior="Test: {input}", effect="effectful")
    backend = DeterministicBackend(
        responses={"Generate": json.dumps(["p1", "p2"])},
        default="effectful output",
    )
    state = init(wf, "test", n_particles=2, backend=backend)
    _, output = sample(state, node, backend, rng=random.Random(42))
    assert output == "effectful output"


def test_observe_updates_weights():
    wf = _make_workflow()
    backend = DeterministicBackend(
        responses={
            "Generate": json.dumps(["good plan", "bad plan", "ok plan"]),
            "Compare": "A",
        },
        default="A",
    )
    state = init(wf, "test", n_particles=3, backend=backend)
    new_state = observe(state, "the test passed", backend)
    assert len(new_state.belief.particles) == 3
    new_state.belief.normalize()
    assert all(abs(p.weight) >= 0 for p in new_state.belief.particles)


def test_observe_triggers_resample_on_low_ess():
    wf = _make_workflow()
    backend = DeterministicBackend(
        responses={"Generate": json.dumps(["winner", "loser1", "loser2", "loser3"])},
        default="A",
    )
    state = init(wf, "test", n_particles=4, backend=backend)
    state.belief.particles[0].weight = 100.0
    state.belief.particles[1].weight = 0.001
    state.belief.particles[2].weight = 0.001
    state.belief.particles[3].weight = 0.001
    new_state = observe(state, "observation", backend)
    new_state.belief.normalize()
    weights = [p.weight for p in new_state.belief.particles]
    assert all(w > 0 for w in weights)


def test_observe_single_particle():
    wf = _make_workflow()
    backend = DeterministicBackend(
        responses={"Generate": json.dumps(["solo"])},
        default="ok",
    )
    state = init(wf, "test", n_particles=1, backend=backend)
    new_state = observe(state, "obs", backend)
    assert len(new_state.belief.particles) == 1


def test_fork_marks_dead_and_rewinds():
    wf = _make_workflow()
    backend = DeterministicBackend(
        responses={
            "Generate": json.dumps(["p1", "p2"]),
            "Summarize": "failed because X",
            "fresh": json.dumps(["new-p1", "new-p2"]),
        },
        default=json.dumps(["rejuv-1", "rejuv-2"]),
    )
    state = init(wf, "test", n_particles=2, backend=backend)
    node_a = wf.nodes[0]
    state, _ = sample(state, node_a, backend, rng=random.Random(42))
    state.pointer = "b"
    node_b = wf.nodes[1]
    state, _ = sample(state, node_b, backend, rng=random.Random(42))
    state.pointer = "c"

    new_state = fork(state, k=2, backend=backend)
    assert len(new_state.belief.particles) == 2
    for p in new_state.belief.particles:
        assert abs(p.weight - 0.5) < 1e-9


def test_fork_generates_new_briefs():
    wf = _make_workflow()
    backend = DeterministicBackend(
        responses={
            "Generate": json.dumps(["old-1", "old-2", "old-3"]),
            "Summarize": "lesson",
            "fresh": json.dumps(["new-1", "new-2", "new-3"]),
        },
        default=json.dumps(["r1", "r2", "r3"]),
    )
    state = init(wf, "test", n_particles=3, backend=backend)
    state.pointer = "b"
    new_state = fork(state, k=1, backend=backend)
    assert len(new_state.belief.particles) == 3


def test_round_trip_init_sample_observe_fork_sample():
    wf = _make_workflow()
    backend = DeterministicBackend(
        responses={
            "Generate": json.dumps(["plan-A", "plan-B"]),
            "Compare": "A",
            "Summarize": "learned X",
        },
        default=json.dumps(["fresh-1", "fresh-2"]),
    )
    state = init(wf, "test input", n_particles=2, backend=backend)
    assert state.pointer == "a"

    state, out1 = sample(state, wf.nodes[0], backend, rng=random.Random(1))
    assert state.step == 1

    state = observe(state, "observation 1", backend)

    state.pointer = "b"
    state = fork(state, k=1, backend=backend)

    backend_2 = DeterministicBackend(default="final output")
    state, out2 = sample(state, wf.nodes[1], backend_2, rng=random.Random(2))
    assert state.step == 2
    assert out2 == "final output"


class TestExtractJson:
    def test_strips_json_fence(self):
        raw = '```json\n["a", "b", "c"]\n```'
        assert _extract_json(raw) == '["a", "b", "c"]'

    def test_strips_bare_fence(self):
        raw = '```\n["a", "b"]\n```'
        assert _extract_json(raw) == '["a", "b"]'

    def test_passes_through_plain_json(self):
        raw = '["a", "b"]'
        assert _extract_json(raw) == '["a", "b"]'

    def test_strips_surrounding_whitespace(self):
        raw = '  \n ["a"]  \n '
        assert _extract_json(raw) == '["a"]'

    def test_fence_with_surrounding_text(self):
        raw = 'Here is the JSON:\n```json\n{"key": "val"}\n```\nDone.'
        assert _extract_json(raw) == '{"key": "val"}'


def test_init_handles_markdown_fenced_json():
    wf = _make_workflow()
    fenced = '```json\n["plan-A", "plan-B", "plan-C"]\n```'
    backend = DeterministicBackend(responses={"Generate": fenced})
    state = init(wf, "test input", n_particles=3, backend=backend)
    assert len(state.belief.particles) == 3
    assert state.belief.particles[0].brief == "plan-A"


def test_fork_handles_markdown_fenced_json():
    wf = _make_workflow()
    fenced_init = '```json\n["p1", "p2"]\n```'
    fenced_rejuv = '```json\n["new-1", "new-2"]\n```'
    backend = DeterministicBackend(
        responses={
            "diverse": fenced_init,
            "Summarize": "failed because X",
        },
        default=fenced_rejuv,
    )
    state = init(wf, "test", n_particles=2, backend=backend)
    state.pointer = "b"
    new_state = fork(state, k=1, backend=backend)
    assert len(new_state.belief.particles) == 2
    assert new_state.belief.particles[0].brief == "new-1"
