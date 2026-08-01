"""Tests for pfexec.ir — intermediate representation."""

from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec


def test_node_spec_defaults():
    n = NodeSpec(id="a", spec="do stuff", theta_prior="prompt {input}")
    assert n.id == "a"
    assert n.effect == "pure"
    assert n.tools == []
    assert n.input_schema == {}
    assert n.output_schema == {}


def test_node_spec_effectful():
    n = NodeSpec(id="b", spec="run tests", theta_prior="test {input}", effect="effectful")
    assert n.effect == "effectful"


def test_edge_spec():
    e = EdgeSpec(source="a", target="b")
    assert e.condition is None
    e2 = EdgeSpec(source="a", target="b", condition="x > 0")
    assert e2.condition == "x > 0"


def test_workflow_spec_creation(linear_workflow: WorkflowSpec):
    assert linear_workflow.name == "linear"
    assert len(linear_workflow.nodes) == 3
    assert len(linear_workflow.edges) == 2
    assert linear_workflow.entry == "a"


def test_json_round_trip(linear_workflow: WorkflowSpec):
    s = linear_workflow.to_json()
    restored = WorkflowSpec.from_json(s)
    assert restored.name == linear_workflow.name
    assert len(restored.nodes) == len(linear_workflow.nodes)
    assert len(restored.edges) == len(linear_workflow.edges)
    assert restored.entry == linear_workflow.entry
    for orig, rest in zip(linear_workflow.nodes, restored.nodes):
        assert orig.id == rest.id
        assert orig.spec == rest.spec
        assert orig.theta_prior == rest.theta_prior
        assert orig.effect == rest.effect


def test_json_round_trip_with_tools():
    wf = WorkflowSpec(
        name="with-tools",
        nodes=[
            NodeSpec(
                id="n1",
                spec="search",
                theta_prior="find {input}",
                tools=["web_search", "file_read"],
                effect="effectful",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
            ),
        ],
        edges=[],
        entry="n1",
    )
    restored = WorkflowSpec.from_json(wf.to_json())
    assert restored.nodes[0].tools == ["web_search", "file_read"]
    assert restored.nodes[0].input_schema["properties"]["q"]["type"] == "string"


def test_validate_ok(linear_workflow: WorkflowSpec):
    assert linear_workflow.validate() == []


def test_validate_bad_entry():
    wf = WorkflowSpec(
        name="bad",
        nodes=[NodeSpec(id="a", spec="x", theta_prior="p")],
        edges=[],
        entry="missing",
    )
    issues = wf.validate()
    assert any("entry" in i for i in issues)


def test_validate_bad_edge_source():
    wf = WorkflowSpec(
        name="bad",
        nodes=[NodeSpec(id="a", spec="x", theta_prior="p")],
        edges=[EdgeSpec(source="missing", target="a")],
        entry="a",
    )
    issues = wf.validate()
    assert any("source" in i and "missing" in i for i in issues)


def test_validate_bad_edge_target():
    wf = WorkflowSpec(
        name="bad",
        nodes=[NodeSpec(id="a", spec="x", theta_prior="p")],
        edges=[EdgeSpec(source="a", target="missing")],
        entry="a",
    )
    issues = wf.validate()
    assert any("target" in i and "missing" in i for i in issues)


def test_branching_workflow(branching_workflow: WorkflowSpec):
    assert branching_workflow.validate() == []
    assert len(branching_workflow.edges) == 4
    conditional = [e for e in branching_workflow.edges if e.condition]
    assert len(conditional) == 2
