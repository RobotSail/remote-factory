"""Tests for factory/workflow/tool.py — tool-based workflow execution."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    Edge,
    FnNode,
    GateNode,
    Study,
    VerdictType,
    Workflow,
)
from factory.workflow.registry import WorkflowRegistry
from factory.workflow.tool import (
    _find_reloop_target,
    _format_gate_task,
    _format_node_task,
    tool_init,
    tool_next,
    tool_status,
    tool_submit,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    WorkflowRegistry.reset()
    yield
    WorkflowRegistry.reset()


def _simple_workflow() -> Workflow:
    """A minimal workflow: study -> researcher -> gate -> builder."""
    return Workflow(
        name="test-simple",
        start_node="study",
        nodes={
            "study": Study(
                id="study",
                command="factory study {project_path}",
                writes={".factory/strategy/observations.md"},
            ),
            "researcher": AgentNode(
                id="researcher",
                role=AgentRole.RESEARCHER,
                prompt_template="Research the project at {project_path}",
                writes={".factory/reviews/researcher-latest.md"},
            ),
            "gate_research": GateNode(
                id="gate_research",
                evaluator_type="agent",
                gate_prompt="Review research output",
                reads={".factory/reviews/researcher-latest.md"},
            ),
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build the project",
                writes={".factory/reviews/builder-latest.md"},
            ),
        },
        edges=[
            Edge(source="study", target="researcher"),
            Edge(source="researcher", target="gate_research"),
            Edge(source="gate_research", target="builder", condition=VerdictType.PROCEED),
            Edge(source="gate_research", target="researcher", condition=VerdictType.RELOOP),
        ],
    )


def _fn_gate_workflow() -> Workflow:
    """Workflow with an fn-type gate for auto-evaluation."""
    return Workflow(
        name="test-fn-gate",
        start_node="builder",
        nodes={
            "builder": AgentNode(
                id="builder",
                role=AgentRole.BUILDER,
                prompt_template="Build",
                writes={".factory/reviews/builder-latest.md"},
            ),
            "gate_review": GateNode(
                id="gate_review",
                evaluator_type="fn",
                evaluator_command="echo PROCEED",
                reads={".factory/reviews/builder-latest.md"},
            ),
            "archivist": AgentNode(
                id="archivist",
                role=AgentRole.ARCHIVIST,
                prompt_template="Archive results",
                writes={".factory/archive/build.md"},
                blocking=False,
            ),
        },
        edges=[
            Edge(source="builder", target="gate_review"),
            Edge(source="gate_review", target="archivist", condition=VerdictType.PROCEED),
            Edge(source="gate_review", target="builder", condition=VerdictType.RELOOP),
        ],
    )


def _register_workflow(wf: Workflow) -> None:
    """Helper to register a workflow in the registry."""
    from factory.workflow.registry import WorkflowEntry
    WorkflowRegistry._entries[wf.name] = WorkflowEntry(
        name=wf.name,
        description="test workflow",
        path="<test>",
        source="builtin",
        _workflow_fn=lambda _wf=wf: _wf,
    )


class TestToolInit:
    def test_init_creates_state(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()

        session_dir = tool_init("test-simple", tmp_path)

        state_path = Path(session_dir) / "state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["workflow_name"] == "test-simple"
        assert state["status"] == "active"
        assert state["pointer_idx"] == 0
        assert len(state["session_id"]) == 12
        assert "study" in state["topo_order"]

    def test_init_unknown_workflow(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown workflow"):
            tool_init("nonexistent", tmp_path)

    def test_init_filters_join_nodes(self, tmp_path: Path) -> None:
        """JoinNodes should be excluded from topo_order."""
        from factory.workflow.primitives import JoinNode
        wf = Workflow(
            name="test-join",
            start_node="a",
            nodes={
                "a": FnNode(id="a", command="echo a"),
                "join": JoinNode(id="join", sources=["a"]),
                "b": FnNode(id="b", command="echo b"),
            },
            edges=[
                Edge(source="a", target="join"),
                Edge(source="join", target="b"),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()

        tool_init("test-join", tmp_path)
        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "join" not in state["topo_order"]
        assert "a" in state["topo_order"]
        assert "b" in state["topo_order"]


class TestToolNext:
    def test_next_returns_first_node(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_next(tmp_path)

        assert "Node: study" in result
        assert "Type: Study" in result

    def test_next_returns_done_when_completed(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["status"] = "completed"
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        result = tool_next(tmp_path)
        assert result.startswith("DONE")

    def test_next_completes_when_past_end(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["pointer_idx"] = len(state["topo_order"])
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        result = tool_next(tmp_path)
        assert "DONE" in result


class TestToolSubmit:
    def test_submit_stores_output(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_submit(tmp_path, "study", "Observations: project looks good")

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert state["completed"]["study"] == "Observations: project looks good"
        assert result == "CONTINUE"

    def test_submit_writes_agent_output_files(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Advance past study first
        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["pointer_idx"] = 1  # researcher
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        tool_submit(tmp_path, "researcher", "Research findings here")

        output_file = tmp_path / ".factory" / "reviews" / "researcher-latest.md"
        assert output_file.exists()
        assert output_file.read_text() == "Research findings here"

    def test_submit_returns_gate_for_agent_gate(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["pointer_idx"] = 1  # researcher
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        result = tool_submit(tmp_path, "researcher", "Research done")
        assert result.startswith("GATE")
        assert "gate_research" in result
        assert "PROCEED" in result

    def test_submit_fn_gate_proceed(self, tmp_path: Path) -> None:
        wf = _fn_gate_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-fn-gate", tmp_path)

        result = tool_submit(tmp_path, "builder", "Built successfully")

        assert result == "CONTINUE"
        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert state["gate_results"]["gate_review"] == "PROCEED"

    def test_submit_fn_gate_halt(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-halt",
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build",
                ),
                "gate_fail": GateNode(
                    id="gate_fail",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL: tests broken",
                ),
            },
            edges=[
                Edge(source="builder", target="gate_fail"),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-halt", tmp_path)

        result = tool_submit(tmp_path, "builder", "Built")
        assert result.startswith("HALT")
        assert "FAIL" in result

    def test_submit_fn_gate_reloop(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-reloop",
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build",
                ),
                "gate_check": GateNode(
                    id="gate_check",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL: needs fixes",
                ),
            },
            edges=[
                Edge(source="builder", target="gate_check"),
                Edge(source="gate_check", target="builder", condition=VerdictType.RELOOP),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-reloop", tmp_path)

        result = tool_submit(tmp_path, "builder", "First attempt")
        assert result.startswith("RETRY")
        assert "attempt 1/3" in result

    def test_submit_fn_gate_reloop_max_iterations(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-max-iter",
            start_node="builder",
            nodes={
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build",
                ),
                "gate_check": GateNode(
                    id="gate_check",
                    evaluator_type="fn",
                    evaluator_command="echo FAIL: still broken",
                ),
            },
            edges=[
                Edge(source="builder", target="gate_check"),
                Edge(source="gate_check", target="builder", condition=VerdictType.RELOOP),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-max-iter", tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        state["iteration_counts"]["gate_check->builder"] = 3
        (tmp_path / ".factory" / "tool_session" / "state.json").write_text(
            json.dumps(state)
        )

        result = tool_submit(tmp_path, "builder", "Fourth attempt")
        assert result.startswith("HALT")

    def test_submit_user_gate_approval(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-user-gate",
            start_node="strategist",
            nodes={
                "strategist": AgentNode(
                    id="strategist",
                    role=AgentRole.STRATEGIST,
                    prompt_template="Strategize",
                ),
                "gate_approval": GateNode(
                    id="gate_approval",
                    evaluator_type="user",
                    gate_prompt="Approve this strategy?",
                ),
                "builder": AgentNode(
                    id="builder",
                    role=AgentRole.BUILDER,
                    prompt_template="Build",
                ),
            },
            edges=[
                Edge(source="strategist", target="gate_approval"),
                Edge(source="gate_approval", target="builder", condition=VerdictType.PROCEED),
            ],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-user-gate", tmp_path)

        result = tool_submit(tmp_path, "strategist", "Strategy ready")
        assert result.startswith("APPROVAL_NEEDED")
        assert "Approve this strategy?" in result

    def test_submit_returns_done_at_end(self, tmp_path: Path) -> None:
        wf = Workflow(
            name="test-single",
            start_node="study",
            nodes={
                "study": Study(id="study", command="factory study {project_path}"),
            },
            edges=[],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-single", tmp_path)

        result = tool_submit(tmp_path, "study", "Done studying")
        assert result == "DONE"


class TestToolStatus:
    def test_status_no_session(self, tmp_path: Path) -> None:
        result = tool_status(tmp_path)
        assert "No active session" in result

    def test_status_active_session(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        result = tool_status(tmp_path)
        assert "Workflow: test-simple" in result
        assert "Status:   active" in result
        assert "Progress: 0/" in result

    def test_status_with_completed_nodes(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)
        tool_submit(tmp_path, "study", "Observations here")

        result = tool_status(tmp_path)
        assert "Progress: 1/" in result
        assert "[study]" in result
        assert "Completed nodes:" in result

    def test_status_with_gate_results(self, tmp_path: Path) -> None:
        wf = _fn_gate_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-fn-gate", tmp_path)
        tool_submit(tmp_path, "builder", "Built")

        result = tool_status(tmp_path)
        assert "Gates:" in result
        assert "PROCEED" in result


class TestAutoComplete:
    def test_next_auto_completes_agent_with_review_file(self, tmp_path: Path) -> None:
        """If an agent's review file exists but submit wasn't called, next skips it."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Submit study to advance past it
        tool_submit(tmp_path, "study", "Observations done")

        # Simulate agent ran but submit was skipped: write the review file directly
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("Research findings here")

        # tool_next should auto-complete the researcher and return the gate
        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "researcher" in state["completed"]
        assert state["completed"]["researcher"] == "Research findings here"
        assert "gate_research" in result

    def test_next_auto_completes_study_with_observations(self, tmp_path: Path) -> None:
        """If observations.md exists but submit wasn't called, next skips the study."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Write observations file directly (simulating study ran but submit skipped)
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text(
            "Detailed observations about the project that exceed the minimum length threshold"
        )

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "study" in state["completed"]
        assert "researcher" in result

    def test_next_auto_completes_fn_with_output_files(self, tmp_path: Path) -> None:
        """If a FnNode's declared output files exist, next skips it."""
        wf = Workflow(
            name="test-fn-auto",
            start_node="fn1",
            nodes={
                "fn1": FnNode(
                    id="fn1",
                    command="echo hello",
                    writes={".factory/output.md"},
                ),
                "fn2": FnNode(id="fn2", command="echo done"),
            },
            edges=[Edge(source="fn1", target="fn2")],
        )
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-fn-auto", tmp_path)

        # Write the output file directly
        (tmp_path / ".factory" / "output.md").write_text("Generated output")

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "fn1" in state["completed"]
        assert "fn2" in result

    def test_next_does_not_auto_complete_empty_review(self, tmp_path: Path) -> None:
        """Empty review files should not trigger auto-complete."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        tool_submit(tmp_path, "study", "Observations done")

        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("")

        result = tool_next(tmp_path)
        assert "researcher" in result
        assert "Type: Agent" in result

    def test_next_auto_completes_multiple_consecutive(self, tmp_path: Path) -> None:
        """Auto-complete should chain through multiple skippable nodes."""
        wf = _simple_workflow()
        _register_workflow(wf)
        (tmp_path / ".factory").mkdir()
        tool_init("test-simple", tmp_path)

        # Write both study observations and researcher review
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "observations.md").write_text(
            "Detailed observations about the project that exceed the minimum length threshold"
        )
        reviews_dir = tmp_path / ".factory" / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        (reviews_dir / "researcher-latest.md").write_text("Research findings")

        result = tool_next(tmp_path)

        state = json.loads(
            (tmp_path / ".factory" / "tool_session" / "state.json").read_text()
        )
        assert "study" in state["completed"]
        assert "researcher" in state["completed"]
        assert "gate_research" in result


class TestHelpers:
    def test_find_reloop_target(self) -> None:
        wf = _simple_workflow()
        target = _find_reloop_target(wf, "gate_research")
        assert target == "researcher"

    def test_find_reloop_target_none(self) -> None:
        wf = _simple_workflow()
        target = _find_reloop_target(wf, "study")
        assert target is None

    def test_format_node_task_agent(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        node = wf.nodes["researcher"]
        result = _format_node_task("researcher", node, wf, {}, tmp_path)
        assert "Type: Agent (researcher)" in result
        assert "Model:" in result
        assert "Timeout:" in result

    def test_format_node_task_study(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        node = wf.nodes["study"]
        result = _format_node_task("study", node, wf, {}, tmp_path)
        assert "Type: Study" in result
        assert "Command:" in result

    def test_format_node_task_gate(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        node = wf.nodes["gate_research"]
        result = _format_node_task("gate_research", node, wf, {}, tmp_path)
        assert "Type: Gate (agent)" in result

    def test_format_node_task_fn(self, tmp_path: Path) -> None:
        node = FnNode(id="fn1", command="echo hello", notes="test note")
        wf = Workflow(
            name="test", start_node="fn1",
            nodes={"fn1": node}, edges=[],
        )
        result = _format_node_task("fn1", node, wf, {}, tmp_path)
        assert "Type: Function" in result
        assert "Notes: test note" in result

    def test_format_node_task_fork(self, tmp_path: Path) -> None:
        from factory.workflow.primitives import ForkNode
        node = ForkNode(id="fork1", targets=["a", "b"])
        wf = Workflow(
            name="test", start_node="fork1",
            nodes={"fork1": node}, edges=[],
        )
        result = _format_node_task("fork1", node, wf, {}, tmp_path)
        assert "Type: Fork" in result
        assert "a, b" in result

    def test_format_gate_task(self, tmp_path: Path) -> None:
        wf = _simple_workflow()
        _register_workflow(wf)
        gate = wf.nodes["gate_research"]
        state = {"workflow_name": "test-simple"}
        result = _format_gate_task("gate_research", gate, state, tmp_path)
        assert "Gate: gate_research" in result
        assert "PROCEED" in result
        assert "RETRY" in result
        assert "researcher" in result
