"""Tests for pfexec.tool — CLI tool interface."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pfexec.dist.cc.belief_io import read_state
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec


def _build_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        name="test_workflow",
        nodes=[
            NodeSpec(
                id="decompose",
                spec="Decompose a complex question into sub-questions",
                theta_prior="Decompose this question into simpler parts: {input}",
            ),
            NodeSpec(
                id="retrieve",
                spec="Retrieve information to answer sub-questions",
                theta_prior="Find answers to these sub-questions: {input}",
            ),
            NodeSpec(
                id="answer",
                spec="Synthesize a final answer from retrieved information",
                theta_prior="Given the retrieved facts, answer: {input}",
            ),
        ],
        edges=[
            EdgeSpec(source="decompose", target="retrieve"),
            EdgeSpec(source="retrieve", target="answer"),
        ],
        entry="decompose",
    )


def _write_workflow(tmp: str) -> Path:
    wf = _build_workflow()
    wf_path = Path(tmp) / "workflow.json"
    wf_path.write_text(wf.to_json())
    return wf_path


def _run_tool(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pfexec.tool", *args],
        capture_output=True, text=True, input=input_text, timeout=30,
    )


def _init_session(wf_path: Path) -> str:
    result = _run_tool(
        "init",
        "--workflow", str(wf_path),
        "--input", "What is the capital of France?",
        "--particles", "3",
        "--backend", "mock",
    )
    assert result.returncode == 0, f"init failed: {result.stderr}"
    return result.stdout.strip()


def test_tool_init():
    with tempfile.TemporaryDirectory() as tmp:
        wf_path = _write_workflow(tmp)
        session_dir = _init_session(wf_path)

        session = Path(session_dir)
        assert session.is_dir()
        assert (session / "state.json").exists()
        assert (session / "workflow.json").exists()
        assert (session / "config.json").exists()

        state = read_state(session / "state.json")
        assert len(state.belief.particles) == 3
        assert state.user_input == "What is the capital of France?"

        config = json.loads((session / "config.json").read_text())
        assert config["n_particles"] == 3
        assert config["observe_mode"] == "full"
        assert config["order"] == ["decompose", "retrieve", "answer"]


def test_tool_next():
    with tempfile.TemporaryDirectory() as tmp:
        wf_path = _write_workflow(tmp)
        session_dir = _init_session(wf_path)

        result = _run_tool("next", "--session", session_dir)
        assert result.returncode == 0, f"next failed: {result.stderr}"

        output = result.stdout
        assert "Phase 1: decompose" in output
        assert "Role:" in output
        assert "Task:" in output


def test_tool_submit():
    with tempfile.TemporaryDirectory() as tmp:
        wf_path = _write_workflow(tmp)
        session_dir = _init_session(wf_path)

        result = _run_tool(
            "submit", "--session", session_dir,
            "--node", "decompose", "--backend", "mock",
            input_text="Sub-question 1 and 2",
        )
        assert result.returncode == 0, f"submit failed: {result.stderr}"

        state = read_state(Path(session_dir) / "state.json")
        assert "decompose" in state.node_outputs
        assert state.node_outputs["decompose"] == "Sub-question 1 and 2"
        assert state.step == 1

        config = json.loads((Path(session_dir) / "config.json").read_text())
        assert config["pointer_idx"] == 1


def test_tool_submit_continue():
    with tempfile.TemporaryDirectory() as tmp:
        wf_path = _write_workflow(tmp)
        session_dir = _init_session(wf_path)

        result = _run_tool(
            "submit", "--session", session_dir,
            "--node", "decompose", "--backend", "mock",
            input_text="Sub-question 1 and 2",
        )
        assert result.returncode == 0
        assert "CONTINUE" in result.stdout


def test_tool_submit_done():
    with tempfile.TemporaryDirectory() as tmp:
        wf_path = _write_workflow(tmp)
        session_dir = _init_session(wf_path)

        for nid in ["decompose", "retrieve", "answer"]:
            result = _run_tool(
                "submit", "--session", session_dir,
                "--node", nid, "--backend", "mock",
                input_text=f"output for {nid}",
            )
            assert result.returncode == 0, f"submit {nid} failed: {result.stderr}"

        assert "DONE" in result.stdout


def test_tool_status():
    with tempfile.TemporaryDirectory() as tmp:
        wf_path = _write_workflow(tmp)
        session_dir = _init_session(wf_path)

        _run_tool(
            "submit", "--session", session_dir,
            "--node", "decompose", "--backend", "mock",
            input_text="Sub-question 1 and 2",
        )

        result = _run_tool("status", "--session", session_dir)
        assert result.returncode == 0, f"status failed: {result.stderr}"

        output = result.stdout
        assert "Current node:" in output
        assert "Particles:" in output
        assert "Node outputs:" in output
        assert "decompose:" in output


def test_tool_next_done_after_all():
    with tempfile.TemporaryDirectory() as tmp:
        wf_path = _write_workflow(tmp)
        session_dir = _init_session(wf_path)

        for nid in ["decompose", "retrieve", "answer"]:
            _run_tool(
                "submit", "--session", session_dir,
                "--node", nid, "--backend", "mock",
                input_text=f"output for {nid}",
            )

        result = _run_tool("next", "--session", session_dir)
        assert result.returncode == 0
        assert "DONE" in result.stdout
        assert "output for answer" in result.stdout


def test_runner_tool_mock():
    from pfexec.dist.cc.runner_tool import run as run_tool
    from pfexec.engine import EngineConfig

    workflow = _build_workflow()
    config = EngineConfig(n_particles=3, tau=0.0, max_steps=20, max_forks=1)
    result = run_tool(workflow, "What is X?", config, backend_mode="mock")

    assert result.terminated_by == "complete"
    assert result.steps_taken == 3
    assert len(result.all_outputs) == 3
    for nid in ["decompose", "retrieve", "answer"]:
        assert nid in result.final_state.node_outputs
        assert f"mock output for {nid}" in result.final_state.node_outputs[nid]
