"""Tests for pfexec.dist.cc — Claude Code backend compiler."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pfexec.dist.cc.belief_io import read_state, state_from_dict, state_to_dict, write_state
from pfexec.dist.cc.compiler import compile
from pfexec.dist.cc.runner import _run_agentic, _run_orchestrated, run
from pfexec.dist.cc.skill_gen import generate, generate_agentic
from pfexec.engine import EngineConfig
from pfexec.examples.multi_step_qa import build_workflow
from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


def _workflow():
    return build_workflow()


def _config(**overrides):
    defaults = dict(n_particles=3, tau=0.0, max_steps=20, max_forks=1, rewind_steps=2)
    defaults.update(overrides)
    return EngineConfig(**defaults)


def test_compile_creates_session_dir():
    workflow = _workflow()
    config = _config()
    session = compile(workflow, config, "What is the capital of France?", backend_mode="mock")

    assert session.root.is_dir()
    assert session.skill_path.exists()
    assert session.belief_path.exists()
    assert session.workflow_path.exists()
    assert session.config_path.exists()
    assert session.run_script.exists()
    assert session.trace_dir.is_dir()
    assert session.node_outputs_dir.is_dir()
    assert session.hooks_dir.is_dir()
    assert (session.root / "state.json").exists()
    assert (session.root / "input.txt").exists()
    assert (session.root / "input.txt").read_text() == "What is the capital of France?"


def test_skill_gen_produces_valid_md():
    workflow = _workflow()
    config = _config()
    md = generate(workflow, config)

    assert "multi_step_qa" in md
    assert "decompose" in md
    assert "retrieve" in md
    assert "answer" in md
    assert "pre_step.sh" in md
    assert "post_step.sh" in md
    assert "hooks/prompt.txt" in md
    assert "node_outputs/" in md
    assert "fork_status.txt" in md


def test_belief_io_round_trip():
    belief = Belief(particles=[
        Particle(brief="strategy-A", weight=0.6, evidence="saw X"),
        Particle(brief="strategy-B", weight=0.4, evidence="saw Y"),
    ])
    trace = TraceTree(root=TraceNode(
        node_id="decompose",
        checkpoint_id="init",
        children=[TraceNode(node_id="retrieve", checkpoint_id="step-1")],
    ))
    state = ExecutionState(
        pointer="retrieve",
        belief=belief,
        trace=trace,
        step=1,
        budget_remaining=49,
        user_input="What is X?",
        node_outputs={"decompose": "Sub-questions: A, B"},
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        write_state(path, state)
        loaded = read_state(path)

    assert loaded.pointer == "retrieve"
    assert loaded.step == 1
    assert loaded.budget_remaining == 49
    assert loaded.user_input == "What is X?"
    assert loaded.node_outputs == {"decompose": "Sub-questions: A, B"}
    assert len(loaded.belief.particles) == 2
    assert loaded.belief.particles[0].brief == "strategy-A"
    assert loaded.belief.particles[1].brief == "strategy-B"
    assert loaded.trace.root.node_id == "decompose"
    assert len(loaded.trace.root.children) == 1
    assert loaded.trace.root.children[0].node_id == "retrieve"


def test_belief_io_init_cli():
    workflow = _workflow()

    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp) / "session"
        session_dir.mkdir()
        wf_path = session_dir / "workflow.json"
        wf_path.write_text(workflow.to_json())

        result = subprocess.run(
            [sys.executable, "-m", "pfexec.dist.cc.belief_io",
             "init",
             "--session", str(session_dir),
             "--workflow", str(wf_path),
             "--input", "What is the capital of France?",
             "--particles", "3",
             "--backend", "mock"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        state_path = session_dir / "state.json"
        assert state_path.exists()
        state = read_state(state_path)
        assert state.pointer == "decompose"
        assert len(state.belief.particles) == 3
        assert state.user_input == "What is the capital of France?"

        assert (session_dir / "belief.json").exists()
        assert (session_dir / "trace" / "root.json").exists()


def test_belief_io_sample_cli():
    workflow = _workflow()
    config = _config()
    session = compile(workflow, config, "What is the capital of France?", backend_mode="mock")

    result = subprocess.run(
        [sys.executable, "-m", "pfexec.dist.cc.belief_io",
         "sample",
         "--session", str(session.root),
         "--node", "decompose",
         "--backend", "mock"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    hint_path = session.hooks_dir / "hint.txt"
    assert hint_path.exists()

    prompt_path = session.hooks_dir / "prompt.txt"
    assert prompt_path.exists()
    assert len(prompt_path.read_text()) > 0


def test_hooks_are_executable():
    workflow = _workflow()
    config = _config()
    session = compile(workflow, config, "test input", backend_mode="mock")

    pre_step = session.hooks_dir / "pre_step.sh"
    post_step = session.hooks_dir / "post_step.sh"

    assert pre_step.exists()
    assert post_step.exists()
    assert os.access(pre_step, os.X_OK)
    assert os.access(post_step, os.X_OK)

    pre_content = pre_step.read_text()
    assert "pfexec.dist.cc.belief_io" in pre_content
    assert "sample" in pre_content

    post_content = post_step.read_text()
    assert "observe" in post_content
    assert "fork-check" in post_content


def test_dry_run_produces_result():
    workflow = _workflow()
    config = _config()
    result = run(workflow, "What is the capital of France?", config, mode="dry-run")

    assert result.terminated_by == "complete"
    assert result.steps_taken == 3
    assert result.forks_triggered == 0
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.final_state.pointer is not None
    assert len(result.final_state.node_outputs) == 3


def test_orchestrated_dry_run():
    workflow = _workflow()
    config = _config()
    result = _run_orchestrated(workflow, "What is the capital of France?", config,
                               backend_mode="mock")

    assert result.terminated_by == "complete"
    assert result.steps_taken == 3
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.final_state.pointer is not None
    assert len(result.final_state.node_outputs) == 3
    for nid in ["decompose", "retrieve", "answer"]:
        assert nid in result.final_state.node_outputs
        assert result.final_state.node_outputs[nid] == "mock answer"


def test_orchestrated_preserves_node_outputs_on_disk():
    workflow = _workflow()
    config = _config()
    session = compile(workflow, config, "test", backend_mode="mock")
    result = _run_orchestrated(workflow, "test", config, backend_mode="mock")

    assert result.steps_taken == 3
    assert len(result.all_outputs) == 3


def test_agentic_skill_gen():
    workflow = _workflow()
    config = _config()

    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)
        md = generate_agentic(workflow, config, session_dir)

        assert "pfexec Workflow" in md
        assert "decompose" in md
        assert "retrieve" in md
        assert "answer" in md
        assert "## Phase 1:" in md
        assert "## Completion" in md
        assert str(session_dir) in md


def test_agentic_skill_has_protocol():
    workflow = _workflow()
    config = _config()

    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)
        md = generate_agentic(workflow, config, session_dir)

        assert "Available Tools (optional)" in md
        assert "Write your result under" in md
        assert "run automatically via hooks" in md
        assert "## Completion" in md


def test_agentic_settings_generated():
    workflow = _workflow()
    config = _config()
    result = _run_agentic(workflow, "What is X?", config, backend_mode="mock")

    session_root = result.final_state.trace.root.node_id
    # Find the session dir from the state file written during the run
    # The agentic runner creates settings in the session dir
    # We verify via a fresh compile + generate_settings call
    from pfexec.dist.cc.hooks import generate_settings

    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)
        (session_dir / "hooks").mkdir(parents=True)
        (session_dir / "node_outputs").mkdir()
        generate_settings(session_dir, config, "mock")

        settings_path = session_dir / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings
        assert "PostToolUse" in settings["hooks"]
        hooks = settings["hooks"]["PostToolUse"]
        assert len(hooks) == 1
        assert hooks[0]["matcher"] == "Write"
        assert "write_observer.sh" in hooks[0]["hooks"][0]["command"]


def test_agentic_write_observer_executable():
    from pfexec.dist.cc.hooks import generate_settings

    workflow = _workflow()
    config = _config()

    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)
        (session_dir / "hooks").mkdir(parents=True)
        generate_settings(session_dir, config, "mock")

        observer = session_dir / "hooks" / "write_observer.sh"
        assert observer.exists()
        assert os.access(observer, os.X_OK)
        content = observer.read_text()
        assert "pfexec.dist.cc.belief_io observe" in content
        assert "pfexec.dist.cc.belief_io fork-check" in content
        assert str(session_dir) in content


def test_agentic_dry_run():
    workflow = _workflow()
    config = _config()
    result = _run_agentic(workflow, "What is X?", config, backend_mode="mock")

    assert result.terminated_by == "complete"
    assert len(result.all_outputs) == 3
    for nid in ["decompose", "retrieve", "answer"]:
        assert nid in result.final_state.node_outputs


def test_session_dir_cleanup():
    workflow = _workflow()
    config = _config()
    session = compile(workflow, config, "test", backend_mode="mock")

    assert session.root.exists()
    assert "pfexec-session-" in session.root.name
    assert session.root.parent == Path(tempfile.gettempdir())


def test_belief_io_hint_cli():
    workflow = _workflow()
    config = _config()
    session = compile(workflow, config, "What is the capital of France?", backend_mode="mock")

    result = subprocess.run(
        [sys.executable, "-m", "pfexec.dist.cc.belief_io",
         "hint",
         "--session", str(session.root),
         "--node", "decompose"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_belief_io_hint_prints_hint():
    """cmd_hint prints a hint when the top particle has a meaningful brief."""
    from pfexec.dist.cc.belief_io import cmd_hint

    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)
        state = ExecutionState(
            pointer="decompose",
            belief=Belief(particles=[
                Particle(brief="chain-of-thought reasoning", weight=0.6),
                Particle(brief="keyword matching", weight=0.4),
            ]),
            trace=TraceTree(root=TraceNode(node_id="root")),
            user_input="test",
        )
        write_state(session_dir / "state.json", state)

        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cmd_hint(session_dir, "decompose")
        output = f.getvalue()
        assert "[pfexec:" in output
        assert "chain-of-thought reasoning" in output
        assert "keyword matching" in output


def test_belief_io_hint_skips_plan_briefs():
    """cmd_hint produces no output when top particle has plan-* brief."""
    from pfexec.dist.cc.belief_io import cmd_hint

    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)
        state = ExecutionState(
            pointer="decompose",
            belief=Belief(particles=[
                Particle(brief="plan-0", weight=0.5),
                Particle(brief="plan-1", weight=0.5),
            ]),
            trace=TraceTree(root=TraceNode(node_id="root")),
            user_input="test",
        )
        write_state(session_dir / "state.json", state)

        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cmd_hint(session_dir, "decompose")
        assert f.getvalue() == ""


def test_agentic_v3_dry_run():
    from pfexec.dist.cc.runner_agentic import run as run_agentic_v3

    workflow = _workflow()
    config = _config()
    result = run_agentic_v3(workflow, "What is X?", config, backend_mode="mock")

    assert result.terminated_by == "complete"
    assert result.steps_taken == 3
    assert len(result.all_outputs) == 3
    for nid in ["decompose", "retrieve", "answer"]:
        assert nid in result.final_state.node_outputs


def test_agentic_v3_generates_hinted_skill():
    from pfexec.dist.cc.runner_agentic import generate_hinted_skill_md

    workflow = _workflow()
    state = ExecutionState(
        pointer="decompose",
        belief=Belief(particles=[
            Particle(brief="systematic decomposition", weight=0.6),
            Particle(brief="keyword search", weight=0.25),
            Particle(brief="analogy reasoning", weight=0.15),
        ]),
        trace=TraceTree(root=TraceNode(node_id="root")),
        user_input="test",
    )

    md = generate_hinted_skill_md(workflow, state)

    assert "pfexec Workflow" in md
    assert "pfexec hint:" in md
    assert "systematic decomposition" in md
    assert "decompose" in md
    assert "retrieve" in md
    assert "answer" in md
    assert "### Output:" in md
    assert "node_outputs/" not in md
    assert "### Final Answer" in md
    # Fix 2: hint only at Phase 1
    assert md.count("pfexec hint:") == 1
    assert "## Phase 1: decompose\n[pfexec hint:" in md
    # Fix 3: preamble present when hints exist
    assert "Strategy hints from the pfexec engine" in md


def test_agentic_v3_no_hint_for_uniform_particles():
    """Uniform particle weights produce no hints and no hint preamble."""
    from pfexec.dist.cc.runner_agentic import _format_initial_hints, generate_hinted_skill_md

    workflow = _workflow()
    state = ExecutionState(
        pointer="decompose",
        belief=Belief(particles=[
            Particle(brief="strategy-a", weight=1.0),
            Particle(brief="strategy-b", weight=1.0),
            Particle(brief="strategy-c", weight=1.0),
        ]),
        trace=TraceTree(root=TraceNode(node_id="root")),
        user_input="test",
    )

    assert _format_initial_hints(state) == {}

    md = generate_hinted_skill_md(workflow, state)
    assert "pfexec hint:" not in md
    assert "Strategy hints from the pfexec engine" not in md
    assert "## Phase 1: decompose" in md
    assert "## Phase 2:" in md


def test_agentic_v3_bare_mode_no_hooks():
    from pfexec.dist.cc.runner_agentic import run as run_agentic_v3

    workflow = _workflow()
    config = _config()
    result = run_agentic_v3(workflow, "What is X?", config, backend_mode="mock")

    assert result.terminated_by == "complete"
    assert result.steps_taken == 3


def test_agentic_v3_parse_output():
    from pfexec.dist.cc.runner_agentic import _parse_output

    workflow = _workflow()
    raw = (
        "### Output: decompose\nSub-questions here\n"
        "### Output: retrieve\nRetrieved info\n"
        "### Output: answer\nParis\n"
        "### Final Answer\nParis"
    )
    node_outputs, final = _parse_output(raw, workflow)

    assert "decompose" in node_outputs
    assert "retrieve" in node_outputs
    assert "answer" in node_outputs
    assert final == "Paris"
