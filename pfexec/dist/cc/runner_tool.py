"""Tool-based runner — Claude drives the loop via pfexec CLI.

Claude gets Bash access and interacts with the pfexec engine through
`python -m pfexec.tool` commands (init, next, submit). The engine
internals (particles, beliefs) stay hidden behind the tool interface.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from pfexec.dist.cc.belief_io import read_state
from pfexec.dist.cc.skill_gen import _terminal_nodes, _topo_order
from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import WorkflowSpec
from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


def run(workflow: WorkflowSpec, user_input: str, config: EngineConfig,
        backend_mode: str = "claude") -> EngineResult:
    tmpdir = tempfile.mkdtemp(prefix="pfexec-toolrun-")
    wf_path = Path(tmpdir) / "workflow.json"
    wf_path.write_text(workflow.to_json())

    tool_cmd = "python -m pfexec.tool"
    init_cmd = [
        "python", "-m", "pfexec.tool", "init",
        "--workflow", str(wf_path),
        "--input", user_input,
        "--particles", str(config.n_particles),
        "--tau", str(config.tau),
        "--max-forks", str(config.max_forks),
        "--observe-mode", config.observe_mode,
        "--backend", backend_mode,
    ]

    if backend_mode == "mock":
        result = subprocess.run(init_cmd, capture_output=True, text=True, timeout=60)
        session_dir = result.stdout.strip()
        return _mock_loop(session_dir, workflow, config)

    result = subprocess.run(init_cmd, capture_output=True, text=True, timeout=60)
    session_dir = result.stdout.strip()

    system_prompt = (
        f"You are solving a problem step by step using the pfexec workflow engine.\n"
        f"\n"
        f"Commands:\n"
        f"  {tool_cmd} next --session {session_dir}\n"
        f"  {tool_cmd} submit --session {session_dir} --node <NODE_ID> <<'PFEXEC'\n"
        f"  <your output>\n"
        f"  PFEXEC\n"
        f"\n"
        f"Workflow:\n"
        f"1. Run \"next\" to see your current task\n"
        f"2. Think about the task and produce your answer\n"
        f"3. Run \"submit\" with your answer\n"
        f"4. Repeat until the engine says DONE\n"
        f"5. If the engine says FORK, it will provide a lesson — incorporate it and continue\n"
        f"\n"
        f"When done, output the final answer as plain text."
    )

    claude_result = subprocess.run(
        ["claude",
         "--system-prompt", system_prompt,
         "--allowedTools", "Bash",
         "--dangerously-skip-permissions",
         "-p", f"Solve: {user_input}. Start by running the next command."],
        capture_output=True, text=True,
        timeout=config.max_steps * 120,
    )

    raw_output = claude_result.stdout.strip()

    state_path = Path(session_dir) / "state.json"
    if state_path.exists():
        state = read_state(state_path)
    else:
        belief = Belief(particles=[Particle(brief="", weight=1.0)])
        trace = TraceTree(root=TraceNode(node_id="root"))
        state = ExecutionState(
            pointer="",
            belief=belief,
            trace=trace,
            user_input=user_input,
        )

    order = _topo_order(workflow)
    all_outputs = [state.node_outputs[nid] for nid in order if nid in state.node_outputs]
    steps_taken = len(state.node_outputs)

    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]
    final_answer = state.node_outputs.get(terminal_id, "")
    if not final_answer and all_outputs:
        final_answer = all_outputs[-1]

    return EngineResult(
        final_state=state,
        output=final_answer,
        steps_taken=steps_taken,
        forks_triggered=0,
        terminated_by="complete",
        all_outputs=all_outputs,
    )


def _mock_loop(session_dir: str, workflow: WorkflowSpec, config: EngineConfig) -> EngineResult:
    """Simulate the tool loop with mock backend for testing."""
    order = _topo_order(workflow)

    for nid in order:
        subprocess.run(
            ["python", "-m", "pfexec.tool", "next", "--session", session_dir],
            capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["python", "-m", "pfexec.tool", "submit", "--session", session_dir,
             "--node", nid, "--backend", "mock"],
            input=f"mock output for {nid}", capture_output=True, text=True, timeout=30,
        )

    state = read_state(Path(session_dir) / "state.json")
    all_outputs = [state.node_outputs.get(nid, "") for nid in order]
    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]

    return EngineResult(
        final_state=state,
        output=state.node_outputs.get(terminal_id, "mock output"),
        steps_taken=len(state.node_outputs),
        forks_triggered=0,
        terminated_by="complete",
        all_outputs=all_outputs,
    )
