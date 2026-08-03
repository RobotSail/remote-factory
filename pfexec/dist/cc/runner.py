"""Execute compiled pfexec sessions."""

from __future__ import annotations

import subprocess

from pfexec.dist.cc.belief_io import read_state, write_state
from pfexec.dist.cc.compiler import compile
from pfexec.engine import EngineConfig, EngineResult, run as engine_run
from pfexec.ir import WorkflowSpec
from pfexec.llm import DeterministicBackend
from pfexec.state import ExecutionState


def _build_result(state: ExecutionState, workflow: WorkflowSpec, steps: int,
                  forks: int, terminated_by: str, outputs: list[str]) -> EngineResult:
    terminal_ids = _terminal_nodes(workflow)
    output = ""
    for tid in terminal_ids:
        if tid in state.node_outputs:
            output = state.node_outputs[tid]
            break
    if not output and outputs:
        output = outputs[-1]

    return EngineResult(
        final_state=state,
        output=output,
        steps_taken=steps,
        forks_triggered=forks,
        terminated_by=terminated_by,
        all_outputs=outputs,
    )


def _terminal_nodes(workflow: WorkflowSpec) -> list[str]:
    sources = {e.source for e in workflow.edges}
    return [n.id for n in workflow.nodes if n.id not in sources]


def run(
    workflow: WorkflowSpec,
    user_input: str,
    config: EngineConfig,
    mode: str = "pfexec",
) -> EngineResult:
    if mode == "dry-run":
        return _run_dry(workflow, user_input, config)
    elif mode == "deterministic":
        return _run_claude(workflow, user_input,
                           EngineConfig(n_particles=1, tau=0.0, max_steps=config.max_steps,
                                        max_forks=0, rewind_steps=config.rewind_steps),
                           backend_mode="claude")
    else:
        return _run_claude(workflow, user_input, config, backend_mode="claude")


def _run_dry(workflow: WorkflowSpec, user_input: str, config: EngineConfig) -> EngineResult:
    session = compile(workflow, config, user_input, backend_mode="mock")

    backend = DeterministicBackend(default="ok")
    result = engine_run(workflow, user_input, backend, config)

    write_state(session.root / "state.json", result.final_state)
    for node_id, output in result.final_state.node_outputs.items():
        (session.node_outputs_dir / f"{node_id}.txt").write_text(output)

    verified_state = read_state(session.root / "state.json")

    return EngineResult(
        final_state=verified_state,
        output=result.output,
        steps_taken=result.steps_taken,
        forks_triggered=result.forks_triggered,
        terminated_by=result.terminated_by,
        all_outputs=result.all_outputs,
    )


def _run_claude(workflow: WorkflowSpec, user_input: str, config: EngineConfig,
                backend_mode: str) -> EngineResult:
    session = compile(workflow, config, user_input, backend_mode=backend_mode)

    result = subprocess.run(
        ["bash", str(session.run_script), user_input],
        capture_output=True,
        text=True,
        timeout=config.max_steps * 60,
    )

    state_path = session.root / "state.json"
    if state_path.exists():
        state = read_state(state_path)
    else:
        state = read_state(session.root / "state.json")

    outputs: list[str] = []
    for node in workflow.nodes:
        out_file = session.node_outputs_dir / f"{node.id}.txt"
        if out_file.exists():
            outputs.append(out_file.read_text())

    terminal = _terminal_nodes(workflow)
    output = ""
    for tid in terminal:
        out_file = session.node_outputs_dir / f"{tid}.txt"
        if out_file.exists():
            output = out_file.read_text()
            break
    if not output:
        output = result.stdout.strip()

    return EngineResult(
        final_state=state,
        output=output,
        steps_taken=config.max_steps - state.budget_remaining,
        forks_triggered=0,
        terminated_by="complete",
        all_outputs=outputs,
    )
