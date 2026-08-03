"""Execute compiled pfexec sessions."""

from __future__ import annotations

import subprocess

from pfexec.dist.cc.belief_io import _get_backend, read_state, write_state
from pfexec.dist.cc.compiler import compile
from pfexec.dist.cc.skill_gen import _topo_order
from pfexec.engine import EngineConfig, EngineResult, run as engine_run
from pfexec.ir import WorkflowSpec
from pfexec.llm import DeterministicBackend, LLMBackend
from pfexec.primitives import fork, observe
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


def _claude_call(prompt: str, system: str = "",
                 backend: LLMBackend | None = None) -> str:
    if backend is not None:
        return backend.call(prompt, system=system)

    cmd = [
        "claude", "--bare",
        "--disallowedTools", "Bash Read Edit Write Agent NotebookEdit WebFetch WebSearch",
    ]
    if system:
        cmd.extend(["--system-prompt", system])
    cmd.extend(["-p", prompt])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return f"ERROR: {result.stderr}"
    return result.stdout.strip()


def _run_orchestrated(workflow: WorkflowSpec, user_input: str, config: EngineConfig,
                      backend_mode: str = "claude") -> EngineResult:
    session = compile(workflow, config, user_input, backend_mode=backend_mode)

    node_map = {n.id: n for n in workflow.nodes}
    order = _topo_order(workflow)
    state = read_state(session.root / "state.json")
    state.budget_remaining = config.max_steps
    outputs: list[str] = []
    forks_triggered = 0
    visited: set[str] = set()

    task_backend: LLMBackend | None = None
    if backend_mode == "mock":
        task_backend = DeterministicBackend(default="mock answer")

    belief_backend = _get_backend(backend_mode)

    idx = 0
    while idx < len(order) and state.budget_remaining > 0:
        nid = order[idx]
        if nid in visited:
            idx += 1
            continue
        visited.add(nid)
        node = node_map[nid]

        if not state.node_outputs:
            data_input = state.user_input
        else:
            data_input = list(state.node_outputs.values())[-1]

        prompt = node.theta_prior.replace("{input}", data_input)

        state.belief.normalize()
        n_particles = len(state.belief.particles)
        if n_particles > 1:
            chosen = max(state.belief.particles, key=lambda p: p.weight)
            uniform = 1.0 / n_particles
            if (chosen.brief and not chosen.brief.startswith("plan-")
                    and chosen.weight > uniform * 1.2):
                prompt = f"[Strategy hint: {chosen.brief}]\n\n{prompt}"

        output = _claude_call(
            prompt, system=node.spec,
            backend=task_backend,
        )

        outputs.append(output)
        state.node_outputs[nid] = output
        (session.node_outputs_dir / f"{nid}.txt").write_text(output)

        state.step += 1
        state.budget_remaining -= 1

        if n_particles > 1:
            state = observe(state, output, belief_backend)

        if node.effect == "effectful" and forks_triggered < config.max_forks:
            state.belief.normalize()
            weights = sorted((p.weight for p in state.belief.particles), reverse=True)
            top_k = weights[:3]
            score = sum(top_k) / len(top_k) if top_k else 0.0
            if score < config.tau:
                state = fork(state, config.rewind_steps, belief_backend)
                forks_triggered += 1
                rewind_nid = state.pointer
                if rewind_nid in order:
                    idx = order.index(rewind_nid)
                    visited.discard(rewind_nid)
                    write_state(session.root / "state.json", state)
                    continue

        write_state(session.root / "state.json", state)
        idx += 1

    if state.budget_remaining <= 0:
        terminated_by = "budget"
    else:
        terminated_by = "complete"

    return _build_result(state, workflow, config.max_steps - state.budget_remaining,
                         forks_triggered, terminated_by, outputs)


def _run_agentic(workflow: WorkflowSpec, user_input: str, config: EngineConfig,
                 backend_mode: str = "claude") -> EngineResult:
    from pfexec.dist.cc.skill_gen import generate_agentic

    session = compile(workflow, config, user_input, backend_mode=backend_mode)

    skill_md = generate_agentic(workflow, config, session.root)
    session.skill_path.write_text(skill_md)

    if backend_mode == "mock":
        mock_backend = DeterministicBackend(default="mock agentic output")
        for node in workflow.nodes:
            out_file = session.node_outputs_dir / f"{node.id}.txt"
            out_file.write_text(mock_backend.call(f"Execute {node.id}"))

        state = read_state(session.root / "state.json")
        for node in workflow.nodes:
            state.node_outputs[node.id] = (session.node_outputs_dir / f"{node.id}.txt").read_text()
            state.step += 1
            state.budget_remaining -= 1
        write_state(session.root / "state.json", state)
    else:
        subprocess.run(
            ["claude", "--bare",
             "--allowedTools", "Bash(python *) Write Read",
             "--system-prompt-file", str(session.skill_path),
             "-p", f"Execute the {workflow.name} workflow for this input: {user_input}"],
            capture_output=True, text=True,
            timeout=config.max_steps * 60,
        )

    state = read_state(session.root / "state.json")

    outputs: list[str] = []
    for node in workflow.nodes:
        out_file = session.node_outputs_dir / f"{node.id}.txt"
        if out_file.exists():
            outputs.append(out_file.read_text())

    return _build_result(state, workflow, config.max_steps - state.budget_remaining,
                         0, "complete", outputs)


def run(
    workflow: WorkflowSpec,
    user_input: str,
    config: EngineConfig,
    mode: str = "orchestrated",
) -> EngineResult:
    if mode == "dry-run":
        return _run_dry(workflow, user_input, config)
    elif mode == "deterministic":
        return _run_orchestrated(
            workflow, user_input,
            EngineConfig(n_particles=1, tau=0.0, max_steps=config.max_steps,
                         max_forks=0, rewind_steps=config.rewind_steps),
            backend_mode="claude",
        )
    elif mode == "orchestrated":
        return _run_orchestrated(workflow, user_input, config, backend_mode="claude")
    elif mode == "agentic":
        return _run_agentic(workflow, user_input, config, backend_mode="claude")
    elif mode == "pfexec":
        return _run_orchestrated(workflow, user_input, config, backend_mode="claude")
    else:
        return _run_orchestrated(workflow, user_input, config, backend_mode="claude")


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
