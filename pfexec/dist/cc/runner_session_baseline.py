"""Session baseline — Claude Code session with SKILL.md, no engine.

Same SKILL.md as agentic v2 (prose phases, save to node_outputs/).
Same --allowedTools 'Bash Read Write'. NO hooks, NO init, NO observe,
NO fork. Just workflow structure in a single session.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from pfexec.dist.cc.skill_gen import _terminal_nodes, _topo_order, generate_agentic
from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import WorkflowSpec
from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


def run(workflow: WorkflowSpec, user_input: str, config: EngineConfig,
        backend_mode: str = 'claude') -> EngineResult:
    session_dir = Path(tempfile.mkdtemp(prefix='pfexec-session-baseline-'))
    (session_dir / 'node_outputs').mkdir()

    skill_md = generate_agentic(workflow, config, session_dir, backend_mode=backend_mode)
    skill_path = session_dir / 'SKILL.md'
    skill_path.write_text(skill_md)

    order = _topo_order(workflow)
    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]

    if backend_mode == 'mock':
        for nid in order:
            (session_dir / 'node_outputs' / f'{nid}.txt').write_text(f'mock output for {nid}')
        raw_output = ''
    else:
        result = subprocess.run(
            ['claude',
             '--system-prompt-file', str(skill_path),
             '--allowedTools', 'Bash Read Write',
             '--dangerously-skip-permissions',
             '-p', f'Execute the {workflow.name} workflow for: {user_input}'],
            capture_output=True, text=True,
            timeout=config.max_steps * 120,
            cwd=str(session_dir),
        )
        raw_output = result.stdout.strip()

    node_outputs: dict[str, str] = {}
    all_outputs: list[str] = []
    for nid in order:
        out_file = session_dir / 'node_outputs' / f'{nid}.txt'
        if out_file.exists():
            text = out_file.read_text().strip()
            if text:
                node_outputs[nid] = text
                all_outputs.append(text)

    steps_taken = len(node_outputs)

    final_answer = node_outputs.get(terminal_id, '')
    if not final_answer and all_outputs:
        final_answer = all_outputs[-1]
    if not final_answer and raw_output:
        final_answer = raw_output.split('\n')[-1].strip()

    if final_answer:
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', final_answer)
        cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
        cleaned = cleaned.strip()
        lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
        if lines:
            final_answer = lines[-1]

    belief = Belief(particles=[Particle(brief='', weight=1.0)])
    trace = TraceTree(root=TraceNode(node_id='root'))
    state = ExecutionState(
        pointer=terminal_id,
        belief=belief,
        trace=trace,
        step=steps_taken,
        budget_remaining=config.max_steps - steps_taken,
        user_input=user_input,
        node_outputs=node_outputs,
    )

    return EngineResult(
        final_state=state,
        output=final_answer,
        steps_taken=steps_taken,
        forks_triggered=0,
        terminated_by='complete',
        all_outputs=all_outputs,
    )
