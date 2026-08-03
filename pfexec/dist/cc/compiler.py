"""Top-level compiler — WorkflowSpec + EngineConfig to session directory."""

from __future__ import annotations

import json
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path

from pfexec.dist.cc.belief_io import cmd_init
from pfexec.dist.cc.hooks import generate_hooks
from pfexec.dist.cc.session import SessionDir
from pfexec.dist.cc.skill_gen import generate
from pfexec.engine import EngineConfig
from pfexec.ir import WorkflowSpec


def compile(
    workflow: WorkflowSpec,
    config: EngineConfig,
    user_input: str,
    backend_mode: str = "claude",
    session_dir: Path | None = None,
) -> SessionDir:
    if session_dir is None:
        session_dir = Path(tempfile.mkdtemp(prefix="pfexec-session-"))

    session = SessionDir.from_root(session_dir)
    session.ensure_dirs()

    session.workflow_path.write_text(workflow.to_json())
    session.config_path.write_text(json.dumps(asdict(config), indent=2))

    skill_md = generate(workflow, config)
    session.skill_path.write_text(skill_md)

    generate_hooks(session_dir, config, backend_mode=backend_mode)

    cmd_init(session_dir, session.workflow_path, user_input, config.n_particles, backend_mode)

    input_path = session_dir / "input.txt"
    input_path.write_text(user_input)

    session.run_script.write_text(
        '#!/bin/bash\n'
        'SESSION_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'QUESTION="${1:-$(cat "$SESSION_DIR/input.txt")}"\n'
        'claude --bare --system-prompt-file "$SESSION_DIR/SKILL.md" -p "$QUESTION"\n'
    )
    session.run_script.chmod(
        session.run_script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )

    return session
