"""Generate shell hook scripts for pfexec session directories."""

from __future__ import annotations

import stat
from pathlib import Path

from pfexec.engine import EngineConfig


def generate_hooks(session_dir: Path, engine_config: EngineConfig,
                   backend_mode: str = "claude") -> None:
    hooks_dir = session_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    pre_step = hooks_dir / "pre_step.sh"
    pre_step.write_text(
        '#!/bin/bash\n'
        'NODE_ID=$1\n'
        'SESSION_DIR="$(cd "$(dirname "$0")/.." && pwd)"\n'
        f'python -m pfexec.dist.cc.belief_io sample '
        f'--session "$SESSION_DIR" --node "$NODE_ID" '
        f'--backend {backend_mode}\n'
    )
    pre_step.chmod(pre_step.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    post_step = hooks_dir / "post_step.sh"
    post_step.write_text(
        '#!/bin/bash\n'
        'NODE_ID=$1\n'
        'SESSION_DIR="$(cd "$(dirname "$0")/.." && pwd)"\n'
        f'python -m pfexec.dist.cc.belief_io observe '
        f'--session "$SESSION_DIR" --node "$NODE_ID" '
        f'--backend {backend_mode}\n'
        f'python -m pfexec.dist.cc.belief_io fork-check '
        f'--session "$SESSION_DIR" --node "$NODE_ID" '
        f'--tau {engine_config.tau} --max-forks {engine_config.max_forks} '
        f'--backend {backend_mode} '
        f'> "$SESSION_DIR/hooks/fork_status.txt"\n'
    )
    post_step.chmod(post_step.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
