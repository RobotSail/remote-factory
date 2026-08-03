"""Generate shell hook scripts for pfexec session directories."""

from __future__ import annotations

import json
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


def generate_settings(session_dir: Path, config: EngineConfig,
                      backend_mode: str = "claude") -> None:
    hooks_dir = session_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    observer_path = hooks_dir / "write_observer.sh"
    observer_path.write_text(
        '#!/bin/bash\n'
        f'SESSION_DIR="{session_dir}"\n'
        'for f in "$SESSION_DIR/node_outputs/"*.txt; do\n'
        '    [ -f "$f" ] || continue\n'
        '    NODE_ID=$(basename "$f" .txt)\n'
        '    MARKER="$SESSION_DIR/hooks/.observed_${NODE_ID}"\n'
        '    if [ ! -f "$MARKER" ]; then\n'
        f'        python3 -m pfexec.dist.cc.belief_io observe'
        f' --session "$SESSION_DIR" --node "$NODE_ID"'
        f' --backend {backend_mode} 2>/dev/null\n'
        f'        python3 -m pfexec.dist.cc.belief_io fork-check'
        f' --session "$SESSION_DIR" --node "$NODE_ID"'
        f' --tau {config.tau} --max-forks {config.max_forks}'
        f' --backend {backend_mode}'
        f' > "$SESSION_DIR/hooks/fork_status.txt" 2>/dev/null\n'
        '        touch "$MARKER"\n'
        '    fi\n'
        'done\n'
    )
    observer_path.chmod(
        observer_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )

    claude_dir = session_dir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    settings = {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Write",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"bash {observer_path}",
                        }
                    ],
                }
            ]
        }
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))
