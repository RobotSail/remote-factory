"""Session directory layout for compiled pfexec workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SessionDir:
    root: Path
    skill_path: Path
    belief_path: Path
    trace_dir: Path
    node_outputs_dir: Path
    hooks_dir: Path
    run_script: Path
    workflow_path: Path
    config_path: Path

    @classmethod
    def from_root(cls, root: Path) -> SessionDir:
        return cls(
            root=root,
            skill_path=root / "SKILL.md",
            belief_path=root / "belief.json",
            trace_dir=root / "trace",
            node_outputs_dir=root / "node_outputs",
            hooks_dir=root / "hooks",
            run_script=root / "run.sh",
            workflow_path=root / "workflow.json",
            config_path=root / "config.json",
        )

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(exist_ok=True)
        self.node_outputs_dir.mkdir(exist_ok=True)
        self.hooks_dir.mkdir(exist_ok=True)
