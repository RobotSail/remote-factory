"""Dynamic-dispatch whitelist for dead-code analysis.

Defines default patterns for known false-positive sources and loads
user extensions from .factory/dead_code_whitelist.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from factory.models import WhitelistPattern

DEFAULT_PATTERNS: list[WhitelistPattern] = [
    WhitelistPattern(
        pattern_type="regex",
        pattern=r"^cmd_",
        reason="CLI handler dict dispatch targets",
    ),
    WhitelistPattern(
        pattern_type="regex",
        pattern=r"^test_",
        reason="pytest test functions invoked by pytest discovery",
    ),
    WhitelistPattern(
        pattern_type="module_glob",
        pattern="conftest.py",
        reason="pytest fixtures invoked by pytest DI",
    ),
    WhitelistPattern(
        pattern_type="literal_name",
        pattern="__init__",
        reason="Python module initializer",
    ),
    WhitelistPattern(
        pattern_type="literal_name",
        pattern="main",
        reason="Common entry point name",
    ),
    WhitelistPattern(
        pattern_type="regex",
        pattern=r"^_.*_validator$",
        reason="Pydantic field validators invoked by framework",
    ),
    WhitelistPattern(
        pattern_type="regex",
        pattern=r"^model_config$",
        reason="Pydantic model configuration attribute",
    ),
    WhitelistPattern(
        pattern_type="decorator",
        pattern="@app.route",
        reason="Flask/FastAPI route handlers invoked by framework",
    ),
    WhitelistPattern(
        pattern_type="decorator",
        pattern="@router",
        reason="FastAPI router endpoints invoked by framework",
    ),
    WhitelistPattern(
        pattern_type="decorator",
        pattern="@pytest.fixture",
        reason="pytest fixtures invoked by DI",
    ),
    WhitelistPattern(
        pattern_type="module_glob",
        pattern="__init__.py",
        reason="Module re-exports (public API surface)",
    ),
    WhitelistPattern(
        pattern_type="regex",
        pattern=r"^_coerce_",
        reason="Pydantic field validators with coerce prefix",
    ),
    WhitelistPattern(
        pattern_type="literal_name",
        pattern="meta",
        reason="Workflow registry introspection attribute",
    ),
]


def load_whitelist(project_path: Path) -> list[WhitelistPattern]:
    patterns = list(DEFAULT_PATTERNS)
    user_file = project_path / ".factory" / "dead_code_whitelist.json"
    if user_file.exists():
        try:
            data = json.loads(user_file.read_text())
            for item in data:
                patterns.append(WhitelistPattern.model_validate(item))
        except Exception:
            pass
    return patterns


def matches_whitelist(
    symbol_name: str,
    file_path: str,
    patterns: list[WhitelistPattern],
) -> WhitelistPattern | None:
    for p in patterns:
        if p.pattern_type == "literal_name":
            if symbol_name == p.pattern:
                return p
        elif p.pattern_type == "regex":
            if re.search(p.pattern, symbol_name):
                return p
        elif p.pattern_type == "module_glob":
            if file_path.endswith(p.pattern):
                return p
        elif p.pattern_type == "decorator":
            pass
    return None
