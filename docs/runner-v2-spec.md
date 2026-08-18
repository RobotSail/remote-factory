# Runner Abstraction v2 — Technical Specification

## Architecture

```
CLI / invoke_agent()
        │
    AgentRunRequest (prompt, task, cwd, timeout, model, skip_permissions, role, session_name, project_path, extras)
        │
        ▼
    Runner Protocol
    ├── metadata() → RunnerMeta
    ├── build_command(request) → (cmd[], env{}, tmp_files[])
    ├── headless(request) → AgentRunResult
    └── interactive_run(request) → int
        │
        └── ClaudeRunner    ── system prompt via --append-system-prompt-file
        │
        ▼
    run_subprocess() (shared executor)
        ── asyncio.create_subprocess_exec
        ── stdin=DEVNULL
        ── timeout + kill
        ── streaming via tee_stream
        ── returns AgentRunResult
```

## Data Models

### AgentRunRequest (factory/models.py)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| prompt | str | — | Agent role system prompt |
| task | str | — | The specific task to execute |
| cwd | Path | — | Working directory |
| timeout | float | 600.0 | Max seconds before kill |
| model | str \| None | None | Model override |
| skip_permissions | bool | True | Auto-approve all actions |
| role | str | "unknown" | Agent role name for logging |
| session_name | str \| None | None | Session identifier |
| project_path | Path \| None | None | Project root for .factory/ access |
| extras | dict[str, object] | {} | Runner-specific config (tmux_persist, etc.) |

### AgentRunResult (factory/models.py)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| stdout | str | — | Captured output |
| return_code | int | — | Process exit code |
| usage | AgentUsage \| None | None | Token telemetry |
| metadata | dict[str, object] | {} | stderr, runner-specific data |

### RunnerMeta (factory/runners/protocol.py)

| Field | Type | Default |
|-------|------|---------|
| name | str | — |
| display_name | str | — |
| binary | str | — |
| install_hint | str | — |
| required_env_vars | list[str] | [] |
| supports_model_override | bool | True |
| supports_interactive | bool | True |
| supports_streaming | bool | True |
| supports_usage_telemetry | bool | False |
| supports_session_name | bool | False |

Methods: `is_available() → bool` (shutil.which), `check_auth() → bool` (env vars check)

## Plugin Discovery

Third-party runners register via Python entry points:

```toml
# In third-party package pyproject.toml:
[project.entry-points."factory.runners"]
myrunner = "my_package:MyRunner"
```

Discovery: `importlib.metadata.entry_points(group="factory.runners")`

- Built-in runner (`claude`) registered in `_RUNNERS` dict
- Entry points loaded once via `_load_entrypoints()` with `_entrypoints_loaded` guard
- Built-in runners take precedence on name collision
- Load failures logged at debug level, do not crash
- CLI choices generated dynamically from `get_available_runners().keys()`

## ClaudeRunner Feature Matrix

| Feature | Claude |
|---------|--------|
| Headless mode | -p task |
| System prompt (proper slot) | --append-system-prompt-file |
| Model override | --model |
| Permissions skip | --dangerously-skip-permissions |
| Token telemetry | JSON usage block |
| JSON output | --output-format json |
| Session naming | --name |
| tmux persistence | Yes |
| Background dispatch | Yes (--bg) |

## System Prompt Handling

The factory agent system has two levels of prompts:

1. **Project-level instructions** (CLAUDE.md) — read automatically by Claude Code from the project directory.

2. **Per-agent role prompts** (e.g., "You are the Researcher agent...") — resolved by `factory/agents/runner.py` via `resolve_prompt(role, project_path)`. Delivered via `--append-system-prompt-file` into Claude's system prompt slot.

## Dry-Run Mode

Claude runner uses mocked subprocess in tests (no dedicated dry-run env var).

## Files

| File | Purpose |
|------|---------|
| factory/models.py | AgentRunRequest, AgentRunResult, AgentUsage models |
| factory/runners/protocol.py | Runner protocol, RunnerMeta |
| factory/runners/__init__.py | Registry, plugin discovery, get_runner() |
| factory/runners/_subprocess.py | Shared subprocess executor, make_dry_run_result |
| factory/runners/_stream.py | Streaming output, ANSI stripping |
| factory/runners/claude.py | ClaudeRunner |
| tests/test_runner_e2e.py | E2E tests with real API calls |
| tests/test_runners.py | Unit tests (mocked subprocess) |
