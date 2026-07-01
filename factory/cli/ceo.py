"""CLI ceo commands — thin dispatcher delegating to extracted modules."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import structlog
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from factory.cli._ceo_helpers import (
    _execute_ceo,
    _resolve_ceo_project,
    _validate_ceo_flags,
    _validate_late_flags,
)
from factory.cli._mode_handlers import (
    _auto_detect_mode,
    _resolve_model,
    handle_deep_qa_mode,
    handle_review_mode,
)
from factory.cli._path_resolver import _resolve_focus_issue

log = structlog.get_logger()


# ── subcommand handlers ──────────────────────────────────────


def cmd_ceo(args: argparse.Namespace) -> int:
    """Launch the Factory CEO agent to orchestrate a project."""
    from factory.user_config import load_config

    profile = getattr(args, "profile", None)
    load_config(profile=profile)

    raw_path: str | None = getattr(args, "path", None)

    validated = _validate_ceo_flags(args)
    if isinstance(validated, int):
        return validated
    mode, headless, bg, bg_agents, prompt_file, focus, dir_name, refine_request = validated

    assert raw_path is not None

    if mode == "review":
        return handle_review_mode(args, raw_path, headless)
    if mode == "deep-qa":
        return handle_deep_qa_mode(args, raw_path, headless)

    resolved = _resolve_ceo_project(raw_path, mode, headless, bg, focus, dir_name, prompt_file)
    if isinstance(resolved, int):
        return resolved
    (project_path, context, design_idea, research_ideation,
     deferred_spec, needs_materialize, design_existing, create_description,
     update_existing_mode) = resolved

    no_github = getattr(args, "no_github", False)
    issue_number: int | None = None
    issue_url: str | None = None
    if focus:
        from factory.issue import is_issue_ref

        if is_issue_ref(focus) and no_github:
            print(
                "Error: --focus resolved to an issue reference, but --no-github is set. "
                "Issue fetching requires GitHub/GitLab CLI access.",
                file=sys.stderr,
            )
            return 1
        issue_resolved = _resolve_focus_issue(focus, project_path)
        if issue_resolved:
            title, context, issue_number, issue_url = issue_resolved
            focus = f"{title} (issue #{issue_number})"

    force_fresh = mode == "auto-fresh"
    if mode in ("auto", "auto-fresh"):
        mode = _auto_detect_mode(
            project_path,
            has_prompt=bool(prompt_file or context),
            force_fresh=force_fresh,
        )

    err = _validate_late_flags(
        mode, focus, prompt_file, research_ideation,
        design_existing, project_path, no_github, issue_number,
    )
    if err is not None:
        return err

    if design_existing:
        banner_mode = "design"
    elif mode in ("design", "research") and (design_idea or research_ideation):
        banner_mode = "ideation"
    else:
        banner_mode = mode

    return _execute_ceo(
        args=args,
        project_path=project_path,
        context=context,
        mode=mode,
        banner_mode=banner_mode,
        headless=headless,
        bg=bg,
        bg_agents=bg_agents,
        focus=focus,
        prompt_file=prompt_file,
        design_idea=design_idea,
        design_existing=design_existing,
        research_ideation=research_ideation,
        create_description=create_description,
        update_existing_mode=update_existing_mode,
        deferred_spec=deferred_spec,
        needs_materialize=needs_materialize,
        refine_request=refine_request,
        issue_number=issue_number,
        issue_url=issue_url,
        no_github=no_github,
        raw_path=raw_path,
    )


def cmd_refactory(args: argparse.Namespace) -> int:
    """Launch the re:factory persistent supervisor agent."""
    import shutil

    from factory.agents.runner import resolve_prompt
    from factory.refactory import get_session_id, setup_workspace

    claude_path = shutil.which("claude")
    if not claude_path:
        print("Error: 'claude' CLI not found. Install Claude Code first.", file=sys.stderr)
        return 1

    project_path = Path(getattr(args, "path", None) or Path.cwd()).resolve()

    setup_workspace(project_path)
    reset = getattr(args, "reset", False)
    session_file = project_path / ".refactory" / "session.json"
    is_new_session = reset or not session_file.exists()
    session_id = get_session_id(project_path, reset=reset)
    model = getattr(args, "model", None)

    prompt = resolve_prompt("refactory")
    prompt_tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        prefix="refactory-prompt-",
        delete=False,
    )
    prompt_tmp.write(prompt)
    prompt_tmp.close()

    if is_new_session:
        cmd = [
            "claude",
            "--session-id",
            session_id,
            "--append-system-prompt-file",
            prompt_tmp.name,
            "--disallowedTools",
            "Agent",
            "--dangerously-skip-permissions",
        ]
    else:
        cmd = [
            "claude",
            "--resume",
            session_id,
            "--append-system-prompt-file",
            prompt_tmp.name,
            "--disallowedTools",
            "Agent",
            "--dangerously-skip-permissions",
        ]

    if model:
        cmd.extend(["--model", model])

    os.chdir(project_path)
    os.execvp("claude", cmd)
    return 0


# ── tmux integration ──────────────────────────────────────────


_TMUX_SESSION_PREFIX = "factory-"


_TMUX_SESSIONS_FILE = Path("~/.factory/tmux_sessions.json").expanduser()


def _tmux_session_name(project_path: Path) -> str:
    """Derive a tmux session name from a project path."""
    path_hash = hashlib.sha1(str(project_path).encode()).hexdigest()[:6]
    return f"{_TMUX_SESSION_PREFIX}{project_path.name}-{path_hash}"


def _load_tmux_session_mapping() -> dict[str, str]:
    """Load the session→project mapping from ~/.factory/tmux_sessions.json."""
    if _TMUX_SESSIONS_FILE.exists():
        try:
            return json.loads(_TMUX_SESSIONS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_tmux_session_mapping(session: str, project_path: str) -> None:
    """Save a session→project mapping entry to ~/.factory/tmux_sessions.json."""
    mapping = _load_tmux_session_mapping()
    mapping[session] = project_path
    _TMUX_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TMUX_SESSIONS_FILE.write_text(json.dumps(mapping, indent=2))


def _tmux_available() -> bool:
    """Check if tmux is installed."""
    try:
        subprocess.run(["tmux", "-V"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _tmux_session_alive(session: str) -> bool:
    """Check if a tmux session exists and is alive."""
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            capture_output=True,
        ).returncode
        == 0
    )


def _build_tmux_run_args(args: argparse.Namespace, project_path: Path, model: str | None) -> str:
    """Build the 'factory ceo ...' command string from parsed args."""
    parts = [f"factory ceo {project_path}"]
    if args.mode:
        parts.append(f"--mode {args.mode}")
    if model:
        parts.append(f"--model {shlex.quote(model)}")
    if getattr(args, "no_github", False):
        parts.append("--no-github")
    if getattr(args, "profile", None):
        parts.append(f"--profile {shlex.quote(args.profile)}")
    if getattr(args, "focus", None):
        parts.append(f"--focus {shlex.quote(args.focus)}")
    if getattr(args, "refine", None):
        parts.append(f"--refine {shlex.quote(args.refine)}")
    if getattr(args, "clean_pr", None) is True:
        parts.append("--clean-pr")
    elif getattr(args, "clean_pr", None) is False:
        parts.append("--no-clean-pr")
    if getattr(args, "runner", None):
        parts.append(f"--runner {shlex.quote(args.runner)}")
    if getattr(args, "prompt", None):
        parts.append(f"--prompt {shlex.quote(args.prompt)}")
    if getattr(args, "branch", None):
        parts.append(f"--branch {shlex.quote(args.branch)}")
    if getattr(args, "min_growth", None) is not None:
        parts.append(f"--min-growth {args.min_growth}")
    if getattr(args, "max_new", None) is not None:
        parts.append(f"--max-new {args.max_new}")
    if getattr(args, "discover_only", False):
        parts.append("--discover-only")
    if getattr(args, "bg_agents", False):
        parts.append("--bg-agents")
    if getattr(args, "tmux_persist", False):
        parts.append("--tmux-persist")
    if getattr(args, "use_profile", False):
        parts.append("--use-profile")
    return " ".join(parts)


def cmd_tmux(args: argparse.Namespace) -> int:
    """Launch factory run inside a detached tmux session."""
    if not _tmux_available():
        print("Error: tmux is not installed.", file=sys.stderr)
        return 1

    project_path = Path(args.path).resolve()
    session = args.session or _tmux_session_name(project_path)

    check = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
    )
    if check.returncode == 0:
        if args.attach:
            print(f"Attaching to existing session: {session}")
            os.execvp("tmux", ["tmux", "attach-session", "-t", session])
        print(f"Session '{session}' already running. Use --attach or:")
        print(f"  tmux attach -t {session}")
        return 0

    _ENV_PREFIXES = (
        "FACTORY_",
        "ANTHROPIC_",
        "BOBSHELL_",
        "OPENAI_",
        "CODEX_",
        "CLAUDE_CODE_",
        "CLOUD_ML_",
    )
    run_cmd_parts = []
    for key, val in sorted(os.environ.items()):
        if key.startswith(_ENV_PREFIXES):
            run_cmd_parts.append(f"export {key}={shlex.quote(val)}")
    run_cmd_parts.append(f"export PATH={shlex.quote(os.environ.get('PATH', '/usr/bin'))}")

    model = _resolve_model(args)
    run_args = _build_tmux_run_args(args, project_path, model)
    run_cmd_parts.append(run_args)
    shell_cmd = " && ".join(run_cmd_parts)

    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-x", "200", "-y", "50", shell_cmd],
    )
    if result.returncode != 0:
        print(f"Error: failed to create tmux session '{session}'", file=sys.stderr)
        return 1

    _save_tmux_session_mapping(session, str(project_path))

    time.sleep(3)

    if not _tmux_session_alive(session):
        print(f"Error: session '{session}' exited immediately after launch", file=sys.stderr)
        return 1

    capture = subprocess.run(
        ["tmux", "capture-pane", "-t", session, "-p"],
        capture_output=True,
        text=True,
    )
    if capture.returncode == 0:
        pane_text = capture.stdout
        _error_markers = ("Error:", "exited", "no server")
        if any(marker in pane_text for marker in _error_markers):
            log.warning("tmux_post_dispatch_warning", session=session)
            print(f"Warning: session '{session}' may have errors:", file=sys.stderr)
            for line in pane_text.strip().splitlines()[-10:]:
                print(f"  {line}", file=sys.stderr)

    print(f"Factory launched in tmux session: {session}")
    print(f"  tmux attach -t {session}    # attach")
    print(f"  tmux kill-session -t {session}  # stop")

    if args.attach:
        os.execvp("tmux", ["tmux", "attach-session", "-t", session])

    return 0


def cmd_tmux_ls(args: argparse.Namespace) -> int:
    """List running factory tmux sessions."""
    if not _tmux_available():
        print("Error: tmux is not installed.", file=sys.stderr)
        return 1

    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}\t#{session_created}\t#{session_windows}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("No tmux sessions running.")
        return 0

    mapping = _load_tmux_session_mapping()
    factory_sessions = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        name = parts[0]
        if name.startswith(_TMUX_SESSION_PREFIX):
            created = (
                datetime.fromtimestamp(int(parts[1])).strftime("%Y-%m-%d %H:%M")
                if len(parts) > 1
                else "?"
            )
            project = mapping.get(name, "?")
            factory_sessions.append({"session": name, "started": created, "project": project})

    if not factory_sessions:
        if getattr(args, "json_output", False):
            print("[]")
        else:
            print("No factory sessions running.")
        return 0

    if getattr(args, "json_output", False):
        print(json.dumps(factory_sessions, indent=2))
    else:
        print(f"{'Session':<35} {'Started':<20} {'Project'}")
        print("-" * 80)
        for s in factory_sessions:
            print(f"{s['session']:<35} {s['started']:<20} {s['project']}")
    return 0


def cmd_tmux_capture(args: argparse.Namespace) -> int:
    """Capture recent output from a factory tmux session."""
    if not _tmux_available():
        print("Error: tmux is not installed.", file=sys.stderr)
        return 1

    session = getattr(args, "session", None)
    if not session and getattr(args, "path", None):
        project_path = Path(args.path).resolve()
        mapping = _load_tmux_session_mapping()
        for s, p in mapping.items():
            if Path(p).resolve() == project_path:
                session = s
                break
        if not session:
            session = _tmux_session_name(project_path)

    if not session:
        print("Error: specify --session or path to identify the session", file=sys.stderr)
        return 1

    if not _tmux_session_alive(session):
        print(f"Error: session '{session}' not found", file=sys.stderr)
        return 1

    lines = getattr(args, "lines", -100)
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", session, "-p", "-S", str(lines)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: failed to capture pane for '{session}'", file=sys.stderr)
        return 1

    print(result.stdout, end="")
    return 0


def cmd_tmux_stop(args: argparse.Namespace) -> int:
    """Stop a factory tmux session."""
    if not _tmux_available():
        print("Error: tmux is not installed.", file=sys.stderr)
        return 1

    if args.session:
        session = args.session
    elif args.path:
        session = _tmux_session_name(Path(args.path).resolve())
    elif getattr(args, "stop_all", False):
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("No tmux sessions running.")
            return 0

        killed = 0
        for name in result.stdout.strip().splitlines():
            if name.startswith(_TMUX_SESSION_PREFIX):
                subprocess.run(["tmux", "kill-session", "-t", name])
                print(f"Stopped: {name}")
                killed += 1

        if killed == 0:
            print("No factory sessions running.")
        else:
            print(f"Stopped {killed} session(s).")
        return 0
    else:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True,
            text=True,
        )
        sessions = []
        if result.returncode == 0:
            for name in result.stdout.strip().splitlines():
                if name.startswith(_TMUX_SESSION_PREFIX):
                    sessions.append(name)
        if sessions:
            print("Factory sessions that would be stopped:")
            for s in sessions:
                print(f"  {s}")
        else:
            print("No factory sessions running.")
        print("\nUse --all to stop all factory sessions.")
        return 1

    check = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
    )
    if check.returncode != 0:
        print(f"Session '{session}' not found.")
        return 1

    mapping = _load_tmux_session_mapping()
    if session not in mapping and not getattr(args, "force", False):
        print(
            f"Warning: session '{session}' is not in the factory session registry.",
            file=sys.stderr,
        )
        print(
            "It may not be a factory-managed session. Use --force to kill it anyway.",
            file=sys.stderr,
        )
        return 1

    subprocess.run(["tmux", "kill-session", "-t", session])
    print(f"Stopped: {session}")
    return 0
