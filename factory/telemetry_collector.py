"""PII-stripping event summarizer for anonymous telemetry.

Takes raw events.jsonl dicts and extracts only allowlisted fields.
Uses a hard allowlist — new event fields are automatically excluded.
"""

from __future__ import annotations

import uuid
from importlib.metadata import version as pkg_version

from factory.models import TelemetryEvent, TelemetrySessionSummary

_ALLOWED_EVENT_TYPES = frozenset({
    "agent.started", "agent.completed", "agent.failed", "agent.timeout",
    "cycle.started", "cycle.completed",
    "cli.invoked",
})

_MAX_FIELD_LEN = 100


def _safe_str(value: object, max_len: int = _MAX_FIELD_LEN) -> str | None:
    if value is None:
        return None
    s = str(value)
    if len(s) > max_len:
        return None
    if "/" in s or "\\" in s:
        return None
    return s


def strip_pii(event: dict) -> TelemetryEvent | None:
    event_type = _safe_str(event.get("type"))
    if event_type is None:
        return None

    if event_type not in _ALLOWED_EVENT_TYPES:
        return None

    data = event.get("data", {}) or {}

    duration = data.get("duration_seconds")
    if duration is not None:
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = None

    return TelemetryEvent(
        command=_safe_str(data.get("command")) or "",
        event_type=event_type,
        success=bool(data.get("success", True)),
        duration_seconds=duration,
        agent_role=_safe_str(data.get("role") or event.get("agent")),
        workflow_mode=_safe_str(data.get("mode")),
        runner=_safe_str(data.get("runner")),
        timestamp=_safe_str(event.get("timestamp")) or "",
    )


def summarize_session(events: list[dict]) -> TelemetrySessionSummary:
    stripped: list[TelemetryEvent] = []
    for ev in events:
        te = strip_pii(ev)
        if te is not None:
            stripped.append(te)

    timestamps = [e.timestamp for e in stripped if e.timestamp]
    started = min(timestamps) if timestamps else ""
    ended = max(timestamps) if timestamps else ""

    try:
        fv = pkg_version("remote-factory")
    except Exception:
        fv = "unknown"

    return TelemetrySessionSummary(
        session_id=str(uuid.uuid4()),
        factory_version=fv,
        events=stripped,
        started_at=started,
        ended_at=ended,
    )
