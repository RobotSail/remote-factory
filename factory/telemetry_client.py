"""Lightweight HTTP client for submitting anonymous telemetry session summaries.

Fire-and-forget: never blocks the CLI, never raises on failure. No hardcoded
default endpoint — data only flows when explicitly configured.
"""

from __future__ import annotations

import os
import threading

import structlog

from factory.models import TelemetrySessionSummary

log = structlog.get_logger()


def _get_endpoint_url() -> str | None:
    url = os.environ.get("FACTORY_TELEMETRY_ENDPOINT")
    if url:
        return url

    try:
        from factory.user_config import load_config
        cfg = load_config()
        telemetry_section = cfg.get("telemetry", {})
        return telemetry_section.get("endpoint") or None
    except Exception:
        return None


def submit_session(summary: TelemetrySessionSummary) -> bool:
    url = _get_endpoint_url()
    if not url:
        log.debug("telemetry_no_endpoint")
        return False

    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=summary.model_dump())
            return 200 <= resp.status_code < 300
    except Exception:
        return False


def submit_async(summary: TelemetrySessionSummary) -> None:
    t = threading.Thread(target=submit_session, args=(summary,), daemon=True)
    t.start()
