"""Consent management for opt-in anonymous telemetry.

Checks DO_NOT_TRACK, FACTORY_TELEMETRY env vars, and persisted consent state.
Default is opt-out (unset = disabled).
"""

from __future__ import annotations

import json
import os
from datetime import UTC
from pathlib import Path

import structlog

from factory.models import TelemetryConsent

log = structlog.get_logger()

CONSENT_PATH: Path = Path("~/.factory/telemetry_consent.json").expanduser()


def _load_consent() -> TelemetryConsent | None:
    if not CONSENT_PATH.exists():
        return None
    try:
        data = json.loads(CONSENT_PATH.read_text())
        return TelemetryConsent.model_validate(data)
    except Exception:
        log.debug("telemetry_consent_load_failed", path=str(CONSENT_PATH))
        return None


def is_telemetry_enabled() -> bool:
    dnt = os.environ.get("DO_NOT_TRACK", "")
    if dnt and dnt not in ("0", "false", "no"):
        return False

    env_override = os.environ.get("FACTORY_TELEMETRY", "")
    if env_override:
        return env_override.lower() in ("1", "true", "yes", "enabled")

    consent = _load_consent()
    if consent is None:
        return False
    return consent.enabled


def set_consent(enabled: bool) -> None:
    from datetime import datetime

    consent = TelemetryConsent(
        enabled=enabled,
        consented_at=datetime.now(UTC).isoformat(),
        version=1,
    )
    CONSENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSENT_PATH.write_text(json.dumps(consent.model_dump(), indent=2) + "\n")
    log.info("telemetry_consent_set", enabled=enabled)
