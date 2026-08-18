"""CLI handler for factory telemetry status|enable|disable subcommands."""

from __future__ import annotations

import argparse
import os


def cmd_telemetry(args: argparse.Namespace) -> int:
    action = getattr(args, "telemetry_action", None)
    if not action:
        print("Usage: factory telemetry {status,enable,disable}")
        return 1

    if action == "status":
        return _status()
    elif action == "enable":
        return _enable()
    elif action == "disable":
        return _disable()
    return 1


def _status() -> int:
    from factory.telemetry_consent import CONSENT_PATH, _load_consent, is_telemetry_enabled

    enabled = is_telemetry_enabled()
    consent = _load_consent()

    dnt = os.environ.get("DO_NOT_TRACK", "")
    env_override = os.environ.get("FACTORY_TELEMETRY", "")
    endpoint = os.environ.get("FACTORY_TELEMETRY_ENDPOINT", "")

    if not endpoint:
        try:
            from factory.user_config import load_config
            cfg = load_config()
            endpoint = cfg.get("telemetry", {}).get("endpoint", "")
        except Exception:
            pass

    print(f"Telemetry enabled: {enabled}")
    print(f"Consent file: {CONSENT_PATH}")
    if consent:
        print(f"  Consented at: {consent.consented_at}")
        print(f"  Version: {consent.version}")
    else:
        print("  No consent recorded")

    if dnt:
        print(f"DO_NOT_TRACK: {dnt} (overrides consent)")
    if env_override:
        print(f"FACTORY_TELEMETRY: {env_override} (overrides consent)")
    print(f"Endpoint: {endpoint or '(not configured — no data will be sent)'}")
    return 0


def _enable() -> int:
    from factory.telemetry_consent import set_consent
    set_consent(enabled=True)
    print("Telemetry enabled.")
    return 0


def _disable() -> int:
    from factory.telemetry_consent import set_consent
    set_consent(enabled=False)
    print("Telemetry disabled. No more data will be sent.")
    return 0
