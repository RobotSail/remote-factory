"""Tests for telemetry consent, collector, client, and CLI commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from factory.models import TelemetrySessionSummary


class TestTelemetryConsent:
    def test_unset_returns_false(self, tmp_path: Path) -> None:
        with patch("factory.telemetry_consent.CONSENT_PATH", tmp_path / "nope.json"):
            from factory.telemetry_consent import is_telemetry_enabled
            assert is_telemetry_enabled() is False

    def test_enabled_consent(self, tmp_path: Path) -> None:
        consent_file = tmp_path / "consent.json"
        consent_file.write_text(json.dumps({
            "enabled": True, "consented_at": "2026-01-01T00:00:00Z", "version": 1
        }))
        with patch("factory.telemetry_consent.CONSENT_PATH", consent_file):
            from factory.telemetry_consent import is_telemetry_enabled
            assert is_telemetry_enabled() is True

    def test_disabled_consent(self, tmp_path: Path) -> None:
        consent_file = tmp_path / "consent.json"
        consent_file.write_text(json.dumps({
            "enabled": False, "consented_at": "2026-01-01T00:00:00Z", "version": 1
        }))
        with patch("factory.telemetry_consent.CONSENT_PATH", consent_file):
            from factory.telemetry_consent import is_telemetry_enabled
            assert is_telemetry_enabled() is False

    def test_do_not_track_overrides(self, tmp_path: Path) -> None:
        consent_file = tmp_path / "consent.json"
        consent_file.write_text(json.dumps({
            "enabled": True, "consented_at": "2026-01-01T00:00:00Z", "version": 1
        }))
        with (
            patch("factory.telemetry_consent.CONSENT_PATH", consent_file),
            patch.dict(os.environ, {"DO_NOT_TRACK": "1"}),
        ):
            from factory.telemetry_consent import is_telemetry_enabled
            assert is_telemetry_enabled() is False

    def test_factory_telemetry_env_override(self, tmp_path: Path) -> None:
        with (
            patch("factory.telemetry_consent.CONSENT_PATH", tmp_path / "nope.json"),
            patch.dict(os.environ, {"FACTORY_TELEMETRY": "true"}, clear=False),
        ):
            from factory.telemetry_consent import is_telemetry_enabled
            assert is_telemetry_enabled() is True

    def test_set_consent(self, tmp_path: Path) -> None:
        consent_file = tmp_path / "consent.json"
        with patch("factory.telemetry_consent.CONSENT_PATH", consent_file):
            from factory.telemetry_consent import set_consent
            set_consent(enabled=True)
            data = json.loads(consent_file.read_text())
            assert data["enabled"] is True
            assert data["version"] == 1


class TestTelemetryCollector:
    def test_strip_pii_basic(self) -> None:
        from factory.telemetry_collector import strip_pii
        event = {
            "type": "agent.completed",
            "timestamp": "2026-01-01T00:00:00Z",
            "project": "/home/user/secret-project",
            "agent": "builder",
            "data": {
                "command": "ceo",
                "success": True,
                "duration_seconds": 42.5,
                "role": "builder",
                "mode": "improve",
                "runner": "claude",
                "some_secret": "should_be_dropped",
            },
        }
        result = strip_pii(event)
        assert result is not None
        assert result.event_type == "agent.completed"
        assert result.command == "ceo"
        assert result.agent_role == "builder"
        assert result.duration_seconds == 42.5
        assert result.workflow_mode == "improve"

    def test_strip_pii_drops_paths(self) -> None:
        from factory.telemetry_collector import strip_pii
        event = {
            "type": "agent.started",
            "timestamp": "2026-01-01T00:00:00Z",
            "data": {"command": "/usr/bin/something"},
        }
        result = strip_pii(event)
        assert result is not None
        assert result.command == ""

    def test_strip_pii_drops_long_strings(self) -> None:
        from factory.telemetry_collector import strip_pii
        event = {
            "type": "cli.invoked",
            "timestamp": "2026-01-01T00:00:00Z",
            "data": {"command": "x" * 200},
        }
        result = strip_pii(event)
        assert result is not None
        assert result.command == ""

    def test_summarize_session(self) -> None:
        from factory.telemetry_collector import summarize_session
        events = [
            {
                "type": "agent.started",
                "timestamp": "2026-01-01T00:00:00Z",
                "data": {"role": "builder"},
            },
            {
                "type": "agent.completed",
                "timestamp": "2026-01-01T00:01:00Z",
                "data": {"role": "builder", "success": True},
            },
        ]
        summary = summarize_session(events)
        assert len(summary.events) == 2
        assert summary.session_id
        assert summary.started_at == "2026-01-01T00:00:00Z"
        assert summary.ended_at == "2026-01-01T00:01:00Z"


class TestTelemetryClient:
    def test_submit_no_endpoint(self) -> None:
        from factory.telemetry_client import submit_session
        summary = TelemetrySessionSummary(
            session_id="test-id",
            factory_version="0.0.0",
            events=[],
        )
        with patch("factory.telemetry_client._get_endpoint_url", return_value=None):
            result = submit_session(summary)
            assert result is False

    def test_submit_success(self) -> None:
        from factory.telemetry_client import submit_session
        summary = TelemetrySessionSummary(
            session_id="test-id",
            factory_version="0.0.0",
            events=[],
        )

        class MockResponse:
            status_code = 200

        class MockClient:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def post(self, url, json=None):
                return MockResponse()

        with (
            patch("factory.telemetry_client._get_endpoint_url", return_value="http://test.local/t"),
            patch("httpx.Client", return_value=MockClient()),
        ):
            result = submit_session(summary)
            assert result is True

    def test_submit_failure_silent(self) -> None:
        from factory.telemetry_client import submit_session
        summary = TelemetrySessionSummary(
            session_id="test-id",
            factory_version="0.0.0",
            events=[],
        )
        with (
            patch("factory.telemetry_client._get_endpoint_url", return_value="http://test.local/t"),
            patch("httpx.Client", side_effect=Exception("network error")),
        ):
            result = submit_session(summary)
            assert result is False


class TestTelemetryCLI:
    def test_status_command(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        import argparse
        from factory.cli.telemetry import cmd_telemetry

        args = argparse.Namespace(telemetry_action="status")
        with patch("factory.telemetry_consent.CONSENT_PATH", tmp_path / "nope.json"):
            ret = cmd_telemetry(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "Telemetry enabled:" in out

    def test_enable_command(self, tmp_path: Path) -> None:
        import argparse
        from factory.cli.telemetry import cmd_telemetry

        consent_file = tmp_path / "consent.json"
        args = argparse.Namespace(telemetry_action="enable")
        with patch("factory.telemetry_consent.CONSENT_PATH", consent_file):
            ret = cmd_telemetry(args)
        assert ret == 0
        data = json.loads(consent_file.read_text())
        assert data["enabled"] is True

    def test_disable_command(self, tmp_path: Path) -> None:
        import argparse
        from factory.cli.telemetry import cmd_telemetry

        consent_file = tmp_path / "consent.json"
        args = argparse.Namespace(telemetry_action="disable")
        with patch("factory.telemetry_consent.CONSENT_PATH", consent_file):
            ret = cmd_telemetry(args)
        assert ret == 0
        data = json.loads(consent_file.read_text())
        assert data["enabled"] is False


class TestTelemetryEventIntegration:
    def test_emit_event_buffers_when_enabled(self, tmp_path: Path) -> None:
        import factory.events as events_mod
        events_mod._telemetry_buffer.clear()
        events_mod._telemetry_flush_registered = False

        factory_dir = tmp_path / "proj" / ".factory"
        factory_dir.mkdir(parents=True)

        with patch("factory.telemetry_consent.is_telemetry_enabled", return_value=True):
            events_mod.emit_event(
                tmp_path / "proj",
                "test.event",
                data={"command": "test"},
            )

        assert len(events_mod._telemetry_buffer) == 1
        events_mod._telemetry_buffer.clear()

    def test_emit_event_no_buffer_when_disabled(self, tmp_path: Path) -> None:
        import factory.events as events_mod
        events_mod._telemetry_buffer.clear()
        events_mod._telemetry_flush_registered = False

        factory_dir = tmp_path / "proj" / ".factory"
        factory_dir.mkdir(parents=True)

        with patch("factory.telemetry_consent.is_telemetry_enabled", return_value=False):
            events_mod.emit_event(
                tmp_path / "proj",
                "test.event",
                data={"command": "test"},
            )

        assert len(events_mod._telemetry_buffer) == 0
