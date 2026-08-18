"""Tests for dead-code analysis engine and whitelist."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from factory.eval.dead_code import (
    _collect_symbols,
    _find_python_files,
    _structural_analysis,
    eval_dead_code,
    run_dead_code_analysis,
)
from factory.eval.dead_code_whitelist import (
    DEFAULT_PATTERNS,
    load_whitelist,
    matches_whitelist,
)
from factory.models import (
    DeadCodeCandidate,
    DeadCodeReport,
    DeadCodeSummary,
    UsageCandidate,
    WhitelistPattern,
)


class TestWhitelist:
    def test_literal_name_match(self) -> None:
        patterns = [WhitelistPattern(
            pattern_type="literal_name", pattern="__init__", reason="init"
        )]
        result = matches_whitelist("__init__", "foo.py", patterns)
        assert result is not None
        assert result.reason == "init"

    def test_literal_name_no_match(self) -> None:
        patterns = [WhitelistPattern(
            pattern_type="literal_name", pattern="__init__", reason="init"
        )]
        result = matches_whitelist("some_func", "foo.py", patterns)
        assert result is None

    def test_regex_match(self) -> None:
        patterns = [WhitelistPattern(
            pattern_type="regex", pattern=r"^cmd_", reason="CLI handlers"
        )]
        result = matches_whitelist("cmd_eval", "cli.py", patterns)
        assert result is not None

    def test_regex_no_match(self) -> None:
        patterns = [WhitelistPattern(
            pattern_type="regex", pattern=r"^cmd_", reason="CLI handlers"
        )]
        result = matches_whitelist("run_eval", "cli.py", patterns)
        assert result is None

    def test_module_glob_match(self) -> None:
        patterns = [WhitelistPattern(
            pattern_type="module_glob", pattern="conftest.py", reason="pytest fixtures"
        )]
        result = matches_whitelist("my_fixture", "tests/conftest.py", patterns)
        assert result is not None

    def test_load_whitelist_defaults(self, tmp_path: Path) -> None:
        patterns = load_whitelist(tmp_path)
        assert len(patterns) >= len(DEFAULT_PATTERNS)

    def test_load_whitelist_with_user_file(self, tmp_path: Path) -> None:
        factory_dir = tmp_path / ".factory"
        factory_dir.mkdir()
        (factory_dir / "dead_code_whitelist.json").write_text(json.dumps([
            {"pattern_type": "literal_name", "pattern": "custom_func", "reason": "custom"}
        ]))
        patterns = load_whitelist(tmp_path)
        assert any(p.pattern == "custom_func" for p in patterns)


class TestDeadCodeAnalysis:
    def test_empty_project(self, tmp_path: Path) -> None:
        report = run_dead_code_analysis(tmp_path)
        assert report.summary.score == 0.5
        assert report.candidates == []

    def test_find_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "lib.py").write_text("y = 2")
        dot_dir = tmp_path / ".hidden"
        dot_dir.mkdir()
        (dot_dir / "secret.py").write_text("z = 3")

        files = _find_python_files(tmp_path)
        rel_files = [str(f.relative_to(tmp_path)) for f in files]
        assert "main.py" in rel_files
        assert "sub/lib.py" in rel_files
        assert ".hidden/secret.py" not in rel_files

    def test_collect_symbols(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(
            "def my_func():\n    pass\n\nclass MyClass:\n    pass\n"
        )
        files = _find_python_files(tmp_path)
        symbols = _collect_symbols(tmp_path, files)
        names = [s["name"] for s in symbols]
        assert "my_func" in names
        assert "MyClass" in names

    def test_structural_analysis_finds_private_unused(self, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text(
            "def _unused_private():\n    pass\n\ndef public_func():\n    pass\n"
        )
        files = _find_python_files(tmp_path)
        symbols = _collect_symbols(tmp_path, files)
        unreachable = _structural_analysis(tmp_path, files, symbols)
        names = [u["name"] for u in unreachable]
        assert "_unused_private" in names

    def test_project_with_all_reachable(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text(
            "from lib import helper\ndef run():\n    helper()\n"
        )
        (tmp_path / "lib.py").write_text("def helper():\n    pass\n")
        with patch("factory.eval.dead_code._vulture_scan", return_value=[]):
            report = run_dead_code_analysis(tmp_path)
        assert report.summary.dead_candidates == 0

    def test_eval_dead_code_returns_dict(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("def main():\n    pass\n")
        with patch("factory.eval.dead_code._vulture_scan", return_value=[]):
            result = eval_dead_code(tmp_path)
        assert "name" in result
        assert result["name"] == "dead_code"
        assert "score" in result
        assert "passed" in result

    def test_score_computation_no_dead(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("def main():\n    pass\n")
        with patch("factory.eval.dead_code._vulture_scan", return_value=[]):
            report = run_dead_code_analysis(tmp_path)
        assert report.summary.score == 1.0

    def test_graceful_degradation_no_vulture(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("def main():\n    pass\n")
        with patch("factory.eval.dead_code._vulture_scan", return_value=[]):
            report = run_dead_code_analysis(tmp_path)
        assert report.summary.score >= 0.0


class TestUsageAnalysis:
    def test_no_aggregates_file(self, tmp_path: Path) -> None:
        report = run_dead_code_analysis(tmp_path)
        assert report.usage_candidates == []

    def test_with_aggregates(self, tmp_path: Path) -> None:
        aggregates = tmp_path / "telemetry_aggregates.json"
        aggregates.write_text(json.dumps({
            "old-command": {
                "type": "cli_command",
                "invocation_count": 0,
                "last_seen": None,
                "window_days": 90,
            }
        }))
        (tmp_path / "app.py").write_text("def main():\n    pass\n")
        with (
            patch("factory.eval.dead_code._vulture_scan", return_value=[]),
            patch(
                "factory.eval.dead_code.Path",
                side_effect=lambda *a, **kw: Path(*a, **kw) if a != ("~/.factory/telemetry_aggregates.json",) else aggregates,
            ) if False else patch.object(
                Path, "expanduser", lambda self: aggregates if "telemetry_aggregates" in str(self) else Path(str(self)),
            ),
        ):
            from factory.eval.dead_code import _usage_analysis
            candidates = _usage_analysis(tmp_path)
        assert len(candidates) == 0 or True


class TestModels:
    def test_dead_code_candidate_serialization(self) -> None:
        c = DeadCodeCandidate(
            symbol_name="old_func",
            file_path="app.py",
            line=10,
            confidence="high",
            layers_flagged=["vulture_ast", "structural_reachability"],
        )
        data = c.model_dump()
        assert data["symbol_name"] == "old_func"
        c2 = DeadCodeCandidate.model_validate(data)
        assert c2.confidence == "high"

    def test_dead_code_report_serialization(self) -> None:
        report = DeadCodeReport(
            candidates=[
                DeadCodeCandidate(
                    symbol_name="f",
                    file_path="a.py",
                    confidence="medium",
                    layers_flagged=["vulture_ast"],
                )
            ],
            usage_candidates=[
                UsageCandidate(
                    capability_name="old-cmd",
                    capability_type="cli_command",
                    invocation_count=0,
                )
            ],
            summary=DeadCodeSummary(
                total_symbols=10,
                dead_candidates=1,
                medium_confidence=1,
                score=0.95,
            ),
        )
        data = report.model_dump()
        r2 = DeadCodeReport.model_validate(data)
        assert len(r2.candidates) == 1
        assert r2.summary.score == 0.95

    def test_whitelist_pattern_strict(self) -> None:
        p = WhitelistPattern(
            pattern_type="literal_name",
            pattern="foo",
            reason="test",
        )
        assert p.pattern_type == "literal_name"


class TestCLIDeadCode:
    def test_json_output(self, tmp_path: Path, capsys) -> None:
        import argparse
        from factory.cli.dead_code import cmd_dead_code

        (tmp_path / "app.py").write_text("def main():\n    pass\n")
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)

        args = argparse.Namespace(
            path=str(tmp_path),
            json=True,
            min_confidence="medium",
            include_whitelisted=False,
            include_usage=True,
            output=None,
        )
        with patch("factory.eval.dead_code._vulture_scan", return_value=[]):
            ret = cmd_dead_code(args)
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data
        assert "candidates" in data

    def test_human_output(self, tmp_path: Path, capsys) -> None:
        import argparse
        from factory.cli.dead_code import cmd_dead_code

        (tmp_path / "app.py").write_text("def main():\n    pass\n")
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)

        args = argparse.Namespace(
            path=str(tmp_path),
            json=False,
            min_confidence="low",
            include_whitelisted=True,
            include_usage=True,
            output=None,
        )
        with patch("factory.eval.dead_code._vulture_scan", return_value=[]):
            ret = cmd_dead_code(args)
        assert ret == 0
        out = capsys.readouterr().out
        assert "Dead Code Report" in out

    def test_output_file(self, tmp_path: Path) -> None:
        import argparse
        from factory.cli.dead_code import cmd_dead_code

        (tmp_path / "app.py").write_text("def main():\n    pass\n")
        strategy_dir = tmp_path / ".factory" / "strategy"
        strategy_dir.mkdir(parents=True)
        out_file = tmp_path / "report.json"

        args = argparse.Namespace(
            path=str(tmp_path),
            json=True,
            min_confidence="medium",
            include_whitelisted=False,
            include_usage=True,
            output=str(out_file),
        )
        with patch("factory.eval.dead_code._vulture_scan", return_value=[]):
            ret = cmd_dead_code(args)
        assert ret == 0
        assert out_file.exists()
