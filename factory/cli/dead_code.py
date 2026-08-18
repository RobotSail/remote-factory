"""CLI handler for factory dead-code <path> subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factory.models import DeadCodeReport


def cmd_dead_code(args: argparse.Namespace) -> int:
    from factory.eval.dead_code import run_dead_code_analysis
    from factory.models import DeadCodeReport

    project_path = Path(args.path).resolve()
    if not project_path.is_dir():
        print(f"Error: {project_path} is not a directory")
        return 1

    min_confidence = getattr(args, "min_confidence", "medium")
    include_whitelisted = getattr(args, "include_whitelisted", False)
    include_usage = getattr(args, "include_usage", True)
    json_output = getattr(args, "json", False)
    output_path = getattr(args, "output", None)

    report = run_dead_code_analysis(project_path)

    confidence_order = {"high": 3, "medium": 2, "low": 1}
    min_conf_val = confidence_order.get(min_confidence, 2)

    filtered_candidates = [
        c for c in report.candidates
        if confidence_order.get(c.confidence, 0) >= min_conf_val
        and (include_whitelisted or not c.whitelisted)
    ]

    filtered_report = DeadCodeReport(
        candidates=filtered_candidates,
        usage_candidates=report.usage_candidates if include_usage else [],
        summary=report.summary,
    )

    strategy_dir = project_path / ".factory" / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "dead-code-report.json").write_text(
        json.dumps(filtered_report.model_dump(), indent=2) + "\n"
    )

    if json_output:
        text = json.dumps(filtered_report.model_dump(), indent=2)
    else:
        text = _format_human_readable(filtered_report, include_usage)

    if output_path:
        Path(output_path).write_text(text + "\n")
        print(f"Report written to {output_path}")
    else:
        print(text)

    (strategy_dir / "dead-code-report.md").write_text(text + "\n")
    return 0


def _format_human_readable(report: DeadCodeReport, include_usage: bool) -> str:
    lines: list[str] = []
    lines.append("# Dead Code Report")
    lines.append("")

    s = report.summary
    lines.append(f"Total symbols analyzed: {s.total_symbols}")
    lines.append(f"Dead code candidates:   {s.dead_candidates}")
    lines.append(f"  High confidence:      {s.high_confidence}")
    lines.append(f"  Medium confidence:     {s.medium_confidence}")
    lines.append(f"  Low confidence:        {s.low_confidence}")
    lines.append(f"Whitelisted:            {s.whitelisted_count}")
    if s.usage_dead_count > 0:
        lines.append(f"Usage-dead (advisory):  {s.usage_dead_count}")
    lines.append(f"Score:                  {s.score:.3f}")
    lines.append("")

    if report.candidates:
        lines.append("## Candidates")
        lines.append("")
        lines.append(f"{'Symbol':<40} {'Location':<30} {'Confidence':<12} {'Layers'}")
        lines.append("-" * 100)
        for c in report.candidates:
            location = f"{c.file_path}:{c.line}" if c.line else c.file_path
            layers = ", ".join(c.layers_flagged)
            wl = " [whitelisted]" if c.whitelisted else ""
            lines.append(f"{c.symbol_name:<40} {location:<30} {c.confidence:<12} {layers}{wl}")
        lines.append("")

    if include_usage and report.usage_candidates:
        lines.append("## Usage-Telemetry Deprecation Candidates (Advisory)")
        lines.append("")
        lines.append(f"{'Capability':<30} {'Type':<20} {'Last Invoked':<25} {'Count'}")
        lines.append("-" * 85)
        for u in report.usage_candidates:
            last = u.last_invoked or "never"
            lines.append(f"{u.capability_name:<30} {u.capability_type:<20} {last:<25} {u.invocation_count}")
        lines.append("")

    return "\n".join(lines)
