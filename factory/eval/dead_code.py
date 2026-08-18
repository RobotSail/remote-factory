"""Layered dead-code analysis engine.

Layer 1 (structural): graph reachability + Vulture AST scanning.
Layer 2 (usage): telemetry-based invocation data consumption.
Coverage is at most a weak optional tiebreaker, never a scoring layer.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import structlog

from factory.eval.dead_code_whitelist import load_whitelist, matches_whitelist
from factory.models import (
    DeadCodeCandidate,
    DeadCodeReport,
    DeadCodeSummary,
    UsageCandidate,
)

log = structlog.get_logger()


def run_dead_code_analysis(project_path: Path) -> DeadCodeReport:
    project_path = project_path.resolve()
    py_files = _find_python_files(project_path)
    if not py_files:
        return _neutral_report()

    all_symbols = _collect_symbols(project_path, py_files)
    if not all_symbols:
        return _neutral_report()

    whitelist = load_whitelist(project_path)

    structural = _structural_analysis(project_path, py_files, all_symbols)
    vulture_results = _vulture_scan(project_path)
    usage_candidates = _usage_analysis(project_path)

    vulture_set: dict[tuple[str, str], int] = {}
    for v in vulture_results:
        key = (v.get("name", ""), v.get("filename", ""))
        vulture_set[key] = v.get("confidence", 60)

    candidates: list[DeadCodeCandidate] = []
    seen: set[tuple[str, str]] = set()

    for sym in structural:
        key = (sym["name"], sym["file"])
        if key in seen:
            continue
        seen.add(key)

        layers: list[str] = ["structural_reachability"]
        confidence = "low"

        v_key = (sym["name"], sym["file"])
        if v_key in vulture_set:
            layers.append("vulture_ast")
            v_conf = vulture_set[v_key]
            if v_conf >= 90:
                confidence = "high"
            elif v_conf >= 70:
                confidence = "medium"
            else:
                confidence = "medium"
        else:
            confidence = "low"

        wl_match = matches_whitelist(sym["name"], sym["file"], whitelist)
        candidates.append(DeadCodeCandidate(
            symbol_name=sym["name"],
            file_path=sym["file"],
            line=sym.get("line"),
            confidence=confidence,
            layers_flagged=layers,
            reachability_evidence=sym.get("evidence", ""),
            whitelisted=wl_match is not None,
            whitelist_reason=wl_match.reason if wl_match else None,
        ))

    for v in vulture_results:
        key = (v.get("name", ""), v.get("filename", ""))
        if key in seen:
            continue
        seen.add(key)

        v_conf = v.get("confidence", 60)
        if v_conf >= 90:
            confidence = "high"
        elif v_conf >= 70:
            confidence = "medium"
        else:
            confidence = "low"

        wl_match = matches_whitelist(v.get("name", ""), v.get("filename", ""), whitelist)
        candidates.append(DeadCodeCandidate(
            symbol_name=v.get("name", ""),
            file_path=v.get("filename", ""),
            line=v.get("lineno"),
            confidence=confidence,
            layers_flagged=["vulture_ast"],
            whitelisted=wl_match is not None,
            whitelist_reason=wl_match.reason if wl_match else None,
        ))

    non_whitelisted = [c for c in candidates if not c.whitelisted]
    total = len(all_symbols)

    weighted = 0.0
    for c in non_whitelisted:
        if c.confidence == "high":
            weighted += 1.0
        elif c.confidence == "medium":
            weighted += 0.5
        else:
            weighted += 0.1

    score = max(0.0, min(1.0, 1.0 - (weighted / total))) if total > 0 else 1.0

    summary = DeadCodeSummary(
        total_symbols=total,
        dead_candidates=len(non_whitelisted),
        high_confidence=sum(1 for c in non_whitelisted if c.confidence == "high"),
        medium_confidence=sum(1 for c in non_whitelisted if c.confidence == "medium"),
        low_confidence=sum(1 for c in non_whitelisted if c.confidence == "low"),
        whitelisted_count=sum(1 for c in candidates if c.whitelisted),
        usage_dead_count=len(usage_candidates),
        score=round(score, 4),
    )

    return DeadCodeReport(
        candidates=candidates,
        usage_candidates=usage_candidates,
        summary=summary,
    )


def eval_dead_code(project_path: Path) -> dict[str, Any]:
    report = run_dead_code_analysis(project_path)
    s = report.summary
    details = (
        f"{s.dead_candidates} dead-code candidates "
        f"({s.high_confidence} high, {s.medium_confidence} medium, {s.low_confidence} low)"
    )
    return {
        "name": "dead_code",
        "score": s.score,
        "weight": 1.0,
        "passed": s.score >= 0.7,
        "details": details,
    }


def _neutral_report() -> DeadCodeReport:
    return DeadCodeReport(
        summary=DeadCodeSummary(score=0.5),
    )


def _find_python_files(project_path: Path) -> list[Path]:
    results: list[Path] = []
    for p in project_path.rglob("*.py"):
        rel = str(p.relative_to(project_path))
        if any(part.startswith(".") for part in rel.split("/")):
            continue
        if rel.startswith(("node_modules/", "__pycache__/")):
            continue
        results.append(p)
    return sorted(results)


def _collect_symbols(project_path: Path, py_files: list[Path]) -> list[dict]:
    symbols: list[dict] = []
    for f in py_files:
        try:
            tree = ast.parse(f.read_text(), filename=str(f))
        except SyntaxError:
            continue
        rel = str(f.relative_to(project_path))
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append({"name": node.name, "file": rel, "line": node.lineno})
    return symbols


def _structural_analysis(
    project_path: Path,
    py_files: list[Path],
    all_symbols: list[dict],
) -> list[dict]:
    imported_names: set[str] = set()
    for f in py_files:
        try:
            tree = ast.parse(f.read_text(), filename=str(f))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.name if alias.asname is None else alias.asname)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name if alias.asname is None else alias.asname
                    imported_names.add(name.split(".")[-1])
            elif isinstance(node, ast.Name):
                imported_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                imported_names.add(node.attr)

    unreachable: list[dict] = []
    for sym in all_symbols:
        if sym["name"].startswith("_") and sym["name"] not in imported_names:
            unreachable.append({
                **sym,
                "evidence": "private symbol not referenced in any import or name lookup",
            })

    return unreachable


def _vulture_scan(project_path: Path) -> list[dict]:
    try:
        import vulture
    except ImportError:
        log.debug("vulture_not_installed")
        return []

    try:
        v = vulture.Vulture()
        v.scan([str(project_path)], exclude=[".factory", "__pycache__", ".git", "node_modules"])
        results: list[dict] = []
        for item in v.get_unused_code():
            rel_path = str(item.filename)
            try:
                rel_path = str(Path(item.filename).relative_to(project_path))
            except ValueError:
                pass
            results.append({
                "name": item.name,
                "filename": rel_path,
                "lineno": item.first_lineno,
                "confidence": item.confidence,
                "message": item.message,
            })
        return results
    except Exception as e:
        log.debug("vulture_scan_failed", error=str(e))
        return []


def _usage_analysis(project_path: Path) -> list[UsageCandidate]:
    aggregates_path = Path("~/.factory/telemetry_aggregates.json").expanduser()
    if not aggregates_path.exists():
        return []

    try:
        data = json.loads(aggregates_path.read_text())
    except Exception:
        return []

    candidates: list[UsageCandidate] = []
    for cap_name, info in data.items():
        if not isinstance(info, dict):
            continue
        count = info.get("invocation_count", 0)
        if count == 0:
            candidates.append(UsageCandidate(
                capability_name=cap_name,
                capability_type=info.get("type", "cli_command"),
                last_invoked=info.get("last_seen"),
                invocation_count=0,
                window_days=info.get("window_days", 90),
            ))

    return candidates
