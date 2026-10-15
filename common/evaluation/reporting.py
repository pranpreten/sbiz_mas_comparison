"""Question-level exports and summaries that retain failures and missing scores."""
from __future__ import annotations

from collections import defaultdict
import csv
import io
from datetime import datetime, timezone
import json
import re
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

SCORE_KEYS = ("requirement_coverage", "evidence_consistency")
CSV_FIELDS = (
    "question_id", "question_type", "domain", "question", "status", "stop_reason",
    "answer", "elapsed_seconds", "input_tokens", "output_tokens", "total_tokens",
    "usage_complete", "model_calls", "tool_calls", "error_type", "run_id",
    "requirement_coverage", "evidence_consistency", "grading_status",
)


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def effective_score(row: dict, key: str):
    """Execution failure without an answer is zero; judge errors remain missing."""
    if row.get("status") == "failed" and not str(row.get("answer") or "").strip():
        return 0.0
    return row.get(key)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Quality means require all questions resolved, including zero for answer failures."""
    total = len(rows)
    complete = [row for row in rows if row["status"] == "completed"]
    partial = [row for row in rows if row["status"] == "partial"]
    result: dict[str, Any] = {
        "questions": total,
        "completed": len(complete),
        "partial": len(partial),
        "failed": total - len(complete) - len(partial),
        "completion_rate": len(complete) / total if total else None,
        "answers_present": sum(bool(row.get("answer")) for row in rows),
    }
    for field in ("elapsed_seconds", "input_tokens", "output_tokens", "total_tokens", *SCORE_KEYS):
        eligible = [row for row in rows if numeric(row.get(field)) and (
            "tokens" not in field or row.get("usage_complete") is True)]
        values = [row[field] for row in eligible]
        if field in SCORE_KEYS:
            values = [effective_score(row, field) for row in rows
                      if numeric(effective_score(row, field))]
        result[field] = {
            "observed": len(values), "missing": total - len(values),
            "mean": (sum(values) / total if total and len(values) == total else None)
                    if field in SCORE_KEYS else (sum(values) / len(values) if values else None),
        }
        if "tokens" in field:
            result[field]["sum_complete_records"] = sum(values) if values else None
    return result


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["question_type"]].append(row)
    return {"overall": summarize(rows), "by_question_type": {
        key: summarize(group) for key, group in sorted(groups.items())}}


def atomic_text(path: Path, content: str, *, encoding="utf-8") -> None:
    """Readers see a complete old or new file, including during feedback refresh."""
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(content, encoding=encoding)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_report(directory: Path, report: dict) -> None:
    """Refresh derived exports in place while preserving run identity and answers."""
    rows = report["questions"]
    report["summary"] = build_summary(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ("'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
                         for key, value in row.items()})
    atomic_text(directory / "questions.csv", stream.getvalue(), encoding="utf-8-sig")
    atomic_text(directory / "summary.json", json.dumps(report["summary"], ensure_ascii=False, indent=2))
    atomic_text(directory / "report.json", json.dumps(report, ensure_ascii=False, indent=2, default=str))


def write_report(rows: list[dict[str, Any]], *, output_root: str | Path,
                 metadata: dict[str, Any]) -> Path:
    """Create a unique run folder, never overwrite an earlier experiment."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = metadata.get('agent_type', 'single_agent')
    if not isinstance(prefix, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', prefix):
        raise ValueError('Invalid architecture name')
    directory = Path(output_root) / f"{prefix}_{stamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "metadata": metadata,
              "summary": build_summary(rows), "questions": rows,
              "metric_note": "Answer execution only; judge latency/tokens excluded. Missing feedback is not zero. Token averages use only usage_complete records."}
    save_report(directory, report)
    return directory
