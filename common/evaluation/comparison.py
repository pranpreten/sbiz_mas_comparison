"""Seven architecture dashboard. Reads reports/feedback; never runs answer models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import monotonic, sleep

from common.agents import ARCHITECTURES
from common.config import PROJECT_ROOT, load_settings
from common.evaluation.reporting import SCORE_KEYS, atomic_text, numeric, save_report, summarize, effective_score

CONDITIONS = ("dataset_name", "workbook_sha256", "implementation_sha256", "model",
              "temperature", "experiment_date", "max_model_calls", "max_tool_calls",
              "max_elapsed_seconds", "model_timeout_seconds", "max_context_chars", "mode")
FIELDS = ("architecture", "label", "state", "questions", "completed", "partial", "failed",
          "requirement_coverage_mean", "requirement_coverage_observed",
          "evidence_consistency_mean", "evidence_consistency_observed", "graded_questions",
          "elapsed_seconds_mean", "elapsed_seconds_observed", "total_tokens_mean",
          "total_tokens_observed", "report_path", "experiment_name", "feedback_checked_at")


def group_key(report):
    meta = report["metadata"]
    conditions = {key: meta.get(key) for key in CONDITIONS}
    # Identity is the actual question set, not merely --limit or its size.
    conditions["question_ids"] = sorted(str(row["question_id"]) for row in report["questions"])
    raw = json.dumps(conditions, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16], conditions


def scan_reports(root):
    groups, ignored = {}, []
    for path in sorted(Path(root).glob("*/report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            meta = report["metadata"]
            if meta.get("agent_type") not in ARCHITECTURES or meta.get("mode") not in {"langsmith", "local"}:
                continue  # Development smoke collections are not experiments.
            rows = report["questions"]
            if not isinstance(rows, list) or not rows or not report.get("created_at"):
                raise ValueError("Missing experiment rows or timestamp")
            ids = [row["question_id"] for row in rows]
            if len(ids) != len(set(ids)) or any(row.get("status") not in {"completed", "partial", "failed"} for row in rows):
                raise ValueError("Invalid question rows")
            # Validate numbers before using an old/manual report in the dashboard.
            for row in rows:
                for key in SCORE_KEYS:
                    if not numeric(row.get(key)) or not 0 <= row[key] <= 1:
                        row[key] = None
            key, conditions = group_key(report)
            group = groups.setdefault(key, {"conditions": conditions, "runs": {}})
            name = meta["agent_type"]
            current = group["runs"].get(name)
            # Refresh does not make an older run supersede a newer experiment.
            if current is None or (report["created_at"], str(path)) > (current[1]["created_at"], str(current[0])):
                group["runs"][name] = (path, report)
        except (ValueError, KeyError, TypeError, OSError):
            ignored.append(str(path))
    return groups, ignored


def newest_group(groups):
    return max(groups, key=lambda key: max(report["created_at"] for _, report in groups[key]["runs"].values())) if groups else None


def comparison_rows(group):
    rows = []
    for name, label in ARCHITECTURES.items():
        item = group["runs"].get(name) if group else None
        row = dict.fromkeys(FIELDS)
        row.update(architecture=name, label=label, state="미실행")
        if item:
            path, report = item
            summary = summarize(report["questions"])
            graded = sum(all(numeric(effective_score(q, key)) for key in SCORE_KEYS) for q in report["questions"])
            row.update({key: summary[key] for key in ("questions", "completed", "partial", "failed")})
            row.update(graded_questions=graded, report_path=str(path.resolve()),
                       experiment_name=report["metadata"].get("experiment_name"),
                       feedback_checked_at=report.get("feedback_checked_at"),
                       state="로컬·미채점" if report["metadata"]["mode"] == "local" else
                             ("채점 완료" if graded == summary["questions"] else "채점 대기/확인 필요"))
            if any(q.get("grading_status") == "feedback_unavailable" for q in report["questions"]):
                row["state"] = "점수 조회 실패·기존 값 유지"
            for field in (*SCORE_KEYS, "elapsed_seconds", "total_tokens"):
                row[field + "_mean"] = summary[field]["mean"]
                row[field + "_observed"] = summary[field]["observed"]
        rows.append(row)
    return rows


def render_html(document):
    esc = lambda value: html.escape(str(value))
    def metric(row, field, percent=False):
        value = row[field + "_mean"]
        if value is None:
            return "—"
        display = f"{value * 100:.1f}%" if percent else f"{value:,.2f}"
        return f"{display}<small>{row[field + '_observed']}/{row['questions']}문항 관측</small>"
    body = []
    for row in document["rows"]:
        source = Path(row["report_path"]).as_uri() if row["report_path"] else None
        label = f'<a href="{esc(source)}">{esc(row["label"])}</a>' if source else esc(row["label"])
        status = esc(row["state"])
        if row["questions"]:
            status += f'<small>점수 확정(실패 0점 포함) {row["graded_questions"]}/{row["questions"]}</small>'
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in [label, status,
            metric(row, SCORE_KEYS[0], True), metric(row, SCORE_KEYS[1], True),
            metric(row, "elapsed_seconds"), metric(row, "total_tokens"),
            "—" if row["questions"] is None else f'{row["partial"]} / {row["failed"]}']) + "</tr>")
    links = " · ".join(f'<a href="{esc(item["html_path"])}">{esc(item["label"])}</a>' for item in document["groups"])
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>7개 에이전트 구조 비교</title><style>body{font:16px system-ui,sans-serif;margin:40px;color:#18233b;background:#f5f7fb}
table{border-collapse:collapse;width:100%;background:white}th,td{padding:15px;text-align:left;border-bottom:1px solid #dce2ed}
th{background:#e7edfa}small{display:block;color:#63718b;font-size:12px;margin-top:5px}a{color:#245fc0}p{line-height:1.7}
pre{white-space:pre-wrap;font-size:12px}section{margin-top:24px}</style>
<h1>7개 에이전트 구조 비교</h1>
<p>동일한 질문·실행 조건에서 구조별 최신 실행을 비교합니다. 미실행·미채점은 0이 아닌 빈칸입니다.<br>
답변 생성 실패는 두 품질 지표에 0점으로 반영합니다. 나머지 문항의 채점이 모두 끝나면 전체 문항 수(본 실험 36)로 나눈 평균을 표시합니다. 채점 오류·대기가 남으면 품질 평균은 빈칸입니다.</p>
<table><thead><tr><th>구조</th><th>상태</th><th>요구사항 충족률</th><th>근거 일치율</th><th>평균 시간(초)</th>
<th>평균 토큰</th><th>부분 / 실패</th></tr></thead><tbody>''' + "".join(body) + '''</tbody></table>
<p>답변 실행의 시간·토큰이며 채점 모델은 제외합니다. 품질 평균은 답변 생성 실패도 분모에 포함합니다. 시간 평균은 기록된 시간 기준입니다.
사용량이 불완전한 문항은 토큰 평균에서 제외하며 관측 수로 표시합니다.</p>
<p>이 페이지는 30초마다 로컬 파일을 다시 표시할 뿐 LangSmith에 접속하지 않습니다.
평가 실행은 결과 저장 후 종료합니다. 서버 채점 점수는 별도로 수동 조회해야 반영됩니다.</p>''' + (
        f'<p>파일 갱신: {esc(datetime.fromisoformat(document["updated_at"]).astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST"))}</p><section>조건별 결과: {links or "아직 실행 없음"}</section>'
        f'<details><summary>현재 비교 조건</summary><pre>{esc(json.dumps(document["conditions"], ensure_ascii=False, indent=2))}</pre></details>'
        f'<p>읽지 못한 보고서: {len(document["ignored_reports"])}개</p></html>')


def write_comparison(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    groups, ignored = scan_reports(root)
    active = newest_group(groups)
    links = [{"html_path": f"comparison_{key}.html", "label": f"{len(group['conditions']['question_ids'])}문항 · {group['conditions']['mode']} · {key[:8]}"}
             for key, group in groups.items()]
    def save(key, stem):
        group = groups.get(key)
        doc = {"updated_at": datetime.now(timezone.utc).isoformat(), "group_id": key,
               "conditions": group["conditions"] if group else {}, "rows": comparison_rows(group),
               "groups": links, "ignored_reports": ignored}
        atomic_text(root / (stem + ".json"), json.dumps(doc, ensure_ascii=False, indent=2))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in doc["rows"]:
            writer.writerow({key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
                             for key, value in row.items()})
        atomic_text(root / (stem + ".csv"), stream.getvalue(), encoding="utf-8-sig")
        atomic_text(root / (stem + ".html"), render_html(doc))
    for key in groups:
        save(key, "comparison_" + key)
    save(active, "comparison")
    return root / "comparison.html"


def refresh_once(root, client, *, group_id=None):
    """Refresh current group's latest runs only; retained older experiments stay intact."""
    from common.evaluation.cli import refresh_feedback
    groups, _ = scan_reports(root)
    group = groups.get(group_id or newest_group(groups))
    pending = 0
    if group:
        for path, report in group["runs"].values():
            if report["metadata"]["mode"] != "langsmith":
                continue
            candidates = [row for row in report["questions"] if row.get("run_id") and
                          (row.get("grading_status") == "feedback_unavailable" or
                           any(not numeric(row.get(key)) for key in SCORE_KEYS))]
            if candidates:
                refresh_feedback(candidates, client)
                report["feedback_checked_at"] = datetime.now(timezone.utc).isoformat()
                save_report(path.parent, report)
            pending += sum(row.get("grading_status") == "feedback_unavailable" or
                           any(not numeric(row.get(key)) for key in SCORE_KEYS) for row in report["questions"] if row.get("run_id"))
    write_comparison(root)
    return pending


def watch_feedback(root, client, *, wait_seconds=1800, interval=30, group_id=None,
                   clock=monotonic, sleeper=sleep, progress=print):
    """Bounded foreground polling; interrupting never loses saved answers."""
    if not numeric(wait_seconds) or not numeric(interval) or wait_seconds < 0 or interval <= 0:
        raise ValueError("Invalid feedback wait settings")
    deadline = clock() + wait_seconds
    while True:
        pending = refresh_once(root, client, group_id=group_id)
        progress(f"비교표 갱신: 추가 채점 확인이 필요한 실행 {pending}건")
        if not pending:
            return True
        remaining = deadline - clock()
        if remaining <= 0:
            progress("자동 조회 대기 종료. 미채점은 빈칸으로 보존했습니다. 다음 실행 또는 비교표 --watch로 이어서 조회할 수 있습니다.")
            return False
        sleeper(min(interval, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description="7개 구조 비교표 생성·서버 점수 조회. 답변/채점 모델 재실행 없음")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--env", type=Path)
    parser.add_argument("--refresh", action="store_true", help="Read pending feedback once")
    parser.add_argument("--watch", action="store_true", help="Poll pending feedback (default: up to 30 minutes)")
    parser.add_argument("--wait-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    if not numeric(args.wait_seconds) or args.wait_seconds < 0:
        parser.error("--wait-seconds must be nonnegative")
    print(f"비교표: {write_comparison(args.output_dir)}")
    if args.refresh or args.watch:
        load_settings(args.env)
        from langsmith import Client
        try:
            watch_feedback(args.output_dir, Client(), wait_seconds=args.wait_seconds if args.watch else 0)
        except KeyboardInterrupt:
            print("조회 중단. 저장된 답변과 비교표는 유지됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
