"""Validate by default; run answer generation only with --local-run or --run.

--local-run disables LangSmith tracing and writes local reports (model API cost
still applies). --run uses the named LangSmith dataset; --sync explicitly permits
question/reference updates. Neither mode changes or creates judge definitions.
"""
from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import hashlib
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from common.config import PROJECT_ROOT, load_settings
from common.evaluation.dataset import EvaluationExample, load_examples, sync_dataset
from common.evaluation.runner import question_inputs, run_experiment
from common.evaluation.reporting import SCORE_KEYS, numeric, write_report, save_report
from common.agents import ARCHITECTURES, create_agent


class EvaluationSetupError(ValueError):
    """Only fixed, credential-free setup messages may be shown to the user."""


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _base_row(example: EvaluationExample) -> dict[str, Any]:
    return {**example.metadata, "question": example.question, "answer": "", "status": "failed",
            "stop_reason": "", "error_type": None, "run_id": None,
            "elapsed_seconds": None, "input_tokens": None, "output_tokens": None,
            "total_tokens": None, "usage_complete": False, "model_calls": None,
            "tool_calls": None, **{key: None for key in SCORE_KEYS}, "grading_status": "not_requested"}


def _apply_state(row: dict, state: dict) -> None:
    answer = state.get("final_answer", "")
    row["answer"] = answer if isinstance(answer, str) else ""
    status = {"success": "completed", "incomplete": "partial", "error": "failed"}.get(state.get("status"), "failed")
    row["status"] = status if status in {"completed", "partial"} and row["answer"].strip() else "failed"
    row["stop_reason"] = state.get("stop_reason", "")
    row["error_type"] = state.get("error_type")
    metrics = state.get("metrics") or {}
    row['execution_metrics'] = deepcopy(metrics)
    row['execution_events'] = deepcopy(state.get('events', []))
    for key in ("input_tokens", "output_tokens", "total_tokens", "model_calls", "tool_calls"):
        value = metrics.get(key)
        row[key] = value if numeric(value) else None
    row["usage_complete"] = metrics.get("usage_complete") is True and (
        metrics.get("responses") == metrics.get("model_calls")
        if "responses" in metrics else row["status"] != "failed")


def run_local_batch(examples: list[EvaluationExample], answer_question: Callable,
                    *, clock=monotonic, progress: Callable | None = None) -> list[dict]:
    """One fresh input per question; failures do not remove rows from denominators."""
    rows = []
    for index, example in enumerate(examples, 1):
        row = _base_row(example)
        start = clock()
        try:
            state = answer_question(deepcopy(example.inputs))
            if not isinstance(state, dict):
                raise TypeError("Expected agent state dictionary")
            _apply_state(row, state)
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, stop_reason="exception")
        row["elapsed_seconds"] = max(0.0, clock() - start)
        rows.append(row)
        if progress:
            progress(index, len(examples), row)
    return rows


class _SelectedDatasetClient:
    """Limit SDK data without changing common runner or the remote dataset."""
    def __init__(self, client, examples):
        self._client, self._examples = client, examples

    def list_examples(self, **kwargs):
        return iter(self._examples)

    def __getattr__(self, name):
        return getattr(self._client, name)


def select_remote_examples(client, dataset_name: str, examples: list[EvaluationExample]) -> list:
    all_remote = list(client.list_examples(dataset_name=dataset_name))
    by_id = {}
    for remote in all_remote:
        question_id = (_field(remote, "metadata", {}) or {}).get("question_id")
        if question_id in by_id:
            raise ValueError("Remote dataset has duplicate question_id metadata")
        if question_id is not None:
            by_id[question_id] = remote
    selected = []
    for example in examples:
        remote = by_id.get(example.question_id)
        if remote is None or question_inputs(remote.inputs) != example.inputs or remote.outputs != example.outputs:
            raise ValueError("Remote dataset differs from workbook; explicitly use --sync to update it")
        selected.append(remote)
    return selected


def run_remote_batch(examples: list[EvaluationExample], answer_question: Callable,
                     *, client, dataset_name: str, metadata: dict, experiment=None) -> tuple[list[dict], str]:
    remote = select_remote_examples(client, dataset_name, examples)
    states = defaultdict(list)
    def capture(inputs):
        state = answer_question(inputs)
        states[question_inputs(inputs)["messages"][0][1]].append(state)
        if state.get("status") == "error":
            raise RuntimeError("Agent could not produce a final answer")
        return state
    report = run_experiment(capture, dataset_name=dataset_name,
        experiment_prefix=metadata.get('agent_type', 'single_agent') + "_36", client=_SelectedDatasetClient(client, remote),
        metadata=metadata, max_concurrency=1, num_repetitions=1,
        **({"experiment": experiment} if experiment is not None else {}))
    source = {example.question_id: example for example in examples}
    rows = []
    for entry, metrics in zip(report.rows, report.metrics()):
        question_id = (_field(entry["example"], "metadata", {}) or {}).get("question_id")
        row = _base_row(source[question_id])
        captured = states.get(row["question"], [])
        if captured:
            _apply_state(row, captured.pop(0))
        row.update(run_id=metrics["run_id"], elapsed_seconds=metrics["latency_seconds"],
                   grading_status="pending_or_unconfigured")
        if metrics["error"]:
            row["status"] = "failed"
            row["error_type"] = row["error_type"] or "target_error"
        # Do not substitute potentially stale aggregate trace tokens for the
        # explicit per-answer usage ledger, especially on failed runs.
        rows.append(row)
    # If the SDK unexpectedly omitted a target, keep it in the denominator.
    seen = {row["question_id"] for row in rows}
    for example in examples:
        if example.question_id not in seen:
            row = _base_row(example)
            row.update(stop_reason="missing_run", grading_status="not_available")
            rows.append(row)
    order = {example.question_id: i for i, example in enumerate(examples)}
    rows.sort(key=lambda row: order[row["question_id"]])
    return rows, report.experiment_name


def resume_snapshot(client, project_id, dataset_name, examples, architecture):
    """Read only: retain ended roots (including failures), retry unfinished/missing examples."""
    project = client.read_project(project_id=project_id)
    if (project.metadata or {}).get("agent_type") != architecture:
        raise EvaluationSetupError("이어하기 실험의 에이전트 유형이 다릅니다.")
    remote = select_remote_examples(client, dataset_name, examples)
    if any(str(r.dataset_id) != str(project.reference_dataset_id) for r in remote):
        raise EvaluationSetupError("이어하기 실험의 데이터셋이 다릅니다.")
    by_remote = {str(r.id): e for r, e in zip(remote, examples)}
    ended = {}
    unfinished = 0
    for run in client.list_runs(project_id=project.id, is_root=True):
        key = str(run.reference_example_id)
        if key not in by_remote:
            raise EvaluationSetupError("실험에 현재 문항과 다른 기록이 있습니다.")
        if run.end_time is None:
            unfinished += 1
            continue
        if key in ended:
            raise EvaluationSetupError("완료 기록이 중복된 문항이 있어 자동 이어하기를 중단합니다.")
        ended[key] = run
    rows = []
    for key, run in ended.items():
        row = _base_row(by_remote[key])
        from common.evaluation.runner import extract_final_answer
        try:
            answer = extract_final_answer(run.outputs or {})
        except ValueError:
            answer = ""
        row.update(answer=answer, status="failed" if run.error or not answer else "completed",
            stop_reason="restored_remote_run", run_id=str(run.id),
            elapsed_seconds=(run.end_time-run.start_time).total_seconds(),
            input_tokens=run.prompt_tokens, output_tokens=run.completion_tokens,
            total_tokens=run.total_tokens, usage_complete=run.total_tokens is not None,
            grading_status="pending_or_unconfigured", restored_from_langsmith=True)
        rows.append(row)
    remaining = [e for r, e in zip(remote, examples) if str(r.id) not in ended]
    return project, rows, remaining, unfinished


def refresh_feedback(rows: list[dict], client) -> None:
    """One read per existing run; never re-run answers or paid evaluators."""
    for row in rows:
        if not row.get("run_id"):
            continue
        try:
            feedback = list(client.list_feedback(run_ids=[row["run_id"]]))
            for key in SCORE_KEYS:
                row[key] = None
                matching = [f for f in feedback if _field(f, "key") == key]
                matching.sort(key=lambda f: str(_field(f, "created_at", "")), reverse=True)
                if matching:
                    value = _field(matching[0], "score")
                    if numeric(value) and 0 <= value <= 1:
                        row[key] = value
            row["grading_status"] = "completed" if all(row[key] is not None for key in SCORE_KEYS) else (
                "partial" if any(row[key] is not None for key in SCORE_KEYS) else (
                    "unscored_feedback" if any(_field(f, "key") in SCORE_KEYS for f in feedback)
                    else "pending_or_unconfigured"))
        except Exception:
            row["grading_status"] = "feedback_unavailable"


def main(argv=None, *, architecture='single_agent') -> int:
    if architecture not in ARCHITECTURES:
        raise ValueError('Unknown architecture')
    parser = argparse.ArgumentParser(description=ARCHITECTURES[architecture] + ": 기본은 36문항 검증만 수행")
    parser.add_argument("--workbook", type=Path, default=PROJECT_ROOT.parent / "예비실험" / "예비실험_질문36문항_v0.6.xlsx")
    parser.add_argument("--env", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", metavar="PROJECT_ID", help="Read-only preview; add --run to execute only unfinished/missing questions")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Run paid-capable answers and upload LangSmith experiment")
    mode.add_argument("--local-run", action="store_true", help="Run paid-capable answers locally with LangSmith tracing disabled")
    mode.add_argument("--upload-only", action="store_true", help="Upload questions/references only; no answer model calls")
    mode.add_argument("--refresh", type=Path, metavar="REPORT_JSON", help="Update saved report with delayed feedback; no model or judge calls")
    parser.add_argument("--sync", action="store_true", help="With --run only: create/update remote dataset from the full workbook")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args(argv)
    if args.resume and (args.limit or args.sync or args.local_run or args.upload_only or args.refresh):
        parser.error("--resume cannot be combined with limit, sync, local-run, upload-only or refresh")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.sync and not args.run:
        parser.error("--sync requires --run")
    if args.refresh:
        load_settings(args.env)
        from langsmith import Client
        report = json.loads(args.refresh.read_text(encoding="utf-8"))
        rows = report["questions"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Report has no question rows")
        refresh_feedback(rows, Client())
        from datetime import datetime, timezone
        from common.evaluation.comparison import write_comparison
        report["feedback_checked_at"] = datetime.now(timezone.utc).isoformat()
        directory = args.refresh.resolve().parent
        save_report(directory, report)
        comparison = write_comparison(directory.parent)
        print(f"채점 결과를 다시 조회했습니다. 모델 호출 없음. 결과: {directory}\n비교표: {comparison}")
        return 0
    all_examples = load_examples(args.workbook)
    examples = all_examples[:args.limit] if args.limit else all_examples
    print(f"검증 완료: 전체 {len(all_examples)}문항 / 선택 {len(examples)}문항")
    if not args.run and not args.local_run and not args.upload_only and not args.resume:
        print("실행하지 않았습니다. --local-run은 모델 호출, --run은 모델 호출과 LangSmith 실험 기록을 실행합니다.")
        return 0
    settings = load_settings(args.env)
    if args.upload_only:
        from langsmith import Client
        sync_dataset(Client(), settings.dataset_name, all_examples, apply=True)
        print(f'{settings.dataset_name}: {len(all_examples)}문항 동기화 완료. 모델 실행 없음. 평가기 두 개를 연결한 뒤 --run을 실행하세요.')
        return 0
    if args.run and not settings.tracing:
        raise EvaluationSetupError(
            "LangSmith 추적이 꺼져 있습니다. main_project/.env의 LANGSMITH_TRACING=true를 설정하세요. "
            "CMD 환경변수가 우선 적용되므로 기존 창에서는 set LANGSMITH_TRACING=true도 실행하세요. 답변 모델은 실행하지 않았습니다.")
    metadata = {"agent_type": architecture, "model": settings.upstage_model,
                "implementation_sha256": implementation_hash(),
                "temperature": settings.temperature,
                "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
                "experiment_date": settings.experiment_date.isoformat(),
                "dataset_name": settings.dataset_name, "selected_questions": len(examples),
                "max_model_calls": settings.max_model_calls, "max_tool_calls": settings.max_tool_calls,
                "max_elapsed_seconds": settings.max_elapsed_seconds,
                "model_timeout_seconds": settings.model_timeout_seconds,
                "max_context_chars": settings.max_context_chars,
                "mode": "langsmith" if args.run else "local"}
    resume_project, retained = None, []
    if args.resume:
        from langsmith import Client
        client = Client()
        resume_project, retained, examples, unfinished = resume_snapshot(
            client, args.resume, settings.dataset_name, all_examples, architecture)
        for key in ("model", "temperature", "workbook_sha256", "experiment_date", "max_model_calls",
                    "max_tool_calls", "max_elapsed_seconds", "model_timeout_seconds", "max_context_chars"):
            if (resume_project.metadata or {}).get(key) != metadata[key]:
                raise EvaluationSetupError(f"기존 실험과 설정이 다릅니다: {key}")
        print(f"이어하기: {resume_project.name} / 완료 보존 {len(retained)} / 실행 대상 {len(examples)} / 미종료 기록 {unfinished}")
        print("실행 대상: " + ", ".join(e.question_id for e in examples))
        if not args.run:
            print("조회만 완료. 모델 호출 없음. --run을 추가하면 남은 문항만 실행합니다.")
            return 0
    agent = create_agent(architecture, settings) if examples else None
    if args.run:
        from langsmith import Client
        client = Client()
        if args.sync:
            sync_dataset(client, settings.dataset_name, all_examples, apply=True)
        if examples:
            rows, experiment_name = run_remote_batch(examples, agent.answer,
                client=client, dataset_name=settings.dataset_name, metadata=metadata,
                **({"experiment": resume_project} if resume_project is not None else {}))
        else:
            rows, experiment_name = [], resume_project.name
        if resume_project is not None:
            rows = retained + rows
            order = {e.question_id: i for i, e in enumerate(all_examples)}
            rows.sort(key=lambda row: order[row["question_id"]])
            metadata["resumed_project_id"] = str(resume_project.id)
            metadata["preserved_questions"] = len(retained)
            metadata["original_implementation_sha256"] = (resume_project.metadata or {}).get("implementation_sha256")
        metadata["experiment_name"] = experiment_name
        print(f"LangSmith 실험: {experiment_name}. 서버 채점은 별도로 완료되며 즉시 점수가 없을 수 있습니다.")
    else:
        from langsmith import tracing_context
        with tracing_context(enabled=False):
            rows = run_local_batch(examples, agent.answer,
                progress=lambda i, total, row: print(f"[{i}/{total}] {row['question_id']}: {row['status']}"))
    directory = write_report(rows, output_root=args.output_dir, metadata=metadata)
    print(f"결과 저장: {directory}")
    print("completed는 실행 완료를 뜻하며 정답 판정이 아닙니다. 미채점 점수는 빈칸입니다.")
    from common.evaluation.comparison import write_comparison
    print(f"7개 구조 비교표: {write_comparison(args.output_dir)}")
    if args.run:
        print("답변 결과를 저장하고 종료합니다. 서버 채점은 LangSmith에서 진행되며 점수는 자동 조회하지 않습니다.")
    return 1 if any(row["status"] == "failed" for row in rows) else 0


def implementation_hash():
    """비밀 설정을 읽지 않고 실행 코드의 버전을 보고서에 기록한다."""
    digest = hashlib.sha256()
    for directory in ('common', *ARCHITECTURES):
        for path in sorted((PROJECT_ROOT / directory).rglob('*.py')):
            digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def entrypoint(architecture='single_agent'):
    try:
        return main(architecture=architecture)
    except EvaluationSetupError as error:
        print(str(error))
        return 1
    except KeyboardInterrupt:
        print("중단했습니다. 이미 전송된 원격 실행은 계속될 수 있습니다.")
        return 130
    except Exception as error:
        print(f"평가 실행 실패 ({type(error).__name__}). 설정·데이터셋·실행 환경을 확인하세요.")
        return 1


if __name__ == '__main__':
    raise SystemExit(entrypoint())
