"""Opt-in live regression matrix. No LangSmith upload or evaluator calls."""
import argparse
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.callbacks import BaseCallbackHandler
from langsmith import tracing_context
from common.config import PROJECT_ROOT, load_settings
from common.agents import ARCHITECTURES, create_agent
from common.evaluation.dataset import load_examples
from common.evaluation.cli import implementation_hash
from common.model import create_model


class Sizes(BaseCallbackHandler):
    def __init__(self, label=""):
        self.calls, self.lock = [], Lock()
        self.label = label

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self.lock:
            self.calls.append({"role": (kwargs.get("metadata") or {}).get("mas_role"),
                "input_chars": len(json.dumps([[m.model_dump() for m in batch] for batch in messages],
                                              ensure_ascii=False, default=str))})
            print("MODEL", self.label, len(self.calls), self.calls[-1]["role"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--max-elapsed-seconds", type=float)
    parser.add_argument("--questions", nargs="+", required=True)
    parser.add_argument("--architectures", nargs="+", choices=[n for n in ARCHITECTURES if n != "single_agent"],
                        default=[n for n in ARCHITECTURES if n != "single_agent"])
    args = parser.parse_args()
    if not args.run:
        print("실제 Solar 모델/읽기 전용 조회를 실행하려면 --run. OpenAI 채점은 호출하지 않습니다.")
        return 0
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    settings = load_settings()
    if args.max_model_calls is not None:
        settings = replace(settings, max_model_calls=args.max_model_calls)
    if args.max_tool_calls is not None:
        settings = replace(settings, max_tool_calls=args.max_tool_calls)
    if args.max_elapsed_seconds is not None:
        settings = replace(settings, max_elapsed_seconds=args.max_elapsed_seconds)
    examples = {e.question_id: e for e in load_examples(PROJECT_ROOT.parent / "예비실험" / "예비실험_질문36문항_v0.6.xlsx")}
    version = implementation_hash()
    folder = PROJECT_ROOT / "outputs" / ("mas_rework_" + version[:12] + "_" + datetime.now(timezone.utc).strftime("%H%M%S"))
    folder.mkdir()
    print("OUTPUT", folder, flush=True)

    def run(name, qid):
        print("START", name, qid, flush=True)
        sizes = Sizes(name + ":" + qid)
        def model_factory(s, timeout=None):
            model = create_model(s, timeout=timeout)
            model.callbacks = [sizes]
            return model
        with tracing_context(enabled=False):
            result = create_agent(name, settings, model_factory=model_factory).answer(examples[qid].inputs)
        record = {"mode": "development_smoke", "implementation_sha256": version,
                  "limits": {"max_model_calls": settings.max_model_calls,
                             "max_tool_calls": settings.max_tool_calls,
                             "max_elapsed_seconds": settings.max_elapsed_seconds},
                  "architecture": name, "question_id": qid, "question": examples[qid].question,
                  "model_inputs": sizes.calls, "result": result}
        (folder / (name + "_" + qid + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {"architecture": name, "question_id": qid, "status": result["status"],
                "stop_reason": result["stop_reason"], "answer_chars": len(result.get("final_answer", "")),
                "max_input_chars": max((c["input_chars"] for c in sizes.calls), default=0),
                "metrics": result["metrics"],
                "tool_errors": [e for e in result["evidence"] if e.get("result", {}).get("status") == "error"],
                "error_events": [e for e in result["events"] if e["event"] in ("error", "synthesis_stop", "model_retry")]}

    rows = []
    jobs = [(name, qid) for qid in args.questions for name in args.architectures]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, *job) for job in jobs]):
            row = future.result()
            rows.append(row)
            print("DONE", row["architecture"], row["question_id"], row["status"], row["stop_reason"],
                  "chars", row["max_input_chars"], "errors", len(row["error_events"]), flush=True)
            (folder / "summary.json").write_text(json.dumps({"implementation_sha256": version, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINISHED", len(rows), flush=True)
    return int(any(r["status"] == "error" or not r["answer_chars"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
