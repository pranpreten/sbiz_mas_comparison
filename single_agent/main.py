"""질문 하나 실행. python -m single_agent.main 또는 python single_agent/main.py."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from common.config import load_settings
from single_agent.service import SingleAgent


def main():
    parser = argparse.ArgumentParser(description="단일 에이전트 질문 하나 실행 (모델 사용 비용 발생)")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--env", type=Path)
    parser.add_argument("--output", type=Path, help="답변·근거·사용량 JSON 저장 경로")
    args = parser.parse_args()
    question = args.question or input("질문: ").strip()
    if not question:
        parser.error("질문을 입력해 주세요.")
    result = SingleAgent(load_settings(args.env)).answer({"messages": [["user", question]]})
    print(result["final_answer"] or "답변을 완료하지 못했습니다.")
    print("상태:", result["status"], "/", result["stop_reason"])
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in result.items() if k != "messages"}
        args.output.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
