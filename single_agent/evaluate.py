"""기존 단일 에이전트 평가 명령의 호환 진입점."""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluation.cli import main, entrypoint, refresh_feedback, run_local_batch, run_remote_batch, select_remote_examples
if __name__ == '__main__':
    raise SystemExit(entrypoint('single_agent'))
