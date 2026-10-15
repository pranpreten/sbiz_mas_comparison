"""실험 전체의 공통 설정. 자격증명은 repr/log에 노출하지 않는다."""
import os
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default="", repr=False)
    upstage_api_key: str = field(default="", repr=False)
    upstage_model: str = "solar-pro4"
    temperature: float = 0.0
    model_timeout_seconds: float = 30.0
    max_tool_calls: int = 24
    max_model_calls: int = 20
    max_elapsed_seconds: float = 300.0
    statement_timeout_ms: int = 10000
    max_context_chars: int = 100000
    experiment_date: date = date(2026, 10, 13)
    dataset_name: str = "sbiz_eval_36"
    langsmith_project: str = "sbiz_main_project"
    tracing: bool = False

    def __post_init__(self):
        for name in ("model_timeout_seconds", "max_tool_calls", "max_model_calls",
                     "max_elapsed_seconds", "statement_timeout_ms", "max_context_chars"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}은 양수여야 합니다.")
        if not math.isfinite(self.temperature):
            raise ValueError("temperature는 유한한 숫자여야 합니다.")
        if self.max_model_calls < 2:
            raise ValueError("최종 답변을 위한 호출을 포함해 max_model_calls >= 2가 필요합니다.")


def load_settings(env_path: str | Path | None = None) -> Settings:
    """기본은 main_project/.env. 다른 프로젝트 설정은 자동으로 가져오지 않는다."""
    from dotenv import load_dotenv

    path = Path(env_path) if env_path is not None else PROJECT_ROOT / ".env"
    if env_path is not None and not path.is_file():
        raise ValueError("지정한 환경 설정 파일이 없습니다.")
    load_dotenv(path, override=False, encoding="utf-8-sig")
    raw_tracing = os.getenv("LANGSMITH_TRACING", "false").lower().strip()
    if raw_tracing not in {"true", "false"}:
        raise ValueError("LANGSMITH_TRACING은 true 또는 false여야 합니다.")
    tracing = raw_tracing == "true"
    if tracing and not os.getenv("LANGSMITH_API_KEY", "").strip():
        raise ValueError("추적을 켜려면 LANGSMITH_API_KEY가 필요합니다.")
    # 기존 LANGCHAIN_TRACING_V2가 외부에서 켜져 있어도 명시한 설정을 따른다.
    os.environ["LANGSMITH_TRACING"] = raw_tracing
    os.environ["LANGCHAIN_TRACING_V2"] = raw_tracing
    os.environ.setdefault("LANGSMITH_PROJECT", "sbiz_main_project")
    return Settings(
        database_url=os.getenv("DATABASE_URL", "").strip(),
        upstage_api_key=os.getenv("UPSTAGE_API_KEY", "").strip(),
        upstage_model=os.getenv("UPSTAGE_MODEL", "solar-pro4").strip(),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0")),
        model_timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "30")),
        max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "24")),
        max_model_calls=int(os.getenv("MAX_MODEL_CALLS", "20")),
        max_elapsed_seconds=float(os.getenv("MAX_ELAPSED_SECONDS", "300")),
        statement_timeout_ms=int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "10000")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "100000")),
        experiment_date=date.fromisoformat(os.getenv("EXPERIMENT_DATE", "2026-10-13")),
        dataset_name=os.getenv("EVAL_DATASET_NAME", "sbiz_eval_36"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "sbiz_main_project"),
        tracing=tracing,
    )
