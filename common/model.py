"""모델 설정만 공통화한다. 역할·프롬프트·그래프는 각 구조가 결정한다."""
from common.config import Settings


def create_model(settings: Settings, *, timeout: float | None = None):
    from langchain_upstage import ChatUpstage

    if not settings.upstage_api_key or not settings.upstage_model:
        raise ValueError("UPSTAGE_API_KEY와 UPSTAGE_MODEL을 설정해 주세요.")
    return ChatUpstage(
        model=settings.upstage_model,
        upstage_api_key=settings.upstage_api_key,
        temperature=settings.temperature,
        timeout=min(timeout, settings.model_timeout_seconds) if timeout is not None
        else settings.model_timeout_seconds,
        # 숨은 재시도가 비교 실험의 호출 수를 달리하지 않게 한다.
        max_retries=0,
    )
