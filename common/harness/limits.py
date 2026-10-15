"""한 질문의 모든 에이전트가 공유할 호출 예산. 질문마다 새 객체를 만든다."""
from threading import RLock
from time import monotonic


class BudgetExceeded(RuntimeError):
    pass


class RunBudget:
    def __init__(self, max_model_calls=12, max_tool_calls=16, max_elapsed_seconds=180,
                 *, clock=monotonic):
        if max_model_calls < 2 or max_tool_calls < 1 or max_elapsed_seconds <= 0:
            raise ValueError("호출·시간 제한이 잘못되었습니다.")
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.max_elapsed_seconds = max_elapsed_seconds
        self.model_calls = 0
        self.tool_calls = 0
        self._clock = clock
        self._start = clock()
        self._lock = RLock()
        self.finalization_reserve_seconds = min(30.0, max_elapsed_seconds * 0.2)

    @property
    def elapsed_seconds(self):
        return self._clock() - self._start

    @property
    def remaining_seconds(self):
        return max(0.0, self.max_elapsed_seconds - self.elapsed_seconds)

    @property
    def should_finalize(self):
        return (self.model_calls >= self.max_model_calls - 1
                or self.tool_calls >= self.max_tool_calls
                or self.remaining_seconds <= self.finalization_reserve_seconds)

    def reserve_model(self, *, final=False):
        with self._lock:
            self._check_time()
            # None이면 역할별 runtime들이 공유하는 예산 잠금 안에서 종료 모드 결정.
            if final is None:
                final = self.should_finalize
            limit = self.max_model_calls if final else self.max_model_calls - 1
            if self.model_calls >= limit:
                raise BudgetExceeded("모델 호출 한도에 도달했습니다.")
            self.model_calls += 1
            return final

    def reserve_tool(self):
        with self._lock:
            self._check_time()
            if self.tool_calls >= self.max_tool_calls:
                raise BudgetExceeded("도구 호출 한도에 도달했습니다.")
            if self.remaining_seconds <= self.finalization_reserve_seconds:
                raise BudgetExceeded("최종 답변을 위해 남은 시간에 추가 조회를 중단합니다.")
            self.tool_calls += 1

    def _check_time(self):
        if self.remaining_seconds <= 0:
            raise BudgetExceeded("질문 실행 시간 한도에 도달했습니다.")
