"""반환된 AI 메시지의 사용량만 합산. 부모/자식 trace를 중복 합산하지 않는다."""
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class UsageLedger:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    responses: int = 0
    missing_usage_responses: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, response):
        usage = getattr(response, "usage_metadata", None)
        with self._lock:
            self.responses += 1
            if not usage or any(usage.get(k) is None for k in
                                ("input_tokens", "output_tokens", "total_tokens")):
                self.missing_usage_responses += 1
                return
            self.input_tokens += usage["input_tokens"]
            self.output_tokens += usage["output_tokens"]
            self.total_tokens += usage["total_tokens"]

    def snapshot(self):
        with self._lock:
            return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                    "total_tokens": self.total_tokens, "responses": self.responses,
                    "usage_complete": self.missing_usage_responses == 0,
                    "missing_usage_responses": self.missing_usage_responses}
