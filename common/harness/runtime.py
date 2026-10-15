"""각 구조의 LangGraph 노드에서 호출하는 공통 모델/도구 실행기."""
import json
from langchain_core.messages import ToolMessage
from common.model import create_model
from .context import build_messages
from .limits import BudgetExceeded, RunBudget
from .tracing import UsageLedger
from .safe_tools import safe_tool


class AgentRuntime:
    def __init__(self, settings, tools, *, budget=None, usage=None, model_factory=create_model):
        self.settings = settings
        self.tools = {tool.name: safe_tool(tool) for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("도구 이름이 중복됩니다.")
        self.budget = budget or RunBudget(settings.max_model_calls, settings.max_tool_calls,
                                         settings.max_elapsed_seconds)
        self.usage = usage or UsageLedger()
        self._model_factory = model_factory
        self._cache = {}
        self.last_model_final = False

    def invoke_model(self, role_prompt, messages, *, final=False, config=None):
        # 하나의 runtime은 한 노드 흐름에서 순차 사용. MAS는 예산/사용량만 공유한다.
        final = self.budget.reserve_model(final=True if final else None)
        self.last_model_final = final
        prompt = build_messages(role_prompt, messages,
                                experiment_date=self.settings.experiment_date,
                                max_chars=self.settings.max_context_chars, final=final)
        model = self._model_factory(self.settings, timeout=self.budget.remaining_seconds)
        if not final:
            model = model.bind_tools(list(self.tools.values()))
        response = model.invoke(prompt, config=config)
        self.usage.record(response)
        return response

    def execute_tools(self, calls, *, config=None):
        """거부·실패한 호출에도 ToolMessage를 반환하여 호출 ID 대응을 유지한다."""
        messages = []
        for call in calls:
            try:
                self.budget.reserve_tool()  # 잘못된 요청도 시도 횟수에 포함한다.
                name, args = call["name"], call["args"]
                if name not in self.tools:
                    result = {"status": "error", "message": "지원하지 않는 도구입니다."}
                else:
                    key = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
                    if key in self._cache:
                        result = {**self._cache[key], "cached": True}
                    else:
                        result = self.tools[name].invoke(args, config=config)
                        if not isinstance(result, dict):
                            result = {"status": "error", "message": "도구 반환 형식이 잘못되었습니다."}
                        if result.get("status") in {"found", "not_found", "ambiguous"}:
                            self._cache[key] = result
            except BudgetExceeded as exc:
                result = {"status": "error", "error_type": "budget_exceeded", "message": str(exc)}
            except Exception as exc:
                # 원문에 접속 정보나 키가 있을 수 있어 오류 종류만 기록한다.
                result = {"status": "error", "error_type": type(exc).__name__,
                          "message": "도구 입력 또는 조회에 실패했습니다. 조건을 확인해 주세요."}
            messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False, default=str),
                                        tool_call_id=call["id"], name=call.get("name", "unknown")))
        return messages

    def metrics(self):
        return {**self.usage.snapshot(), "elapsed_seconds": self.budget.elapsed_seconds,
                "model_calls": self.budget.model_calls, "tool_calls": self.budget.tool_calls}
