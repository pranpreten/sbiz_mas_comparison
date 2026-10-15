import json
import unittest
from types import SimpleNamespace
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from common.config import Settings
from common.harness.context import ContextLimitExceeded, build_messages
from common.harness.limits import BudgetExceeded, RunBudget
from common.harness.runtime import AgentRuntime
from common.harness.tracing import UsageLedger


class HarnessTests(unittest.TestCase):
    def test_budget_reserves_final_and_shares_question_limit(self):
        b = RunBudget(max_model_calls=3, max_tool_calls=2)
        b.reserve_model()
        b.reserve_model()
        self.assertTrue(b.should_finalize)
        with self.assertRaises(BudgetExceeded):
            b.reserve_model()
        b.reserve_model(final=True)
        with self.assertRaises(BudgetExceeded):
            b.reserve_model(final=True)

    def test_deadline_prevents_new_calls(self):
        now = [10.0]
        b = RunBudget(max_elapsed_seconds=5, clock=lambda: now[0])
        now[0] = 15
        with self.assertRaises(BudgetExceeded):
            b.reserve_tool()
        self.assertEqual(b.tool_calls, 0)

    def test_duplicate_lookup_is_cached_but_attempt_is_counted(self):
        invoked = []

        @tool
        def lookup(value: int) -> dict:
            """테스트용 읽기 전용 조회."""
            invoked.append(value)
            return {"status": "found", "data": value}

        runtime = AgentRuntime(Settings(max_tool_calls=2), [lookup])
        responses = runtime.execute_tools([
            {"id": "a", "name": "lookup", "args": {"value": 3}},
            {"id": "b", "name": "lookup", "args": {"value": 3}},
            {"id": "c", "name": "lookup", "args": {"value": 4}},
        ])
        self.assertEqual(invoked, [3])
        self.assertEqual([r.tool_call_id for r in responses], ["a", "b", "c"])
        self.assertTrue(json.loads(responses[1].content)["cached"])
        self.assertEqual(json.loads(responses[2].content)["error_type"], "budget_exceeded")
        self.assertEqual(runtime.budget.tool_calls, 2)

    def test_errors_do_not_leak_exception_credentials(self):
        from langchain_core.callbacks import BaseCallbackHandler
        callback_errors = []

        class Handler(BaseCallbackHandler):
            def on_tool_error(self, error, **kwargs):
                callback_errors.append(str(error))

        @tool
        def fail() -> dict:
            """오류 확인."""
            raise RuntimeError("postgresql://secret-password sk-secret")

        runtime = AgentRuntime(Settings(), [fail])
        result = runtime.execute_tools([{"id": "x", "name": "fail", "args": {}}],
                                       config={"callbacks": [Handler()]})[0]
        self.assertNotIn("secret", result.content)
        self.assertEqual(json.loads(result.content)["status"], "error")
        self.assertEqual(callback_errors, [])

    def test_final_call_has_no_bound_tools(self):
        bound = []

        class Model:
            def bind_tools(self, tools):
                bound.append(True)
                return self

            def invoke(self, messages, config=None):
                return AIMessage(content="답변", usage_metadata={"input_tokens": 2,
                                 "output_tokens": 1, "total_tokens": 3})

        runtime = AgentRuntime(Settings(max_model_calls=2), [],
                               model_factory=lambda settings, timeout: Model())
        for _ in range(2):
            runtime.invoke_model("역할", [HumanMessage(content="질문")])
        self.assertEqual(len(bound), 1)
        self.assertEqual(runtime.metrics()["total_tokens"], 6)

    def test_missing_usage_does_not_look_complete(self):
        ledger = UsageLedger()
        ledger.record(SimpleNamespace(usage_metadata=None))
        self.assertFalse(ledger.snapshot()["usage_complete"])

    def test_context_does_not_silently_drop_history(self):
        with self.assertRaises(ContextLimitExceeded):
            build_messages("역할", [HumanMessage(content="긴 질문" * 100)],
                           experiment_date="2026-10-13", max_chars=100)

    def test_settings_and_database_repr_hide_secrets(self):
        from common.data.database import Database
        self.assertNotIn("secret", repr(Settings(database_url="secret", upstage_api_key="secret")))
        self.assertNotIn("secret", repr(Database("postgresql://secret")))


if __name__ == "__main__":
    unittest.main()
