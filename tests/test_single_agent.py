"""Network-free behavioral tests for the single-agent LangGraph service."""
import json
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from common.config import Settings
from common.evaluation.runner import extract_final_answer
from common.harness.limits import RunBudget
from single_agent.service import SingleAgent


class ScriptedModels:
    """One scripted response per invocation, including structured tool calls."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.timeouts = []

    def __call__(self, settings, timeout=None):
        self.timeouts.append(timeout)
        owner = self

        class Model:
            bound = False

            def bind_tools(self, tools):
                self.bound = True
                return self

            def invoke(self, messages, config=None):
                owner.calls.append({"messages": list(messages), "bound": self.bound})
                response = owner.responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

        return Model()


def answer(text):
    return AIMessage(content=text, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def calls(*items):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}
                                            for name, args, call_id in items])


class SingleAgentTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.settings = Settings()
        self.executed = []
        executed = self.executed

        @tool
        def find_fact(topic: str) -> dict:
            """Read a synthetic, deterministic fact without a database."""
            executed.append(topic)
            return {"status": "found", "data": {"topic": topic, "value": 85},
                    "source": "synthetic", "conditions": {"topic": topic}}

        self.tools = [find_fact]

    def make(self, responses, settings=None):
        model = ScriptedModels(responses)
        service = SingleAgent(settings or self.settings, tools=self.tools, model_factory=model)
        return service, model

    def test_final_answer_contract_and_usage(self):
        service, model = self.make([calls(("find_fact", {"topic": "사실"}, "a")), answer("정답")])
        result = service.answer({"messages": [["user", "질문"]]})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["final_answer"], "정답")
        self.assertEqual(result["metrics"]["model_calls"], 2)
        self.assertEqual(result["metrics"]["tool_calls"], 1)
        self.assertEqual(extract_final_answer(result), "정답")

    def test_batch_tool_ids_pair_before_next_model_and_only_final_is_output(self):
        service, model = self.make([
            calls(("find_fact", {"topic": "서울"}, "call-a"), ("find_fact", {"topic": "부산"}, "call-b")),
            answer("두 지역을 확인했습니다."),
        ])
        result = service.answer({"messages": [["user", "서울 부산 비교"]]})
        tool_messages = [m for m in model.calls[1]["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.tool_call_id for m in tool_messages], ["call-a", "call-b"])
        self.assertEqual(self.executed, ["서울", "부산"])
        self.assertEqual(result["metrics"]["tool_calls"], 2)
        self.assertEqual(extract_final_answer(result), "두 지역을 확인했습니다.")
        self.assertNotIn('"value": 85', result["final_answer"])
        self.assertTrue(result["evidence"])

    def test_each_question_has_fresh_history_budget_and_cache(self):
        service, model = self.make([
            calls(("find_fact", {"topic": "같은조회"}, "a")), answer("첫 답"),
            calls(("find_fact", {"topic": "같은조회"}, "b")), answer("둘째 답"),
        ])
        first = service.answer({"messages": [["user", "첫질문표식"]]})
        second = service.answer({"messages": [["user", "둘째질문표식"]]})
        self.assertEqual(self.executed, ["같은조회", "같은조회"])
        self.assertEqual(first["metrics"]["tool_calls"], 1)
        self.assertEqual(second["metrics"]["tool_calls"], 1)
        self.assertNotIn("첫질문표식", str(model.calls[2]["messages"]))
        self.assertNotIn("첫 답", str(model.calls[2]["messages"]))

    def test_gold_fields_are_never_model_context(self):
        service, model = self.make([answer("답변")])
        service.answer({"messages": [["user", "질문"]],
                        "required_items": ["GOLD_REQUIRED_SECRET"],
                        "reference_evidence": "GOLD_EVIDENCE_SECRET",
                        "reference_outputs": {"answer": "GOLD_NESTED_SECRET"}})
        payload = str(model.calls[0]["messages"])
        self.assertNotIn("GOLD_REQUIRED_SECRET", payload)
        self.assertNotIn("GOLD_EVIDENCE_SECRET", payload)
        self.assertNotIn("GOLD_NESTED_SECRET", payload)

    def test_empty_ai_is_not_success(self):
        service, model = self.make([answer("")])
        result = service.answer({"messages": [["user", "질문"]]})
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["final_answer"])
        self.assertTrue(result["stop_reason"])

    def test_unparseable_tool_arguments_are_not_success(self):
        malformed = AIMessage(content="", invalid_tool_calls=[
            {"name": "find_fact", "args": "{broken", "id": "broken-id", "error": "Invalid JSON", "type": "invalid_tool_call"}])
        service, model = self.make([malformed])
        result = service.answer({"messages": [["user", "질문"]]})
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["final_answer"])
        self.assertEqual(self.executed, [])

    def test_evidence_free_answer_is_incomplete(self):
        service, model = self.make([answer("확인 근거 없이 말한 답")])
        result = service.answer({"messages": [["user", "질문"]]})
        self.assertEqual(result["status"], "incomplete")

    def test_raw_tool_protocol_during_finalization_is_not_graded_as_answer(self):
        service, model = self.make([
            calls(("find_fact", {"topic": "사실"}, "a")),
            answer('<|tool_call:start|>search_law_articles {"keywords":["소기업"]}<|tool_call:end|>'),
        ], replace(self.settings, max_model_calls=2))
        result = service.answer({"messages": [["user", "법적 기준"]]})
        self.assertFalse(model.calls[1]["bound"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["final_answer"], "")
        self.assertEqual(self.executed, ["사실"])
        with self.assertRaises(ValueError):
            extract_final_answer(result)

    def test_provider_exception_is_sanitized(self):
        service, model = self.make([RuntimeError("sk-secret postgres://password")])
        result = service.answer({"messages": [["user", "질문"]]})
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["final_answer"])
        self.assertNotIn("sk-secret", json.dumps(result, default=str))
        self.assertNotIn("postgres://password", json.dumps(result, default=str))

    def test_invalid_tool_call_is_not_success_and_id_is_preserved(self):
        service, model = self.make([
            calls(("find_fact", {"unknown_field": "bad"}, "bad-id")),
            answer("조회에 실패해 확인하지 못했습니다."),
        ])
        result = service.answer({"messages": [["user", "질문"]]})
        self.assertNotEqual(result["status"], "success")
        tool_messages = [m for m in model.calls[1]["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(tool_messages[0].tool_call_id, "bad-id")
        self.assertEqual(json.loads(tool_messages[0].content)["status"], "error")
        self.assertEqual(self.executed, [])

    def test_model_limit_preserves_final_call_without_tools(self):
        service, model = self.make([
            calls(("find_fact", {"topic": "조사"}, "a")), answer("확인한 범위만 답합니다."),
        ], replace(self.settings, max_model_calls=2))
        result = service.answer({"messages": [["user", "질문"]]})
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(model.calls[0]["bound"])
        self.assertFalse(model.calls[1]["bound"])
        self.assertNotEqual(result["status"], "success")

    def test_deadline_reserve_forces_final_without_waiting(self):
        # Deterministic 81/100 seconds elapsed; 20 seconds are reserved for final.
        clock = [0.0]
        budget = RunBudget(max_elapsed_seconds=100, clock=lambda: clock[0])
        clock[0] = 81.0
        service, model = self.make([answer("시간 한도로 확인한 범위만 답합니다.")])
        with patch("common.harness.runtime.RunBudget", return_value=budget):
            result = service.answer({"messages": [["user", "질문"]]})
        self.assertFalse(model.calls[0]["bound"])
        self.assertLessEqual(model.timeouts[0], 19.0)
        self.assertNotEqual(result["status"], "success")

    def test_invalid_question_is_rejected_before_model(self):
        service, model = self.make([])
        for value in ({}, {"messages": []}, {"messages": [["user", ""]]}):
            with self.assertRaises(ValueError):
                service.answer(value)
        self.assertEqual(model.calls, [])

    def test_expired_deadline_prevents_model_call(self):
        clock = [0.0]
        budget = RunBudget(max_elapsed_seconds=100, clock=lambda: clock[0])
        clock[0] = 101.0
        service, model = self.make([])
        with patch("common.harness.runtime.RunBudget", return_value=budget):
            result = service.answer({"messages": [["user", "질문"]]})
        self.assertNotEqual(result["status"], "success")
        self.assertFalse(result["final_answer"])
        self.assertEqual(model.calls, [])

    def test_deadline_crossing_between_routing_and_reservation_marks_incomplete(self):
        # A prior successful lookup makes this fail if the graph uses a stale
        # pre-reservation final-mode flag: the answer would be marked success.
        clock = [0.0]

        class BoundaryBudget(RunBudget):
            def reserve_model(self, *, final=False):
                if self.model_calls == 1:
                    clock[0] = 81.0
                return super().reserve_model(final=final)

        budget = BoundaryBudget(max_elapsed_seconds=100, clock=lambda: clock[0])
        service, model = self.make([
            calls(("find_fact", {"topic": "근거"}, "a")),
            answer("시간 한도로 확인한 근거만 답합니다."),
        ])
        with patch("common.harness.runtime.RunBudget", return_value=budget):
            result = service.answer({"messages": [["user", "질문"]]})
        self.assertEqual(self.executed, ["근거"])
        self.assertTrue(model.calls[0]["bound"])
        self.assertFalse(model.calls[1]["bound"])
        self.assertEqual(result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
