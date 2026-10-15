"""Shared MAS kernel tests; scripted models and in-memory tools only."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Lock
import unittest

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from common.config import Settings
from common.orchestration import Context, MASState, OrchestrationError, merge_reports
from common.orchestration.context import validate_plan


def ai(text="answer", calls=None):
    return AIMessage(content=text, tool_calls=calls or [], usage_metadata={
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12})


def task(task_id="t1", depends=None):
    return {"id": task_id, "instruction": "조회해서 답변", "depends_on": depends or [], "domain": "general"}


def evidence(call_id="a"):
    return {"tool": "lookup", "args": {"name": "서울"}, "tool_call_id": call_id,
            "result": {"status": "found", "source": "test.table", "data": {"value": 85}}}


def report(task_id="t1", status="success"):
    return {"task_id": task_id, "instruction": "조회", "answer": "85개", "evidence": [evidence()],
            "status": status, "stop_reason": "worker_answer"}


class ScriptedFactory:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.lock = Lock()

    def __call__(self, settings, timeout):
        parent = self
        class Model:
            tools = None
            response_format = None
            def bind(self, **kwargs):
                self.response_format = kwargs.get("response_format")
                return self
            def bind_tools(self, tools):
                self.tools = [tool.name for tool in tools]
                return self
            def invoke(self, messages, config=None):
                with parent.lock:
                    parent.calls.append({"messages": messages, "config": config, "tools": self.tools,
                                         "response_format": self.response_format})
                    response = parent.responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response
        return Model()


@tool
def lookup(name: str) -> dict:
    """Return test evidence without external access."""
    return {"status": "found", "source": "test.table", "data": {"name": name, "value": 85}}


class PlanTests(unittest.TestCase):
    def test_valid_dag_and_prior_dependency(self):
        payload = {"tasks": [task("t2", ["t1"]), task("t1", ["previous"])]}
        parsed = validate_plan(json.dumps(payload), prior_reports={"previous": report("previous")})
        self.assertEqual(parsed[0]["depends_on"], ["t1"])

    def test_reject_cycle_unknown_duplicates_and_malformed(self):
        invalid = [
            {"tasks": [task("t1", ["t2"]), task("t2", ["t1"])]},
            {"tasks": [task("t1", ["unknown"])]},
            {"tasks": [task("t1", ["t1"])]},
            {"tasks": [task(), task()]},
            {"tasks": [{**task(), "domain": "invented"}]},
            {"tasks": [{**task(), "domain": []}]},
            {"tasks": [task()], "answer": "gold"},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(OrchestrationError):
                validate_plan(json.dumps(payload))
        with self.assertRaises(OrchestrationError):
            validate_plan('Here is the plan {"tasks": []}')
        with self.assertRaises(OrchestrationError):
            validate_plan('{"tasks":[],"tasks":[]}')

    def test_empty_plan_is_explicit_finish_only_after_reports(self):
        with self.assertRaises(OrchestrationError):
            validate_plan('{"tasks":[]}')
        self.assertEqual(validate_plan('{"tasks":[]}', prior_reports={"t1": report()}), [])

    def test_context_plan_has_no_keyword_fallback(self):
        model = ScriptedFactory([ai("not json"), ai("still not json")])
        ctx = Context(Settings(), tools=[], model_factory=model)
        with self.assertRaisesRegex(OrchestrationError, "invalid_plan_json"):
            ctx.plan("정책과 상권 질문", "분해")
        self.assertEqual(ctx.metrics()["model_calls"], 2)
        self.assertEqual(ctx.events_snapshot()[-1]["code"], "invalid_plan_json")

    def test_malformed_and_multiple_json_are_regenerated_and_usage_counted(self):
        valid = json.dumps({"tasks": [task()]})
        for invalid in (valid[:-1], valid + "\n" + valid):
            with self.subTest(invalid=invalid):
                model = ScriptedFactory([ai(invalid), ai(valid)])
                ctx = Context(Settings(), tools=[], model_factory=model)
                self.assertEqual(ctx.plan("조회 질문", "분해"), [task()])
                self.assertEqual(ctx.metrics()["model_calls"], 2)
                self.assertEqual(ctx.metrics()["total_tokens"], 24)
                for call in model.calls:
                    self.assertEqual(call["response_format"]["type"], "json_schema")
                    self.assertTrue(call["response_format"]["json_schema"]["strict"])
                feedback = json.loads(model.calls[1]["messages"][1].content)["validation_feedback"]
                self.assertEqual(feedback["previous_plan"], invalid)
                self.assertEqual(feedback["error"], "invalid_plan_json")

    def test_semantic_error_is_retried_without_repairing_dependencies(self):
        bad = json.dumps({"tasks": [task("t1", ["unknown"])]})
        model = ScriptedFactory([ai(bad), ai(json.dumps({"tasks": [task()]}))])
        ctx = Context(Settings(), tools=[], model_factory=model)
        self.assertEqual(ctx.plan("질문", "분해"), [task()])
        feedback = json.loads(model.calls[1]["messages"][1].content)["validation_feedback"]
        self.assertEqual(feedback["error"], "unknown_or_self_dependency")

    def test_retry_cannot_consume_final_call_reservation(self):
        model = ScriptedFactory([ai("invalid"), ai("최종 답변")])
        ctx = Context(Settings(max_model_calls=2), tools=[], model_factory=model)
        with self.assertRaisesRegex(OrchestrationError, "budget_exceeded"):
            ctx.plan("질문", "분해")
        self.assertEqual(ctx.metrics()["model_calls"], 1)
        self.assertEqual(ctx.synthesize("질문", {"t1": report()})["final_answer"], "최종 답변")
        self.assertIsNone(model.calls[-1]["response_format"])

    def test_provider_failure_does_not_trigger_plan_retry(self):
        model = ScriptedFactory([RuntimeError("private provider error")])
        ctx = Context(Settings(), tools=[], model_factory=model)
        with self.assertRaisesRegex(OrchestrationError, "model_error:RuntimeError"):
            ctx.plan("질문", "분해")
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(ctx.metrics()["usage_complete"])

    def test_independent_plan_dependency_violation_is_regenerated(self):
        bad = {"tasks": [task("t1"), task("t2", ["t1"])]}
        good = {"tasks": [task("t1"), task("t2")]}
        model = ScriptedFactory([ai(json.dumps(bad)), ai(json.dumps(good))])
        ctx = Context(Settings(), tools=[], model_factory=model)
        result = ctx.plan("비교 후 후속 조회", "독립 작업", independent=True)
        self.assertTrue(all(not t["depends_on"] for t in result))
        feedback = json.loads(model.calls[1]["messages"][1].content)["validation_feedback"]
        self.assertEqual(feedback["error"], "unexpected_plan_dependencies")
        schema = model.calls[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["tasks"]["items"]["properties"]["depends_on"]["maxItems"], 0)
        # Independent-plan constraints must not leak into sequential planning.
        sequential = ScriptedFactory([ai(json.dumps(bad))])
        seq_ctx = Context(Settings(), tools=[], model_factory=sequential)
        self.assertEqual(seq_ctx.plan("비교 후 조회", "순차 작업")[1]["depends_on"], ["t1"])
        seq_schema = sequential.calls[0]["response_format"]["json_schema"]["schema"]
        self.assertNotIn("maxItems", seq_schema["properties"]["tasks"]["items"]["properties"]["depends_on"])


class WorkerTests(unittest.TestCase):
    def test_worker_tool_ids_paired_all_tools_available_and_snapshot_isolated(self):
        call = {"id": "call1", "name": "lookup", "args": {"name": "서울"}}
        model = ScriptedFactory([ai("", [call]), ai("85개")])
        ctx = Context(Settings(), tools=[lookup], model_factory=model)
        actual = ctx.work("질문", task())
        self.assertEqual(actual["status"], "success")
        self.assertEqual(actual["answer"], "85개")
        self.assertEqual(model.calls[0]["tools"], ["lookup"])
        replies = [m for m in model.calls[1]["messages"] if m.type == "tool"]
        self.assertEqual([reply.tool_call_id for reply in replies], ["call1"])
        self.assertEqual(ctx.metrics()["total_tokens"], 24)
        snap = ctx.reports_snapshot()
        snap["t1"]["evidence"].clear()
        self.assertEqual(len(ctx.reports_snapshot()["t1"]["evidence"]), 1)

    def test_fixed_turn_handoff_returns_raw_evidence_without_fabricated_answer(self):
        model = ScriptedFactory([ai("", [{"id": "a", "name": "lookup", "args": {"name": "서울"}}])])
        ctx = Context(Settings(), tools=[lookup], model_factory=model)
        actual = ctx.work("질문", task(), max_steps=1)
        self.assertEqual(actual["status"], "success")
        self.assertEqual(actual["answer"], "")
        self.assertEqual(actual["stop_reason"], "evidence_handoff")
        self.assertEqual(actual["evidence"][0]["result"]["data"]["value"], 85)

    def test_global_tool_denial_is_not_successful_handoff(self):
        calls = [{"id": "a", "name": "lookup", "args": {"name": "서울"}},
                 {"id": "b", "name": "lookup", "args": {"name": "부산"}}]
        ctx = Context(Settings(max_tool_calls=1), tools=[lookup], model_factory=ScriptedFactory([ai("", calls)]))
        actual = ctx.work("질문", task(), max_steps=1)
        self.assertEqual(actual["status"], "incomplete")
        self.assertEqual(actual["evidence"][1]["result"]["error_type"], "budget_exceeded")

    def test_model_budget_denial_after_retrieval_is_partial(self):
        model = ScriptedFactory([ai("", [{"id": "a", "name": "lookup", "args": {"name": "서울"}}])])
        ctx = Context(Settings(max_model_calls=2), tools=[lookup], model_factory=model)
        actual = ctx.work("질문", task(), max_steps=2)
        self.assertEqual(actual["status"], "incomplete")
        self.assertEqual(actual["stop_reason"], "budget_exceeded")
        self.assertEqual(len(actual["evidence"]), 1)

    def test_context_overflow_after_retrieval_keeps_evidence_partial(self):
        @tool
        def large_lookup() -> dict:
            """Large test evidence."""
            return {"status": "found", "source": "test.table", "data": "x" * 20000}
        model = ScriptedFactory([ai("", [{"id": "a", "name": "large_lookup", "args": {}}])])
        ctx = Context(Settings(max_context_chars=10000), tools=[large_lookup], model_factory=model)
        actual = ctx.work("질문", task(), max_steps=2)
        self.assertEqual(actual["status"], "incomplete")
        self.assertEqual(actual["stop_reason"], "context_limit")
        self.assertEqual(len(actual["evidence"][0]["result"]["data"]), 20000)

    def test_dependency_failure_prevents_model_call(self):
        model = ScriptedFactory([])
        ctx = Context(Settings(), tools=[], model_factory=model)
        actual = ctx.work("질문", task("t2", ["t1"]))
        self.assertEqual(actual["stop_reason"], "dependency_not_ready")
        self.assertEqual(ctx.metrics()["model_calls"], 0)

    def test_invalid_and_raw_tool_protocol_are_not_answers(self):
        responses = [AIMessage(content="bad", invalid_tool_calls=[{"name": "lookup", "args": "bad", "id": "x", "error": "bad"}]),
                     ai("<|tool_call_begin|> lookup")]
        for response in responses:
            with self.subTest(response=response):
                ctx = Context(Settings(), tools=[], model_factory=ScriptedFactory([response]))
                result = ctx.work("질문", task())
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["answer"], "")

    def test_provider_error_is_sanitized_and_usage_unknown(self):
        ctx = Context(Settings(), tools=[], model_factory=ScriptedFactory([RuntimeError("sk-secret-postgresql")]))
        result = ctx.work("질문", task())
        self.assertEqual(result["status"], "error")
        self.assertNotIn("sk-secret", json.dumps(result))
        self.assertNotIn("sk-secret", json.dumps(ctx.events_snapshot()))
        self.assertFalse(ctx.metrics()["usage_complete"])
        self.assertEqual(ctx.metrics()["unobserved_model_calls"], 1)

    def test_prior_reports_only_known_fields_reach_worker(self):
        prior = report("previous")
        prior["reference_evidence"] = "gold must stay out"
        prior["required_items"] = ["hidden rubric"]
        model = ScriptedFactory([ai("근거 기반 답변")])
        ctx = Context(Settings(), tools=[], model_factory=model)
        ctx.work("질문", task("t2", ["previous"]), {"previous": prior})
        body = model.calls[0]["messages"][1].content
        self.assertNotIn("gold must stay out", body)
        self.assertNotIn("hidden rubric", body)
        self.assertIn("test.table", body)


class SharedBudgetTests(unittest.TestCase):
    def test_timeout_retries_once_counts_attempts_and_keeps_usage_unknown(self):
        Timeout = type("OpenAITimeoutError", (Exception,), {})
        model = ScriptedFactory([Timeout("private details"), ai("복구된 답변")])
        ctx = Context(Settings(), tools=[], model_factory=model)
        result = ctx.synthesize("질문", {"t1": report()})
        self.assertEqual(result["final_answer"], "복구된 답변")
        self.assertEqual(ctx.metrics()["model_calls"], 2)
        self.assertEqual(ctx.metrics()["unobserved_model_calls"], 1)
        self.assertFalse(ctx.metrics()["usage_complete"])
        self.assertTrue(any(e["event"] == "model_retry" for e in ctx.events_snapshot()))
        self.assertEqual(model.calls[0]["messages"], model.calls[1]["messages"])
        self.assertNotIn("private details", str(ctx.events_snapshot()))

    def test_repeated_timeout_is_bounded_and_does_not_replay_tools(self):
        Timeout = type("OpenAITimeoutError", (Exception,), {})
        model = ScriptedFactory([Timeout(), Timeout()])
        ctx = Context(Settings(), tools=[], model_factory=model)
        result = ctx.synthesize("질문", {"t1": report()})
        self.assertEqual(result["status"], "error")
        self.assertEqual(ctx.metrics()["model_calls"], 2)
        self.assertEqual(ctx.metrics()["tool_calls"], 0)

    def test_timeout_retry_does_not_take_reserved_final_slot(self):
        Timeout = type("OpenAITimeoutError", (Exception,), {})
        model = ScriptedFactory([Timeout(), ai("최종")])
        ctx = Context(Settings(max_model_calls=2), tools=[], model_factory=model)
        with self.assertRaisesRegex(OrchestrationError, "model_error"):
            ctx.text("질문", "생각")
        self.assertEqual(ctx.metrics()["model_calls"], 1)
        self.assertEqual(ctx.synthesize("질문", {"t1": report()})["final_answer"], "최종")

    def test_last_call_reserved_only_for_one_synthesis(self):
        model = ScriptedFactory([ai("one"), ai("two"), ai("final")])
        ctx = Context(Settings(max_model_calls=3), tools=[], model_factory=model)
        ctx.text("질문", "생각")
        ctx.text("질문", "검토")
        with self.assertRaisesRegex(OrchestrationError, "budget_exceeded"):
            ctx.text("질문", "추가")
        with self.assertRaisesRegex(OrchestrationError, "reserved"):
            ctx.text("질문", "추가", final=True)
        final = ctx.synthesize("질문", {"t1": report()})
        self.assertEqual(final["final_answer"], "final")
        self.assertEqual(final["status"], "success")
        second = ctx.synthesize("질문", {"t1": report()})
        self.assertEqual(second["stop_reason"], "synthesis_already_attempted")
        self.assertEqual(ctx.metrics()["model_calls"], 3)
        self.assertTrue(all(call["tools"] is None for call in model.calls))

    def test_concurrent_workers_share_global_budget(self):
        model = ScriptedFactory([ai("answer") for _ in range(3)])
        ctx = Context(Settings(max_model_calls=3), tools=[], model_factory=model)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: ctx.work("질문", task(f"t{i}"), max_steps=1), range(4)))
        self.assertEqual(ctx.metrics()["model_calls"], 2)
        self.assertEqual(sum(r["stop_reason"] == "budget_exceeded" for r in results), 2)
        ctx.synthesize("질문", ctx.reports_snapshot())
        self.assertEqual(ctx.metrics()["model_calls"], 3)
        self.assertEqual(sum(role["model_calls"] for role in ctx.metrics()["roles"]), 3)

    def test_context_overflow_fails_without_silent_truncation(self):
        model = ScriptedFactory([])
        ctx = Context(Settings(max_context_chars=20), tools=[], model_factory=model)
        result = ctx.synthesize("긴 질문" * 100, {"t1": report()})
        self.assertEqual(result["stop_reason"], "context_limit")
        self.assertEqual(ctx.metrics()["model_calls"], 0)


class SynthesisTests(unittest.TestCase):
    def test_full_evidence_deduplicated_and_failed_work_not_success(self):
        first, second = report(), report("t2", status="error")
        second["evidence"][0]["tool_call_id"] = "different-id"
        model = ScriptedFactory([ai("부분 답변")])
        ctx = Context(Settings(), tools=[], model_factory=model)
        result = ctx.synthesize("질문", {"t1": first, "t2": second})
        body = json.loads(model.calls[0]["messages"][1].content)
        self.assertEqual(len(body["original_evidence"]), 1)
        self.assertEqual(body["original_evidence"][0]["result"]["data"], {"value": 85})
        self.assertEqual(body["reports"]["t2"]["original_evidence_indices"], [0])
        self.assertEqual(result["status"], "incomplete")

    def test_replaced_reports_do_not_erase_original_tool_evidence(self):
        model = ScriptedFactory([ai("", [{"id": "a", "name": "lookup", "args": {"name": "서울"}}]), ai("final")])
        ctx = Context(Settings(), tools=[lookup], model_factory=model)
        ctx.work("질문", task(), max_steps=1)
        replacement = report()
        replacement["evidence"] = []
        ctx.synthesize("질문", {"t1": replacement})
        body = json.loads(model.calls[-1]["messages"][1].content)
        self.assertEqual(body["original_evidence"][0]["result"]["data"]["name"], "서울")

    def test_reducer_keeps_parallel_peers_and_does_not_mutate(self):
        left, right = {"a": {"answer": "old"}}, {"b": {"answer": "new"}}
        merged = merge_reports(left, right)
        merged["a"]["answer"] = "changed"
        self.assertEqual(left["a"]["answer"], "old")
        self.assertEqual(set(merged), {"a", "b"})

    def test_model_config_keeps_graph_callback_lineage(self):
        model = ScriptedFactory([ai("draft")])
        ctx = Context(Settings(), tools=[], model_factory=model)
        builder = StateGraph(MASState)
        builder.add_node("draft", lambda state: {"final_answer": ctx.text(state["question"], "draft")})
        builder.add_edge(START, "draft")
        builder.add_edge("draft", END)
        builder.compile().invoke({"question": "질문"})
        config = model.calls[0]["config"]
        self.assertIsNotNone(config.get("callbacks"))
        self.assertEqual(config["run_name"], "mas_discussion")
        self.assertEqual(config["metadata"]["langgraph_node"], "draft")


if __name__ == "__main__":
    unittest.main()
