"""Real shared Context + real LangGraph, scripted local model/tool integration."""
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import json
from threading import Lock
import unittest
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from common.config import Settings
from common.evaluation.runner import make_target
from common.harness.limits import RunBudget
from common.orchestration import Context


@tool
def fixture_lookup(topic: str) -> dict:
    """Read a local fixture, with a real provenance-shaped result."""
    return {"status": "found", "data": {"topic": topic, "value": 7},
            "source": "fixture.source", "conditions": {"period": "2024"}}


class ToolCallback(BaseCallbackHandler):
    def __init__(self):
        self.parents = []

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self.parents.append(parent_run_id)


class ScriptedFactory:
    """Protocol-valid model double; routes by assigned role, never calls a provider."""
    def __init__(self, fail_tool=False, invalid_final=False):
        self.calls = []
        self.lock = Lock()
        self.fail_tool = fail_tool
        self.invalid_final = invalid_final

    def __call__(self, settings, *, timeout=None):
        owner = self

        class Model:
            def bind(self, **kwargs):
                return self

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, config=None):
                role = (config or {}).get("run_name", "")
                with owner.lock:
                    call_no = len(owner.calls) + 1
                    owner.calls.append({"role": role, "messages": list(messages),
                                        "config": config, "timeout": timeout})
                body = json.loads(next(m.content for m in messages if m.type == "human"))
                system = messages[0].content
                usage = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
                if role == "mas_planner":
                    if "동적 조정자" in system and body.get("reports"):
                        tasks = []
                    elif "하위 조정자" in system:
                        tasks = [{"id": "sub", "instruction": "retrieve assigned scope",
                                  "depends_on": [], "domain": "general"}]
                    else:
                        tasks = [{"id": "data", "instruction": "retrieve statistics",
                                  "depends_on": [], "domain": "statistics"},
                                 {"id": "doc", "instruction": "retrieve policy",
                                  "depends_on": [], "domain": "policy"}]
                    return AIMessage(content=json.dumps({"tasks": tasks}), usage_metadata=usage)
                if role.startswith("mas_worker:"):
                    if any(isinstance(m, ToolMessage) for m in messages) or body.get("prior_reports"):
                        return AIMessage(content="확인한 원본 자료의 값은 7입니다.", usage_metadata=usage)
                    tool_name = "unknown_tool" if owner.fail_tool else "fixture_lookup"
                    return AIMessage(content="", tool_calls=[{"id": f"call{call_no}", "name": tool_name,
                                                              "args": {"topic": "test"}}],
                                     usage_metadata=usage)
                if role == "mas_synthesizer" and owner.invalid_final:
                    return AIMessage(content="<|tool_call|>unfinished", usage_metadata=usage)
                return AIMessage(content="근거를 종합한 실제 최종 답변입니다.", usage_metadata=usage)

        return Model()


def initial():
    return {"question": "compare and explain", "reports": {}, "tasks": [],
            "final_answer": "", "status": "running", "stop_reason": ""}


class RealContextIntegrationTests(unittest.TestCase):
    def setUp(self):
        tracing = patch.dict("os.environ", {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})
        tracing.start()
        self.addCleanup(tracing.stop)

    def test_all_six_graphs_execute_real_context_with_one_final_and_trace_parent(self):
        for architecture in ("mas_central_parallel", "mas_sequential", "mas_review",
                             "mas_decentralized", "mas_hierarchical", "mas_dynamic"):
            with self.subTest(architecture=architecture):
                factory, callback = ScriptedFactory(), ToolCallback()
                ctx = Context(Settings(), [fixture_lookup], model_factory=factory)
                # This test verifies topology/call accounting, not elapsed time.
                # Keep host suspend/resume from consuming the question budget.
                ctx.budget = RunBudget(12, 16, 180, clock=lambda: 0.0)
                graph = import_module(architecture + ".graph").build_graph(ctx)
                result = graph.invoke(initial(), config={"recursion_limit": 80, "callbacks": [callback]})
                self.assertTrue(result["final_answer"])
                self.assertEqual(result["status"], "success")
                self.assertEqual(sum(c["role"] == "mas_synthesizer" for c in factory.calls), 1)
                metrics = ctx.metrics()
                self.assertLessEqual(metrics["model_calls"], 12)
                self.assertLessEqual(metrics["tool_calls"], 16)
                self.assertEqual(metrics["total_tokens"], metrics["responses"] * 3)
                self.assertTrue(callback.parents)
                self.assertTrue(all(parent is not None for parent in callback.parents))
                for call in factory.calls:
                    self.assertIn("langgraph_node", call["config"]["metadata"])
                    self.assertIsNotNone(call["config"].get("callbacks"))

    def test_parallel_workers_cannot_consume_last_model_slot_or_exceed_tool_budget(self):
        factory = ScriptedFactory()
        ctx = Context(Settings(max_model_calls=5, max_tool_calls=2), [fixture_lookup], model_factory=factory)
        ctx.budget = RunBudget(5, 2, 180, clock=lambda: 0.0)
        def work(i):
            return ctx.work("question", {"id": f"task{i}", "instruction": "lookup", "depends_on": []}, max_steps=2)
        with ThreadPoolExecutor(max_workers=12) as pool:
            reports = list(pool.map(work, range(20)))
        self.assertEqual(ctx.metrics()["model_calls"], 4)
        self.assertLessEqual(ctx.metrics()["tool_calls"], 2)
        result = ctx.synthesize("question", {r["task_id"]: r for r in reports})
        self.assertTrue(result["final_answer"])
        self.assertEqual(ctx.metrics()["model_calls"], 5)
        self.assertNotEqual(result["status"], "success")
        second = ctx.synthesize("question", {})
        self.assertEqual(second["stop_reason"], "synthesis_already_attempted")
        self.assertEqual(ctx.metrics()["model_calls"], 5)

    def test_time_reserve_blocks_worker_but_keeps_synthesis_possible(self):
        factory = ScriptedFactory()
        ctx = Context(Settings(max_elapsed_seconds=100), [fixture_lookup], model_factory=factory)
        now = [0.0]
        ctx.budget = RunBudget(12, 16, 100, clock=lambda: now[0])
        now[0] = 85.0
        report = ctx.work("question", {"id": "a", "instruction": "lookup", "depends_on": []})
        self.assertEqual(report["stop_reason"], "budget_exceeded")
        self.assertEqual(factory.calls, [])
        result = ctx.synthesize("question", {"a": report})
        self.assertTrue(result["final_answer"])
        self.assertEqual(len(factory.calls), 1)
        self.assertEqual(factory.calls[0]["role"], "mas_synthesizer")

    def test_error_report_and_protocol_failure_never_become_success(self):
        factory = ScriptedFactory(fail_tool=True, invalid_final=True)
        ctx = Context(Settings(), [fixture_lookup], model_factory=factory)
        ctx.budget = RunBudget(12, 16, 180, clock=lambda: 0.0)
        report = ctx.work("question", {"id": "a", "instruction": "lookup", "depends_on": []})
        self.assertNotEqual(report["status"], "success")
        self.assertEqual(report["evidence"][0]["result"]["status"], "error")
        result = ctx.synthesize("question", {"a": report})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["final_answer"], "")
        self.assertEqual(result["stop_reason"], "invalid_synthesis_answer")

    def test_evaluation_input_adapter_removes_gold_before_real_model(self):
        factory = ScriptedFactory()
        ctx = Context(Settings(), [fixture_lookup], model_factory=factory)
        ctx.budget = RunBudget(12, 16, 180, clock=lambda: 0.0)
        graph = import_module("mas_central_parallel.graph").build_graph(ctx)
        def answer(inputs):
            state = initial()
            state["question"] = inputs["messages"][0][1]
            return graph.invoke(state)
        target = make_target(answer)
        target({"messages": [["user", "QUESTION_ONLY"]],
                "required_items": ["GOLD_SENTINEL"], "reference_evidence": "GOLD_SENTINEL",
                "question_type": "GOLD_SENTINEL"})
        for call in factory.calls:
            self.assertNotIn("GOLD_SENTINEL", str([m.content for m in call["messages"]]))


if __name__ == "__main__":
    unittest.main()
