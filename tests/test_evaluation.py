"""Offline evaluation-contract tests; never contact LangSmith or call models."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from uuid import uuid4

from common.evaluation.dataset import EvaluationExample, load_examples, sync_dataset, validate_examples
from common.evaluation.runner import ExperimentRun, extract_final_answer, make_target, run_experiment


def example(question_id="1-01"):
    return EvaluationExample(question_id, "단일 분야", "상권", "카페 몇 개?", ("업소 수 제시",), "85개")


class DatasetTests(unittest.TestCase):
    def test_actual_workbook_has_36_valid_separated_examples(self):
        path = Path(__file__).resolve().parents[2] / "예비실험" / "예비실험_질문36문항_v0.6.xlsx"
        examples = load_examples(path)
        self.assertEqual(len(examples), 36)
        self.assertEqual(len({x.question_id for x in examples}), 36)
        self.assertEqual(set(examples[0].inputs), {"messages"})
        self.assertEqual(examples[0].inputs["messages"][0][0], "user")
        self.assertEqual(len(examples[0].required_items), 2)
        self.assertFalse(examples[0].required_items[0].startswith("1."))
        self.assertIn("정책", examples[0].metadata["domain"])

    def test_empty_duplicate_and_missing_reference_rejected(self):
        with self.assertRaises(ValueError):
            validate_examples([])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_examples([example(), example()])
        with self.assertRaises(ValueError):
            EvaluationExample("x", "type", "domain", "q", ("item",), "")

    def test_preview_never_mutates(self):
        client = Mock()
        client.has_dataset.return_value = False
        plan = sync_dataset(client, "test", [example()])
        self.assertEqual(plan["create"], ["1-01"])
        client.create_dataset.assert_not_called()
        client.create_example.assert_not_called()
        client.update_example.assert_not_called()

    def test_update_retains_dataset_and_unowned_metadata(self):
        client = Mock()
        client.has_dataset.return_value = True
        client.read_dataset.return_value = NS(id=uuid4())
        old = NS(id=uuid4(), inputs={"messages": [["user", "old"]]}, outputs={},
                 metadata={"question_id": "1-01", "owner_note": "keep"})
        unrelated = NS(id=uuid4(), inputs={}, outputs={}, metadata={"question_id": "other"})
        client.list_examples.return_value = [old, unrelated]
        plan = sync_dataset(client, "test", [example()], apply=True)
        self.assertEqual(plan["update"], ["1-01"])
        client.create_dataset.assert_not_called()
        client.update_example.assert_called_once()
        self.assertEqual(client.update_example.call_args.kwargs["metadata"]["owner_note"], "keep")
        client.delete_example.assert_not_called()
        client.delete_dataset.assert_not_called()

    def test_unchanged_remote_no_write(self):
        item = example()
        client = Mock()
        client.has_dataset.return_value = True
        client.read_dataset.return_value = NS(id=uuid4())
        client.list_examples.return_value = [NS(id=uuid4(), inputs=item.inputs, outputs=item.outputs, metadata=item.metadata)]
        self.assertEqual(sync_dataset(client, "test", [item], apply=True)["unchanged"], [item.question_id])
        client.update_example.assert_not_called()
        client.create_example.assert_not_called()

    def test_create_stable_ids_and_duplicate_remote_preflight(self):
        item = example()
        client = Mock()
        client.has_dataset.return_value = False
        client.create_dataset.return_value = NS(id=uuid4())
        sync_dataset(client, "test", [item], apply=True)
        first_id = client.create_example.call_args.kwargs["example_id"]
        sync_dataset(client, "test", [item], apply=True)
        self.assertEqual(first_id, client.create_example.call_args.kwargs["example_id"])
        client.has_dataset.return_value = True
        client.list_examples.return_value = [NS(metadata=item.metadata), NS(metadata=item.metadata)]
        client.update_example.reset_mock()
        with self.assertRaisesRegex(ValueError, "Duplicate remote"):
            sync_dataset(client, "test", [item], apply=True)
        client.update_example.assert_not_called()


class TargetTests(unittest.TestCase):
    def test_only_question_reaches_target_and_output_only_final_ai(self):
        received = []
        def agent(inputs):
            received.append(inputs)
            inputs["messages"].append(["assistant", "mutated"])
            return {"messages": [{"type": "tool", "content": "secret retrieval"},
                                 {"type": "ai", "content": "85개"}]}
        source = {**example().inputs, **example().outputs, "domain": "상권"}
        target = make_target(agent)
        self.assertEqual(target(source), {"messages": [{"type": "ai", "content": "85개"}]})
        self.assertEqual(set(received[0]), {"messages"})
        self.assertEqual(len(source["messages"]), 1)
        target(source)
        self.assertIsNot(received[0], received[1])

    def test_tool_or_pending_tool_not_misread_as_final_answer(self):
        for last in ({"type": "tool", "content": "85"},
                     {"type": "ai", "content": "조회합니다", "tool_calls": [{"name": "search"}]},
                     {"type": "ai", "content": "", "tool_calls": []}):
            with self.subTest(last=last), self.assertRaises(ValueError):
                extract_final_answer({"messages": [{"type": "ai", "content": "earlier answer"}, last]})

    def test_reasoning_blocks_excluded_and_object_message_supported(self):
        content = [{"type": "reasoning", "text": "hidden"}, {"type": "text", "text": "85개"}]
        self.assertEqual(extract_final_answer({"messages": [NS(type="ai", content=content, tool_calls=[])]}), "85개")

    def test_plain_text_tool_protocol_is_rejected_in_every_output_contract(self):
        raw = '<|tool_call:start|>search_law_articles {"keywords":["소기업"]}<|tool_call:end|>'
        values = [raw, {"final_answer": raw},
                  {"messages": [{"type": "ai", "content": raw}]},
                  {"messages": [NS(type="ai", content=[{"type": "text", "text": raw}], tool_calls=[])]},
                  {"final_answer": "답변 뒤에 " + raw},
                  {"final_answer": "<|tool_call:start"}]
        for value in values:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "tool-call protocol"):
                extract_final_answer(value)
        self.assertEqual(extract_final_answer("tool_call은 도구 호출을 뜻합니다."),
                         "tool_call은 도구 호출을 뜻합니다.")

    def test_invalid_tool_call_metadata_is_not_a_final_answer(self):
        with self.assertRaises(ValueError):
            extract_final_answer({"messages": [{"type": "ai", "content": "조회 중",
                "invalid_tool_calls": [{"name": "search_law_articles", "args": "{broken"}]}]})

    def test_errors_propagate_to_sdk(self):
        def broken(inputs):
            raise RuntimeError("target failure")
        with self.assertRaisesRegex(RuntimeError, "target failure"):
            make_target(broken)(example().inputs)


class RunnerTests(unittest.TestCase):
    def test_validate_before_any_model_execution(self):
        client = Mock()
        client.list_examples.return_value = [NS(inputs=example().inputs, outputs={})]
        target = Mock()
        with self.assertRaises(ValueError):
            run_experiment(target, dataset_name="test", experiment_prefix="single", client=client)
        target.assert_not_called()
        client.evaluate.assert_not_called()

    def test_results_keep_errors_and_no_duplicate_local_evaluators(self):
        client = Mock()
        item = example()
        client.list_examples.return_value = [NS(inputs=item.inputs, outputs=item.outputs)]
        class Results(list):
            experiment_name = "test-run"
        client.evaluate.return_value = Results([{"run": NS(error="failed"), "evaluation_results": {}}])
        report = run_experiment(lambda inputs: "answer", dataset_name="test", experiment_prefix="single", client=client)
        self.assertEqual(report.answer_errors, 1)
        self.assertEqual(report.experiment_name, "test-run")
        self.assertNotIn("evaluators", client.evaluate.call_args.kwargs)
        self.assertEqual(client.evaluate.call_args.kwargs["error_handling"], "log")

    def test_metrics_unknown_usage_is_not_zero(self):
        start = datetime.now(timezone.utc)
        run = NS(id=uuid4(), start_time=start, end_time=start + timedelta(seconds=2), error=None, total_tokens=None)
        report = ExperimentRun("test", [{"run": run}], None)
        self.assertEqual(report.metrics()[0]["latency_seconds"], 2)
        self.assertIsNone(report.metrics()[0]["total_tokens"])


if __name__ == "__main__":
    unittest.main()
