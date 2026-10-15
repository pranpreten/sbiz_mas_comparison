"""Offline batch/export checks. Model calls, uploads and live feedback are mocked."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from common.evaluation.dataset import EvaluationExample
from single_agent.evaluate import main, refresh_feedback, run_local_batch, run_remote_batch, select_remote_examples
from single_agent.reporting import build_summary, write_report


def sample(question_id="1-01", question_type="단일 분야"):
    return EvaluationExample(question_id, question_type, "상권", f"질문 {question_id}", ("필수항목",), "정답근거")


def result(status="success", answer="85개", complete=True):
    return {"status": status, "final_answer": answer, "stop_reason": "answer",
            "metrics": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                        "usage_complete": complete, "model_calls": 1, "tool_calls": 1}}


class BatchTests(unittest.TestCase):
    def test_questions_only_and_failures_retained(self):
        examples = [sample(), sample("1-02"), sample("1-03")]
        received = []
        def agent(inputs):
            received.append(inputs)
            if len(received) == 2:
                raise RuntimeError("sk-secret-must-not-export")
            return result("success" if len(received) == 1 else "incomplete")
        clock = iter([1, 2, 3, 5, 6, 9]).__next__
        rows = run_local_batch(examples, agent, clock=clock)
        self.assertEqual([r["status"] for r in rows], ["completed", "failed", "partial"])
        self.assertEqual([r["elapsed_seconds"] for r in rows], [1, 2, 3])
        self.assertEqual(set(received[0]), {"messages"})
        self.assertNotIn("sk-secret", json.dumps(rows))
        self.assertTrue(all(r["requirement_coverage"] is None for r in rows))
        summary = build_summary(rows)["overall"]
        self.assertEqual(summary["questions"], 3)
        self.assertEqual(summary["completion_rate"], 1 / 3)
        self.assertEqual(summary["requirement_coverage"], {"observed": 1, "missing": 2, "mean": None})

    def test_quality_failure_zero_and_grading_missing(self):
        from common.evaluation.reporting import summarize
        rows = [dict(status="completed", answer="answer", requirement_coverage=.8,
                     evidence_consistency=.6) for _ in range(35)]
        rows.append(dict(status="failed", answer="", requirement_coverage=1))
        summary = summarize(rows)
        self.assertAlmostEqual(summary["requirement_coverage"]["mean"], 28 / 36)
        self.assertEqual(summary["evidence_consistency"]["observed"], 36)
        rows[0]["requirement_coverage"] = None
        self.assertIsNone(summarize(rows)["requirement_coverage"]["mean"])
        self.assertIsNone(rows[0]["requirement_coverage"])

    def test_empty_final_and_unknown_usage_not_success_or_zero(self):
        rows = run_local_batch([sample()], lambda inputs: result("error", "", False))
        self.assertEqual(rows[0]["status"], "failed")
        summary = build_summary(rows)["overall"]["total_tokens"]
        self.assertIsNone(summary["mean"])
        self.assertEqual(summary["missing"], 1)

    def test_failed_api_call_without_usage_not_reported_as_complete_zero(self):
        state = result("error", "")
        state["metrics"].update(responses=0, model_calls=1, input_tokens=0, output_tokens=0, total_tokens=0)
        rows = run_local_batch([sample()], lambda inputs: state)
        self.assertFalse(rows[0]["usage_complete"])
        self.assertIsNone(build_summary(rows)["overall"]["total_tokens"]["mean"])

    def test_report_unique_and_csv_formula_safe(self):
        rows = run_local_batch([sample()], lambda inputs: result(answer="=DANGEROUS()"))
        with TemporaryDirectory() as tmp:
            first = write_report(rows, output_root=tmp, metadata={"mode": "local"})
            second = write_report(rows, output_root=tmp, metadata={})
            self.assertNotEqual(first, second)
            doc = json.loads((first / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["questions"][0]["answer"], "=DANGEROUS()")
            self.assertIn("'=DANGEROUS()", (first / "questions.csv").read_text(encoding="utf-8-sig"))
            self.assertTrue((first / "summary.json").is_file())

    def test_dry_run_does_not_read_credentials_or_create_agent(self):
        with patch("common.evaluation.cli.load_settings") as settings:
            self.assertEqual(main(["--limit", "2"]), 0)
            settings.assert_not_called()

    def test_sync_requires_explicit_remote_run(self):
        with self.assertRaises(SystemExit):
            main(["--sync"])


class RemoteTests(unittest.TestCase):
    def client(self, examples):
        client = Mock()
        client.list_examples.return_value = [NS(id=f"id-{e.question_id}", inputs=e.inputs, outputs=e.outputs,
                                                metadata=e.metadata) for e in examples]
        client.list_feedback.return_value = []
        return client

    def test_remote_drift_blocks_before_model(self):
        client = self.client([sample()])
        client.list_examples.return_value[0].outputs = {"reference_evidence": "wrong"}
        with self.assertRaisesRegex(ValueError, "--sync"):
            select_remote_examples(client, "test", [sample()])
        client.evaluate.assert_not_called()

    def test_remote_subset_and_late_score(self):
        first, second = sample(), sample("2-01", "독립 복합")
        client = self.client([first, second])
        class Results(list):
            experiment_name = "experiment"
        def evaluate(target, **kwargs):
            output = Results()
            for item in kwargs["data"]:
                answer = target(item.inputs)
                now = datetime.now(timezone.utc)
                output.append({"example": item, "run": NS(id="run1", start_time=now,
                    end_time=now + timedelta(seconds=2), error=None, outputs=answer)})
            return output
        client.evaluate.side_effect = evaluate
        rows, name = run_remote_batch([first], lambda inputs: result(), client=client,
                                      dataset_name="test", metadata={})
        self.assertEqual(name, "experiment")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["grading_status"], "pending_or_unconfigured")
        self.assertIsNone(rows[0]["evidence_consistency"])
        client.list_feedback.assert_not_called()
        client.list_feedback.return_value = [NS(key="requirement_coverage", score=1.0, created_at="2"),
                                             NS(key="evidence_consistency", score=0.5, created_at="2")]
        refresh_feedback(rows, client)
        self.assertEqual(rows[0]["grading_status"], "completed")
        self.assertEqual(rows[0]["evidence_consistency"], 0.5)
        client.evaluate.assert_called_once()

    def test_failed_new_grade_does_not_retain_old_score(self):
        rows = run_local_batch([sample()], lambda inputs: result())
        rows[0].update(run_id="run1", requirement_coverage=1.0)
        client = Mock()
        client.list_feedback.return_value = [NS(key="requirement_coverage", score=None, created_at="2"),
                                             NS(key="requirement_coverage", score=1.0, created_at="1")]
        refresh_feedback(rows, client)
        self.assertIsNone(rows[0]["requirement_coverage"])
        self.assertEqual(rows[0]["grading_status"], "unscored_feedback")


if __name__ == "__main__":
    unittest.main()
