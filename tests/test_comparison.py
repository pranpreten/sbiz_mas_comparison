"""Comparison lifecycle and delayed feedback: no provider calls or network."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from common.config import Settings
from common.agents import ARCHITECTURES
from common.evaluation.comparison import (comparison_rows, group_key, main, refresh_once,
    scan_reports, watch_feedback, write_comparison)
from common.evaluation.reporting import save_report, write_report
from common.evaluation.cli import main as evaluate
from test_single_evaluation import sample, result, run_local_batch


def report(root, architecture="single_agent", count=2, **changes):
    rows = run_local_batch([sample(f"q{i}") for i in range(count)], lambda _: result())
    for i, row in enumerate(rows):
        row.update(run_id=f"{architecture}-{i}", grading_status="pending_or_unconfigured")
    meta = dict(agent_type=architecture, mode="langsmith", model="fixed", temperature=0,
                workbook_sha256="book", implementation_sha256="code", dataset_name="test",
                max_model_calls=12, max_tool_calls=16, selected_questions=count)
    meta.update(changes)
    directory = write_report(rows, output_root=root, metadata=meta)
    return directory, json.loads((directory / "report.json").read_text(encoding="utf-8"))


class ComparisonTests(unittest.TestCase):
    def test_empty_seven_rows_and_incremental_fill(self):
        with TemporaryDirectory() as root:
            write_comparison(root)
            def table():
                return json.loads((Path(root) / "comparison.json").read_text(encoding="utf-8"))["rows"]
            self.assertEqual(len(table()), 7)
            self.assertTrue(all(row["state"] == "미실행" for row in table()))
            for index, name in enumerate(ARCHITECTURES, 1):
                report(root, name)
                write_comparison(root)
                self.assertEqual(sum(row["questions"] is not None for row in table()), index)
            self.assertTrue((Path(root) / "comparison.csv").read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_conditions_and_question_ids_do_not_mix(self):
        with TemporaryDirectory() as root:
            _, first = report(root)
            report(root, "mas_sequential", count=1)
            report(root, "mas_dynamic", model="different")
            report(root, "mas_review", implementation_sha256="changed")
            report(root, "mas_hierarchical", mode="local")
            directory, different = report(root, "mas_central_parallel")
            different["questions"][0]["question_id"] = "another-question"
            save_report(directory, different)
            groups, _ = scan_reports(root)
            self.assertEqual(len(groups), 6)
            self.assertEqual(len(groups[group_key(first)[0]]["runs"]), 1)

    def test_rerun_replaces_row_but_feedback_update_does_not_change_identity(self):
        with TemporaryDirectory() as root:
            old_dir, old = report(root)
            new_dir, new = report(root)
            old["feedback_checked_at"] = "2099-01-01T00:00:00Z"
            save_report(old_dir, old)
            groups, _ = scan_reports(root)
            self.assertEqual(groups[group_key(new)[0]]["runs"]["single_agent"][0].parent, new_dir)
            self.assertTrue((old_dir / "report.json").exists())

    def test_delayed_grades_update_same_reports_without_model(self):
        with TemporaryDirectory() as root:
            directory, original = report(root)
            client = Mock()
            client.list_feedback.return_value = []
            self.assertEqual(refresh_once(root, client), 2)
            client.list_feedback.return_value = [NS(key="requirement_coverage", score=1, created_at="2"),
                                                 NS(key="evidence_consistency", score=.5, created_at="2")]
            self.assertEqual(refresh_once(root, client), 0)
            saved = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["created_at"], original["created_at"])
            self.assertEqual(saved["questions"][0]["answer"], original["questions"][0]["answer"])
            self.assertEqual(saved["summary"]["overall"]["evidence_consistency"]["mean"], .5)
            client.evaluate.assert_not_called()
            self.assertEqual(len(list(Path(root).glob("*/report.json"))), 1)

    def test_failed_partial_and_missing_scores_remain_visible(self):
        with TemporaryDirectory() as root:
            directory, doc = report(root, count=3)
            doc["questions"][0].update(status="failed", answer="", usage_complete=False, run_id=None)
            doc["questions"][1].update(status="partial", requirement_coverage=0)
            doc["questions"][2].update(requirement_coverage=1)
            save_report(directory, doc)
            groups, _ = scan_reports(root)
            row = comparison_rows(groups[group_key(doc)[0]])[0]
            self.assertEqual((row["questions"], row["failed"], row["partial"]), (3, 1, 1))
            self.assertAlmostEqual(row["requirement_coverage_mean"], 1 / 3)
            self.assertEqual(row["requirement_coverage_observed"], 3)
            self.assertIsNone(row["evidence_consistency_mean"])
            self.assertEqual(row["total_tokens_observed"], 2)

    def test_polling_stops_on_completion_and_has_deadline(self):
        with patch("common.evaluation.comparison.refresh_once", side_effect=[2, 1, 0]) as refresh:
            sleeps = Mock()
            self.assertTrue(watch_feedback("unused", Mock(), clock=lambda: 0, sleeper=sleeps, progress=Mock()))
            self.assertEqual(sleeps.call_count, 2)
        with patch("common.evaluation.comparison.refresh_once", return_value=1):
            self.assertFalse(watch_feedback("unused", Mock(), wait_seconds=0, progress=Mock()))
        with self.assertRaises(ValueError):
            watch_feedback("unused", Mock(), wait_seconds=float("nan"))

    def test_network_error_keeps_previous_grade_and_reports_unavailable(self):
        with TemporaryDirectory() as root:
            directory, doc = report(root)
            doc["questions"][0]["requirement_coverage"] = .75
            save_report(directory, doc)
            client = Mock()
            client.list_feedback.side_effect = RuntimeError("secret-do-not-export")
            self.assertEqual(refresh_once(root, client), 2)
            saved = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["questions"][0]["requirement_coverage"], .75)
            self.assertNotIn("secret-do-not-export", json.dumps(saved))
            table = json.loads((Path(root) / "comparison.json").read_text(encoding="utf-8"))
            self.assertIn("조회 실패", table["rows"][0]["state"])

    def test_smoke_and_corrupt_reports_not_used(self):
        with TemporaryDirectory() as root:
            report(root, mode="development_smoke")
            broken = Path(root) / "broken"
            broken.mkdir()
            (broken / "report.json").write_text("{", encoding="utf-8")
            with patch("common.evaluation.comparison.load_settings") as settings:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--output-dir", root]), 0)
                settings.assert_not_called()
            groups, ignored = scan_reports(root)
            self.assertFalse(groups)
            self.assertEqual(len(ignored), 1)

    def test_invalid_score_and_html_are_not_trusted(self):
        with TemporaryDirectory() as root:
            directory, doc = report(root, model="<script>alert(1)</script>")
            doc["questions"][0].update(requirement_coverage=10, evidence_consistency=True)
            save_report(directory, doc)
            path = write_comparison(root)
            rendered = path.read_text(encoding="utf-8")
            self.assertNotIn("<script>", rendered)
            self.assertIn("&lt;script&gt;", rendered)
            groups, _ = scan_reports(root)
            row = comparison_rows(groups[group_key(doc)[0]])[0]
            self.assertIsNone(row["requirement_coverage_mean"])

    def test_remote_evaluation_saves_and_exits_without_polling(self):
        with TemporaryDirectory() as root, \
             patch("common.evaluation.cli.load_settings", return_value=Settings(tracing=True)), \
             patch("common.evaluation.cli.create_agent") as agent, \
             patch("langsmith.Client"), \
             patch("common.evaluation.cli.run_remote_batch") as batch, \
             patch("common.evaluation.comparison.watch_feedback") as watch, \
             contextlib.redirect_stdout(io.StringIO()):
            batch.return_value = (run_local_batch([sample()], lambda _: result()), "test-experiment")
            self.assertEqual(evaluate(["--run", "--limit", "1", "--output-dir", root]), 0)
            watch.assert_not_called()
            self.assertTrue((Path(root) / "comparison.html").exists())
            agent.return_value.answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
