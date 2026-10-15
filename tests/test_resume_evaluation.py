"""Offline resume selection: no model calls or remote mutations."""
import unittest
from types import SimpleNamespace as NS
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from common.evaluation.cli import resume_snapshot, EvaluationSetupError
from common.evaluation.runner import run_experiment
from tests.test_single_evaluation import sample

class ResumeTests(unittest.TestCase):
    def fixture(self):
        es=[sample(f"3-{i:02}") for i in range(1,37)]
        rs=[NS(id=str(i),dataset_id="dataset",inputs=e.inputs,outputs=e.outputs,metadata=e.metadata) for i,e in enumerate(es)]
        now=datetime.now(timezone.utc)
        runs=[NS(id="run"+str(i),reference_example_id=str(i),end_time=now if i<25 else None,start_time=now,error=None,outputs={"messages":[{"type":"ai","content":"answer"}]},prompt_tokens=10,completion_tokens=2,total_tokens=12) for i in range(26)]
        c=Mock(); c.read_project.return_value=NS(id="project",reference_dataset_id="dataset",metadata={"agent_type":"mas_hierarchical"})
        c.list_examples.return_value=rs;c.list_runs.return_value=runs
        return c,es,runs
    def test_preserve25_resume11(self):
        c,es,runs=self.fixture()
        _,rows,pending,n=resume_snapshot(c,"project","dataset",es,"mas_hierarchical")
        self.assertEqual(len(rows),25);self.assertEqual(pending,es[25:]);self.assertEqual(n,1)
        c.evaluate.assert_not_called()
    def test_ended_failure_is_not_retried(self):
        c,es,runs=self.fixture();runs[0].error="failure";runs[0].outputs={}
        _,rows,pending,_=resume_snapshot(c,"project","dataset",es,"mas_hierarchical")
        self.assertEqual(rows[0]["status"],"failed");self.assertNotIn(es[0],pending)
    def test_duplicate_completed_refused(self):
        c,es,runs=self.fixture();runs.append(runs[0])
        with self.assertRaises(EvaluationSetupError):resume_snapshot(c,"project","dataset",es,"mas_hierarchical")
    def test_extends_existing_project_without_prefix(self):
        c,es,_=self.fixture();result=Mock();result.experiment_name="existing";result.__iter__=Mock(return_value=iter([]));c.evaluate.return_value=result
        project=c.read_project.return_value
        run_experiment(lambda x:"unused",dataset_name="dataset",experiment_prefix="unused",client=c,experiment=project)
        self.assertIs(c.evaluate.call_args.kwargs["experiment"],project)
        self.assertNotIn("experiment_prefix",c.evaluate.call_args.kwargs)
