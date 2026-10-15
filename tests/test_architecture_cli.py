"""실행 구조와 업로드/채점 경계를 검사한다. 외부 호출 없음."""
import contextlib
import io
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest.mock import patch, Mock
from common.agents import ARCHITECTURES
from common.config import Settings
from common.evaluation.cli import main, EvaluationSetupError, entrypoint
from common.evaluation.reporting import write_report


class ArchitectureCLITests(unittest.TestCase):
    def test_disabled_tracing_has_actionable_error_before_agent_creation(self):
        with patch('common.evaluation.cli.load_settings', return_value=Settings(tracing=False)), \
             patch('common.evaluation.cli.create_agent') as agent, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(EvaluationSetupError, 'LANGSMITH_TRACING=true'):
                main(['--run', '--limit', '1'])
            agent.assert_not_called()

    def test_entrypoint_prints_safe_setup_message(self):
        output = io.StringIO()
        with patch('common.evaluation.cli.main', side_effect=EvaluationSetupError('LANGSMITH_TRACING=true')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(entrypoint(), 1)
        self.assertIn('LANGSMITH_TRACING=true', output.getvalue())

    def test_all_seven_validate_without_agent_keys_or_remote(self):
        with patch('common.evaluation.cli.load_settings') as settings, patch('common.evaluation.cli.create_agent') as agent:
            for name in ARCHITECTURES:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([], architecture=name), 0)
            settings.assert_not_called()
            agent.assert_not_called()

    def test_upload_only_syncs_all_questions_without_model_or_graph(self):
        with patch('common.evaluation.cli.load_settings', return_value=Settings()), \
             patch('langsmith.Client') as client, \
             patch('common.evaluation.cli.sync_dataset') as sync, \
             patch('common.evaluation.cli.create_agent') as agent, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--upload-only','--limit','2'], architecture='mas_dynamic'), 0)
            sync.assert_called_once()
            self.assertEqual(len(sync.call_args.args[2]), 36)
            self.assertTrue(sync.call_args.kwargs['apply'])
            agent.assert_not_called()
            client.return_value.evaluate.assert_not_called()

    def test_reports_identify_architecture_and_keep_unscored_missing(self):
        with TemporaryDirectory() as root:
            for name in ARCHITECTURES:
                path=write_report([],output_root=root,metadata={'agent_type':name})
                self.assertTrue(path.name.startswith(name+'_'))
                data=json.loads((path/'report.json').read_text(encoding='utf-8'))
                self.assertIsNone(data['summary']['overall']['requirement_coverage']['mean'])
            with self.assertRaises(ValueError):
                write_report([],output_root=root,metadata={'agent_type':'../escape'})


if __name__ == '__main__':
    unittest.main()
