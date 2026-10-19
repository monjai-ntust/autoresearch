"""A clean commit and selected path identify tracked configuration without copies."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stages import comparison_matrix, preparation
from utils.common.config import PipelineConfig
from utils.common.constants import MATCHER_ID, PROTOCOL_ID, WORKFLOW_ID
from utils.common.environment import inspect_checkout, run_doctor
from utils.common.paths import RunLayout
from utils.encoder_comparison.config import load_comparison_config


ROOT = Path(__file__).resolve().parents[1]


class SourceIdentityTests(unittest.TestCase):
    def test_matrix_records_selected_paths_without_static_config_or_profile_copies(self):
        config = load_comparison_config(ROOT)
        layout = RunLayout(ROOT, 'source-identity-fixture')

        def git(*args, **kwargs):
            return subprocess.CompletedProcess(args, 0, 'a' * 40 if args[0] == 'rev-parse' else '')

        with (mock.patch('utils.rag.runner.run_git', side_effect=git),
              mock.patch.object(comparison_matrix, 'sha256_file', side_effect=AssertionError('tracked file hashing'))):
            actual = comparison_matrix.matrix_identity(layout, config)
        self.assertEqual(actual, {'matrix_id': layout.run_id, 'source_commit': 'a' * 40,
            'config': 'resources/configs/encoder-comparison.json',
            'pipeline_config': 'resources/configs/pipeline.json',
            'members': {arm: f'{layout.run_id}-{arm}' for arm in config.arms}})

    def test_matrix_rejects_either_untracked_selected_config_before_namespace_creation(self):
        config = load_comparison_config(ROOT)
        layout = RunLayout(ROOT, 'untracked-source-identity-fixture')
        for rejected in (config.source_path, config.pipeline.source_path):
            with self.subTest(rejected=rejected):
                def git(*args, **kwargs):
                    status = int(args[0] == 'ls-files' and args[-1] == rejected)
                    return subprocess.CompletedProcess(args, status,
                        'a' * 40 if args[0] == 'rev-parse' else '')

                with (mock.patch('utils.rag.runner.run_git', side_effect=git),
                      self.assertRaisesRegex(ValueError, 'tracked by the source commit')):
                    comparison_matrix.matrix_identity(layout, config)
                self.assertFalse(layout.run_root.exists())

    def test_real_git_fresh_doctor_and_checkout_recovery_need_no_config_snapshot(self):
        (ROOT / 'output').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'output') as temporary:
            root = Path(temporary)
            config_path = root / 'resources/configs/pipeline.json'
            config_path.parent.mkdir(parents=True)
            value = {'protocol_id': PROTOCOL_ID, 'workflow_id': WORKFLOW_ID, 'matcher_id': MATCHER_ID}
            config_path.write_text(json.dumps(value) + '\n', encoding='utf-8', newline='\n')
            (root / '.gitignore').write_text('/output/\n/ignored.json\n', encoding='utf-8')

            def git(*args):
                return subprocess.run(['git', *args], cwd=root, check=True,
                    capture_output=True, text=True).stdout.strip()

            git('init', '-q')
            git('config', 'user.name', 'Fixture')
            git('config', 'user.email', 'fixture@example.invalid')
            git('config', 'core.autocrlf', 'false')
            git('add', '.gitignore', 'resources/configs/pipeline.json')
            git('commit', '-qm', 'fixture source')
            first_commit = git('rev-parse', 'HEAD')
            ignored_path = root / 'ignored.json'
            ignored_path.write_text(json.dumps(value) + '\n', encoding='utf-8')
            ignored, admitted = inspect_checkout(RunLayout(root, 'ignored'), PipelineConfig(ignored_path, value))
            self.assertFalse(admitted)
            self.assertEqual(next(item['status'] for item in ignored['checks']
                if item['check_id'] == 'selected-config-tracked'), 'fail')
            self.assertFalse((root / 'output').exists())
            (root / 'README.md').write_text('A fresh checkout-only interruption.\n', encoding='utf-8')
            self.assertFalse((root / 'output').exists())
            layout = RunLayout(root, 'fresh')
            config = PipelineConfig(config_path, value)
            blocked, passed = run_doctor(layout, config)
            self.assertFalse(passed)
            self.assertEqual(blocked['source']['commit'], first_commit)
            self.assertEqual(blocked['source']['config'], {'path': 'resources/configs/pipeline.json'})
            self.assertFalse(layout.resolve('inputs/config').exists())
            git('add', 'README.md')
            git('commit', '-qm', 'finish checkout source')
            preparation.initialize(layout, config)
            fresh = json.loads(layout.resolve('manifests/00-checkout-manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(fresh['source']['commit'], git('rev-parse', 'HEAD'))
            self.assertEqual(fresh['status'], 'pass')
            recovery = layout.resolve('manifests/checkout-recovery')
            self.assertFalse((recovery / 'config.json').exists())
            self.assertFalse(layout.resolve('inputs/config').exists())
            receipt = json.loads(next(recovery.glob('receipt-*.json')).read_text(encoding='utf-8'))
            self.assertEqual(set(receipt['artifacts']), {'previous_checkout', 'fresh_checkout'})


if __name__ == '__main__':
    unittest.main()
