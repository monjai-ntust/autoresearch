"""Administrative hints cannot block a scientifically authenticated workflow."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from utils.common import config as config_module
from utils.common.artifact_io import DataContractError
from utils.verifier.verifier import _environment_manifest
from utils.verifier.requests import _prompt_bundle

ROOT = Path(__file__).resolve().parents[1]


class CheckoutRequirementsTests(unittest.TestCase):
    def load(self, value):
        with mock.patch.object(config_module, 'load_json', return_value=value):
            return config_module.load_pipeline_config(ROOT, 'resources/configs/pipeline.json')

    def original(self):
        return json.loads((ROOT / 'resources/configs/pipeline.json').read_text(encoding='utf-8'))

    def test_documented_versions_are_optional_and_do_not_gate_actual_runtime(self):
        original = self.original()
        for environment in (None, {'python': '3.12.4', 'uv': '0.11.3'}):
            with self.subTest(environment=environment):
                value = copy.deepcopy(original)
                if environment is None:
                    value.pop('environment')
                else:
                    value['environment'] = environment
                loaded = self.load(value)
                loaded.value['verifier']['registry_manifest_sha256'] = 'a' * 64
                loaded.value['verifier']['model_blob_sha256'] = 'b' * 64
                with mock.patch('utils.verifier.verifier._command_output',
                                return_value={'status': 'unavailable', 'stdout': None}):
                    actual = _environment_manifest(loaded, 'VER-SIMPLE', 'replay')
                self.assertTrue(actual['python']['observed'])
                self.assertNotIn('expected', actual['python'])
                self.assertNotIn('expected_uv', actual['software'])
                self.assertIn('packages', actual['software'])

    def test_unused_metadata_text_does_not_gate_scientific_configuration(self):
        value = self.original()
        value['notes'] = 'TBD_user_notes'
        self.assertEqual(self.load(value).value['notes'], 'TBD_user_notes')
        value['training_seeds'] = [42]
        with self.assertRaisesRegex(DataContractError, 'training_seeds'):
            self.load(value)

    def test_cloned_resource_hash_decorations_do_not_gate_actual_prompt_identity(self):
        original = self.load(self.original())
        value = self.original()
        for field in list(value['verifier']):
            if field.endswith(('_prompt_sha256', '_response_schema_sha256', '_bundle_sha256')):
                value['verifier'].pop(field)
        loaded = self.load(value)
        for mode in ('simple', 'corrective'):
            with self.subTest(mode=mode), mock.patch(
                    'utils.common.artifact_io.sha256_file', side_effect=AssertionError('cloned file hashing')):
                actual = _prompt_bundle(loaded, mode)
            self.assertEqual(actual, _prompt_bundle(original, mode))

    def test_comparison_uses_semantic_config_without_a_cloned_file_hash_pin(self):
        from utils.encoder_comparison import config as comparison
        path = ROOT / 'resources/configs/encoder-comparison.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['canonical_pipeline_config'].pop('sha256')
        actual_hash = comparison.sha256_file
        with mock.patch.object(comparison, 'load_json', return_value=value), \
             mock.patch.object(comparison, 'sha256_file', wraps=actual_hash) as hashes:
            loaded = comparison.load_comparison_config(ROOT)
        self.assertEqual(loaded.pipeline.value, self.original())
        hashes.assert_called_once_with(path)

    def test_entrypoints_disable_bytecode_without_a_required_cli_flag(self):
        for script in ('pipeline.py', 'encoder_comparison.py', 'rag.py'):
            with self.subTest(script=script):
                program = '''import runpy,sys
sys.dont_write_bytecode=False
sys.argv=[sys.argv[1],'--help']
try:
    runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0, exc.code
assert sys.dont_write_bytecode
'''
                result = subprocess.run([sys.executable, '-I', '-c', program, script],
                                        cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
