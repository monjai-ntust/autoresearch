"""Cold-start policy and restart repeatability without pretrained model work."""
from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import DataLoader

from utils.common.artifact_io import DataContractError, load_json
from utils.encoder import runtime
from utils.encoder.data import ResumableRandomSampler
from utils.encoder.trainer import cycle
from utils.encoder_comparison.trainer import _cycle


ROOT = Path(__file__).resolve().parents[1]


class NumericalDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.get_rng_state()
        self.python_rng = random.getstate()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        self.cudnn_benchmark = torch.backends.cudnn.benchmark
        self.cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self.precision = torch.get_float32_matmul_precision()
        self.addCleanup(self._restore)

    def _restore(self):
        torch.set_rng_state(self.rng)
        random.setstate(self.python_rng)
        torch.use_deterministic_algorithms(self.deterministic, warn_only=self.warn_only)
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = self.cudnn_tf32
        torch.set_float32_matmul_precision(self.precision)

    def test_policy_precedes_cuda_work_and_rejects_late_or_conflicting_setup(self):
        events = []
        fake = SimpleNamespace(
            cuda=SimpleNamespace(is_initialized=lambda: False),
            backends=SimpleNamespace(cudnn=SimpleNamespace()),
            use_deterministic_algorithms=lambda enabled, warn_only: events.append(
                ('strict', enabled, warn_only, os.environ.get('CUBLAS_WORKSPACE_CONFIG'))),
            set_float32_matmul_precision=lambda value: events.append(('precision', value)),
            are_deterministic_algorithms_enabled=lambda: True,
            is_deterministic_algorithms_warn_only_enabled=lambda: False,
            get_float32_matmul_precision=lambda: 'highest',
        )
        with (mock.patch.dict(sys.modules, {'torch': fake}),
              mock.patch.object(runtime, '_configured_torch', None),
              mock.patch.dict(os.environ, {}, clear=True)):
            runtime.configure_numerics()
            self.assertEqual(events, [('strict', True, False, ':4096:8'), ('precision', 'highest')])
            self.assertTrue(fake.backends.cudnn.deterministic)
            self.assertFalse(fake.backends.cudnn.benchmark)
            self.assertTrue(fake.backends.cudnn.allow_tf32)
            fake.cuda.is_initialized = lambda: True
            runtime.configure_numerics()  # Already configured process is safe.
            with mock.patch.object(runtime, '_configured_torch', None):
                with self.assertRaisesRegex(DataContractError, 'before CUDA initialization'):
                    runtime.configure_numerics()
            fake.backends.cudnn.deterministic = False
            with self.assertRaisesRegex(DataContractError, 'changed after CUDA initialization'):
                runtime.configure_numerics()
            self.assertFalse(fake.backends.cudnn.deterministic)
            fake.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            with self.assertRaisesRegex(DataContractError, 'incompatible external setting'):
                runtime.configure_numerics()
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':16:8')
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
            with self.assertRaisesRegex(DataContractError, 'overrides the recorded'):
                runtime.configure_numerics()

    def test_runtime_is_configured_before_device_observation_and_bound_once(self):
        (ROOT / 'output').mkdir(exist_ok=True)
        with (tempfile.TemporaryDirectory(dir=ROOT / 'output') as temporary,
              mock.patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}),
              mock.patch.object(torch.cuda, 'is_available', return_value=False)):
            root = Path(temporary)
            prepared = root / 'data-prepared'
            prepared.mkdir()
            original = runtime.configure_numerics
            calls = []

            def configure():
                calls.append('configured')
                original()

            def observe():
                self.assertEqual(calls[-1], 'configured')
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                return False

            with (mock.patch.object(runtime, 'configure_numerics', side_effect=configure),
                  mock.patch.object(torch.cuda, 'is_available', side_effect=observe)):
                reference = runtime.record_runtime(prepared)
                saved = (root / reference['path']).read_bytes()
                self.assertEqual(reference, runtime.record_runtime(prepared))
            self.assertEqual(saved, (root / reference['path']).read_bytes())
            environment = load_json(root / reference['path'])
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertIsNone(environment['nvidia_driver_versions'])
            for redundant in ('sdpa_backends_enabled', 'default_dtype',
                              'autocast_enabled', 'autocast_dtype',
                              'matmul_reduced_precision_reduction',
                              'nvidia_driver_version_source'):
                self.assertNotIn(redundant, environment)
            self.assertNotIn('PYTHONHASHSEED', environment['numerical_environment_variables'])
            self.assertEqual(environment['packages']['torch'], version('torch'))
            self.assertNotIn('deterministic_algorithms', environment)
            self.assertNotIn('float32_matmul_precision', environment)
            self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', environment['numerical_environment_variables'])
            with mock.patch.dict(os.environ, {'OMP_NUM_THREADS': 'fixture-changed-thread-policy'}):
                with self.assertRaisesRegex(DataContractError, 'durable evidence differs'):
                    runtime.record_runtime(prepared)

    def test_unsupported_operation_raises_instead_of_degrading_to_warning(self):
        with mock.patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            runtime.configure_numerics()
            with self.assertRaisesRegex(RuntimeError, 'deterministic'):
                torch.zeros(4).put_(torch.tensor([1, 1]), torch.tensor([2., 3.]))

    def test_model_runtime_records_actual_attention_dtype_without_rng_draws(self):
        (ROOT / 'output').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'output') as temporary:
            prepared = Path(temporary) / 'data-prepared'
            prepared.mkdir()
            model = torch.nn.Linear(2, 2)
            model.backbone = SimpleNamespace(bert=SimpleNamespace(
                config=SimpleNamespace(_attn_implementation='sdpa')))
            before = torch.get_rng_state()
            runtime.record_model_runtime(prepared, model)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            path = prepared.parent / 'manifests/encoder-model-runtime.json'
            value = load_json(path)
            self.assertEqual(value['attention_implementation'], 'sdpa')
            self.assertEqual(value['parameter_dtypes'], ['torch.float32'])
            runtime.record_model_runtime(prepared, model)
            model.backbone.bert.config._attn_implementation = 'eager'
            with self.assertRaisesRegex(DataContractError, 'durable evidence differs'):
                runtime.record_model_runtime(prepared, model)
            model = torch.nn.Linear(2, 2)
            path.unlink()
            runtime.record_model_runtime(prepared, model)
            self.assertIsNone(load_json(path)['backbone_class'])
            self.assertIsNone(load_json(path)['attention_implementation'])

    def test_partial_and_boundary_restart_preserve_actual_cpu_training(self):
        def run(cycle_factory, stop=None, restart=None):
            torch.manual_seed(43)
            model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(0.4), torch.nn.Linear(3, 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            data = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
            sampler = ResumableRandomSampler(data, seed=19)
            loader = DataLoader(data, batch_size=2, sampler=sampler, num_workers=0)
            start = 0
            iterator = cycle_factory(loader)
            if restart:
                model.load_state_dict(restart['model'])
                optimizer.load_state_dict(restart['optimizer'])
                sampler.load_state_dict(restart['sampler'])
                torch.set_rng_state(restart['rng'])
                iterator = runtime.continue_training(loader, iterator)
                start = restart['next_step']
            losses = []
            for step in range(start, stop or 9):
                optimizer.zero_grad()
                loss = model(next(iterator)).square().mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().clone())
            return losses, copy.deepcopy({
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'sampler': sampler.state_dict(), 'rng': torch.get_rng_state(),
                'next_step': stop or 9})

        with mock.patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            runtime.configure_numerics()
            for cycle_factory in (cycle, _cycle):
                expected_losses, expected = run(cycle_factory)
                for stop in (2, 3, 5, 6):
                    with self.subTest(cycle=cycle_factory.__name__, stop=stop):
                        prefix, saved = run(cycle_factory, stop=stop)
                        suffix, actual = run(cycle_factory, restart=saved)
                        self.assertTrue(torch.equal(torch.stack(prefix + suffix), torch.stack(expected_losses)))
                        for key in expected['model']:
                            self.assertTrue(torch.equal(expected['model'][key], actual['model'][key]), key)
                        self.assertTrue(torch.equal(expected['rng'], actual['rng']))
                        self.assertEqual(expected['sampler']['order'], actual['sampler']['order'])
                        self.assertEqual(expected['sampler']['cursor'], actual['sampler']['cursor'])
                        self.assertTrue(torch.equal(expected['sampler']['generator_state'], actual['sampler']['generator_state']))

    def test_independent_processes_have_identical_policy_and_cpu_tensor_results(self):
        program = '''import json,sys,torch
sys.path.insert(0,sys.argv[1])
from utils.encoder.runtime import configure_numerics
configure_numerics()
torch.manual_seed(71)
model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Dropout(.3),torch.nn.Linear(3,1))
optimizer=torch.optim.SGD(model.parameters(),lr=.01)
for _ in range(3):
    optimizer.zero_grad(); loss=model(torch.ones(2,2)).square().mean(); loss.backward(); optimizer.step()
print(json.dumps({'parameters':[value.tolist() for value in model.parameters()], 'strict':torch.are_deterministic_algorithms_enabled(),'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),'precision':torch.get_float32_matmul_precision()}))
'''
        environment = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        first = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(ROOT)],
            check=True, capture_output=True, text=True, env=environment)
        second = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(ROOT)],
            check=True, capture_output=True, text=True, env=environment)
        self.assertEqual(json.loads(first.stdout), json.loads(second.stdout))


if __name__ == '__main__':
    unittest.main()
