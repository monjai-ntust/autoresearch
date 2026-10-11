"""Canonical continuation rejects foreign or incomplete durable evidence."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stages.preparation import require_canonical_namespace
from utils.common.artifact_io import (DataContractError, atomic_write_bytes,
    atomic_write_json, sha256_file, validate_stage_receipt)
from utils.common.execution import checked_process, require_active_dispatch
from utils.common.paths import PathContractError, RunLayout
from utils.encoder.cache import write_cache_manifest
from utils.encoder.runtime import restart_identity
from utils.encoder import trainer

ROOT = Path(__file__).resolve().parents[1]


class CanonicalContinuationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'output').mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'output')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.layout = RunLayout(self.root, 'canonical')
        self.layout.create()

    def _inputs(self, layout):
        for relative in ('manifests/00-checkout-manifest.json',
                'manifests/02-input-acquisition-manifest.json',
                'manifests/03-data-preparation-manifest.json',
                'manifests/huggingface-model-cache.json',
                'manifests/encoder-runtime.json', 'inputs/config/pipeline.json',
                'data-prepared/split-manifest.json', 'data-prepared/train.jsonl',
                'data-prepared/development.jsonl'):
            atomic_write_json(layout.resolve(relative), {})

    def test_restart_binding_detects_foreign_run_and_input_drift(self):
        self._inputs(self.layout)
        expected = restart_identity(self.layout.resolve('data-prepared'))
        foreign = RunLayout(self.root, 'foreign')
        foreign.create()
        self._inputs(foreign)
        self.assertNotEqual(expected, restart_identity(foreign.resolve('data-prepared')))
        atomic_write_json(self.layout.resolve('data-prepared/train.jsonl'), {'changed': True})
        self.assertNotEqual(expected, restart_identity(self.layout.resolve('data-prepared')))

    def test_restart_writer_embeds_identity_without_changing_tensor_payload(self):
        self._inputs(self.layout)
        path = self.layout.resolve('checkpoints/seed-42/restart-state.pt')
        state = {'format_version': 'train-span-restart-2.0', 'encoder': {'value': 7}}
        trainer._atomic_torch_save(state, path)
        actual = trainer.torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(actual.pop('run_identity'), restart_identity(self.layout.resolve('data-prepared')))
        self.assertEqual(actual, state)
        self.assertNotIn('run_identity', state)

    def test_private_worker_requires_current_accounted_dispatch(self):
        arguments = ['_train-encoder', '--run-id', 'canonical', '--training-seed', '42']
        command = ['python', '-B', 'pipeline.py', *arguments]
        label = 'canonical-train-seed-42'
        with self.assertRaises(DataContractError):
            require_active_dispatch(self.layout, arguments, label=label)
        def child(argv, **kwargs):
            with mock.patch('utils.common.execution.os.getppid', return_value=os.getpid()):
                require_active_dispatch(self.layout, arguments, label=label)
                with self.assertRaises(DataContractError):
                    require_active_dispatch(self.layout, [*arguments[:-1], '43'], label=label)
        checked_process(self.layout, command, label=label, command_runner=child)
        with self.assertRaises(DataContractError):
            require_active_dispatch(self.layout, arguments, label=label)

    def test_canonical_workflow_rejects_comparison_roots(self):
        require_canonical_namespace(self.layout)
        for relative in ('manifests/comparison-matrix.json',
                         'manifests/comparison-matrix-link.json',
                         'manifests/encoder-comparison-selection.json'):
            path = self.layout.resolve(relative)
            atomic_write_json(path, {})
            with self.assertRaisesRegex(DataContractError, 'comparison namespace'):
                require_canonical_namespace(self.layout)
            path.unlink()

    def test_candidate_receipt_requires_recorded_output_hashes(self):
        checkout = self.layout.resolve('manifests/00-checkout-manifest.json')
        atomic_write_json(checkout, {})
        candidates = 'predictions/test/seed-42-candidates.jsonl'
        ledger = 'predictions/test/seed-42-prediction-ledger.jsonl'
        for path in (candidates, ledger):
            atomic_write_bytes(self.layout.resolve(path), b'original\n')
        manifest = {'stage': 'model-generate-candidates', 'execution_mode': 'live',
                    'candidates_output': candidates, 'receipt_version': 1,
                    'run_id': self.layout.run_id, 'checkout_sha256': sha256_file(checkout),
                    'inputs': {'prediction_ledger': {'path': ledger,
                               'sha256': sha256_file(self.layout.resolve(ledger))}},
                    'output_hashes': {p: sha256_file(self.layout.resolve(p)) for p in (candidates, ledger)}}
        path = self.layout.resolve('manifests/candidates.json')
        validate_stage_receipt(self.layout, path, manifest)
        manifest['output_hashes'].pop(candidates)
        with self.assertRaisesRegex(DataContractError, 'coverage'):
            validate_stage_receipt(self.layout, path, manifest)
        manifest['output_hashes'][candidates] = sha256_file(self.layout.resolve(candidates))
        atomic_write_bytes(self.layout.resolve(candidates), b'tampered\n')
        with self.assertRaisesRegex(DataContractError, 'differs from its producer'):
            validate_stage_receipt(self.layout, path, manifest)

    def test_redirected_run_root_is_rejected_before_locks_or_writes(self):
        target = self.root / 'foreign'
        target.mkdir()
        alias = self.layout.output_root / 'alias'
        if os.name == 'nt':
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                f'New-Item -ItemType Junction -Path {quote(alias)} -Target {quote(target)} | Out-Null'],
                check=True, capture_output=True)
        else:
            alias.symlink_to(target, target_is_directory=True)
        self.addCleanup(lambda: alias.rmdir() if os.name == 'nt' else alias.unlink())
        with self.assertRaisesRegex(PathContractError, 'run root'):
            RunLayout(self.root, 'alias')
        self.assertEqual(list(target.iterdir()), [])

    def test_cache_rejects_foreign_link_but_allows_internal_blob_link(self):
        cache = self.layout.resolve('inputs/huggingface')
        cache.mkdir(parents=True, exist_ok=True)
        blob = cache / 'blob'
        atomic_write_bytes(blob, b'model')
        link = cache / 'weight'
        # Mock link inspection on Windows, where file symlinks require privileges.
        original_is_symlink, original_resolve = Path.is_symlink, Path.resolve
        def is_link(path):
            return path == link or original_is_symlink(path)
        def resolve(path, *args, **kwargs):
            return target if path == link else original_resolve(path, *args, **kwargs)
        atomic_write_bytes(link, b'placeholder')
        target = self.root / 'foreign-model'
        atomic_write_bytes(target, b'model')
        with (mock.patch.object(Path, 'is_symlink', is_link),
              mock.patch.object(Path, 'resolve', resolve), mock.patch('os.readlink', return_value='blob')):
            with self.assertRaisesRegex(DataContractError, 'outside'):
                write_cache_manifest(self.layout, model='fixture', revision='revision')
            target = blob
            manifest = write_cache_manifest(self.layout, model='fixture', revision='revision')
            self.assertEqual(manifest['file_count'], 2)


if __name__ == '__main__':
    unittest.main()
