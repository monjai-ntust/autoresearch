"""Focused tests for the run-local Hugging Face cache contract."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from hf_cache import verify_cache_manifest, write_cache_manifest
from phase_b_io import DataContractError, atomic_write_bytes
from paths import RunLayout, discover_source_root


SOURCE_ROOT = discover_source_root(Path(__file__))
MODEL = "microsoft/deberta-large"
REVISION = "28c23d9eb93ea6cf11f845501ab7aeb2a497658b"


class HuggingFaceCacheTests(unittest.TestCase):
    def setUp(self):
        self.layout = RunLayout(SOURCE_ROOT, f"test-hf-cache-{uuid.uuid4().hex}")
        self.layout.create()

    def tearDown(self):
        shutil.rmtree(self.layout.run_root)

    def test_manifest_freezes_run_local_files_for_offline_replay(self):
        atomic_write_bytes(
            self.layout.resolve("inputs/huggingface/models--fixture/blobs/model.bin"),
            b"model bytes\n",
        )
        written = write_cache_manifest(
            self.layout, model=MODEL, revision=REVISION
        )
        verified = verify_cache_manifest(
            self.layout, model=MODEL, revision=REVISION
        )
        self.assertEqual(verified, written)
        self.assertTrue(verified["offline_replay_ready"])
        self.assertEqual(verified["file_count"], 1)

    def test_cache_content_drift_fails_closed(self):
        model_path = self.layout.resolve(
            "inputs/huggingface/models--fixture/blobs/model.bin"
        )
        atomic_write_bytes(model_path, b"model bytes\n")
        write_cache_manifest(self.layout, model=MODEL, revision=REVISION)
        atomic_write_bytes(model_path, b"changed model bytes\n")
        with self.assertRaisesRegex(DataContractError, "inventory drifted"):
            verify_cache_manifest(self.layout, model=MODEL, revision=REVISION)


if __name__ == "__main__":
    unittest.main()
