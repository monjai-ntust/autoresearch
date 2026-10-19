"""Focused tests for the run-local Hugging Face cache contract."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from utils.encoder.cache import verify_cache_manifest, write_cache_manifest
from utils.common.artifact_io import DataContractError, atomic_write_bytes
from utils.common.paths import RunLayout, discover_source_root


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

    def test_failure_names_added_removed_and_changed_paths_without_refreezing(self):
        cache = self.layout.resolve("inputs/huggingface/models--fixture")
        atomic_write_bytes(cache / "blobs/model.bin", b"model bytes\n")
        atomic_write_bytes(cache / "snapshots/revision/config.json", b"original config\n")
        write_cache_manifest(self.layout, model=MODEL, revision=REVISION)
        manifest_path = self.layout.resolve("manifests/huggingface-model-cache.json")
        original_manifest = manifest_path.read_bytes()
        atomic_write_bytes(cache / "blobs/model.bin", b"changed model bytes\n")
        (cache / "snapshots/revision/config.json").unlink()
        atomic_write_bytes(cache / "refs/pr/1", b"converted-revision\n")
        for validate in (write_cache_manifest, verify_cache_manifest):
            with self.subTest(validate=validate.__name__), self.assertRaises(DataContractError) as raised:
                validate(self.layout, model=MODEL, revision=REVISION)
            message = str(raised.exception)
            self.assertIn("added: 'models--fixture/refs/pr/1'", message)
            self.assertIn("removed: 'models--fixture/snapshots/revision/config.json'", message)
            self.assertIn("changed: 'models--fixture/blobs/model.bin'", message)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)

    def test_inventory_diagnostic_is_bounded_for_many_added_files(self):
        cache = self.layout.resolve("inputs/huggingface/models--fixture")
        atomic_write_bytes(cache / "blobs/model.bin", b"model bytes\n")
        write_cache_manifest(self.layout, model=MODEL, revision=REVISION)
        for index in range(20):
            atomic_write_bytes(cache / f"refs/pr/{index:02}", b"converted-revision\n")
        with self.assertRaises(DataContractError) as raised:
            verify_cache_manifest(self.layout, model=MODEL, revision=REVISION)
        message = str(raised.exception)
        self.assertIn("(20 total)", message)
        self.assertIn("refs/pr/04", message)
        self.assertNotIn("refs/pr/05", message)


if __name__ == "__main__":
    unittest.main()
