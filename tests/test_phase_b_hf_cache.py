"""Focused tests for the run-local Hugging Face cache contract."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from hf_cache import verify_cache_manifest, write_cache_manifest
from phase_b_io import DataContractError, atomic_write_bytes, sha256_file
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

    def _create_symlink_or_skip(self, link: Path, target: str | Path) -> None:
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")

    def test_manifest_hashes_contained_relative_symlink_target_bytes(self):
        blob = self.layout.resolve(
            "inputs/huggingface/models--fixture/blobs/model.bin"
        )
        atomic_write_bytes(blob, b"model bytes\n")
        link = self.layout.run_root / (
            "inputs/huggingface/models--fixture/snapshots/revision/model.bin"
        )
        relative_target = str(Path("..") / ".." / "blobs" / "model.bin")
        self._create_symlink_or_skip(link, relative_target)

        manifest = write_cache_manifest(self.layout, model=MODEL, revision=REVISION)
        record = next(row for row in manifest["files"] if row["kind"] == "symlink")
        self.assertEqual(record["target"], relative_target)
        self.assertEqual(record["bytes"], blob.stat().st_size)
        self.assertEqual(record["sha256"], sha256_file(blob))
        self.assertEqual(
            verify_cache_manifest(self.layout, model=MODEL, revision=REVISION),
            manifest,
        )

    def test_absolute_symlink_target_fails_closed(self):
        blob = self.layout.resolve(
            "inputs/huggingface/models--fixture/blobs/model.bin"
        )
        atomic_write_bytes(blob, b"model bytes\n")
        link = self.layout.run_root / "inputs/huggingface/absolute-model.bin"
        self._create_symlink_or_skip(link, blob.resolve())
        with self.assertRaisesRegex(DataContractError, "target is absolute"):
            write_cache_manifest(self.layout, model=MODEL, revision=REVISION)

    def test_escaping_symlink_target_fails_closed(self):
        outside = self.layout.resolve("outside-cache.bin")
        atomic_write_bytes(outside, b"outside bytes\n")
        link = self.layout.run_root / "inputs/huggingface/escaping-model.bin"
        self._create_symlink_or_skip(link, "../../outside-cache.bin")
        with self.assertRaisesRegex(DataContractError, "escapes the cache root"):
            write_cache_manifest(self.layout, model=MODEL, revision=REVISION)

    def test_broken_symlink_target_fails_closed(self):
        link = self.layout.run_root / "inputs/huggingface/broken-model.bin"
        self._create_symlink_or_skip(link, "missing-model.bin")
        with self.assertRaisesRegex(DataContractError, "broken or escapes"):
            write_cache_manifest(self.layout, model=MODEL, revision=REVISION)


if __name__ == "__main__":
    unittest.main()
