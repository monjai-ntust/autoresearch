"""Regression tests for the frozen secondary-evidence reconciliation stage."""

from __future__ import annotations

import shutil
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from config import load_pipeline_config
from phase_b_io import DataContractError, atomic_write_json, load_json
from paths import RunLayout, discover_source_root
from reconciliation import audit_ledger, reconcile_section5_evidence


SOURCE_ROOT = discover_source_root(Path(__file__))


@contextmanager
def _temporary_run():
    run_id = f"test-reconcile-{uuid.uuid4().hex}"
    layout = RunLayout(SOURCE_ROOT, run_id)
    layout.create()
    try:
        atomic_write_json(
            layout.resolve("manifests/00-checkout-manifest.json"),
            {"status": "pass"},
        )
        yield layout
    finally:
        shutil.rmtree(layout.run_root)


class Section5ReconciliationTests(unittest.TestCase):
    def test_frozen_ledgers_reconcile_without_canonical_promotion(self):
        config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")
        with _temporary_run() as layout:
            audit = reconcile_section5_evidence(layout, config)
            self.assertEqual(audit["status"], "reconciled_secondary_evidence")
            self.assertFalse(audit["canonical_publication_eligible"])
            self.assertFalse(audit["publication_execution_admitted"])
            self.assertEqual(audit["summary"]["claim_family_count"], 7)
            self.assertEqual(audit["summary"]["canonical_ready_claim_family_count"], 0)

            by_path = {item["path"]: item for item in audit["ledgers"]}
            primary = by_path["results.tsv"]
            self.assertEqual(primary["blank_lines"], [57, 61])
            self.assertEqual(primary["concatenated_record_lines"], [58, 62])
            self.assertEqual(
                primary["normalized_lf_sha256"],
                "f862fa3b63fce17187990337909016176dd9a8527988b5c0da4bd33bb25ccba5",
            )
            self.assertIn(primary["newline_style"], ("lf", "crlf"))

            persisted = load_json(
                layout.resolve(
                    "audit/section5-evidence-reconciliation.json", must_exist=True
                )
            )
            self.assertEqual(persisted, audit)
            with self.assertRaises(DataContractError):
                reconcile_section5_evidence(layout, config)

    def test_ledger_content_drift_fails_closed(self):
        config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")
        contract = config.section5_evidence["ledgers"][0]
        output_root = SOURCE_ROOT / "output"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-ledger-drift-", dir=output_root) as temp:
            root = Path(temp)
            original = (SOURCE_ROOT / "results.tsv").read_bytes()
            mutated = original.replace(b"DeBERTa-large", b"DeBERTa-Large", 1)
            self.assertNotEqual(mutated, original)
            (root / "results.tsv").write_bytes(mutated)
            with self.assertRaisesRegex(DataContractError, "normalized LF"):
                audit_ledger(root, contract)


if __name__ == "__main__":
    unittest.main()
