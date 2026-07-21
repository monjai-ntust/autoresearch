"""Regression tests for the frozen secondary-evidence reconciliation stage."""

from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from config import load_pipeline_config
from phase_b_io import DataContractError, atomic_write_json, load_json
from paths import RunLayout, discover_source_root
from reconciliation import reconcile_section5_evidence


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
    def test_archived_ledgers_reconcile_without_runtime_or_canonical_promotion(self):
        config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")
        with _temporary_run() as layout:
            audit = reconcile_section5_evidence(layout, config)
            self.assertEqual(audit["status"], "reconciled_secondary_evidence")
            self.assertFalse(audit["canonical_publication_eligible"])
            self.assertFalse(audit["publication_execution_admitted"])
            self.assertEqual(
                audit["publication_execution_gate"],
                "mandatory-passing-b07-pilot-audit",
            )
            self.assertEqual(audit["summary"]["claim_family_count"], 7)
            self.assertEqual(audit["summary"]["canonical_ready_claim_family_count"], 0)
            self.assertTrue(audit["summary"]["historical_ledger_source_files_removed"])
            self.assertTrue(
                audit["summary"]["archived_facts_available_without_external_archive"]
            )

            by_path = {item["path"]: item for item in audit["ledgers"]}
            primary = by_path["results.tsv"]
            self.assertEqual(primary["expected_structure"]["blank_lines"], [57, 61])
            self.assertEqual(
                primary["expected_structure"]["concatenated_record_lines"],
                [58, 62],
            )
            self.assertEqual(
                primary["normalized_lf_sha256"],
                "f862fa3b63fce17187990337909016176dd9a8527988b5c0da4bd33bb25ccba5",
            )
            self.assertEqual(
                primary["source_checkout_sha256"],
                "abef9a3e2bcc344abff8ed1e03b0ab6831d230052f9a49749bb770e41d3f8852",
            )
            self.assertEqual(primary["archival_status"], "hash_verified_external_archive")
            self.assertFalse(primary["checkout_dependency"])

            persisted = load_json(
                layout.resolve(
                    "audit/section5-evidence-reconciliation.json", must_exist=True
                )
            )
            self.assertEqual(persisted, audit)
            with self.assertRaises(DataContractError):
                reconcile_section5_evidence(layout, config)

    def test_archived_register_digest_drift_fails_closed(self):
        config = load_pipeline_config(SOURCE_ROOT, "configs/phase_b_path_a.json")
        config.section5_evidence["ledgers"][0]["source_checkout_sha256"] = "not-a-digest"
        with _temporary_run() as layout:
            with self.assertRaisesRegex(DataContractError, "source SHA-256"):
                reconcile_section5_evidence(layout, config)


if __name__ == "__main__":
    unittest.main()
