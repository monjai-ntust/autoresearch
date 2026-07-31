import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from graph_rag_eval.runner import (
    ConfigError,
    RunIdentityError,
    create_context,
    doctor,
    evaluate,
    prepare,
)


ROOT = Path(__file__).resolve().parents[1]


class GraphRagBackwardCompatibilityTests(unittest.TestCase):
    run_ids = (
        "unittest-graph-rag-identity-resume",
        "unittest-graph-rag-identity-occupied",
        "unittest-graph-rag-intrinsic-only",
        "unittest-graph-rag-existing-parent",
    )

    def tearDown(self):
        for run_id in self.run_ids:
            path = ROOT / "output" / run_id
            if path.exists():
                shutil.rmtree(path)

    def test_legacy_config_bytes_remain_frozen_and_loadable(self):
        expected = {
            "phase_b_graph_rag_synthetic.json": (
                "9f4399e91f8c53fb93d903f8107476ba2c48085115ffe8ab91786019ebd22bd7"
            ),
            "phase_b_graph_rag_code_accord.json": (
                "9001ade53ffa38ee17375c5c357ef5c301abd6c2b9818d13ccb7602683386ecc"
            ),
        }
        for name, digest in expected.items():
            with self.subTest(name=name):
                path = ROOT / "configs" / name
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                context = create_context(path)
                self.assertEqual(context.config["schema_version"], "rag-run-config-1.0")

    def test_matching_stage_resume_succeeds_and_changed_config_fails_closed(self):
        run_id = self.run_ids[0]
        context = create_context(
            "configs/phase_b_graph_rag_synthetic.json",
            run_id=run_id,
        )
        self.assertEqual(doctor(context)["status"], "ready")
        identity_path = (
            ROOT
            / "output"
            / run_id
            / "graph-rag"
            / "manifests"
            / "run-identity.json"
        )
        original_identity = identity_path.read_bytes()
        self.assertEqual(prepare(context)["status"], "prepared")
        self.assertEqual(identity_path.read_bytes(), original_identity)

        output = ROOT / "output"
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            changed_path = Path(temporary) / "changed-config.json"
            changed = json.loads(
                (ROOT / "configs/phase_b_graph_rag_synthetic.json").read_text(
                    encoding="utf-8"
                )
            )
            changed["adapter"]["options"]["variant"] = "identity-mismatch"
            changed_path.write_text(
                json.dumps(changed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            changed_context = create_context(changed_path, run_id=run_id)
            with self.assertRaisesRegex(RunIdentityError, "choose a new run ID"):
                doctor(changed_context)
        self.assertEqual(identity_path.read_bytes(), original_identity)

    def test_occupied_unbound_run_id_is_rejected_before_writes(self):
        run_root = ROOT / "output" / self.run_ids[1] / "graph-rag"
        run_root.mkdir(parents=True)
        sentinel = run_root / "user-owned.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        context = create_context(
            "configs/phase_b_graph_rag_synthetic.json",
            run_id=self.run_ids[1],
        )
        with self.assertRaisesRegex(RunIdentityError, "no Graph RAG identity"):
            doctor(context)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_fresh_graph_rag_child_can_coexist_with_existing_parent_files(self):
        parent = ROOT / "output" / self.run_ids[3]
        parent.mkdir(parents=True)
        sentinel = parent / "phase-b-parent.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        context = create_context(
            "configs/phase_b_graph_rag_synthetic.json",
            run_id=self.run_ids[3],
        )
        self.assertEqual(doctor(context)["status"], "ready")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertTrue(
            (parent / "graph-rag/manifests/run-identity.json").is_file()
        )

    def test_intrinsic_only_scope_emits_exact_extraction_without_qa(self):
        output = ROOT / "output"
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            config_path = Path(temporary) / "intrinsic-config.json"
            config = json.loads(
                (ROOT / "configs/phase_b_graph_rag_synthetic.json").read_text(
                    encoding="utf-8"
                )
            )
            config["protocol"]["evaluation_scope"] = "intrinsic_graph_only"
            config["protocol"]["required_capabilities"] = [
                "documents",
                "gold_entities",
                "gold_relations",
                "gold_triples",
                "predicted_graph",
            ]
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            context = create_context(config_path, run_id=self.run_ids[2])
            result = evaluate(context)
        self.assertEqual(result["status"], "intrinsic_evaluation_complete")
        metrics_path = (
            ROOT
            / "output"
            / self.run_ids[2]
            / "graph-rag"
            / "metrics"
            / "aggregate.jsonl"
        )
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        extraction = next(row for row in rows if row["metric"] == "exact_extraction")
        self.assertGreater(
            extraction["value"]["entity"]["micro"]["false_negative"],
            0,
        )
        self.assertGreater(
            extraction["value"]["end_to_end_relation"]["micro"]["false_positive"],
            0,
        )
        self.assertTrue(
            (
                ROOT
                / "output"
                / self.run_ids[2]
                / "graph-rag"
                / "tables"
                / "extraction-results.tsv"
            ).is_file()
        )

    def test_ready_checkpoint_requires_hash_and_no_blockers_before_run_claim(self):
        output = ROOT / "output"
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            config_path = Path(temporary) / "invalid-checkpoint.json"
            config = json.loads(
                (ROOT / "configs/phase_b_graph_rag_synthetic.json").read_text(
                    encoding="utf-8"
                )
            )
            config["checkpoint"] = {
                "schema_version": "rag-checkpoint-manifest-1.0",
                "strategy": "dataset-specific",
                "status": "ready",
                "checkpoint_id": "invalid-no-hash",
                "checkpoint_sha256": None,
                "architecture": "fixture",
                "base_model": {"model_id": "fixture", "revision": "sha256:fixture"},
                "tokenizer": {
                    "tokenizer_id": "fixture",
                    "revision": "sha256:fixture",
                },
                "training": {
                    "dataset_id": "fixture",
                    "split_id": "fixture",
                    "recipe_id": "fixture",
                    "seed": 42,
                    "selection_metric": "fixture",
                },
                "label_vocabulary": {
                    "entity_types": ["Fixture"],
                    "relation_types": ["fixture"],
                },
                "blocked_reasons": [],
            }
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "exact SHA-256"):
                create_context(config_path, run_id="unittest-invalid-checkpoint")
        self.assertFalse((output / "unittest-invalid-checkpoint").exists())


if __name__ == "__main__":
    unittest.main()
