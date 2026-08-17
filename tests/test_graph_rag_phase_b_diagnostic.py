from __future__ import annotations

from dataclasses import replace
import json
import shutil
import unittest
from pathlib import Path

from graph_rag_eval.phase_b_artifacts import (
    PhaseBArtifactError,
    load_phase_b_run,
    strict_counts,
)
from graph_rag_eval.phase_b_diagnostic import (
    PhaseBDiagnosticError,
    _bind_child,
    _build_canonical,
    _retrieval_matrix,
    build_phase_b_graphs,
    create_phase_b_diagnostic_context,
    recompute_diagnostic_metrics,
    run_phase_b_diagnostic,
)
from tests.phase_b_diagnostic_fixture import (
    build_phase_b_run,
    write_test_config,
)


ROOT = Path(__file__).resolve().parents[1]


class PhaseBGraphRagDiagnosticTests(unittest.TestCase):
    run_id = "unittest-phase-b-graph-rag"

    def setUp(self):
        self.output = ROOT / "output"
        self.output.mkdir(exist_ok=True)
        self.parent = self.output / self.run_id
        self.config_path = self.output / f"{self.run_id}-config.json"
        if self.parent.exists():
            shutil.rmtree(self.parent)
        self.config_path.unlink(missing_ok=True)
        self.fixture = build_phase_b_run(self.parent, self.run_id)
        write_test_config(ROOT, self.config_path)
        self.context = create_phase_b_diagnostic_context(
            self.config_path,
            run_id=self.run_id,
        )

    def tearDown(self):
        if self.parent.exists():
            shutil.rmtree(self.parent)
        self.config_path.unlink(missing_ok=True)

    def _load(self):
        return load_phase_b_run(
            self.parent,
            self.context.config,
            run_id=self.run_id,
        )

    def test_phase_b_ingestion_and_condition_mapping_recompute_exactly(self):
        parent = self._load()
        self.assertEqual(parent.seeds, tuple(range(42, 50)))
        self.assertEqual(parent.selected_threshold, 0.5)
        self.assertEqual(len(parent.verified_artifacts), 18)
        counts = strict_counts(parent.emitted, parent.gold)
        for condition, expected in self.fixture["expected_counts"].items():
            self.assertEqual(counts[condition][42], expected)
        self.assertEqual(
            counts["VER-CONFIDENCE"][42],
            {"tp": 2, "fp": 1, "fn": 0},
        )
        self.assertEqual(
            counts["VER-SIMPLE"][42],
            {"tp": 2, "fp": 0, "fn": 0},
        )
        self.assertEqual(
            counts["VER-CORRECTIVE"][42],
            {"tp": 1, "fp": 0, "fn": 1},
        )

    def test_parent_hash_mismatch_refuses_before_child_creation(self):
        candidates = self.parent / "predictions/test/candidates.jsonl"
        candidates.write_bytes(candidates.read_bytes() + b" ")
        with self.assertRaisesRegex(PhaseBArtifactError, "hash mismatch"):
            self._load()
        self.assertFalse((self.parent / "graph-rag").exists())

    def test_child_is_confined_and_existing_parent_namespace_is_supported(self):
        self.assertEqual(self.context.child_root, self.parent / "graph-rag")
        parent_bytes = {
            path.relative_to(self.parent).as_posix(): path.read_bytes()
            for path in self.parent.rglob("*")
            if path.is_file()
        }
        with self.assertRaisesRegex(PhaseBDiagnosticError, "invalid Phase B run ID"):
            create_phase_b_diagnostic_context(
                self.config_path,
                run_id="../escape",
            )
        dotted = create_phase_b_diagnostic_context(
            self.config_path,
            run_id="valid.phase-b-run",
        )
        self.assertEqual(
            dotted.child_root,
            self.output / "valid.phase-b-run/graph-rag",
        )
        result = run_phase_b_diagnostic(self.context)
        self.assertEqual(result["status"], "diagnostic_complete")
        self.assertTrue(
            (self.parent / "graph-rag/manifests/complete.json").is_file()
        )
        self.assertTrue(
            (self.parent / "manifests/score-manifest.json").is_file()
        )
        public_probe = json.loads(
            (
                self.parent / "graph-rag/canonical/probes-public.jsonl"
            ).read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertNotIn("cluster_id", public_probe)
        self.assertNotIn("source_document_id", public_probe)
        self.assertNotIn("evidence_id", public_probe)
        for relative, payload in parent_bytes.items():
            self.assertEqual(
                (self.parent / relative).read_bytes(),
                payload,
                relative,
            )

    def test_partial_child_resumes_and_completed_child_reverifies(self):
        parent = self._load()
        _bind_child(self.context, parent)
        self.assertFalse(
            (self.parent / "graph-rag/manifests/complete.json").exists()
        )
        first = run_phase_b_diagnostic(self.context)
        self.assertEqual(first["resume_status"], "completed_or_resumed_partial")
        before = (
            self.parent / "graph-rag/manifests/artifacts.json"
        ).read_bytes()
        second = run_phase_b_diagnostic(self.context)
        self.assertEqual(second["resume_status"], "already_complete_verified")
        self.assertEqual(
            before,
            (self.parent / "graph-rag/manifests/artifacts.json").read_bytes(),
        )

    def test_completed_child_content_drift_is_refused(self):
        run_phase_b_diagnostic(self.context)
        table = self.parent / "graph-rag/tables/diagnostic-summary.tsv"
        table.write_bytes(table.read_bytes() + b"changed\n")
        with self.assertRaisesRegex(
            PhaseBDiagnosticError,
            "completed Graph RAG artifact is missing or changed",
        ):
            run_phase_b_diagnostic(self.context)

    def test_completed_child_rejects_unmanifested_file(self):
        run_phase_b_diagnostic(self.context)
        extra = self.parent / "graph-rag/unmanifested.txt"
        extra.write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(
            PhaseBDiagnosticError,
            "child file set differs",
        ):
            run_phase_b_diagnostic(self.context)

    def test_graph_construction_is_deterministic_and_gold_is_private_oracle(self):
        parent = self._load()
        canonical = _build_canonical(parent, self.context.config)
        gold_a, predicted_a = build_phase_b_graphs(
            parent, canonical, self.context.config
        )
        gold_b, predicted_b = build_phase_b_graphs(
            parent, canonical, self.context.config
        )
        self.assertEqual(gold_a.graph_id, gold_b.graph_id)
        self.assertEqual(
            predicted_a["confidence_filtered"][42].graph_id,
            predicted_b["confidence_filtered"][42].graph_id,
        )
        self.assertNotEqual(
            gold_a.graph_id,
            predicted_a["confidence_filtered"][42].graph_id,
        )
        self.assertEqual(
            len(predicted_a["confidence_filtered"][42].triples),
            3,
        )
        private_rebound = replace(
            parent,
            score_manifest_sha256="f" * 64,
            verified_artifacts=tuple(
                replace(item, sha256="e" * 64)
                if item.path == self.context.config["artifacts"]["private_gold"]
                else item
                for item in parent.verified_artifacts
            ),
        )
        gold_private, predicted_private = build_phase_b_graphs(
            private_rebound,
            canonical,
            self.context.config,
        )
        self.assertEqual(
            predicted_a["confidence_filtered"][42].graph_id,
            predicted_private["confidence_filtered"][42].graph_id,
        )
        self.assertNotEqual(gold_a.graph_id, gold_private.graph_id)

    def test_probe_queries_exclude_private_target_and_leakage_sentinel_fails_closed(self):
        parent = self._load()
        canonical = _build_canonical(parent, self.context.config)
        targets = {item.question.question_id: item for item in canonical.probes}
        for question_id, target in targets.items():
            query = target.question.public_view()
            normalized_query = " ".join(query.text.casefold().split())
            self.assertEqual(
                dict(query.public_metadata),
                {"template_id": self.context.config["probe"]["template_id"]},
            )
            for private in (
                target.tail,
                target.relation,
                target.source_document_id,
                target.chunk_id,
                target.gold_triple_id,
            ):
                self.assertNotIn(
                    " ".join(private.casefold().split()),
                    normalized_query,
                    question_id,
                )
        gold, predicted = build_phase_b_graphs(
            parent, canonical, self.context.config
        )
        target = canonical.probes[0]
        # Regression for the historical target `met`: scanning serialized JSON
        # produced a false positive from the structural key `public_metadata`.
        incidental_json_key_collision = replace(target, tail="met")
        _retrieval_matrix(
            parent,
            replace(canonical, probes=(incidental_json_key_collision,)),
            gold,
            predicted,
            self.context.config,
        )
        leaked = replace(
            target,
            question=replace(
                target.question,
                text=f"What fact is stated about {target.head}? {target.tail}",
            ),
        )
        with self.assertRaisesRegex(
            PhaseBDiagnosticError, "private target reached tested query"
        ):
            _retrieval_matrix(
                parent,
                replace(canonical, probes=(leaked,)),
                gold,
                predicted,
                self.context.config,
            )

        leaked_metadata = replace(
            target,
            question=replace(
                target.question,
                public_metadata={
                    "template_id": self.context.config["probe"]["template_id"],
                    "answer_hint": target.tail,
                },
            ),
        )
        with self.assertRaisesRegex(
            PhaseBDiagnosticError, "public metadata differs from the fixed template"
        ):
            _retrieval_matrix(
                parent,
                replace(canonical, probes=(leaked_metadata,)),
                gold,
                predicted,
                self.context.config,
            )

    def test_independent_metric_recomputation_preserves_na_denominators(self):
        rows = [
            {
                "condition": "text_bm25",
                "failure": None,
                "metrics": {
                    "fact_preserved": {
                        "status": "not_applicable",
                        "value": None,
                    },
                    "fact_retrieved_at_k": {
                        "status": "not_applicable",
                        "value": None,
                        "reciprocal_rank": None,
                    },
                    "source_support_at_k": {
                        "status": "available",
                        "value": value,
                        "reciprocal_rank": float(value),
                    },
                    "support_or_fact_at_k": {
                        "status": "available",
                        "value": value,
                    },
                },
            }
            for value in (1, 0)
        ]
        aggregate = {
            row["metric"]: row for row in recompute_diagnostic_metrics(rows)
        }
        self.assertNotIn("fact_preserved", aggregate)
        self.assertNotIn("fact_retrieved_at_k", aggregate)
        self.assertEqual(aggregate["source_support_at_k"]["numerator"], 1.0)
        self.assertEqual(aggregate["source_support_at_k"]["denominator"], 2)
        self.assertEqual(aggregate["source_support_at_k"]["value"], 0.5)

    def test_differing_partial_child_artifact_is_never_overwritten(self):
        parent = self._load()
        _bind_child(self.context, parent)
        path = self.parent / "graph-rag/manifests/config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"user":"owned"}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            PhaseBDiagnosticError, "refuses to overwrite differing child artifact"
        ):
            run_phase_b_diagnostic(self.context)
        self.assertEqual(path.read_text(encoding="utf-8"), '{"user":"owned"}\n')


if __name__ == "__main__":
    unittest.main()
