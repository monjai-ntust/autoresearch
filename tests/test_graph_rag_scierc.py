import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from graph_rag_eval.adapters.scierc import (
    ENTITY_TYPES,
    RELATION_TYPES,
    SciERCAdapter,
)
from graph_rag_eval.evaluation.extraction import evaluate_extraction
from graph_rag_eval.graphs.matching import match_graphs
from graph_rag_eval.graphs.snapshots import build_snapshot
from graph_rag_eval.registry import load_adapter
from graph_rag_eval.runner import create_context, doctor


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _fixture_rows():
    return {
        "train": [
            {
                "doc_key": "TRAIN-1",
                "sentences": [["Neural", "models", "use", "attention", "."]],
                "ner": [[[0, 1, "Method"], [3, 3, "Method"]]],
                "relations": [[[0, 1, 3, 3, "USED-FOR"]]],
                "clusters": [],
            }
        ],
        "dev": [
            {
                "doc_key": "DEV-1",
                "sentences": [["Accuracy", "is", "a", "metric", "."]],
                "ner": [[[0, 0, "Metric"]]],
                "relations": [[]],
                "clusters": [],
            }
        ],
        "test": [
            {
                "doc_key": "TEST-1",
                "sentences": [["Systems", "compare", "metrics", "."]],
                "ner": [[[0, 0, "Task"], [2, 2, "Metric"]]],
                "relations": [[[0, 0, 2, 2, "COMPARE"]]],
                "clusters": [],
            }
        ],
    }


def _prediction_rows():
    return [
        {
            "schema_version": "rag-extraction-prediction-1.0",
            "record_type": "entity",
            "doc_key": "TEST-1",
            "start": 0,
            "end": 0,
            "entity_type": "Task",
            "confidence": 0.95,
        },
        {
            "schema_version": "rag-extraction-prediction-1.0",
            "record_type": "entity",
            "doc_key": "TEST-1",
            "start": 2,
            "end": 2,
            "entity_type": "Metric",
            "confidence": 0.90,
        },
        {
            "schema_version": "rag-extraction-prediction-1.0",
            "record_type": "entity",
            "doc_key": "TEST-1",
            "start": 1,
            "end": 1,
            "entity_type": "Method",
            "confidence": 0.40,
        },
        {
            "schema_version": "rag-extraction-prediction-1.0",
            "record_type": "relation",
            "doc_key": "TEST-1",
            "sentence_index": 0,
            "head": {"start": 2, "end": 2, "entity_type": "Metric"},
            "tail": {"start": 0, "end": 0, "entity_type": "Task"},
            "relation_type": "COMPARE",
            "confidence": 0.88,
        },
        {
            "schema_version": "rag-extraction-prediction-1.0",
            "record_type": "relation",
            "doc_key": "TEST-1",
            "sentence_index": 0,
            "head": {"start": 0, "end": 0, "entity_type": "Task"},
            "tail": {"start": 1, "end": 1, "entity_type": "Method"},
            "relation_type": "USED-FOR",
            "confidence": 0.35,
        },
    ]


class SciERCAdapterTests(unittest.TestCase):
    def _make_fixture(self, temporary, *, predictions=True, rows=None):
        run_root = Path(temporary) / "graph-rag"
        data_root = run_root / "inputs" / "scierc"
        split_rows = rows or _fixture_rows()
        hashes = {}
        counts = {}
        for split, values in split_rows.items():
            path = data_root / f"{split}.json"
            _write_jsonl(path, values)
            hashes[f"{split}.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
            counts[split] = len(values)
        if predictions:
            _write_jsonl(
                run_root / "inputs" / "scierc-predictions.jsonl",
                _prediction_rows(),
            )
        adapter = SciERCAdapter(
            run_root=str(run_root),
            fixture_mode=True,
            expected_split_sha256=hashes,
            expected_document_counts=counts,
        )
        return adapter

    def test_native_mapping_preserves_splits_mentions_relations_and_no_qa(self):
        output = ROOT / "output"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            adapter = self._make_fixture(temporary)
            report = adapter.validate()
            self.assertTrue(report.ready, report.issues)
            self.assertIn("predicted_graph", report.capabilities)
            bundle = adapter.load()
            self.assertEqual(len(bundle.documents), 3)
            self.assertEqual(len(bundle.chunks), 3)
            self.assertEqual(len(bundle.entities), 5)
            self.assertEqual(len(bundle.relations), len(RELATION_TYPES))
            self.assertEqual(
                next(item for item in bundle.relations if item.label == "COMPARE").direction,
                "undirected",
            )
            self.assertEqual(len(bundle.triples), 2)
            self.assertEqual(bundle.questions, ())
            self.assertEqual(
                {item.split_name for item in bundle.split_membership},
                {"train", "dev", "test"},
            )
            self.assertEqual(
                {item.entity_type for item in bundle.entities},
                {"Method", "Metric", "Task"},
            )
            neural = next(
                item for item in bundle.entities if item.surface_forms == ("Neural models",)
            )
            self.assertEqual(neural.source_spans[0][1:], (0, 13))
            predicted = adapter.generated_graph(bundle)
            self.assertEqual(len(predicted.entities), 3)
            self.assertEqual(len(predicted.triples), 2)

    def test_exact_extraction_reports_false_positives_and_missing_gold(self):
        output = ROOT / "output"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            adapter = self._make_fixture(temporary)
            bundle = adapter.load()
            predicted = adapter.generated_graph(bundle)
            descriptor_hash = "0" * 64
            corpus_hash = "1" * 64
            adapter_hash = "2" * 64
            gold = build_snapshot(
                dataset_id=bundle.descriptor.dataset_id,
                condition="gold",
                entities=bundle.entities,
                relations=bundle.relations,
                triples=bundle.triples,
                descriptor_sha256=descriptor_hash,
                corpus_sha256=corpus_hash,
                adapter_sha256=adapter_hash,
                extractor_sha256="3" * 64,
                construction_recipe="fixture-gold",
            )
            generated = build_snapshot(
                dataset_id=bundle.descriptor.dataset_id,
                condition="generated",
                entities=predicted.entities,
                relations=predicted.relations,
                triples=predicted.triples,
                descriptor_sha256=descriptor_hash,
                corpus_sha256=corpus_hash,
                adapter_sha256=adapter_hash,
                extractor_sha256="4" * 64,
                construction_recipe="fixture-predicted",
            )
            metrics = evaluate_extraction(generated, gold)
            intrinsic = match_graphs(generated, gold)
            self.assertEqual(intrinsic.triples.true_positive, 1)
            self.assertEqual(
                metrics["entity"]["micro"]["true_positive"],
                2,
            )
            self.assertEqual(metrics["entity"]["micro"]["false_positive"], 1)
            self.assertEqual(metrics["entity"]["micro"]["false_negative"], 3)
            self.assertEqual(
                metrics["end_to_end_relation"]["micro"]["true_positive"],
                1,
            )
            self.assertEqual(
                metrics["end_to_end_relation"]["micro"]["false_positive"],
                1,
            )
            self.assertEqual(
                metrics["end_to_end_relation"]["micro"]["false_negative"],
                1,
            )

    def test_missing_predictions_is_blocked_not_a_zero_result(self):
        output = ROOT / "output"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            adapter = self._make_fixture(temporary, predictions=False)
            report = adapter.validate()
            self.assertTrue(report.ready)
            self.assertIn(
                "missing-scierc-predictions",
                {issue.code for issue in report.issues},
            )
            self.assertNotIn("predicted_graph", report.capabilities)
            bundle = adapter.load()
            self.assertEqual(adapter.generated_graph(bundle).triples, ())

    def test_overlap_hash_override_and_path_escape_fail_closed(self):
        output = ROOT / "output"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            rows = _fixture_rows()
            rows["dev"][0]["doc_key"] = "TRAIN-1"
            adapter = self._make_fixture(temporary, rows=rows)
            self.assertFalse(adapter.validate().ready)
        with self.assertRaisesRegex(ValueError, "cannot override official"):
            SciERCAdapter(
                run_root=str(output / "official"),
                expected_split_sha256={
                    "train.json": "0" * 64,
                    "dev.json": "1" * 64,
                    "test.json": "2" * 64,
                },
            )
        with self.assertRaisesRegex(ValueError, "traversal-free"):
            SciERCAdapter(
                run_root=str(output / "escape"),
                data_root="../outside",
            )

    def test_registry_loads_scierc_without_core_dataset_branch(self):
        output = ROOT / "output"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            adapter = self._make_fixture(temporary)
            loaded = load_adapter(
                "graph_rag_eval.adapters.scierc:SciERCAdapter",
                {
                    "run_root": str(adapter.run_root),
                    "fixture_mode": True,
                    "expected_split_sha256": adapter.expected_split_sha256,
                    "expected_document_counts": adapter.expected_document_counts,
                },
            )
            self.assertEqual(loaded.adapter_id, "scierc-native-extraction")
            self.assertEqual(tuple(ENTITY_TYPES), ENTITY_TYPES)
            self.assertTrue(loaded.validate().ready)

    def test_tracked_readiness_config_is_blocked_without_data_or_checkpoint(self):
        run_id = "unittest-scierc-readiness"
        run_root = ROOT / "output" / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        try:
            context = create_context(
                "configs/phase_d_graph_rag_scierc.json",
                run_id=run_id,
            )
            result = doctor(context)
            self.assertEqual(result["status"], "blocked")
            codes = {item["code"] for item in result["gates"]}
            self.assertIn("missing-scierc-split", codes)
            self.assertIn("checkpoint-not-ready", codes)
            checkpoint = json.loads(
                (
                    context.run_root
                    / "manifests"
                    / "checkpoint.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["strategy"], "dataset-specific")
            self.assertEqual(
                checkpoint["base_model"]["revision"],
                "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
            )
        finally:
            if run_root.exists():
                shutil.rmtree(run_root)


if __name__ == "__main__":
    unittest.main()
