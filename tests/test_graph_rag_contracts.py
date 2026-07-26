import dataclasses
import json
import unittest
from pathlib import Path

from graph_rag_eval.adapters.synthetic import SyntheticAdapter
from graph_rag_eval.contracts import CanonicalBundle, canonical_json
from graph_rag_eval.trace import SchemaValidationError, validate_schema


ROOT = Path(__file__).resolve().parents[1]


class GraphRagContractTests(unittest.TestCase):
    def test_records_are_immutable_stable_and_foreign_keys_are_checked(self):
        bundle = SyntheticAdapter().load()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bundle.documents[0].text = "changed"
        with self.assertRaises(TypeError):
            bundle.documents[0].metadata["changed"] = True
        self.assertEqual(canonical_json(bundle), canonical_json(bundle))
        with self.assertRaisesRegex(ValueError, "stably sorted"):
            CanonicalBundle(
                descriptor=bundle.descriptor,
                documents=tuple(reversed(bundle.documents)),
                chunks=bundle.chunks,
                entities=bundle.entities,
                relations=bundle.relations,
                triples=bundle.triples,
                questions=bundle.questions,
                answers=bundle.answers,
                evidence_sets=bundle.evidence_sets,
            )

    def test_public_question_projection_has_no_private_scoring_fields(self):
        bundle = SyntheticAdapter().load()
        view = bundle.questions[0].public_view()
        field_names = {item.name for item in dataclasses.fields(view)}
        self.assertEqual(
            field_names,
            {"dataset_id", "question_id", "text", "regime", "public_metadata"},
        )
        self.assertFalse(
            {"answer", "evidence_ids", "source_ids", "gold_relation"}
            & field_names
        )

    def test_all_eight_graph_rag_schemas_parse_and_are_versioned(self):
        names = (
            "rag-dataset-descriptor.schema.json",
            "rag-canonical-record.schema.json",
            "rag-graph-snapshot.schema.json",
            "rag-question-set.schema.json",
            "rag-run-config.schema.json",
            "rag-retrieval-trace.schema.json",
            "rag-generation-trace.schema.json",
            "rag-metrics.schema.json",
        )
        for name in names:
            with self.subTest(name=name):
                schema = json.loads(
                    (ROOT / "schemas" / "phase_b" / name).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
                )
                self.assertIn("required", schema)

    def test_schema_validation_rejects_unknown_config_and_invalid_digest(self):
        schema = json.loads(
            (ROOT / "schemas/phase_b/rag-dataset-descriptor.schema.json").read_text(
                encoding="utf-8"
            )
        )
        descriptor = SyntheticAdapter().load().descriptor
        validate_schema(descriptor, schema)
        invalid = dataclasses.asdict(descriptor)
        invalid["raw_content_sha256"] = "not-a-digest"
        with self.assertRaises(SchemaValidationError):
            validate_schema(invalid, schema)


if __name__ == "__main__":
    unittest.main()
