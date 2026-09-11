import argparse
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "phase_e_rag_contract.json").read_text(encoding="utf-8"))


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "phase_e_runner", ROOT / "run_phase_e_rag.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhaseERunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_source_contract_is_valid(self):
        result = self.runner.validate_source_contract(CONTRACT, require_clean=False)
        self.assertEqual(result["branch"], "publication-legacy-methodology")
        self.assertEqual(result["baseline"], CONTRACT["source_baseline"]["commit"])

    def test_run_id_cannot_escape_output(self):
        for value in ("../escape", "nested/run", ".", "-leading"):
            with self.subTest(value=value):
                with self.assertRaises(self.runner.PhaseEError):
                    self.runner.validate_run_id(value)

    def test_model_manifest_uses_canonical_json_identity(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"z":2,"a":1}'
        with mock.patch.object(
            self.runner.urllib.request, "urlopen", return_value=response
        ) as urlopen:
            manifest, canonical, digest = self.runner.fetch_model_manifest(
                "http://localhost:11434"
            )

        self.assertEqual(manifest, {"z": 2, "a": 1})
        self.assertEqual(canonical, b'{"a":1,"z":2}\n')
        self.assertEqual(digest, hashlib.sha256(canonical).hexdigest())
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/show")
        self.assertEqual(request.data, b'{"model":"qwen3:32b"}\n')

    def test_dry_run_validates_without_model_call_or_output(self):
        record = {
            "doc_id": 1,
            "sentence": "Head is part of Tail.",
            "gold_triples": [
                {"head_text": "Head", "tail_text": "Tail", "relation": "part-of"}
            ],
            "predicted_triples": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "inference.jsonl"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            test_contract = copy.deepcopy(CONTRACT)
            test_contract["artifacts"]["inference"] = {
                "sha256": self.runner.sha256_file(input_path),
                "records": 1,
            }
            test_contract["evaluator"]["max_questions"] = 1
            run_id = "phase-e-dry-run-unit"
            run_dir = self.runner.OUTPUT_ROOT / run_id
            self.assertFalse(run_dir.exists())
            args = argparse.Namespace(
                run_id=run_id,
                inference=input_path,
                confidence_graph=None,
                verified_graph=None,
                block_verified=True,
                verified_graph_sha256=None,
                ollama_url="http://localhost:11434",
                ollama_show_sha256=None,
                dry_run=True,
            )
            output = io.StringIO()
            with mock.patch.object(
                self.runner,
                "validate_source_contract",
                return_value={
                    "head": "test-head",
                    "branch": "publication-legacy-methodology",
                    "baseline": "test-baseline",
                },
            ):
                with mock.patch.object(
                    self.runner,
                    "validate_environment",
                    return_value={"python": "test", "curl": "test"},
                ):
                    with mock.patch.object(self.runner, "fetch_model_manifest") as model:
                        with redirect_stdout(output):
                            result = self.runner.run_command(args, test_contract)

            self.assertEqual(result, 0)
            model.assert_not_called()
            self.assertIn('"verified": "blocked-by-explicit-flag"', output.getvalue())
            self.assertFalse(run_dir.exists())

    def test_remote_ollama_endpoint_is_rejected(self):
        with self.assertRaisesRegex(self.runner.PhaseEError, "only a loopback"):
            self.runner.fetch_model_manifest("https://example.com:11434")

    def test_rag_output_validation_rejects_failed_model_calls(self):
        modes = CONTRACT["evaluator"]["modes"]
        count = CONTRACT["evaluator"]["max_questions"]
        output = {
            "metadata": {"n_questions": count, "model": "qwen3:32b"},
            "accuracy": {mode: 0.0 for mode in modes},
            "correct_counts": {mode: 0 for mode in modes},
            "results": {
                mode: [
                    {"q": "q", "pred": "answer", "gold": "gold", "correct": False}
                    for _ in range(count)
                ]
                for mode in modes
            },
        }
        output["results"]["hybrid"][0]["pred"] = "ERROR: timeout"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps(output), encoding="utf-8")
            with self.assertRaisesRegex(
                self.runner.PhaseEError, "contains failed model calls"
            ):
                self.runner.validate_rag_output(path, CONTRACT)

    def test_confidence_graph_metadata_is_frozen(self):
        graph_contract = CONTRACT["artifacts"]["confidence_graph"]
        graph = {
            "metadata": dict(graph_contract["metadata"]),
            "nodes": [{"id": f"node-{index}"} for index in range(58)],
            "edges": [
                {"head": "head", "relation": "part-of", "tail": "tail"}
                for _ in range(39)
            ],
        }
        graph["metadata"]["conf_threshold"] = 0.5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            path.write_text(json.dumps(graph), encoding="utf-8")
            with self.assertRaisesRegex(
                self.runner.PhaseEError, "Graph metadata mismatch"
            ):
                self.runner.validate_graph(
                    path,
                    graph_contract["nodes"],
                    graph_contract["edges"],
                    graph_contract["metadata"],
                )


if __name__ == "__main__":
    unittest.main()
