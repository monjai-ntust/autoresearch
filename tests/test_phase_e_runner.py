import argparse
import copy
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


def verifier_manifest(model_name="example-model:tag"):
    return {
        "condition_id": "VERIFIER-CONDITION",
        "execution_mode": "live",
        "protocol_id": "VERIFIER-PROTOCOL",
        "model": {
            "identity_verified": True,
            "name": model_name,
            "tag_digest": "a" * 64,
            "registry_manifest_sha256": "a" * 64,
            "blob_sha256": "b" * 64,
            "model_blob_sha256": "b" * 64,
            "details": {
                "family": "example-family",
                "parameter_size": "12.3B",
                "quantization_level": "Q4_EXAMPLE",
            },
        },
    }


def mocked_http_response(value):
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(value).encode(
        "utf-8"
    )
    return response


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

    def test_model_identity_policy_has_no_pinned_runtime_value(self):
        self.assertEqual(
            CONTRACT["evaluator"]["model_identity_policy"],
            "match-hash-verified-completed-verifier-environment-manifest",
        )
        self.assertNotIn("model", CONTRACT["evaluator"])
        runner_source = (ROOT / "run_phase_e_rag.py").read_text(encoding="utf-8")
        self.assertNotIn("qwen3:32b", runner_source)

    def test_run_id_cannot_escape_output(self):
        for value in ("../escape", "nested/run", ".", "-leading"):
            with self.subTest(value=value):
                with self.assertRaises(self.runner.PhaseEError):
                    self.runner.validate_run_id(value)

    def test_model_identity_is_derived_from_hash_verified_verifier_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verifier-environment.json"
            path.write_text(json.dumps(verifier_manifest()), encoding="utf-8")
            manifest_hash = self.runner.sha256_file(path)
            identity = self.runner.load_verifier_model_identity(path, manifest_hash)

        self.assertEqual(identity["name"], "example-model:tag")
        self.assertEqual(identity["tag_digest"], "a" * 64)
        self.assertEqual(identity["blob_sha256"], "b" * 64)
        self.assertEqual(identity["verifier_environment_sha256"], manifest_hash)

    def test_model_match_uses_verifier_tag_and_blob_not_whole_show_hash(self):
        expected = {
            "name": "example-model:tag",
            "tag_digest": "a" * 64,
            "blob_sha256": "b" * 64,
            "details": {
                "family": "example-family",
                "parameter_size": "12.3B",
                "quantization_level": "Q4_EXAMPLE",
            },
            "verifier_environment_sha256": "c" * 64,
            "verifier_condition_id": "VERIFIER-CONDITION",
            "verifier_protocol_id": "VERIFIER-PROTOCOL",
        }
        tags = {
            "models": [
                {
                    "name": expected["name"],
                    "model": expected["name"],
                    "digest": expected["tag_digest"],
                }
            ]
        }
        show_hashes = []
        parameter_orders = (
            "top_p 0.95\ntemperature 0.6",
            "temperature 0.6\ntop_p 0.95",
        )
        for parameters in parameter_orders:
            show = {
                "details": expected["details"],
                "modelfile": f"FROM /models/blobs/sha256-{expected['blob_sha256']}\n",
                "parameters": parameters,
            }
            with mock.patch.object(
                self.runner.urllib.request,
                "urlopen",
                side_effect=[
                    mocked_http_response(tags),
                    mocked_http_response(show),
                ],
            ) as urlopen:
                evidence, tags_bytes, show_bytes = self.runner.fetch_matching_model(
                    "http://localhost:11434", expected
                )

            self.assertTrue(evidence["identity_verified"])
            self.assertEqual(evidence["name"], expected["name"])
            self.assertEqual(evidence["tag_digest"], expected["tag_digest"])
            self.assertEqual(evidence["blob_sha256"], expected["blob_sha256"])
            self.assertEqual(tags_bytes, self.runner.canonical_json_bytes(tags))
            self.assertEqual(show_bytes, self.runner.canonical_json_bytes(show))
            show_hashes.append(evidence["show_response_sha256"])
            requests = [call.args[0] for call in urlopen.call_args_list]
            self.assertEqual(requests[0].full_url, "http://localhost:11434/api/tags")
            self.assertEqual(requests[0].method, "GET")
            self.assertEqual(requests[1].full_url, "http://localhost:11434/api/show")
            self.assertEqual(
                requests[1].data, b'{"model":"example-model:tag"}\n'
            )

        self.assertNotEqual(show_hashes[0], show_hashes[1])

    def test_model_match_rejects_a_different_tag_digest(self):
        expected = {
            "name": "example-model:tag",
            "tag_digest": "a" * 64,
            "blob_sha256": "b" * 64,
            "details": {},
            "verifier_environment_sha256": "c" * 64,
            "verifier_condition_id": None,
            "verifier_protocol_id": None,
        }
        tags = {
            "models": [
                {
                    "name": expected["name"],
                    "digest": "d" * 64,
                }
            ]
        }
        with mock.patch.object(
            self.runner.urllib.request,
            "urlopen",
            return_value=mocked_http_response(tags),
        ):
            with self.assertRaisesRegex(
                self.runner.PhaseEError, "differs from the completed verifier run"
            ):
                self.runner.fetch_matching_model(
                    "http://localhost:11434", expected
                )

    def test_model_match_rejects_a_different_model_blob(self):
        expected = {
            "name": "example-model:tag",
            "tag_digest": "a" * 64,
            "blob_sha256": "b" * 64,
            "details": {
                "family": "example-family",
                "parameter_size": "12.3B",
                "quantization_level": "Q4_EXAMPLE",
            },
            "verifier_environment_sha256": "c" * 64,
            "verifier_condition_id": None,
            "verifier_protocol_id": None,
        }
        tags = {
            "models": [
                {
                    "name": expected["name"],
                    "digest": expected["tag_digest"],
                }
            ]
        }
        show = {
            "details": expected["details"],
            "modelfile": f"FROM /models/blobs/sha256-{'d' * 64}\n",
        }
        with mock.patch.object(
            self.runner.urllib.request,
            "urlopen",
            side_effect=[mocked_http_response(tags), mocked_http_response(show)],
        ):
            with self.assertRaisesRegex(
                self.runner.PhaseEError, "blob identity differs"
            ):
                self.runner.fetch_matching_model(
                    "http://localhost:11434", expected
                )

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
            verifier_path = Path(tmp) / "verifier-environment.json"
            verifier_path.write_text(
                json.dumps(verifier_manifest()), encoding="utf-8"
            )
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
                verifier_environment_manifest=verifier_path,
                verifier_environment_sha256=self.runner.sha256_file(verifier_path),
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
                    with mock.patch.object(
                        self.runner, "fetch_matching_model"
                    ) as model:
                        with redirect_stdout(output):
                            result = self.runner.run_command(args, test_contract)

            self.assertEqual(result, 0)
            model.assert_not_called()
            self.assertIn('"verified": "blocked-by-explicit-flag"', output.getvalue())
            self.assertFalse(run_dir.exists())

    def test_remote_ollama_endpoint_is_rejected(self):
        expected = {
            "name": "example-model:tag",
            "tag_digest": "a" * 64,
            "blob_sha256": "b" * 64,
            "details": {},
        }
        with self.assertRaisesRegex(self.runner.PhaseEError, "only a loopback"):
            self.runner.fetch_matching_model("https://example.com:11434", expected)

    def test_rag_output_validation_rejects_failed_model_calls(self):
        modes = CONTRACT["evaluator"]["modes"]
        count = CONTRACT["evaluator"]["max_questions"]
        output = {
            "metadata": {"n_questions": count, "model": "example-model:tag"},
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
                self.runner.validate_rag_output(path, CONTRACT, "example-model:tag")

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
