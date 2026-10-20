"""Model-free evidence for server identity and fresh same-run drift guards."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from utils.common.artifact_io import DataContractError, atomic_write_json, atomic_write_jsonl, load_json, sha256_file
from utils.common.config import load_pipeline_config
from utils.common.paths import RunLayout
from utils.rag import runner
from utils.verifier.runtime import bind_runtime, load_runtime, runtime_identity, runtime_matches
from utils.verifier.verifier import HttpResult, _environment_manifest, _verify_live_model, run_verifier


ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "http://localhost:11434"
PARAMETER_BLOCKS = (
    'stop                           "<|im_start|>"\nstop                           "<|im_end|>"',
    "temperature                    0.6",
    "top_k                          20",
    "top_p                          0.95",
    "repeat_penalty                 1",
)


class OllamaRuntimeTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(self.temporary.cleanup)
        self.layout = RunLayout(Path(self.temporary.name), "server-runtime-fixture")
        self.layout.create()
        self.config = load_pipeline_config(ROOT, "resources/configs/pipeline.json")
        self.blob = self.layout.resolve("inputs/ollama/fixture-model")
        self.blob.parent.mkdir(parents=True)
        self.blob.write_bytes(b"model-free runtime fixture")
        verifier = self.config.value["verifier"]
        verifier["model_blob_sha256"] = sha256_file(self.blob)
        verifier["registry_manifest_sha256"] = "a" * 64
        self.model = {"name": verifier["model"], "tag_digest": "a" * 64,
            "blob_sha256": sha256_file(self.blob), "details": {
                "family": "qwen3", "parameter_size": "32.8B", "quantization_level": "Q4_K_M"}}
        self.show = {"details": self.model["details"],
            "modelfile": f"FROM /models/blobs/sha256-{self.model['blob_sha256']}\n",
            "template": "{{ .Messages }}", "parameters": "num_ctx 4096\n",
            "system": "", "model_info": {"general.architecture": "qwen3",
                "qwen3.context_length": 4096, "qwen3.rope.freq_base": 10000,
                "general.name": "cosmetic model label"},
            "capabilities": ["completion", "thinking"]}
        self.version = {"version": "0.fixture.server"}
        self.calls = []

    def transport(self, method, url, payload, timeout):
        self.calls.append(url)
        if url.endswith("/api/tags"):
            body = {"models": [{"name": self.model["name"], "digest": self.model["tag_digest"]}]}
        elif url.endswith("/api/show"):
            body = self.show
        elif url.endswith("/api/version"):
            body = self.version
        else:
            self.fail(f"unexpected identity endpoint {url}")
        return HttpResult(200, copy.deepcopy(body), hashlib.sha256(b"fixture").hexdigest())

    def verify(self):
        return _verify_live_model(self.layout, self.config, "simple", ORIGIN,
                                  self.blob, self.transport)

    def test_actual_server_version_and_defaults_are_bound_once_per_run(self):
        evidence = self.verify()
        reference = evidence["ollama_runtime"]
        original = self.layout.resolve(reference["path"]).read_bytes()
        observed = load_runtime(self.layout.run_root, reference)
        self.assertEqual(observed["server_version"], self.version["version"])
        self.assertEqual(observed["generation_defaults"]["template"], self.show["template"])
        self.assertEqual(set(observed), {"origin", "server_version", "model", "generation_defaults"})
        self.assertEqual(set(observed["generation_defaults"]), {"template", "parameters", "system", "model_info"})
        self.assertEqual(set(observed["generation_defaults"]["model_info"]), {
            "general.architecture", "qwen3.context_length", "qwen3.rope.freq_base"})
        document = load_json(self.layout.resolve(reference["path"]))
        self.assertEqual(set(document), {"run_id", "identity"})
        self.verify()
        self.assertEqual(self.layout.resolve(reference["path"]).read_bytes(), original)
        self.assertEqual(sum(url.endswith("/api/version") for url in self.calls), 2)
        self.assertFalse(any(url.endswith("/api/ps") for url in self.calls))

    def test_stage_environment_has_no_client_hardware_or_software_probes(self):
        with mock.patch("subprocess.run", side_effect=AssertionError("client probe")):
            environment = _environment_manifest(self.config, "VER-SIMPLE", "live")
        self.assertEqual(environment, {"condition_id": "VER-SIMPLE", "execution_mode": "live",
                                      "model": {"identity_verified": False}})

    def test_configured_server_proof_requires_no_local_ollama_cli(self):
        with mock.patch("subprocess.run", side_effect=AssertionError("local CLI invoked")):
            self.verify()
        self.assertEqual(self.calls, [ORIGIN + suffix for suffix in ("/api/tags", "/api/show", "/api/version")])
        self.assertFalse(self.layout.resolve("verifier/simple/model/ollama-modelfile.txt").exists())

    def test_server_modelfile_must_name_one_exact_run_local_weight_blob(self):
        for modelfile in ("FROM another-model:tag\n", self.show["modelfile"] * 2,
                          "FROM sha256:" + "f" * 64):
            with self.subTest(modelfile=modelfile):
                self.show["modelfile"] = modelfile
                with self.assertRaisesRegex(DataContractError, "frozen model blob"):
                    self.verify()
                self.assertFalse(self.layout.resolve("runtime/ollama.json").exists())

    def test_full_run_manifest_refers_to_shared_identity_without_copying_it(self):
        from stages.verifier import _write_full_run_manifest
        reference = self.verify()["ollama_runtime"]
        config = SimpleNamespace(path=self.layout.source_root / "resources/configs/pipeline.json")
        with mock.patch("stages.verifier.subprocess.run", return_value=SimpleNamespace(stdout="c" * 40)):
            _write_full_run_manifest(self.layout, config)
            path = self.layout.resolve("manifests/pipeline-run.json")
            original = path.read_bytes()
            document = load_json(path)
            self.assertEqual(document, {"schema_version": "pipeline-run-2", "run_id": self.layout.run_id,
                "source_commit": "c" * 40, "config": "resources/configs/pipeline.json",
                "ollama_runtime": reference})
            _write_full_run_manifest(self.layout, config)
            self.assertEqual(path.read_bytes(), original)
            runtime_path = self.layout.resolve(reference["path"])
            runtime = load_json(runtime_path)
            runtime["identity"]["server_version"] = "changed"
            atomic_write_json(runtime_path, runtime)
            with self.assertRaisesRegex(DataContractError, "differs at ollama_runtime"):
                _write_full_run_manifest(self.layout, config)

    def test_live_fixture_seals_minimal_runtime_and_blocks_later_stage_drift(self):
        from tests.test_verifier import _candidate, _sentence, _write_inputs
        self.show["parameters"] = "\n".join(PARAMETER_BLOCKS)
        atomic_write_json(self.layout.resolve("manifests/00-checkout-manifest.json"), {"source": {"commit": "c" * 40}})
        sentences, candidates = _write_inputs(self.layout)
        warm_sentences = self.layout.resolve("data-prepared/development.jsonl")
        warm_candidates = self.layout.resolve("predictions/dev/warmup.jsonl")
        atomic_write_jsonl(warm_sentences, [_sentence("warm", split="development")])
        atomic_write_jsonl(warm_candidates, [_candidate("warm", split="development")])
        generated = []

        def transport(method, url, payload, timeout):
            if not url.endswith("/api/chat"):
                return self.transport(method, url, payload, timeout)
            generated.append(payload)
            content = {"action": "KEEP", "reason_code": "SUPPORTED"}
            if "corrected" in payload["format"]["properties"]:
                content["corrected"] = None
            body = {"message": {"role": "assistant", "content": json.dumps(content)},
                    "done_reason": "stop"}
            return HttpResult(200, body, hashlib.sha256(b"fixture reply").hexdigest())

        arguments = {"layout": self.layout, "config": self.config, "execution_mode": "live",
            "sentences_path": sentences, "candidates_path": candidates, "warmup_sentences_path": warm_sentences,
            "warmup_candidates_path": warm_candidates, "ollama_url": ORIGIN, "model_blob_path": self.blob,
            "transport": transport, "sleep": lambda _: None}
        with mock.patch("subprocess.run", side_effect=AssertionError("local CLI invoked")):
            manifest = run_verifier(mode="simple", **arguments)
        environment = load_json(self.layout.resolve("verifier/simple/environment-manifest.json"))
        self.assertEqual(set(environment), {"condition_id", "execution_mode", "model"})
        self.assertEqual(set(environment["model"]), {
            "identity_verified", "ollama_runtime", "tags_response_sha256", "show_response_sha256"})
        reference = environment["model"]["ollama_runtime"]
        self.assertEqual(manifest["outputs"][reference["path"]], reference["sha256"])
        self.assertFalse(any("modelfile" in key for key in manifest["outputs"]))
        request = load_json(self.layout.resolve("verifier/simple/requests.jsonl"))
        verdict = load_json(self.layout.resolve("verifier/simple/verdicts.jsonl"))
        for record in (request, verdict, manifest):
            self.assertNotIn("prompt_sha256", record)
            self.assertNotIn("decoding_sha256", record)
            self.assertEqual(record["template_id"], "CODE-VERIFIER-1:simple")
        self.assertEqual(verdict["request_sha256"], request["request_sha256"])
        self.assertEqual(len(generated), 2)
        self.version["version"] = "drift-before-next-mode"
        with self.assertRaisesRegex(DataContractError, "runtime drift"):
            run_verifier(mode="corrective", **arguments)
        self.assertEqual(len(generated), 2)
        preserved = {relative: self.layout.resolve(relative).read_bytes() for relative in (
            "runtime/ollama.json", "verifier/simple/requests.jsonl",
            "verifier/simple/responses.jsonl", "verifier/simple/verdicts.jsonl",
            "verifier/simple/warmup-response.jsonl", "verifier/simple/environment-manifest.json",
        )}
        self.version["version"] = "0.fixture.server"
        self.show["parameters"] = "\n".join(
            (PARAMETER_BLOCKS[-1], *PARAMETER_BLOCKS[:-1]))
        with mock.patch("subprocess.run", side_effect=AssertionError("local CLI invoked")):
            run_verifier(mode="corrective", **arguments)
        self.assertEqual(len(generated), 4)
        for relative, original in preserved.items():
            self.assertEqual(self.layout.resolve(relative).read_bytes(), original, relative)

    def test_same_tag_and_weights_cannot_hide_server_or_generation_default_drift(self):
        for field in ("version", "template", "parameters", "system", "model_info"):
            with self.subTest(field=field):
                evidence = self.verify()
                original = self.layout.resolve(evidence["ollama_runtime"]["path"]).read_bytes()
                target = self.version if field == "version" else self.show
                old = copy.deepcopy(target[field])
                target[field] = "different fixture value"
                with self.assertRaisesRegex(DataContractError, "runtime drift"):
                    self.verify()
                target[field] = old
                self.assertEqual(self.layout.resolve(evidence["ollama_runtime"]["path"]).read_bytes(), original)

    def test_parameter_name_permutations_preserve_raw_runtime_bytes_and_reference(self):
        self.show["parameters"] = "\n".join(PARAMETER_BLOCKS)
        evidence = self.verify()
        reference = evidence["ollama_runtime"]
        path = self.layout.resolve(reference["path"])
        original = path.read_bytes()
        for blocks in itertools.permutations(PARAMETER_BLOCKS):
            with self.subTest(blocks=blocks):
                self.show["parameters"] = "\n".join(blocks)
                self.assertEqual(self.verify()["ollama_runtime"], reference)
                self.assertEqual(path.read_bytes(), original)
        # The user's reported rotation is one of these 120 permutations.
        self.assertEqual(load_runtime(self.layout.run_root, reference)[
            "generation_defaults"]["parameters"], "\n".join(PARAMETER_BLOCKS))

    def test_parameter_values_counts_and_per_name_order_still_reject_drift(self):
        parameters = "\n".join(PARAMETER_BLOCKS)
        self.show["parameters"] = parameters
        reference = self.verify()["ollama_runtime"]
        path = self.layout.resolve(reference["path"])
        original = path.read_bytes()
        lines = parameters.splitlines()
        variants = {
            "value": parameters.replace("0.6", "0.7"),
            "stop_order": "\n".join([lines[1], lines[0], *lines[2:]]),
            "stop_content": parameters.replace("<|im_start|>", "<|different|>"),
            "missing_stop": "\n".join(lines[1:]),
            "duplicate_stop": parameters + "\n" + lines[0],
            "added_parameter": parameters + "\nseed 42",
            "missing_parameter": "\n".join(lines[:-1]),
            "quoted_space": parameters.replace('"<|im_end|>"', '"<|im_end|> "'),
            "duplicate_scalar": parameters + "\ntemperature 0.6",
        }
        for label, changed in variants.items():
            with self.subTest(label=label):
                self.show["parameters"] = changed
                with self.assertRaisesRegex(DataContractError, "runtime drift"):
                    self.verify()
                self.assertEqual(path.read_bytes(), original)

    def test_runtime_comparison_does_not_mutate_or_conflate_raw_value_syntax(self):
        self.show["parameters"] = "\n".join(PARAMETER_BLOCKS)
        recorded = runtime_identity(ORIGIN, self.version, self.show, self.model)
        observed = copy.deepcopy(recorded)
        observed["generation_defaults"]["parameters"] = "\n".join(
            (PARAMETER_BLOCKS[-1], *PARAMETER_BLOCKS[:-1]))
        originals = copy.deepcopy((recorded, observed))
        self.assertTrue(runtime_matches(recorded, observed))
        self.assertEqual((recorded, observed), originals)
        for text in ('stop "a b"', 'stop "a  b"', "temperature 0.60"):
            with self.subTest(text=text):
                changed = copy.deepcopy(observed)
                changed["generation_defaults"]["parameters"] = text
                self.assertFalse(runtime_matches(recorded, changed))
        missing = copy.deepcopy(recorded)
        del missing["generation_defaults"]["parameters"]
        self.assertFalse(runtime_matches(recorded, missing))
        added = copy.deepcopy(recorded)
        added["unrecognized_runtime_field"] = True
        self.assertFalse(runtime_matches(recorded, added))

    def test_unknown_parameter_display_syntax_requires_exact_equality(self):
        for text in ("temperature", "temperature 0.6\n\nstop x", "bad-key 1\nstop x"):
            with self.subTest(text=text):
                self.show["parameters"] = text
                recorded = runtime_identity(ORIGIN, self.version, self.show, self.model)
                self.assertTrue(runtime_matches(recorded, copy.deepcopy(recorded)))
                changed = copy.deepcopy(recorded)
                changed["generation_defaults"]["parameters"] = "\n".join(
                    reversed(text.splitlines())) + "\n"
                self.assertFalse(runtime_matches(recorded, changed))

    def test_ordering_equivalence_cannot_hide_foreign_runtime_run_id(self):
        self.show["parameters"] = "\n".join(PARAMETER_BLOCKS)
        reference = self.verify()["ollama_runtime"]
        path = self.layout.resolve(reference["path"])
        document = load_json(path)
        document["run_id"] = "foreign-run"
        atomic_write_json(path, document)
        original = path.read_bytes()
        self.show["parameters"] = "\n".join(reversed(PARAMETER_BLOCKS))
        with self.assertRaisesRegex(DataContractError, "runtime drift"):
            self.verify()
        self.assertEqual(path.read_bytes(), original)

    def test_missing_server_version_fails_without_recording_placeholders(self):
        self.version.clear()
        with self.assertRaisesRegex(DataContractError, "actual server version"):
            self.verify()
        self.assertFalse(self.layout.resolve("runtime/ollama.json").exists())
        self.assertFalse(any(url.endswith("/api/ps") for url in self.calls))

    def test_reference_requires_same_run_physical_hash_bound_artifact(self):
        identity = runtime_identity(ORIGIN, self.version, self.show, self.model)
        reference = bind_runtime(self.layout, identity)
        self.assertEqual(load_runtime(self.layout.run_root, reference), identity)
        with self.assertRaisesRegex(DataContractError, "runtime reference"):
            load_runtime(self.layout.run_root, {**reference, "path": "../runtime/ollama.json"})
        path = self.layout.resolve(reference["path"])
        document = load_json(path)
        document["identity"]["server_version"] = "modified"
        atomic_write_json(path, document)
        with self.assertRaisesRegex(DataContractError, "changed or is missing"):
            load_runtime(self.layout.run_root, reference)

    def test_historical_partial_runs_are_not_given_reconstructed_server_identity(self):
        atomic_write_json(self.layout.resolve("verifier/simple/environment-manifest.json"), {"old": True})
        with self.assertRaisesRegex(DataContractError, "original pinned workflow"):
            self.verify()
        self.assertFalse(self.layout.resolve("runtime/ollama.json").exists())

    def expected_rag_model(self):
        evidence = self.verify()
        return {**self.model, "ollama_runtime": evidence["ollama_runtime"],
            "ollama_runtime_identity": load_runtime(self.layout.run_root, evidence["ollama_runtime"]),
            "verifier_environment_sha256": "b" * 64, "verifier_condition_id": "VER-CORRECTIVE",
            "verifier_protocol_id": "fixture-protocol"}

    def rag_fetch(self, method, url, payload=None):
        response = self.transport(method, url, payload, 30)
        body = runner.canonical_json_bytes(response.json_body)
        return response.json_body, body, runner.sha256_bytes(body)

    def test_rag_checks_actual_server_before_dispatch_and_rejects_drift(self):
        expected = self.expected_rag_model()
        self.calls.clear()
        with mock.patch.object(runner, "fetch_ollama_json", side_effect=self.rag_fetch):
            evidence, _tags, _show = runner.fetch_matching_model(ORIGIN, expected)
            self.assertEqual(evidence["ollama_runtime_sha256"], expected["ollama_runtime"]["sha256"])
            self.assertEqual(set(evidence), {"ollama_runtime_sha256", "tags_response_sha256", "show_response_sha256"})
            self.assertEqual(sum(url.endswith("/api/version") for url in self.calls), 1)
            self.version["version"] = "server-changed-on-resume"
            with self.assertRaisesRegex(runner.PhaseEError, "runtime differs"):
                runner.fetch_matching_model(ORIGIN, expected)

    def test_live_and_stored_rag_accept_display_order_without_rewriting_runtime(self):
        self.show["parameters"] = "\n".join(PARAMETER_BLOCKS)
        expected = self.expected_rag_model()
        reference = expected["ollama_runtime"]
        path = self.layout.resolve(reference["path"])
        original = path.read_bytes()
        self.show["parameters"] = "\n".join(
            (PARAMETER_BLOCKS[-1], *PARAMETER_BLOCKS[:-1]))
        with mock.patch.object(runner, "fetch_ollama_json", side_effect=self.rag_fetch):
            for check_runtime in (True, False):
                evidence, tags_bytes, show_bytes = runner.fetch_matching_model(
                    ORIGIN, expected, check_runtime=check_runtime)
                self.assertEqual(evidence["ollama_runtime_sha256"], reference["sha256"])
                runner._validate_stored_model_response(
                    json.loads(tags_bytes), json.loads(show_bytes), expected)
                self.assertEqual(path.read_bytes(), original)
        self.show["parameters"] = self.show["parameters"].replace("0.6", "0.7")
        with mock.patch.object(runner, "fetch_ollama_json", side_effect=self.rag_fetch):
            with self.assertRaisesRegex(runner.PhaseEError, "runtime differs"):
                runner.fetch_matching_model(ORIGIN, expected)
        with self.assertRaisesRegex(runner.ModelEvidenceError, "generation defaults differ"):
            runner._validate_stored_model_response(
                {"models": [{"name": self.model["name"], "digest": self.model["tag_digest"]}]},
                self.show, expected)
        self.assertEqual(path.read_bytes(), original)

    def test_per_answer_checks_reuse_stage_version_but_still_check_template(self):
        expected = self.expected_rag_model()
        self.calls.clear()
        with mock.patch.object(runner, "fetch_ollama_json", side_effect=self.rag_fetch):
            runner.fetch_matching_model(ORIGIN, expected, check_runtime=False)
            self.assertFalse(any(url.endswith("/api/version") for url in self.calls))
            self.show["template"] = "changed-default"
            with self.assertRaisesRegex(runner.PhaseEError, "runtime differs"):
                runner.fetch_matching_model(ORIGIN, expected, check_runtime=False)
        with self.assertRaisesRegex(runner.ModelEvidenceError, "generation defaults differ"):
            runner._validate_stored_model_response(
                {"models": [{"name": self.model["name"], "digest": self.model["tag_digest"]}]},
                self.show, expected)

    def test_verifier_model_loader_authenticates_shared_runtime(self):
        expected = self.expected_rag_model()
        environment = {"execution_mode": "live", "condition_id": "VER-CORRECTIVE", "model": {
            "identity_verified": True, "ollama_runtime": expected["ollama_runtime"]}}
        path = self.layout.resolve("verifier/corrective/environment-manifest.json")
        atomic_write_json(path, environment)
        restored = runner.load_verifier_model_identity(path, sha256_file(path), run_dir=self.layout.run_root)
        self.assertEqual(restored["ollama_runtime_identity"], expected["ollama_runtime_identity"])
        self.assertEqual(runner.persisted_model_identity(restored), {
            "ollama_runtime": expected["ollama_runtime"], "verifier_environment_sha256": sha256_file(path)})
        with self.assertRaisesRegex(runner.PhaseEError, "selected run root"):
            runner.load_verifier_model_identity(path, sha256_file(path))


if __name__ == "__main__":
    unittest.main()
