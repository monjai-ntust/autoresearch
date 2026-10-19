"""Prevent optional Hub conversion writes without changing pretrained inputs."""

from __future__ import annotations

import ast
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from transformers.utils import is_env_variable_true

from utils.encoder import network as canonical_network
from utils.encoder.cache import configure_model_loading
from utils.encoder_comparison import network as comparison_network


def _probe_binary_resolution(*, use_safetensors=None, local_files_only=False):
    """Execute the installed resolver with every I/O operation replaced.

    Isolating its two function definitions avoids importing optional model
    integrations. No real Hub request, thread, file acquisition or model load
    occurs; the resolver's format and conversion decisions are exercised.
    """
    package = importlib.util.find_spec("transformers")
    source = Path(next(iter(package.submodule_search_locations))) / "modeling_utils.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_add_variant", "_get_resolved_checkpoint_files"}
    ]
    if len(functions) != 2:
        raise AssertionError("the locked Transformers resolver is unavailable")
    isolated = ast.fix_missing_locations(ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
              *functions], type_ignores=[]))
    lookups = []
    revision = "fixture-pinned-revision"
    weight = "/synthetic/pinned/pytorch_model.bin"

    def cached_file(model, filename, **kwargs):
        if kwargs["revision"] != revision:
            raise AssertionError("the resolver changed the pinned model revision")
        lookups.append((filename, kwargs["revision"], kwargs["local_files_only"]))
        return weight if filename == "pytorch_model.bin" else None

    thread = mock.Mock()
    namespace = {
        "os": SimpleNamespace(path=SimpleNamespace(
            isdir=lambda value: False, isfile=lambda value: False, join=os.path.join)),
        "DownloadKwargs": dict,
        "cached_file": cached_file,
        "has_file": lambda *args, **kwargs: False,
        "is_offline_mode": lambda: False,
        "is_env_variable_true": is_env_variable_true,
        "SAFE_WEIGHTS_NAME": "model.safetensors",
        "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json",
        "WEIGHTS_NAME": "pytorch_model.bin",
        "WEIGHTS_INDEX_NAME": "pytorch_model.bin.index.json",
        "Thread": thread,
        "auto_conversion": mock.Mock(side_effect=AssertionError("conversion was executed")),
        "logger": mock.Mock(),
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    files, metadata = namespace["_get_resolved_checkpoint_files"](
        "fixture/encoder", None, None, use_safetensors, None, False,
        download_kwargs={"revision": revision, "local_files_only": local_files_only})
    if files != [weight] or metadata is not None:
        raise AssertionError("the resolver changed its original pinned binary input")
    return {"files": files, "lookups": lookups,
            "conversion_started": thread.return_value.start.called}


class ModelCacheLoadingTests(unittest.TestCase):
    def test_default_binary_load_can_start_conversion_after_return(self):
        with mock.patch.dict(os.environ, {"DISABLE_SAFETENSORS_CONVERSION": "0"}):
            self.assertTrue(_probe_binary_resolution()["conversion_started"])

    def test_explicit_binary_format_does_not_disable_conversion(self):
        with mock.patch.dict(os.environ, {"DISABLE_SAFETENSORS_CONVERSION": "0"}):
            self.assertTrue(_probe_binary_resolution(use_safetensors=False)["conversion_started"])

    def test_local_files_only_does_not_disable_conversion(self):
        with mock.patch.dict(os.environ, {"DISABLE_SAFETENSORS_CONVERSION": "0"}):
            self.assertTrue(_probe_binary_resolution(local_files_only=True)["conversion_started"])

    def test_conversion_guard_preserves_pinned_inputs_and_lookups(self):
        for options in ({}, {"use_safetensors": False}, {"local_files_only": True}):
            with self.subTest(options=options), mock.patch.dict(
                    os.environ, {"DISABLE_SAFETENSORS_CONVERSION": "false"}):
                previous = _probe_binary_resolution(**options)
                configure_model_loading()
                current = _probe_binary_resolution(**options)
                self.assertTrue(previous.pop("conversion_started"))
                self.assertFalse(current.pop("conversion_started"))
                self.assertEqual(current, previous)

    def test_both_network_wrappers_enforce_policy_before_model_load(self):
        class TinyBackbone(torch.nn.Module):
            config = SimpleNamespace(hidden_size=4)

        for module, extra, expected in (
            (canonical_network, {}, {}),
            (comparison_network, {"expected_hidden_size": 4}, {"use_safetensors": False}),
        ):
            with self.subTest(module=module.__name__), mock.patch.dict(
                    os.environ, {"DISABLE_SAFETENSORS_CONVERSION": "false"}):
                def load(model_name, **kwargs):
                    self.assertEqual(os.environ["DISABLE_SAFETENSORS_CONVERSION"], "1")
                    self.assertEqual(model_name, "fixture/encoder")
                    self.assertEqual(kwargs, {"revision": "revision", "cache_dir": "cache",
                                              "local_files_only": True, **expected})
                    return TinyBackbone()

                with mock.patch.object(module.AutoModel, "from_pretrained", side_effect=load) as loader:
                    backbone = module.BertBackbone(
                        "fixture/encoder", model_revision="revision", model_cache_dir="cache",
                        model_local_files_only=True, **extra)
                self.assertEqual(backbone.hidden_size, 4)
                loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
