"""Focused contracts for compatibility-hosted canonical encoder training."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from data.code_accord import ResumableRandomSampler, _load_prepared_examples
from models.bert_kg_encoder import BertBackbone
from train_span import _load_restart, parse_args


def _prepared_record(example_id: str, split: str) -> dict:
    return {
        "protocol_id": "B04-PATH-A-1.3",
        "dataset_id": "CODE-ACCORD-v1.0.0",
        "example_id": example_id,
        "source_document_id": "doc-a",
        "source_country": "UK",
        "split": split,
        "content": "Door is fire rated",
        "processed_content": "Door is fire rated",
        "words": ["Door", "is", "fire", "rated"],
        "entities": [
            {"start": 0, "end": 0, "type": "Object", "text": "Door"},
            {"start": 2, "end": 3, "type": "Quality", "text": "fire rated"},
        ],
        "relations": [
            {
                "head": {"start": 0, "end": 0, "type": "Object", "text": "Door"},
                "relation": "necessity",
                "tail": {
                    "start": 2,
                    "end": 3,
                    "type": "Quality",
                    "text": "fire rated",
                },
                "annotation_rows": [1],
                "head_alignment": "exact",
                "tail_alignment": "exact",
            }
        ],
    }


class PreparedAdapterTests(unittest.TestCase):
    def test_adapter_preserves_jsonl_order_spans_and_relations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            records = [
                _prepared_record("example-b", "train"),
                _prepared_record("example-a", "train"),
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            examples = _load_prepared_examples(path, "train")
            self.assertEqual(
                [example["example_id"] for example in examples],
                ["example-b", "example-a"],
            )
            self.assertEqual(
                examples[0]["ner"],
                [(0, 0, "Object"), (2, 3, "Quality")],
            )
            self.assertEqual(examples[0]["relations"], [((0, 0), (2, 3), 2)])

    def test_adapter_rejects_split_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            path.write_text(
                json.dumps(_prepared_record("example-a", "test")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected 'train'"):
                _load_prepared_examples(path, "train")

    def test_sampler_state_resumes_at_next_unconsumed_example(self):
        sampler = ResumableRandomSampler(list(range(8)), seed=42)
        iterator = iter(sampler)
        consumed = [next(iterator), next(iterator), next(iterator)]
        state = sampler.state_dict()
        expected_remainder = list(iterator)

        resumed = ResumableRandomSampler(list(range(8)), seed=999)
        resumed.load_state_dict(state)
        self.assertEqual(list(iter(resumed)), expected_remainder)
        self.assertEqual(
            sorted(consumed + expected_remainder), list(range(8))
        )

    def test_restart_loads_rng_payload_on_cpu_before_restoring_sampler(self):
        sampler = ResumableRandomSampler(list(range(4)), seed=42)
        next(iter(sampler))
        loader = SimpleNamespace(sampler=sampler)
        state = {
            "format_version": "train-span-restart-1.0",
            "seed": 42,
            "model_name": "microsoft/deberta-large",
            "model_revision": "revision-a",
            "max_steps": 3500,
            "encoder": {},
            "optimizer": {},
            "scheduler": {},
            "next_step": 100,
            "best_metrics": {"triple_f1": 0.1},
            "best_step": 100,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": [],
            "train_sampler_state": sampler.state_dict(),
            "boost_adaptive_triggered": False,
            "boost_adaptive_switched": False,
            "re_head_finetune_active": False,
        }
        args = SimpleNamespace(
            resume_from="restart-state.pt",
            seed=42,
            model_name="microsoft/deberta-large",
            model_revision="revision-a",
            max_steps=3500,
        )
        with patch("train_span.torch.load", return_value=state) as loader_mock:
            _load_restart(args, Mock(), Mock(), Mock(), loader, torch.device("cuda"))
        loader_mock.assert_called_once_with(
            "restart-state.pt", map_location="cpu", weights_only=False
        )


class HistoricalDefaultTests(unittest.TestCase):
    def test_new_training_controls_are_opt_in(self):
        args = parse_args(["--dataset", "accord"])
        self.assertFalse(args.canonical_mode)
        self.assertIsNone(args.prepared_dir)
        self.assertIsNone(args.model_revision)
        self.assertIsNone(args.model_cache_dir)
        self.assertFalse(args.model_local_files_only)
        self.assertFalse(args.skip_test_eval)
        self.assertIsNone(args.save_last_to)
        self.assertIsNone(args.resume_from)

    def test_backbone_revision_is_optional_and_forwarded_only_when_supplied(self):
        fake_model = Mock()
        fake_model.config.hidden_size = 16
        with patch(
            "models.bert_kg_encoder.AutoModel.from_pretrained",
            return_value=fake_model,
        ) as loader:
            BertBackbone("model-a")
            loader.assert_called_once_with("model-a")
        with patch(
            "models.bert_kg_encoder.AutoModel.from_pretrained",
            return_value=fake_model,
        ) as loader:
            BertBackbone("model-a", model_revision="revision-a")
            loader.assert_called_once_with("model-a", revision="revision-a")
        with patch(
            "models.bert_kg_encoder.AutoModel.from_pretrained",
            return_value=fake_model,
        ) as loader:
            BertBackbone(
                "model-a",
                model_revision="revision-a",
                model_cache_dir="cache-a",
                model_local_files_only=True,
            )
            loader.assert_called_once_with(
                "model-a",
                revision="revision-a",
                cache_dir="cache-a",
                local_files_only=True,
            )


if __name__ == "__main__":
    unittest.main()
