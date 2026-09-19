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

from utils.pipeline.encoder.data import ResumableRandomSampler, _load_prepared_examples
from utils.pipeline.encoder.network import BertBackbone, BertKGExtractor
from stages.encoder import _load_restart, parse_args


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
        with patch("stages.encoder.torch.load", return_value=state) as loader_mock:
            _load_restart(args, Mock(), Mock(), Mock(), loader, torch.device("cuda"))
        loader_mock.assert_called_once_with(
            "restart-state.pt", map_location="cpu", weights_only=False
        )


class TrainerSurfaceTests(unittest.TestCase):
    def test_trainer_parser_exposes_only_the_canonical_path(self):
        values = {
            "prepared-dir": "run/data-prepared",
            "model-name": "microsoft/deberta-large",
            "model-revision": "revision-a",
            "model-cache-dir": "run/inputs/huggingface",
            "batch-size": "16",
            "max-length": "128",
            "lr": "3e-5",
            "max-steps": "3500",
            "warmup-steps": "250",
            "max-span-width": "8",
            "re-weight": "1",
            "re-no-rel-weight": "1",
            "neg-sample-ratio": "3",
            "focal-gamma": "2",
            "label-smoothing": ".1",
            "eval-every": "100",
            "seed": "42",
            "re-comparison-boost": "5",
            "re-boost-adaptive-steps": "1000",
            "re-boost-adaptive-threshold": ".35",
            "re-boost-adaptive-threshold2": ".4",
            "re-boost-mid": "3.5",
            "re-boost-end": "2",
            "save-best-to": "checkpoint.pt",
            "save-last-to": "restart.pt",
            "progress-log": "training.log",
            "run-summary-out": "summary.json",
        }
        argv = [item for key, value in values.items() for item in (f"--{key}", value)]
        args = parse_args(argv)
        for removed in (
            "dataset",
            "canonical_mode",
            "synth_jsonl",
            "evidence_gat",
            "use_crf",
        ):
            self.assertFalse(hasattr(args, removed))

    def test_backbone_revision_is_optional_and_forwarded_only_when_supplied(self):
        fake_model = Mock()
        fake_model.config.hidden_size = 16
        with patch(
            "utils.pipeline.encoder.network.AutoModel.from_pretrained",
            return_value=fake_model,
        ) as loader:
            BertBackbone("model-a")
            loader.assert_called_once_with("model-a")
        with patch(
            "utils.pipeline.encoder.network.AutoModel.from_pretrained",
            return_value=fake_model,
        ) as loader:
            BertBackbone("model-a", model_revision="revision-a")
            loader.assert_called_once_with("model-a", revision="revision-a")
        with patch(
            "utils.pipeline.encoder.network.AutoModel.from_pretrained",
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

    def test_canonical_extractor_keeps_only_checkpoint_compatible_heads(self):
        class FakeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(32, 4)
                self.config = SimpleNamespace(hidden_size=4)

            def get_input_embeddings(self):
                return self.embeddings

            def forward(self, *, inputs_embeds, attention_mask):
                del attention_mask
                return SimpleNamespace(last_hidden_state=inputs_embeds)

        with patch(
            "utils.pipeline.encoder.network.AutoModel.from_pretrained",
            return_value=FakeEncoder(),
        ):
            model = BertKGExtractor(
                "model-a",
                num_bio_tags=9,
                num_relations=10,
                num_entity_types=4,
                max_span_width=8,
            )
        keys = set(model.state_dict())
        self.assertIn("ner_head.weight", keys)
        self.assertIn("span_ner_head.weight", keys)
        self.assertIn("re_head.0.weight", keys)
        self.assertIn("adapters.text.word_embeddings.weight", keys)
        self.assertFalse(any("boundary" in key or "evidence" in key for key in keys))
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        logits, spans = model.forward_span_ner(hidden, [None, 0, 1, 2, None], 3, 2)
        self.assertEqual(spans, [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])
        self.assertEqual(tuple(logits.shape), (5, 5))


if __name__ == "__main__":
    unittest.main()
