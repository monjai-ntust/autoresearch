"""Canonical encoder candidate-generation adapter for ``CODE-STRICT-1``.

This module provides encoder training and candidate generation for the main
pipeline. It keeps the retained extraction *semantics*: the encoder proposes
entity spans and relations with confidence scores, and a triple's confidence
is the encoder softmax product. Every emitted record is bound to typed strict
data, split, and checkpoint identities beneath one ignored
``output/<run-id>/`` tree.

The stage has three implemented executions:

* ``dry-run`` validates the prepared sentences and checkpoint identity and
  materializes a deterministic per-sentence inference plan without producing a
  candidate. It contacts no model.
* ``replay`` deterministically transforms a frozen prediction ledger (the raw
  per-sentence span/relation scores an encoder emitted for one training seed)
  into canonical :class:`records.Candidate` records, reproducing the historical
  greedy non-overlapping NER selection and ``min(head, tail) * re`` confidence
  product on typed CODE spans.
* ``live`` runs the retained encoder (``utils.encoder.network`` under the
  configured canonical recipe) over the prepared sentences to produce that prediction ledger,
  then applies the same deterministic ledger->candidate transform as ``replay``.
  Its forward outputs cannot be validated offline, so live execution requires
  the selected run's authenticated accelerator checkpoint and model cache.
"""

from __future__ import annotations

from utils.encoder.options import EncoderOptions

from utils.encoder.runtime import training_environment
from utils.encoder.inference import _live_inference_records

from utils.encoder.candidates import PredictionLedgerIdentity, _SPLIT_ID, _exact_keys, _integer, _load_prediction_ledger, _unit_float

import math
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.common.config import PipelineConfig
from utils.common.constants import MATCHER_ID, PROTOCOL_ID, TRAINING_SEEDS
from utils.encoder.cache import (
    CACHE_RELATIVE,
    MANIFEST_RELATIVE as HF_CACHE_MANIFEST_RELATIVE,
    verify_cache_manifest,
    write_cache_manifest,
)
from utils.common.paths import RunLayout
from utils.common.artifact_io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    iter_jsonl,
    sha256_file,
    canonical_json_bytes,
    atomic_write_bytes,
)
from utils.verifier.requests import PreparedSentence, _load_sentences

EXECUTION_MODES = {"dry-run", "replay", "live"}
TRAIN_EXECUTION_MODES = {"dry-run", "live"}
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CheckpointIdentity:
    training_seed: int
    checkpoint_sha256: str
    checkpoint_step: int
    split_manifest_sha256: str
    max_span_width: int
    archive_sha256: str
    acquisition_manifest_sha256: str
    annotation_bundle_sha256: str
    prepared_dataset_tree_sha256: str
    train_jsonl_sha256: str
    development_jsonl_sha256: str
    config: str
    model_cache_manifest_sha256: str
    model_cache_tree_sha256: str
    source_commit: str
    restart_state_sha256: str


def _sha256_hex(value: Any, label: str) -> str:
    text = value if isinstance(value, str) else None
    if text is None or len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DataContractError(f"{label} must be a 64-character lowercase SHA-256 digest")
    return text


def _load_checkpoint_manifest(path: Path, config: PipelineConfig) -> CheckpointIdentity:
    value = load_json(path)
    label = path.name
    _exact_keys(
        value,
        {
            "schema_version",
            "protocol_id",
            "split_id",
            "training_seed",
            "checkpoint_sha256",
            "checkpoint_step",
            "split_manifest_sha256",
            "archive_sha256",
            "acquisition_manifest_sha256",
            "annotation_bundle_sha256",
            "prepared_dataset_tree_sha256",
            "train_jsonl_sha256",
            "development_jsonl_sha256",
            "config",
            "model_cache_manifest",
            "model_cache_manifest_sha256",
            "model_cache_tree_sha256",
            "source_commit",
            "selected_metric_value",
            "restart_state_sha256",
        },
        label,
        optional={"trainer_sha256", "data_adapter_sha256", "model_helper_sha256",
                  "base_model", "base_model_revision", "max_span_width",
                  "context_between_spans", "selected_metric"},
    )
    if value["schema_version"] not in {
        "encoder-checkpoint-1",
    }:
        raise DataContractError(f"{label}.schema_version is unsupported")
    if value["protocol_id"] != PROTOCOL_ID:
        raise DataContractError(f"{label}.protocol_id differs from {PROTOCOL_ID}")
    if value["split_id"] != _SPLIT_ID:
        raise DataContractError(f"{label}.split_id must be {_SPLIT_ID}")
    seed = _integer(value["training_seed"], f"{label}.training_seed")
    if seed not in TRAINING_SEEDS:
        raise DataContractError(f"{label}.training_seed is outside seeds 42-49")
    training = config.value["training"]
    for field in ("base_model", "base_model_revision", "max_span_width", "context_between_spans"):
        if field in value and value[field] != training[field]:
            raise DataContractError(
                f"{label}.{field} differs from the approved training recipe"
            )
    step = _integer(value["checkpoint_step"], f"{label}.checkpoint_step", minimum=1)
    if value.get("selected_metric", "development_strict_triple_f1") != "development_strict_triple_f1":
        raise DataContractError(f"{label}.selected_metric is unsupported")
    _unit_float(value["selected_metric_value"], f"{label}.selected_metric_value")
    if value["model_cache_manifest"] != HF_CACHE_MANIFEST_RELATIVE:
        raise DataContractError(f"{label}.model_cache_manifest is unsupported")
    if value["config"] != config.source_path:
        raise DataContractError(f"{label}.config differs from the selected source configuration")
    source_commit = value["source_commit"]
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise DataContractError(f"{label}.source_commit must be a full Git object ID")
    return CheckpointIdentity(
        training_seed=seed,
        checkpoint_sha256=_sha256_hex(value["checkpoint_sha256"], f"{label}.checkpoint_sha256"),
        checkpoint_step=step,
        split_manifest_sha256=_sha256_hex(
            value["split_manifest_sha256"], f"{label}.split_manifest_sha256"
        ),
        max_span_width=training["max_span_width"],
        archive_sha256=_sha256_hex(value["archive_sha256"], f"{label}.archive_sha256"),
        acquisition_manifest_sha256=_sha256_hex(
            value["acquisition_manifest_sha256"],
            f"{label}.acquisition_manifest_sha256",
        ),
        annotation_bundle_sha256=_sha256_hex(
            value["annotation_bundle_sha256"],
            f"{label}.annotation_bundle_sha256",
        ),
        prepared_dataset_tree_sha256=_sha256_hex(
            value["prepared_dataset_tree_sha256"],
            f"{label}.prepared_dataset_tree_sha256",
        ),
        train_jsonl_sha256=_sha256_hex(
            value["train_jsonl_sha256"], f"{label}.train_jsonl_sha256"
        ),
        development_jsonl_sha256=_sha256_hex(
            value["development_jsonl_sha256"],
            f"{label}.development_jsonl_sha256",
        ),
        config=value["config"],
        model_cache_manifest_sha256=_sha256_hex(
            value["model_cache_manifest_sha256"],
            f"{label}.model_cache_manifest_sha256",
        ),
        model_cache_tree_sha256=_sha256_hex(
            value["model_cache_tree_sha256"], f"{label}.model_cache_tree_sha256"
        ),
        source_commit=source_commit,
        restart_state_sha256=_sha256_hex(
            value["restart_state_sha256"], f"{label}.restart_state_sha256"
        ),
    )


def _generation_plan(
    sentences: dict[str, PreparedSentence], checkpoint: CheckpointIdentity
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for example_id in sorted(sentences):
        sentence = sentences[example_id]
        plan.append(
            {
                "protocol_id": PROTOCOL_ID,
                "example_id": example_id,
                "split": sentence.split,
                "source_document_id": sentence.source_document_id,
                "token_count": len(sentence.words),
                "training_seed": checkpoint.training_seed,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            }
        )
    return plan


def _validate_checkpoint_run_identity(
    layout: RunLayout, config: PipelineConfig, checkpoint: CheckpointIdentity
) -> None:
    """Reject a checkpoint detached from this run's fetched/prepared corpus."""

    acquisition_path = layout.resolve(
        "manifests/02-input-acquisition-manifest.json", must_exist=True
    )
    preparation_path = layout.resolve(
        "manifests/03-data-preparation-manifest.json", must_exist=True
    )
    split_path = layout.resolve("data-prepared/split-manifest.json", must_exist=True)
    train_path = layout.resolve("data-prepared/train.jsonl", must_exist=True)
    development_path = layout.resolve(
        "data-prepared/development.jsonl", must_exist=True
    )
    acquisition = load_json(acquisition_path)
    preparation = load_json(preparation_path)
    checkout = load_json(
        layout.resolve("manifests/00-checkout-manifest.json", must_exist=True)
    )
    restart_path = layout.resolve(
        f"checkpoints/seed-{checkpoint.training_seed}/restart-state.pt",
        must_exist=True,
    )
    cache_manifest = verify_cache_manifest(
        layout,
        model=config.value["training"]["base_model"],
        revision=config.value["training"]["base_model_revision"],
    )
    cache_manifest_path = layout.resolve(HF_CACHE_MANIFEST_RELATIVE, must_exist=True)
    expected = {
        "archive_sha256": preparation.get("archive_sha256"),
        "acquisition_manifest_sha256": sha256_file(acquisition_path),
        "annotation_bundle_sha256": preparation.get("annotation_bundle_sha256"),
        "prepared_dataset_tree_sha256": preparation.get("dataset_tree_sha256"),
        "split_manifest_sha256": sha256_file(split_path),
        "train_jsonl_sha256": sha256_file(train_path),
        "development_jsonl_sha256": sha256_file(development_path),
        "config": config.source_path,
        "model_cache_manifest_sha256": sha256_file(cache_manifest_path),
        "model_cache_tree_sha256": cache_manifest["tree_sha256"],
        "source_commit": checkout.get("source", {}).get("commit"),
        "restart_state_sha256": sha256_file(restart_path),
    }
    if acquisition.get("archive", {}).get("sha256") != expected["archive_sha256"]:
        raise DataContractError("acquisition/preparation archive identities differ")
    for field, actual in expected.items():
        if getattr(checkpoint, field) != actual:
            raise DataContractError(
                f"checkpoint {field} does not match the selected run"
            )


def generate_candidates(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    execution_mode: str,
    sentences_path: Path,
    checkpoint_manifest_path: Path,
    candidates_out_path: Path,
    prediction_ledger_path: Path | None = None,
    checkpoint_blob_path: Path | None = None,
    base_model: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Plan, replay, or run canonical CODE-STRICT-1 candidate generation for one seed.

    ``live`` execution runs the retained encoder to produce the prediction ledger
    and then applies the same deterministic ledger->candidate transform as
    ``replay``, so a live run and a replay of its produced ledger yield identical
    candidates. Live execution is externally gated (accelerator + checkpoint).
    """

    layout.require_existing()
    if execution_mode not in EXECUTION_MODES:
        raise DataContractError(f"unsupported model execution mode: {execution_mode!r}")
    if execution_mode == "replay" and prediction_ledger_path is None:
        raise DataContractError("replay requires a run-relative prediction ledger")
    if execution_mode != "replay" and prediction_ledger_path is not None:
        raise DataContractError("a supplied prediction ledger is accepted only by replay execution")
    if execution_mode == "live" and checkpoint_blob_path is None:
        raise DataContractError("live execution requires the run-relative checkpoint blob")
    if execution_mode != "live" and checkpoint_blob_path is not None:
        raise DataContractError("a checkpoint blob is accepted only by live execution")

    sentences = _load_sentences(sentences_path)
    checkpoint = _load_checkpoint_manifest(checkpoint_manifest_path, config)
    _validate_checkpoint_run_identity(layout, config, checkpoint)

    plan_path = candidates_out_path.parent / (
        f"seed-{checkpoint.training_seed}-generation-plan.jsonl"
    )
    live_ledger_path = candidates_out_path.parent / (
        f"seed-{checkpoint.training_seed}-prediction-ledger.jsonl"
    )
    sentence_splits = {sentence.split for sentence in sentences.values()}
    if len(sentence_splits) != 1:
        raise DataContractError("candidate generation requires one prepared split per invocation")
    sentence_split = sentence_splits.pop()
    manifest_path = layout.resolve(
        f"manifests/model-generate-candidates-{execution_mode}-"
        f"seed-{checkpoint.training_seed}-{sentence_split}.json"
    )
    planned_outputs = [manifest_path]
    planned_outputs.append(plan_path if execution_mode == "dry-run" else candidates_out_path)
    if execution_mode == "live":
        planned_outputs.append(live_ledger_path)
    if manifest_path.exists():
        previous = load_json(manifest_path)
        if (previous.get("execution_mode") != execution_mode
                or previous.get("training_seed") != checkpoint.training_seed
                or previous.get("inputs", {}).get("checkpoint_manifest", {}).get("sha256") != sha256_file(checkpoint_manifest_path)
                or previous.get("inputs", {}).get("prepared_sentences", {}).get("sha256") != sha256_file(sentences_path)):
            raise DataContractError("candidate producer identity differs")
        if execution_mode != "dry-run":
            ledger_binding = previous.get("inputs", {}).get("prediction_ledger", {})
            ledger = layout.resolve(ledger_binding.get("path", ""), must_exist=True)
            if sha256_file(ledger) != ledger_binding.get("sha256"):
                raise DataContractError("prediction ledger differs from its producer")
            validate_prediction_artifacts(sentences_path=sentences_path, prediction_ledger_path=ledger,
                candidates_path=candidates_out_path, training_seed=checkpoint.training_seed,
                max_span_width=checkpoint.max_span_width,
                input_hashes={k:v["sha256"] for k,v in previous["inputs"].items() if k in {"prepared_sentences","checkpoint_manifest","prediction_ledger"}})
        return previous

    inputs = {
        "prepared_sentences": {
            "path": layout.relative_identity(sentences_path),
            "sha256": sha256_file(sentences_path),
        },
        "checkpoint_manifest": {
            "path": layout.relative_identity(checkpoint_manifest_path),
            "sha256": sha256_file(checkpoint_manifest_path),
        },
        "prediction_ledger": None,
    }
    manifest: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "matcher_id": MATCHER_ID,
        "stage": "model-generate-candidates",
        "execution_mode": execution_mode,
        "created_at_utc": _utc_now(),
        "training_seed": checkpoint.training_seed,
        "checkpoint": {
            "sha256": checkpoint.checkpoint_sha256,
            "step": checkpoint.checkpoint_step,
            "split_manifest_sha256": checkpoint.split_manifest_sha256,
        },
        "inputs": inputs,
        "sentence_count": len(sentences),
    }

    if execution_mode == "dry-run":
        plan = _generation_plan(sentences, checkpoint)
        atomic_write_jsonl(plan_path, plan)
        manifest.update(
            {
                "status": "planned",
                "ledger_sentence_count": None,
                "selected_entity_count": None,
                "candidate_count": 0,
                "duplicate_candidate_collapsed": 0,
                "generation_plan_output": layout.relative_identity(plan_path),
                "candidates_output": None,
            }
        )
        atomic_write_json(manifest_path, manifest)
        return manifest

    if execution_mode == "live":
        blob_digest = sha256_file(checkpoint_blob_path)
        if blob_digest != checkpoint.checkpoint_sha256:
            raise DataContractError(
                "live checkpoint blob hash does not match the checkpoint manifest identity"
            )
        cache_manifest = verify_cache_manifest(
            layout,
            model=config.value["training"]["base_model"],
            revision=config.value["training"]["base_model_revision"],
        )
        cache_manifest_path = layout.resolve(
            HF_CACHE_MANIFEST_RELATIVE, must_exist=True
        )
        if (
            sha256_file(cache_manifest_path) != checkpoint.model_cache_manifest_sha256
            or cache_manifest["tree_sha256"] != checkpoint.model_cache_tree_sha256
        ):
            raise DataContractError(
                "run-local Hugging Face cache differs from the checkpoint identity"
            )
        from utils.common.execution import bind_json
        work_path = manifest_path.with_name(f"work-{manifest_path.name}")
        binding = {"run_id": layout.run_id, "inputs": inputs,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "config": config.source_path, "source_commit": checkpoint.source_commit}
        progress = load_json(work_path) if work_path.exists() else {"binding": binding}
        if not isinstance(progress, dict) or progress.get('binding') != binding:
            raise DataContractError('prediction work belongs to different run or inputs')
        saved_digest = progress.get('prediction_ledger_sha256')
        if live_ledger_path.exists():
            if saved_digest != sha256_file(live_ledger_path):
                raise DataContractError('prediction ledger lacks its original authenticated completion')
        elif saved_digest is not None:
            raise DataContractError('completed prediction ledger is missing')
        bind_json(work_path, progress)
        records = None
        if live_ledger_path.exists():
            validate_prediction_cache(layout, config, sentences_path=sentences_path,
                checkpoint_manifest_path=checkpoint_manifest_path, cache_ledger_path=live_ledger_path)
            records = [r for _, r in iter_jsonl(live_ledger_path)]
        if records is None:
            records = _live_inference_records(
                config,
                checkpoint,
                sentences,
                checkpoint_blob_path,
                base_model or config.value["training"]["base_model"],
                device,
                layout.resolve(CACHE_RELATIVE, must_exist=True),
            )
        atomic_write_jsonl(
            live_ledger_path,
            sorted(records, key=lambda record: record["example_id"]),
        )
        atomic_write_json(work_path, {'binding': binding,
                                     'prediction_ledger_sha256': sha256_file(live_ledger_path)})
        ledger_path = live_ledger_path
    else:
        ledger_path = prediction_ledger_path

    input_hashes = {
        "prepared_sentences": inputs["prepared_sentences"]["sha256"],
        "checkpoint_manifest": inputs["checkpoint_manifest"]["sha256"],
        "prediction_ledger": sha256_file(ledger_path),
    }
    inputs["prediction_ledger"] = {
        "path": layout.relative_identity(ledger_path),
        "sha256": input_hashes["prediction_ledger"],
    }
    candidates, sentence_count, selected_total, duplicate_collapsed = _load_prediction_ledger(
        ledger_path, sentences, checkpoint, input_hashes
    )
    expected = b"".join(canonical_json_bytes(row) for row in candidates)
    if candidates_out_path.exists() and candidates_out_path.read_bytes() != expected:
        raise DataContractError("unsealed candidates differ from the deterministic ledger transform")
    if not candidates_out_path.exists():
        atomic_write_bytes(candidates_out_path, expected)
    manifest.update(
        {
            "status": "completed",
            "ledger_sentence_count": sentence_count,
            "selected_entity_count": selected_total,
            "candidate_count": len(candidates),
            "duplicate_candidate_collapsed": duplicate_collapsed,
            "generation_plan_output": None,
            "candidates_output": layout.relative_identity(candidates_out_path),
        }
    )
    manifest["output_hashes"] = {layout.relative_identity(p): sha256_file(p) for p in (candidates_out_path, live_ledger_path if execution_mode == "live" else prediction_ledger_path) if p is not None and p.exists()}
    atomic_write_json(manifest_path, manifest)
    return manifest


def validate_prediction_cache(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    sentences_path: Path,
    checkpoint_manifest_path: Path,
    cache_ledger_path: Path,
) -> None:
    """Validate an interrupted live ledger without loading the encoder."""

    layout.require_existing()
    sentences = _load_sentences(sentences_path)
    checkpoint = _load_checkpoint_manifest(checkpoint_manifest_path, config)
    _validate_checkpoint_run_identity(layout, config, checkpoint)
    cache_sha = sha256_file(cache_ledger_path)
    _load_prediction_ledger(
        cache_ledger_path,
        sentences,
        checkpoint,
        {
            "prepared_sentences": sha256_file(sentences_path),
            "checkpoint_manifest": sha256_file(checkpoint_manifest_path),
            "prediction_ledger": cache_sha,
        },
    )


def validate_prediction_artifacts(
    *,
    sentences_path: Path,
    prediction_ledger_path: Path,
    candidates_path: Path,
    training_seed: int,
    max_span_width: int,
    input_hashes: dict[str, str],
) -> None:
    """Reconstruct and authenticate an inference ledger's candidate output."""

    sentences = _load_sentences(sentences_path)
    expected, _sentence_count, _selected, _duplicates = _load_prediction_ledger(
        prediction_ledger_path,
        sentences,
        PredictionLedgerIdentity(
            training_seed=training_seed,
            max_span_width=max_span_width,
        ),
        input_hashes,
    )
    observed = [value for _, value in iter_jsonl(candidates_path)]
    if observed != expected:
        raise DataContractError(
            "candidate output differs from deterministic prediction-ledger replay"
        )


def _required_run_file(layout: RunLayout, relative: str) -> Path:
    path = layout.resolve(relative)
    if not path.is_file():
        raise DataContractError(f"live training requires {relative}")
    return path


def canonical_trainer_arguments(layout: RunLayout, config: PipelineConfig, training_seed: int) -> EncoderOptions:
    """Derive every trainer argument from one validated run namespace."""
    if training_seed not in config.value.get('training_seeds', []):
        raise DataContractError(f'training_seed {training_seed} is not approved by the selected run config')
    training = config.value['training']
    boost = training['comparison_boost']
    checkpoint_dir = layout.resolve(f'checkpoints/seed-{training_seed}')
    checkpoint_path = checkpoint_dir / 'checkpoint.pt'
    restart_path = checkpoint_dir / 'restart-state.pt'
    progress_path = layout.resolve(f'logs/model-train-seed-{training_seed}.log')
    summary_path = checkpoint_dir / 'training-summary.json'
    command = EncoderOptions(
        prepared_dir=str(layout.resolve('data-prepared', must_exist=True)),
        model_name=training['base_model'],
        model_revision=training['base_model_revision'],
        model_cache_dir=str(layout.resolve(CACHE_RELATIVE)),
        batch_size=training['batch_size'],
        max_length=training['max_length'],
        lr=float(training['learning_rate']),
        max_steps=training['max_steps'],
        warmup_steps=training['warmup_steps'],
        max_span_width=training['max_span_width'],
        re_weight=float(training['re_loss_weight']),
        re_no_rel_weight=float(training['re_no_rel_weight']),
        neg_sample_ratio=float(training['ner_negative_ratio']),
        focal_gamma=float(training['ner_focal_gamma']),
        eval_every=training['evaluation_every_steps'],
        seed=training_seed,
        label_smoothing=float(training['label_smoothing']),
        re_comparison_boost=float(boost['initial']),
        re_boost_adaptive_steps=boost['adaptive_step'],
        re_boost_adaptive_threshold=float(boost['threshold_low']),
        re_boost_adaptive_threshold2=float(boost['threshold_high']),
        re_boost_mid=float(boost['middle']),
        re_boost_end=float(boost['low']),
        save_best_to=str(checkpoint_path),
        save_last_to=str(restart_path),
        progress_log=str(progress_path),
        run_summary_out=str(summary_path),
    )
    if layout.resolve(HF_CACHE_MANIFEST_RELATIVE).is_file():
        command.model_local_files_only = True
    if restart_path.exists() and (not summary_path.exists()):
        command = replace(command,
            resume_from=str(restart_path),
        )
    return command


def _training_command(
    layout: RunLayout,
    config: PipelineConfig,
    training_seed: int,
) -> list[str]:
    """Invoke the private trainer with namespace keys, never artifact paths."""

    return [
        sys.executable,
        "-B",
        "pipeline.py",
        "_train-encoder",
        "--run-id",
        layout.run_id,
        "--training-seed",
        str(training_seed),
        "--config",
        str(config.path.resolve()),
    ]


def plan_training(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    execution_mode: str,
    training_seed: int,
    command_runner=subprocess.run,
) -> dict[str, Any]:
    """Plan or execute compatibility-hosted canonical training for one seed."""

    layout.require_existing()
    if execution_mode not in TRAIN_EXECUTION_MODES:
        raise DataContractError(f"unsupported model-train execution mode: {execution_mode!r}")
    if training_seed not in TRAINING_SEEDS:
        raise DataContractError(f"training_seed {training_seed} is outside seeds 42-49")

    split = config.value["split"]
    if split.get("split_id") != _SPLIT_ID:
        raise DataContractError(f"config.split.split_id must be {_SPLIT_ID}")

    manifest_name = f"model-train-{execution_mode}-seed-{training_seed}.json"
    manifest_path = layout.resolve(f"manifests/{manifest_name}")
    if manifest_path.exists():
        raise DataContractError(
            "model train refuses to overwrite existing artifact: "
            + layout.relative_identity(manifest_path)
        )

    manifest: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "matcher_id": MATCHER_ID,
        "stage": "model-train",
        "run_id": layout.run_id,
        "execution_mode": execution_mode,
        "status": "planned" if execution_mode == "dry-run" else "completed",
        "created_at_utc": _utc_now(),
        "training_seed": training_seed,
        "config": config.source_path,
        "expected_checkpoint_dir": f"checkpoints/seed-{training_seed}",
        "final_test_selection_forbidden": True,
    }
    if execution_mode == "dry-run":
        atomic_write_json(manifest_path, manifest)
        return manifest

    checkout_path = _required_run_file(layout, "manifests/00-checkout-manifest.json")
    acquisition_path = _required_run_file(
        layout, "manifests/02-input-acquisition-manifest.json"
    )
    preparation_path = _required_run_file(
        layout, "manifests/03-data-preparation-manifest.json"
    )
    checkout = load_json(checkout_path)
    acquisition = load_json(acquisition_path)
    preparation = load_json(preparation_path)
    if checkout.get("status") != "pass":
        raise DataContractError("live training requires a passing checkout manifest")
    if preparation.get("byte_identical_independent_materializations") is not True:
        raise DataContractError("live training requires verified deterministic preparation")
    if preparation.get("acquisition_manifest_sha256") != sha256_file(acquisition_path):
        raise DataContractError("preparation is not bound to this acquisition manifest")
    archive_sha = acquisition.get("archive", {}).get("sha256")
    if preparation.get("archive_sha256") != archive_sha:
        raise DataContractError("acquisition/preparation archive identities differ")

    split_path = _required_run_file(layout, "data-prepared/split-manifest.json")
    train_path = _required_run_file(layout, "data-prepared/train.jsonl")
    development_path = _required_run_file(layout, "data-prepared/development.jsonl")
    checkpoint_dir = layout.resolve(f"checkpoints/seed-{training_seed}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    restart_path = checkpoint_dir / "restart-state.pt"
    summary_path = checkpoint_dir / "training-summary.json"
    checkpoint_manifest_path = checkpoint_dir / "checkpoint-manifest.json"
    progress_path = layout.resolve(f"logs/model-train-seed-{training_seed}.log")
    model_cache_dir = layout.resolve(CACHE_RELATIVE)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = layout.resolve(HF_CACHE_MANIFEST_RELATIVE)
    cache_is_frozen = cache_manifest_path.exists()
    if cache_is_frozen:
        verify_cache_manifest(
            layout,
            model=config.value["training"]["base_model"],
            revision=config.value["training"]["base_model_revision"],
        )
    if checkpoint_path.exists() and not restart_path.exists() and not summary_path.exists():
        raise DataContractError("orphan checkpoint without restart or summary blocks training")

    command = _training_command(layout, config, training_seed)
    if not summary_path.exists():
        try:
            from utils.common.execution import checked_process
            checked_process(layout, command, label=f'canonical-train-seed-{training_seed}',
                            command_runner=command_runner)
        except subprocess.CalledProcessError as exc:
            raise DataContractError(
                f"canonical trainer failed with exit status {exc.returncode}; "
                "rerun the same command to resume from its last complete state"
            ) from exc

    cache_manifest = (
        verify_cache_manifest(
            layout,
            model=config.value["training"]["base_model"],
            revision=config.value["training"]["base_model_revision"],
        )
        if cache_is_frozen
        else write_cache_manifest(
            layout,
            model=config.value["training"]["base_model"],
            revision=config.value["training"]["base_model_revision"],
        )
    )

    for required_path in (checkpoint_path, restart_path, summary_path, progress_path):
        if not required_path.is_file():
            raise DataContractError(
                "canonical trainer did not produce " + layout.relative_identity(required_path)
            )
    summary = load_json(summary_path)
    from utils.encoder.runtime import restart_identity
    if (
        summary.get("status") != "completed"
        or summary.get("seed") != training_seed
        or summary.get("test_evaluated") is not False
        or summary.get('completed_steps') != config.value['training']['max_steps']
        or summary.get('run_identity') != restart_identity(layout.resolve('data-prepared'))
    ):
        raise DataContractError("canonical trainer summary violates the live contract")
    checkpoint_step = summary.get("selected_step")
    selected_metrics = summary.get("selected_metrics", {})
    selected_value = selected_metrics.get("triple_f1")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 1
        or checkpoint_step > config.value['training']['max_steps']
        or isinstance(selected_value, bool)
        or not isinstance(selected_value, (int, float))
        or not math.isfinite(selected_value) or not 0 <= selected_value <= 1
    ):
        raise DataContractError("canonical trainer summary lacks a selected checkpoint metric")

    checkpoint_manifest = {
        "schema_version": "encoder-checkpoint-1",
        "protocol_id": PROTOCOL_ID,
        "split_id": _SPLIT_ID,
        "training_seed": training_seed,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "split_manifest_sha256": sha256_file(split_path),
        "archive_sha256": archive_sha,
        "acquisition_manifest_sha256": sha256_file(acquisition_path),
        "annotation_bundle_sha256": preparation["annotation_bundle_sha256"],
        "prepared_dataset_tree_sha256": preparation["dataset_tree_sha256"],
        "train_jsonl_sha256": sha256_file(train_path),
        "development_jsonl_sha256": sha256_file(development_path),
        "config": config.source_path,
        "model_cache_manifest": HF_CACHE_MANIFEST_RELATIVE,
        "model_cache_manifest_sha256": sha256_file(cache_manifest_path),
        "model_cache_tree_sha256": cache_manifest["tree_sha256"],
        "source_commit": checkout.get("source", {}).get("commit"),
        "selected_metric_value": float(selected_value),
        "restart_state_sha256": sha256_file(restart_path),
    }
    if checkpoint_manifest_path.exists():
        if load_json(checkpoint_manifest_path) != checkpoint_manifest:
            raise DataContractError("existing checkpoint manifest differs from training outputs")
    else:
        atomic_write_json(checkpoint_manifest_path, checkpoint_manifest)
    manifest.update(
        {
            "inputs": {
                "checkout_manifest_sha256": sha256_file(checkout_path),
                "acquisition_manifest_sha256": sha256_file(acquisition_path),
                "preparation_manifest_sha256": sha256_file(preparation_path),
                "split_manifest_sha256": sha256_file(split_path),
                "train_jsonl_sha256": sha256_file(train_path),
                "development_jsonl_sha256": sha256_file(development_path),
                "model_cache_manifest_sha256": sha256_file(cache_manifest_path),
            },
            "outputs": {
                "checkpoint": layout.relative_identity(checkpoint_path),
                "checkpoint_manifest": layout.relative_identity(checkpoint_manifest_path),
                "restart_state": layout.relative_identity(restart_path),
                "training_summary": layout.relative_identity(summary_path),
                "progress_log": layout.relative_identity(progress_path),
                "model_cache_manifest": HF_CACHE_MANIFEST_RELATIVE,
            },
            "resume": {
                "resumed": summary.get("resumed_from") is not None,
                "restart_state_sha256": sha256_file(restart_path),
            },
            "environment": summary.get("runtime_environment", summary.get("environment")),
            "output_hashes": {layout.relative_identity(p): sha256_file(p) for p in (checkpoint_path, checkpoint_manifest_path, restart_path, summary_path, progress_path, cache_manifest_path)},
            }
    )
    atomic_write_json(manifest_path, manifest)
    return manifest
