"""Validation for the frozen Path A configuration and experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from constants import (
    CODE_ACCORD_ANNOTATION_FILES,
    CODE_ACCORD_ARCHIVE,
    CODE_ACCORD_COUNTS,
    CODE_ACCORD_RELATION_SPLIT_AUDIT,
    CODE_ACCORD_REPAIRED_UUID,
    CONDITION_IDS,
    MATCHER_ID,
    MATRIX_REVISION,
    PROTOCOL_ID,
    SECTION5_CLAIM_IDS,
    SECTION5_EVIDENCE_REVISION,
    TRAINING_SEEDS,
    WORKFLOW_ID,
)
from phase_b_io import DataContractError, load_json, sha256_file
from paths import resolve_tracked_path


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be a JSON object")
    return value


def _required(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise DataContractError(f"{label} is missing required field {key!r}")
    return mapping[key]


def _find_tbd(value: Any, location: str = "config") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and "TBD_" in value:
        found.append(location)
    elif isinstance(value, dict):
        for key, nested in value.items():
            found.extend(_find_tbd(nested, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_find_tbd(nested, f"{location}[{index}]"))
    return found


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    value: dict[str, Any]
    matrix_path: Path
    matrix: dict[str, Any]
    section5_evidence_path: Path
    section5_evidence: dict[str, Any]

    @property
    def paths(self) -> dict[str, str]:
        return self.value["runtime_paths"]


def load_pipeline_config(source_root: Path, supplied: str | Path) -> PipelineConfig:
    path = resolve_tracked_path(source_root, supplied)
    value = _object(load_json(path), "config")
    tbd_locations = _find_tbd(value)
    if tbd_locations:
        raise DataContractError(
            "publication configuration contains unresolved TBD fields: "
            + ", ".join(tbd_locations)
        )

    expected_identity = {
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "matcher_id": MATCHER_ID,
    }
    for field, expected in expected_identity.items():
        actual = _required(value, field, "config")
        if actual != expected:
            raise DataContractError(
                f"config {field} mismatch: expected {expected!r}, got {actual!r}"
            )

    dataset = _object(_required(value, "dataset", "config"), "config.dataset")
    expected_dataset = {
        "dataset_id": CODE_ACCORD_ARCHIVE["dataset_id"],
        "zenodo_record": CODE_ACCORD_ARCHIVE["zenodo_record"],
        "archive_name": CODE_ACCORD_ARCHIVE["name"],
        "archive_url": CODE_ACCORD_ARCHIVE["url"],
        "archive_bytes": CODE_ACCORD_ARCHIVE["bytes"],
        "archive_md5": CODE_ACCORD_ARCHIVE["md5"],
        "license": CODE_ACCORD_ARCHIVE["license"],
        "annotation_files": CODE_ACCORD_ANNOTATION_FILES,
        "expected_counts": CODE_ACCORD_COUNTS,
        "relation_split_audit": CODE_ACCORD_RELATION_SPLIT_AUDIT,
        "repaired_uuid": CODE_ACCORD_REPAIRED_UUID,
    }
    if dataset != expected_dataset:
        raise DataContractError("config.dataset differs from the immutable CODE-ACCORD contract")

    seeds = _required(value, "training_seeds", "config")
    if seeds != list(TRAINING_SEEDS):
        raise DataContractError(
            f"config training_seeds must be exactly {list(TRAINING_SEEDS)}"
        )

    split = _object(_required(value, "split", "config"), "config.split")
    expected_split = {
        "split_id": "CODE-SPLIT-1",
        "seed": 42,
        "train_sentences": 586,
        "development_sentences": 103,
        "test_sentences": 173,
    }
    for field, expected in expected_split.items():
        if split.get(field) != expected:
            raise DataContractError(
                f"config.split.{field} must be {expected!r}, got {split.get(field)!r}"
            )

    statistics = _object(
        _required(value, "statistics", "config"), "config.statistics"
    )
    if statistics.get("bootstrap_replicates") != 10000:
        raise DataContractError("config.statistics.bootstrap_replicates must be 10000")
    if statistics.get("bootstrap_seed") != 20260717:
        raise DataContractError("config.statistics.bootstrap_seed must be 20260717")

    expected_grid = [index / 20 for index in range(20)]
    threshold = _object(
        _required(value, "threshold_selection", "config"),
        "config.threshold_selection",
    )
    expected_threshold = {
        "selection_split": "development",
        "grid": expected_grid,
        "objective": "mean_per_seed_development_strict_triple_f1",
        "tie_rule": "higher_threshold",
    }
    for field, expected in expected_threshold.items():
        if threshold.get(field) != expected:
            raise DataContractError(
                f"config.threshold_selection.{field} differs from the approved protocol"
            )

    training = _object(_required(value, "training", "config"), "config.training")
    expected_training = {
        "base_model": "microsoft/deberta-large",
        "base_model_revision": "9a8befc6d3fbfa800e65f5279aa34d27eaf6d1b0",
        "max_steps": 3500,
        "evaluation_every_steps": 100,
        "batch_size": 16,
        "max_length": 128,
        "max_span_width": 8,
        "learning_rate": 3e-5,
        "warmup_steps": 250,
        "ner_negative_ratio": 3.0,
        "ner_focal_gamma": 2.0,
        "label_smoothing": 0.1,
        "re_loss_weight": 1.0,
        "re_no_rel_weight": 1.0,
        "re_focal_gamma": 0.0,
        "re_negative_subsample": 0.0,
        "context_between_spans": True,
        "document_window_size": 1,
        "synthetic_data": "none",
        "checkpoint_metric": "development_strict_triple_f1",
        "checkpoint_tie_rule": "earliest_step",
    }
    for field, expected in expected_training.items():
        if training.get(field) != expected:
            raise DataContractError(f"config.training.{field} differs from the approved recipe")
    expected_boost = {
        "initial": 5.0,
        "adaptive_step": 1000,
        "threshold_low": 0.35,
        "threshold_high": 0.4,
        "middle": 3.5,
        "low": 2.0,
        "middle_applies_to_gate_batch_only": True,
        "post_gate_boost_after_middle": 2.0,
    }
    if training.get("comparison_boost") != expected_boost:
        raise DataContractError("config.training.comparison_boost differs from A20")

    verifier = _object(_required(value, "verifier", "config"), "config.verifier")
    expected_verifier = {
        "model": "qwen3:32b",
        "registry_manifest_sha256": (
            "030ee887880fc378860c2dd35101da424377520441ae4bfe7be6deff8ade7840"
        ),
        "model_blob_sha256": "3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312",
        "prompt_status": "materialized_b06",
        "prompt_revision": "CODE-VERIFIER-1",
        "system_prompt": "prompts/phase_b/code-verifier-1-system.txt",
        "system_prompt_sha256": "d5be0378a4f0c68bb7e8038ca4008490fafb27934c0996b8d022200567126a04",
        "simple_prompt": "prompts/phase_b/code-verifier-1-simple.txt",
        "simple_prompt_sha256": "6f53d575c990571a882e6e6b0fabf4a8c2ca7049b8af2fb0cc15abd1f54aa537",
        "corrective_prompt": "prompts/phase_b/code-verifier-1-corrective.txt",
        "corrective_prompt_sha256": (
            "180e39c5a6cebd51620ee378a45c4da1cc2a5727d06de24a6717445f58fd36a3"
        ),
        "simple_response_schema": "schemas/phase_b/verifier-simple-response.schema.json",
        "simple_response_schema_sha256": (
            "86a93ff4227d65800e76704454f0e1b1a8c7729df506c18cc66a1018cbc253d9"
        ),
        "corrective_response_schema": "schemas/phase_b/verifier-corrective-response.schema.json",
        "corrective_response_schema_sha256": (
            "ff2a0e41359c230c98b8f754e57c28a4518d8bfbdd17114d8b743ab932065236"
        ),
        "simple_bundle_sha256": "1f663794ac3c2b43398da9afc24db0d1e8ac19d6648f110f540d5b9726bcc6c5",
        "corrective_bundle_sha256": (
            "369cd3681d3339879872ff2a91d7b1ac89ce4ffbf184473c59b7799f57d36203"
        ),
        "stream": False,
        "think": False,
        "temperature": 0,
        "seed": 42,
        "top_k": 1,
        "top_p": 1,
        "min_p": 0,
        "repeat_penalty": 1,
        "num_ctx": 8192,
        "num_predict": 256,
        "timeout_seconds": 300,
        "attempts": 3,
        "backoff_seconds": [2, 4],
    }
    for field, expected in expected_verifier.items():
        if verifier.get(field) != expected:
            raise DataContractError(f"config.verifier.{field} differs from the approved protocol")
    verifier_resources = {
        "system_prompt": "system_prompt_sha256",
        "simple_prompt": "simple_prompt_sha256",
        "corrective_prompt": "corrective_prompt_sha256",
        "simple_response_schema": "simple_response_schema_sha256",
        "corrective_response_schema": "corrective_response_schema_sha256",
    }
    for path_field, hash_field in verifier_resources.items():
        resource = resolve_tracked_path(source_root, verifier[path_field])
        if sha256_file(resource) != verifier[hash_field]:
            raise DataContractError(
                f"config.verifier.{hash_field} does not match {verifier[path_field]}"
            )

    environment = _object(
        _required(value, "environment", "config"), "config.environment"
    )
    if environment.get("python") != "3.10.20" or environment.get("uv") != "0.11.26":
        raise DataContractError("config environment must pin Python 3.10.20 and uv 0.11.26")

    runtime_paths = _object(
        _required(value, "runtime_paths", "config"), "config.runtime_paths"
    )
    for field in (
        "sentences",
        "warmup_sentences",
        "warmup_candidates",
        "split_manifest",
        "development_gold",
        "development_candidate_index",
        "development_pilot_candidates",
        "verifier_pilot_selection",
        "gold",
        "candidates",
        "simple_verdicts",
        "corrective_verdicts",
        "threshold_selection",
    ):
        raw = runtime_paths.get(field)
        if (
            not isinstance(raw, str)
            or not raw
            or Path(raw).is_absolute()
            or ".." in Path(raw).parts
        ):
            raise DataContractError(
                f"config.runtime_paths.{field} must be a nonempty run-relative path"
            )

    matrix_raw = _required(value, "experiment_matrix", "config")
    if not isinstance(matrix_raw, str):
        raise DataContractError("config.experiment_matrix must be a tracked relative path")
    matrix_path = resolve_tracked_path(source_root, matrix_raw)
    matrix = _object(load_json(matrix_path), "experiment matrix")
    _validate_matrix(matrix)

    evidence_raw = _required(value, "section5_evidence_register", "config")
    if not isinstance(evidence_raw, str):
        raise DataContractError(
            "config.section5_evidence_register must be a tracked relative path"
        )
    evidence_path = resolve_tracked_path(source_root, evidence_raw)
    evidence = _object(load_json(evidence_path), "Section 5 evidence register")
    _validate_section5_evidence_identity(evidence)
    return PipelineConfig(
        path=path,
        value=value,
        matrix_path=matrix_path,
        matrix=matrix,
        section5_evidence_path=evidence_path,
        section5_evidence=evidence,
    )


def _validate_section5_evidence_identity(evidence: dict[str, Any]) -> None:
    expected = {
        "register_revision": SECTION5_EVIDENCE_REVISION,
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "authority": "secondary_only",
        "canonical_publication_eligible": False,
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            raise DataContractError(
                f"Section 5 evidence register {field} must be {expected_value!r}"
            )
    claims = evidence.get("claim_families")
    if not isinstance(claims, list):
        raise DataContractError("Section 5 evidence register claim_families must be an array")
    ids = [claim.get("claim_id") for claim in claims if isinstance(claim, dict)]
    if ids != list(SECTION5_CLAIM_IDS):
        raise DataContractError(
            "Section 5 evidence register claim families must be ordered exactly as "
            f"{list(SECTION5_CLAIM_IDS)}"
        )


def _validate_matrix(matrix: dict[str, Any]) -> None:
    expected = {
        "matrix_revision": MATRIX_REVISION,
        "protocol_id": PROTOCOL_ID,
        "workflow_id": WORKFLOW_ID,
        "matcher_id": MATCHER_ID,
    }
    for field, expected_value in expected.items():
        if matrix.get(field) != expected_value:
            raise DataContractError(
                f"experiment matrix {field} must be {expected_value!r}"
            )
    if matrix.get("training_seeds") != list(TRAINING_SEEDS):
        raise DataContractError("experiment matrix seed list differs from protocol")
    rows = matrix.get("conditions")
    if not isinstance(rows, list):
        raise DataContractError("experiment matrix conditions must be an array")
    ids = [row.get("condition_id") for row in rows if isinstance(row, dict)]
    if ids != list(CONDITION_IDS):
        raise DataContractError(
            f"experiment matrix conditions must be ordered exactly as {list(CONDITION_IDS)}"
        )
    for row in rows:
        for field in (
            "condition_id",
            "status",
            "baseline_id",
            "verifier_mode",
            "candidate_decision",
            "end_to_end_emission",
            "claim_ids",
            "publication_targets",
        ):
            if field not in row:
                raise DataContractError(
                    f"experiment matrix row {row.get('condition_id')} lacks {field}"
                )
    expected_modes = {
        "VER-RAW": ("PRIMARY_REFERENCE", None, "none"),
        "VER-CONFIDENCE": ("PRIMARY", "VER-RAW", "none"),
        "VER-SIMPLE": ("PRIMARY", "VER-RAW", "simple"),
        "VER-CORRECTIVE": ("PRIMARY", "VER-RAW", "corrective"),
    }
    for row in rows:
        expected_status, expected_baseline, expected_mode = expected_modes[row["condition_id"]]
        if (
            row["status"] != expected_status
            or row["baseline_id"] != expected_baseline
            or row["verifier_mode"] != expected_mode
        ):
            raise DataContractError(
                f"experiment matrix row {row['condition_id']} changes its frozen role"
            )
