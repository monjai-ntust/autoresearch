"""Fail-closed Phase G encoder-profile and recipe configuration."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.common.artifact_io import (
    DataContractError,
    atomic_write_json,
    load_json,
    sha256_file,
)
from utils.common.config import PipelineConfig, load_pipeline_config
from utils.common.constants import (
    MATCHER_ID,
    PROTOCOL_ID,
    TRAINING_SEEDS,
)
from utils.common.paths import RunLayout, resolve_tracked_path


SCHEMA_VERSION = "phase-g-encoder-comparison-1.2"
SELECTION_SCHEMA_VERSION = "phase-g-encoder-selection-1.0"
DEFAULT_CONFIG = "resources/configs/encoder-comparison.json"
HISTORICAL_COMMIT = "9feafa4029e65ab48ecfb2f452f4b0fabbff0826"
HISTORICAL_BLOBS = {
    "trainer": ("train_span.py", "104ca6e4afcb5122228fff5c73227c6b3b3edb40"),
    "network": (
        "models/bert_kg_encoder.py",
        "d05678f00097cb11176dacef6fade0b1b3f557de",
    ),
    "data_adapter": (
        "data/code_accord.py",
        "0b11fc6bf2f16512864b4dbf77e7a610d8fa8fdb",
    ),
}
PROFILE_FIELDS = {
    "profile_id",
    "model",
    "revision",
    "hidden_size",
    "tokenizer_files",
    "weight",
    "base_parameter_count_source",
    "license",
    "source",
}
RECIPE_FIELDS = {
    "recipe_id",
    "source_classification",
    "max_steps",
    "evaluation_every_steps",
    "batch_size",
    "max_length",
    "max_span_width",
    "learning_rate",
    "warmup_steps",
    "ner_negative_ratio",
    "ner_focal_gamma",
    "bio_loss_weight",
    "label_smoothing",
    "re_loss_weight",
    "re_no_rel_weight",
    "context_between_spans",
    "comparison_boost",
}
ARM_FIELDS = {
    "arm_id",
    "row_label",
    "claim_role",
    "profile_id",
    "recipe_id",
    "historical_reported_development",
}
BASE_PARAMETER_COUNT_SOURCE = (
    "sum_numel_from_pinned_pytorch_model_bin_state_dict_after_digest_verification"
)
EXPECTED_ARM_IDENTITIES = {
    "bert-base-common": (
        "BERT-base",
        "backbone_capacity",
        "bert-base-uncased",
        "historical-common-base",
    ),
    "deberta-base-common": (
        "DeBERTa-base",
        "backbone_capacity",
        "deberta-base-v1",
        "historical-common-base",
    ),
    "deberta-large-common": (
        "DeBERTa-large",
        "backbone_capacity_and_recipe_reference",
        "deberta-large-v1",
        "historical-common-base",
    ),
    "deberta-large-a20-a21-a12": (
        "DeBERTa-large + comparison-weight schedule + span-NER label smoothing + inter-span context",
        "recipe_method",
        "deberta-large-v1",
        "historical-a20-a21-a12",
    ),
}
EXPECTED_STATISTICAL_PROTOCOL = {
    "primary_endpoint": "per_seed_CODE_STRICT_1_test_f1",
    "primary_effect": "mean_of_eight_paired_seed_f1_differences",
    "paired_test": "two_sided_exact_wilcoxon_signed_rank",
    "multiplicity": "holm",
    "alpha": 0.05,
    "paired_seed_bootstrap_replicates": 10000,
    "hierarchical_seed_document_bootstrap_replicates": 10000,
    "bootstrap_seed": 20260919,
    "confidence_level": 0.95,
    "minimum_meaningful_absolute_f1": 0.01,
    "reported_per_arm": [
        "per_seed_f1",
        "mean_f1",
        "sample_sd_f1",
        "best_seed_f1",
        "pooled_tp",
        "pooled_fp",
        "pooled_fn",
        "pooled_precision",
        "pooled_recall",
        "pooled_f1",
    ],
    "paired_contrasts": [
        ["deberta-base-common", "bert-base-common"],
        ["deberta-large-common", "bert-base-common"],
        ["deberta-large-common", "deberta-base-common"],
        ["deberta-large-a20-a21-a12", "deberta-large-common"],
    ],
    "missing_run_policy": (
        "retain every outcome; do not replace a failed seed; an arm missing any "
        "seed is incomplete and excluded from confirmatory contrasts"
    ),
    "historical_statistics_role": "provenance_only",
}
EXPECTED_EXECUTION_LIMITS = {
    "maximum_total_gpu_hours": 48,
    "minimum_free_storage_bytes_before_full_run": 200000000000,
    "stop_on_first_nonrecoverable_arm_failure": True,
    "final_test_access": (
        "only after all four arms pass smoke validation and all development runs "
        "are complete or dispositioned"
    ),
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise DataContractError(
            f"{label} field contract differs; missing={missing}, extra={extra}"
        )


def _full_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ):
        raise DataContractError(f"{label} must be {length} lowercase hex characters")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value):
        raise DataContractError(f"{label} must be a lowercase stable identifier")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DataContractError(f"{label} must be a positive integer")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataContractError(f"{label} must be numeric")
    numeric = float(value)
    if numeric < minimum:
        raise DataContractError(f"{label} must be >= {minimum}")
    return numeric


@dataclass(frozen=True)
class EncoderProfile:
    profile_id: str
    model: str
    revision: str
    hidden_size: int
    tokenizer_files: tuple[str, ...]
    weight_path: str
    weight_bytes: int
    weight_sha256: str
    base_parameter_count_source: str
    license: str
    source: str

    def manifest(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model": self.model,
            "revision": self.revision,
            "hidden_size": self.hidden_size,
            "tokenizer_files": list(self.tokenizer_files),
            "weight": {
                "path": self.weight_path,
                "bytes": self.weight_bytes,
                "sha256": self.weight_sha256,
            },
            "base_parameter_count_source": self.base_parameter_count_source,
            "license": self.license,
            "source": self.source,
        }


@dataclass(frozen=True)
class ComparisonRecipe:
    recipe_id: str
    value: dict[str, Any]


@dataclass(frozen=True)
class Table1Arm:
    arm_id: str
    row_label: str
    claim_role: str
    profile_id: str
    recipe_id: str
    historical_reported_development: dict[str, float | None]

    def manifest(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "row_label": self.row_label,
            "claim_role": self.claim_role,
            "profile_id": self.profile_id,
            "recipe_id": self.recipe_id,
            "historical_reported_development": copy.deepcopy(
                self.historical_reported_development
            ),
        }


@dataclass(frozen=True)
class ComparisonConfig:
    path: Path
    sha256: str
    value: dict[str, Any]
    pipeline: PipelineConfig
    profiles: dict[str, EncoderProfile]
    recipes: dict[str, ComparisonRecipe]
    arms: dict[str, Table1Arm]

    def select_arm(self, arm_id: str) -> "ComparisonSelection":
        try:
            arm = self.arms[arm_id]
        except KeyError as exc:
            raise DataContractError(
                f"unknown approved Table 1 arm: {arm_id!r}"
            ) from exc
        profile_id = arm.profile_id
        recipe_id = arm.recipe_id
        try:
            profile = self.profiles[profile_id]
        except KeyError as exc:
            raise DataContractError(
                f"unknown encoder comparison profile: {profile_id!r}"
            ) from exc
        try:
            recipe = self.recipes[recipe_id]
        except KeyError as exc:
            raise DataContractError(
                f"unknown encoder comparison recipe: {recipe_id!r}"
            ) from exc
        return ComparisonSelection(
            config=self,
            arm=arm,
            profile=profile,
            recipe=recipe,
        )

    def select(self, profile_id: str, recipe_id: str) -> "ComparisonSelection":
        matches = [
            arm
            for arm in self.arms.values()
            if arm.profile_id == profile_id and arm.recipe_id == recipe_id
        ]
        if len(matches) != 1:
            raise DataContractError(
                "encoder profile/recipe pair is not one approved Table 1 arm"
            )
        return self.select_arm(matches[0].arm_id)


@dataclass(frozen=True)
class ComparisonSelection:
    config: ComparisonConfig
    arm: Table1Arm
    profile: EncoderProfile
    recipe: ComparisonRecipe

    @property
    def training(self) -> dict[str, Any]:
        recipe = copy.deepcopy(self.recipe.value)
        recipe.pop("recipe_id")
        recipe.pop("source_classification")
        return {
            "base_model": self.profile.model,
            "base_model_revision": self.profile.revision,
            **recipe,
        }

    def pipeline_view(self) -> PipelineConfig:
        value = copy.deepcopy(self.config.pipeline.value)
        training = self.training
        boost = training.pop("comparison_boost")
        training["comparison_boost"] = {
            "initial": boost["initial"],
            "adaptive_step": boost["adaptive_step"],
            "threshold_low": boost["threshold_low"],
            "threshold_high": boost["threshold_high"],
            "middle": boost["middle"],
            "low": boost["end"],
        }
        training.pop("bio_loss_weight")
        value["training"] = training
        return PipelineConfig(
            path=self.config.path,
            value=value,
            matrix_path=self.config.pipeline.matrix_path,
            matrix=self.config.pipeline.matrix,
            section5_evidence_path=self.config.pipeline.section5_evidence_path,
            section5_evidence=self.config.pipeline.section5_evidence,
        )

    def manifest(self, run_id: str | None = None) -> dict[str, Any]:
        manifest = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "matcher_id": MATCHER_ID,
            "split_id": "CODE-SPLIT-1",
            "comparison_config": {
                "path": self.config.path.relative_to(
                    self.config.pipeline.path.parents[2]
                ).as_posix(),
                "sha256": self.config.sha256,
            },
            "historical_source": copy.deepcopy(
                self.config.value["historical_source"]
            ),
            "table1_arm": self.arm.manifest(),
            "encoder_profile": self.profile.manifest(),
            "recipe": copy.deepcopy(self.recipe.value),
        }
        if run_id is not None:
            manifest["run_id"] = run_id
        return manifest


def _validate_profile(value: Any, index: int) -> EncoderProfile:
    label = f"encoder_profiles[{index}]"
    profile = _object(value, label)
    _exact_keys(profile, PROFILE_FIELDS, label)
    profile_id = _identifier(profile["profile_id"], f"{label}.profile_id")
    model = profile["model"]
    if not isinstance(model, str) or not model:
        raise DataContractError(f"{label}.model must be a nonempty string")
    revision = _full_hex(profile["revision"], 40, f"{label}.revision")
    hidden_size = _positive_integer(profile["hidden_size"], f"{label}.hidden_size")
    tokenizer_files = profile["tokenizer_files"]
    if (
        not isinstance(tokenizer_files, list)
        or not tokenizer_files
        or not all(isinstance(item, str) and item for item in tokenizer_files)
        or len(set(tokenizer_files)) != len(tokenizer_files)
    ):
        raise DataContractError(f"{label}.tokenizer_files must be unique file names")
    weight = _object(profile["weight"], f"{label}.weight")
    _exact_keys(weight, {"path", "bytes", "sha256"}, f"{label}.weight")
    if weight["path"] != "pytorch_model.bin":
        raise DataContractError(f"{label}.weight.path must be pytorch_model.bin")
    weight_bytes = _positive_integer(weight["bytes"], f"{label}.weight.bytes")
    weight_sha256 = _full_hex(weight["sha256"], 64, f"{label}.weight.sha256")
    parameter_source = profile["base_parameter_count_source"]
    if parameter_source != BASE_PARAMETER_COUNT_SOURCE:
        raise DataContractError(
            f"{label}.base_parameter_count_source differs from the approved rule"
        )
    license_name = profile["license"]
    source = profile["source"]
    if not isinstance(license_name, str) or not license_name:
        raise DataContractError(f"{label}.license must be nonempty")
    if not isinstance(source, str) or not source.startswith("https://huggingface.co/"):
        raise DataContractError(f"{label}.source must be an official Hugging Face URL")
    return EncoderProfile(
        profile_id=profile_id,
        model=model,
        revision=revision,
        hidden_size=hidden_size,
        tokenizer_files=tuple(tokenizer_files),
        weight_path=weight["path"],
        weight_bytes=weight_bytes,
        weight_sha256=weight_sha256,
        base_parameter_count_source=parameter_source,
        license=license_name,
        source=source,
    )


def _validate_recipe(value: Any, index: int) -> ComparisonRecipe:
    label = f"recipes[{index}]"
    recipe = _object(value, label)
    _exact_keys(recipe, RECIPE_FIELDS, label)
    recipe_id = _identifier(recipe["recipe_id"], f"{label}.recipe_id")
    if recipe["source_classification"] != "historical_result_producing":
        raise DataContractError(
            f"{label}.source_classification must preserve historical provenance"
        )
    for field in (
        "max_steps",
        "evaluation_every_steps",
        "batch_size",
        "max_length",
        "max_span_width",
        "warmup_steps",
    ):
        _positive_integer(recipe[field], f"{label}.{field}")
    for field in (
        "learning_rate",
        "ner_negative_ratio",
        "ner_focal_gamma",
        "bio_loss_weight",
        "label_smoothing",
        "re_loss_weight",
        "re_no_rel_weight",
    ):
        _number(recipe[field], f"{label}.{field}")
    if not isinstance(recipe["context_between_spans"], bool):
        raise DataContractError(f"{label}.context_between_spans must be boolean")
    boost = _object(recipe["comparison_boost"], f"{label}.comparison_boost")
    _exact_keys(
        boost,
        {
            "schedule",
            "initial",
            "end",
            "adaptive_step",
            "threshold_low",
            "threshold_high",
            "middle",
        },
        f"{label}.comparison_boost",
    )
    schedule = boost["schedule"]
    if schedule not in {"disabled", "pipeline-a20"}:
        raise DataContractError(
            f"{label}.comparison_boost.schedule must be disabled or pipeline-a20"
        )
    for field in ("initial", "end", "threshold_low", "threshold_high", "middle"):
        _number(boost[field], f"{label}.comparison_boost.{field}")
    _positive_integer(boost["adaptive_step"], f"{label}.comparison_boost.adaptive_step")
    if boost["threshold_low"] >= boost["threshold_high"]:
        raise DataContractError(f"{label} comparison thresholds must be increasing")
    if schedule == "disabled" and not all(
        float(boost[field]) == 1.0 for field in ("initial", "end", "middle")
    ):
        raise DataContractError(
            f"{label} disabled comparison boost must keep every multiplier at 1.0"
        )
    return ComparisonRecipe(recipe_id=recipe_id, value=copy.deepcopy(recipe))


def _optional_metric(value: Any, label: str) -> float | None:
    if value is None:
        return None
    metric = _number(value, label)
    if metric > 1.0:
        raise DataContractError(f"{label} must be <= 1.0")
    return metric


def _validate_arm(
    value: Any,
    index: int,
    profiles: dict[str, EncoderProfile],
    recipes: dict[str, ComparisonRecipe],
) -> Table1Arm:
    label = f"table1_arms[{index}]"
    arm = _object(value, label)
    _exact_keys(arm, ARM_FIELDS, label)
    arm_id = _identifier(arm["arm_id"], f"{label}.arm_id")
    row_label = arm["row_label"]
    claim_role = arm["claim_role"]
    if not isinstance(row_label, str) or not row_label:
        raise DataContractError(f"{label}.row_label must be nonempty")
    if claim_role not in {
        "backbone_capacity",
        "backbone_capacity_and_recipe_reference",
        "recipe_method",
    }:
        raise DataContractError(f"{label}.claim_role is not approved")
    profile_id = _identifier(arm["profile_id"], f"{label}.profile_id")
    recipe_id = _identifier(arm["recipe_id"], f"{label}.recipe_id")
    if profile_id not in profiles or recipe_id not in recipes:
        raise DataContractError(f"{label} references an unknown profile or recipe")
    reported = _object(
        arm["historical_reported_development"],
        f"{label}.historical_reported_development",
    )
    _exact_keys(
        reported,
        {"mean_f1", "sample_sd", "best_f1"},
        f"{label}.historical_reported_development",
    )
    historical = {
        key: _optional_metric(
            reported[key], f"{label}.historical_reported_development.{key}"
        )
        for key in ("mean_f1", "sample_sd", "best_f1")
    }
    return Table1Arm(
        arm_id=arm_id,
        row_label=row_label,
        claim_role=claim_role,
        profile_id=profile_id,
        recipe_id=recipe_id,
        historical_reported_development=historical,
    )


def load_comparison_config(
    source_root: Path, supplied: str | Path = DEFAULT_CONFIG
) -> ComparisonConfig:
    path = resolve_tracked_path(source_root, supplied)
    value = _object(load_json(path), "encoder comparison config")
    _exact_keys(
        value,
        {
            "schema_version",
            "protocol_id",
            "matcher_id",
            "dataset_id",
            "split_id",
            "canonical_pipeline_config",
            "historical_source",
            "encoder_profiles",
            "recipes",
            "table1_arms",
            "statistical_protocol",
            "execution_limits",
            "training_seeds",
        },
        "encoder comparison config",
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "matcher_id": MATCHER_ID,
        "dataset_id": "CODE-ACCORD-v1.0.0",
        "split_id": "CODE-SPLIT-1",
        "training_seeds": list(TRAINING_SEEDS),
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise DataContractError(
                f"encoder comparison config {field} must be {expected_value!r}"
            )

    historical = _object(value["historical_source"], "historical_source")
    _exact_keys(historical, {"commit", *HISTORICAL_BLOBS}, "historical_source")
    if historical["commit"] != HISTORICAL_COMMIT:
        raise DataContractError("historical source commit differs from Phase G authority")
    for name, (source_path, blob) in HISTORICAL_BLOBS.items():
        record = _object(historical[name], f"historical_source.{name}")
        if record != {"path": source_path, "git_blob_sha1": blob}:
            raise DataContractError(
                f"historical_source.{name} differs from the frozen source crosswalk"
            )

    pipeline_record = _object(
        value["canonical_pipeline_config"], "canonical_pipeline_config"
    )
    _exact_keys(pipeline_record, {"path", "sha256"}, "canonical_pipeline_config")
    pipeline_path = resolve_tracked_path(source_root, pipeline_record["path"])
    if sha256_file(pipeline_path) != _full_hex(
        pipeline_record["sha256"], 64, "canonical_pipeline_config.sha256"
    ):
        raise DataContractError("canonical pipeline config hash differs from comparison pin")
    pipeline = load_pipeline_config(source_root, pipeline_path)

    raw_profiles = value["encoder_profiles"]
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise DataContractError("encoder_profiles must be a nonempty array")
    profiles = [_validate_profile(item, index) for index, item in enumerate(raw_profiles)]
    profile_map = {profile.profile_id: profile for profile in profiles}
    if len(profile_map) != len(profiles):
        raise DataContractError("encoder profile IDs must be unique")

    raw_recipes = value["recipes"]
    if not isinstance(raw_recipes, list) or not raw_recipes:
        raise DataContractError("recipes must be a nonempty array")
    recipes = [_validate_recipe(item, index) for index, item in enumerate(raw_recipes)]
    recipe_map = {recipe.recipe_id: recipe for recipe in recipes}
    if len(recipe_map) != len(recipes):
        raise DataContractError("comparison recipe IDs must be unique")

    raw_arms = value["table1_arms"]
    if not isinstance(raw_arms, list) or not raw_arms:
        raise DataContractError("table1_arms must be a nonempty array")
    arms = [
        _validate_arm(item, index, profile_map, recipe_map)
        for index, item in enumerate(raw_arms)
    ]
    arm_map = {arm.arm_id: arm for arm in arms}
    if len(arm_map) != len(arms):
        raise DataContractError("Table 1 arm IDs must be unique")
    pairs = {(arm.profile_id, arm.recipe_id) for arm in arms}
    if len(pairs) != len(arms):
        raise DataContractError("Table 1 profile/recipe pairs must be unique")

    actual_arm_identities = {
        arm.arm_id: (
            arm.row_label,
            arm.claim_role,
            arm.profile_id,
            arm.recipe_id,
        )
        for arm in arms
    }
    if actual_arm_identities != EXPECTED_ARM_IDENTITIES:
        raise DataContractError("Table 1 arms differ from the user-approved matrix")

    statistical_protocol = _object(
        value["statistical_protocol"], "statistical_protocol"
    )
    execution_limits = _object(value["execution_limits"], "execution_limits")
    if statistical_protocol != EXPECTED_STATISTICAL_PROTOCOL:
        raise DataContractError(
            "statistical protocol differs from the user-approved protocol"
        )
    if execution_limits != EXPECTED_EXECUTION_LIMITS:
        raise DataContractError("execution limits differ from the user-approved limits")
    return ComparisonConfig(
        path=path,
        sha256=sha256_file(path),
        value=value,
        pipeline=pipeline,
        profiles=profile_map,
        recipes=recipe_map,
        arms=arm_map,
    )


def bind_selection(layout: RunLayout, selection: ComparisonSelection) -> dict[str, Any]:
    layout.require_existing()
    manifest = selection.manifest(layout.run_id)
    path = layout.resolve("manifests/encoder-comparison-selection.json")
    if path.is_file():
        if load_json(path) != manifest:
            raise DataContractError(
                "run is already bound to a different encoder comparison selection"
            )
    else:
        atomic_write_json(path, manifest)
    return manifest


def validate_selection(layout: RunLayout, selection: ComparisonSelection) -> dict[str, Any]:
    path = layout.resolve(
        "manifests/encoder-comparison-selection.json", must_exist=True
    )
    manifest = load_json(path)
    if manifest != selection.manifest(layout.run_id):
        raise DataContractError(
            "run-local encoder comparison selection differs from configuration"
        )
    return manifest
