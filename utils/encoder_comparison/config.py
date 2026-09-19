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


SCHEMA_VERSION = "phase-g-encoder-comparison-1.0"
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
            "license": self.license,
            "source": self.source,
        }


@dataclass(frozen=True)
class ComparisonRecipe:
    recipe_id: str
    value: dict[str, Any]


@dataclass(frozen=True)
class ComparisonConfig:
    path: Path
    sha256: str
    value: dict[str, Any]
    pipeline: PipelineConfig
    profiles: dict[str, EncoderProfile]
    recipes: dict[str, ComparisonRecipe]

    def select(self, profile_id: str, recipe_id: str) -> "ComparisonSelection":
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
        return ComparisonSelection(config=self, profile=profile, recipe=recipe)


@dataclass(frozen=True)
class ComparisonSelection:
    config: ComparisonConfig
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
            "initial",
            "end",
            "adaptive_step",
            "threshold_low",
            "threshold_high",
            "middle",
        },
        f"{label}.comparison_boost",
    )
    for field in ("initial", "end", "threshold_low", "threshold_high", "middle"):
        _number(boost[field], f"{label}.comparison_boost.{field}")
    _positive_integer(boost["adaptive_step"], f"{label}.comparison_boost.adaptive_step")
    if boost["threshold_low"] >= boost["threshold_high"]:
        raise DataContractError(f"{label} comparison thresholds must be increasing")
    return ComparisonRecipe(recipe_id=recipe_id, value=copy.deepcopy(recipe))


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
    return ComparisonConfig(
        path=path,
        sha256=sha256_file(path),
        value=value,
        pipeline=pipeline,
        profiles=profile_map,
        recipes=recipe_map,
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
