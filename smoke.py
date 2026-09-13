"""Non-publication one-candidate Phase B smoke-artifact helpers.

This module deliberately never creates canonical evidence.  It uses one real
seed-42 candidate and expands it into pseudo-seeds only to exercise downstream
eight-seed parsing, threshold, and scoring code in an isolated smoke run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from constants import TRAINING_SEEDS
from phase_b_io import atomic_write_json, atomic_write_jsonl, iter_jsonl, load_json, sha256_file
from paths import RunLayout, discover_source_root
from records import Candidate, candidate_id_for


def _rows(path: Path) -> list[dict]:
    return [value for _, value in iter_jsonl(path)]


def _one(path: Path, label: str) -> dict:
    rows = _rows(path)
    if not rows:
        raise ValueError(f"{label} has no candidates")
    return min(rows, key=lambda value: (value["candidate_id"], value["example_id"]))


def _clone_candidate(row: dict, seed: int) -> dict:
    candidate = Candidate.from_mapping(row, "smoke candidate")
    clone = dict(row)
    clone["training_seed"] = seed
    clone["candidate_id"] = candidate_id_for(seed, candidate.triple)
    return clone


def _validate_stage_seal(
    layout: RunLayout, producer_path: Path, outputs: object
) -> None:
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("same-run producer must bind at least one output")
    for relative, expected in outputs.items():
        path = layout.resolve(relative, must_exist=True)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"same-run producer output differs: {relative}")
    seal_path = producer_path.with_name(f"same-run-{producer_path.name}")
    expected_seal = {
        "schema_version": "phase-b-same-run-stage-seal-1.0",
        "run_id": layout.run_id,
        "producer_manifest": {
            "path": layout.relative_identity(producer_path),
            "sha256": sha256_file(producer_path),
        },
        "outputs": outputs,
    }
    if load_json(seal_path) != expected_seal:
        raise ValueError("producer stage seal has a different run identity")


def _validate_candidate_producer(
    layout: RunLayout, path: Path, seed: int
) -> Path:
    relative = layout.relative_identity(path)
    matches = []
    for manifest_path in layout.resolve("manifests").glob(
        f"model-generate-candidates-live-seed-{seed}-*.json"
    ):
        manifest = load_json(manifest_path)
        if (
            isinstance(manifest, dict)
            and manifest.get("stage") == "model-generate-candidates"
            and manifest.get("execution_mode") == "live"
            and manifest.get("status") == "completed"
            and manifest.get("training_seed") == seed
            and manifest.get("candidates_output") == relative
        ):
            try:
                outputs = {relative: sha256_file(path)}
                ledger = manifest.get("inputs", {}).get("prediction_ledger")
                if isinstance(ledger, dict) and isinstance(ledger.get("path"), str):
                    outputs[ledger["path"]] = ledger.get("sha256")
                _validate_stage_seal(
                    layout, manifest_path, outputs
                )
            except (OSError, ValueError):
                continue
            matches.append(manifest_path)
    if len(matches) != 1:
        raise ValueError(
            f"{relative} is not authenticated by exactly one same-run live producer"
        )
    return matches[0]


def _validate_hash_map(layout: RunLayout, bindings: object, label: str) -> None:
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError(f"{label} must contain artifact hashes")
    for relative, expected in bindings.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"{label} contains an invalid binding")
        path = layout.resolve(relative, must_exist=True)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} artifact differs: {relative}")


def _validate_prepare_manifest(layout: RunLayout, seed: int | None) -> dict:
    path = layout.resolve("predictions/smoke/manifest.json", must_exist=True)
    manifest = load_json(path)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "run_id", "seed", "inputs", "outputs"}
        or manifest.get("schema_version") != "phase-b-smoke-candidates-1.0"
        or manifest.get("run_id") != layout.run_id
        or not isinstance(manifest.get("seed"), int)
        or (seed is not None and manifest.get("seed") != seed)
    ):
        raise ValueError("smoke candidate manifest has a different run identity")
    expected_outputs = {
        "predictions/smoke/development-candidates.jsonl",
        "predictions/smoke/test-candidates.jsonl",
        "predictions/smoke/test-live-candidate.jsonl",
        "predictions/smoke/warmup-candidate.jsonl",
    }
    if set(manifest.get("outputs", {})) != expected_outputs:
        raise ValueError("smoke candidate manifest has an incomplete output set")
    inputs = manifest.get("inputs", {})
    required_inputs = {
        f"predictions/dev/seed-{manifest['seed']}-candidates.jsonl",
        f"predictions/smoke/test-seed-{manifest['seed']}-candidates.jsonl",
    }
    if (
        not isinstance(inputs, dict)
        or len(inputs) != 4
        or not required_inputs <= set(inputs)
    ):
        raise ValueError("smoke candidate manifest has an incomplete producer input set")
    _validate_hash_map(layout, manifest.get("inputs"), "smoke candidate inputs")
    _validate_hash_map(layout, manifest.get("outputs"), "smoke candidate outputs")
    return manifest


def prepare(layout: RunLayout, seed: int = 42) -> None:
    layout.require_existing()
    source_dev = layout.resolve(
        f"predictions/dev/seed-{seed}-candidates.jsonl", must_exist=True
    )
    generated_test = layout.resolve(
        f"predictions/smoke/test-seed-{seed}-candidates.jsonl", must_exist=True
    )
    root = layout.resolve("predictions/smoke")
    manifest_path = root / "manifest.json"
    outputs = [
        root / "development-candidates.jsonl",
        root / "test-candidates.jsonl",
        root / "test-live-candidate.jsonl",
        root / "warmup-candidate.jsonl",
    ]
    if manifest_path.exists() or any(path.exists() for path in outputs):
        _validate_prepare_manifest(layout, seed)
        return
    dev_producer = _validate_candidate_producer(layout, source_dev, seed)
    test_producer = _validate_candidate_producer(layout, generated_test, seed)
    dev = _one(source_dev, "development candidate input")
    test = _one(generated_test, "test candidate input")
    dev_clones = [_clone_candidate(dev, seed) for seed in TRAINING_SEEDS]
    test_clones = [_clone_candidate(test, seed) for seed in TRAINING_SEEDS]
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(root / "development-candidates.jsonl", dev_clones)
    atomic_write_jsonl(root / "test-candidates.jsonl", test_clones)
    atomic_write_jsonl(root / "test-live-candidate.jsonl", [_clone_candidate(test, 42)])
    atomic_write_jsonl(root / "warmup-candidate.jsonl", [_clone_candidate(dev, 42)])
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "phase-b-smoke-candidates-1.0",
            "run_id": layout.run_id,
            "seed": seed,
            "inputs": {
                layout.relative_identity(path): sha256_file(path)
                for path in (source_dev, generated_test, dev_producer, test_producer)
            },
            "outputs": {
                layout.relative_identity(path): sha256_file(path) for path in outputs
            },
        },
    )


def clone_verdicts(layout: RunLayout, mode: str) -> None:
    layout.require_existing()
    if mode not in {"simple", "corrective"}:
        raise ValueError("smoke verifier mode must be simple or corrective")
    source = layout.resolve(f"verifier/smoke/{mode}/verdicts.jsonl", must_exist=True)
    candidates = layout.resolve(
        "predictions/smoke/test-candidates.jsonl", must_exist=True
    )
    prepare_manifest = layout.resolve(
        "predictions/smoke/manifest.json", must_exist=True
    )
    _validate_prepare_manifest(layout, None)
    destination = layout.resolve(
        f"verifier/smoke/{mode}/pseudo-seed-verdicts.jsonl"
    )
    destination_manifest = layout.resolve(
        f"verifier/smoke/{mode}/pseudo-seed-verdicts.manifest.json"
    )
    producer_path = layout.resolve(
        f"manifests/verifier-smoke-{mode}-live.json", must_exist=True
    )
    producer = load_json(producer_path)
    relative_source = layout.relative_identity(source)
    if (
        not isinstance(producer, dict)
        or producer.get("condition_id") != f"VER-{mode.upper()}"
        or producer.get("execution_mode") != "live"
        or producer.get("status") != "completed"
        or producer.get("outputs", {}).get(relative_source) != sha256_file(source)
    ):
        raise ValueError("smoke verdict is not authenticated by its same-run live producer")
    _validate_stage_seal(layout, producer_path, producer.get("outputs"))
    if destination.exists() or destination_manifest.exists():
        manifest = load_json(destination_manifest)
        expected = {
            "schema_version": "phase-b-smoke-verdict-clones-1.0",
            "run_id": layout.run_id,
            "mode": mode,
            "inputs": {
                relative_source: sha256_file(source),
                layout.relative_identity(candidates): sha256_file(candidates),
                layout.relative_identity(prepare_manifest): sha256_file(prepare_manifest),
                layout.relative_identity(producer_path): sha256_file(producer_path),
            },
            "output": {
                "path": layout.relative_identity(destination),
                "sha256": sha256_file(destination),
            },
        }
        if manifest != expected:
            raise ValueError("smoke verdict clone manifest differs from this run")
        return
    verdicts = _rows(source)
    if len(verdicts) != 1:
        raise ValueError(f"{source} must contain exactly one smoke verdict")
    clones = _rows(candidates)
    base = verdicts[0]
    output = []
    for candidate in clones:
        clone = dict(base)
        clone["candidate_id"] = candidate["candidate_id"]
        clone["training_seed"] = candidate["training_seed"]
        output.append(clone)
    atomic_write_jsonl(destination, output)
    atomic_write_json(
        destination_manifest,
        {
            "schema_version": "phase-b-smoke-verdict-clones-1.0",
            "run_id": layout.run_id,
            "mode": mode,
            "inputs": {
                relative_source: sha256_file(source),
                layout.relative_identity(candidates): sha256_file(candidates),
                layout.relative_identity(prepare_manifest): sha256_file(prepare_manifest),
                layout.relative_identity(producer_path): sha256_file(producer_path),
            },
            "output": {
                "path": layout.relative_identity(destination),
                "sha256": sha256_file(destination),
            },
        },
    )


def seal_threshold(layout: RunLayout) -> None:
    layout.require_existing()
    threshold = layout.resolve(
        "predictions/smoke/threshold-selection.json", must_exist=True
    )
    candidates = layout.resolve(
        "predictions/smoke/development-candidates.jsonl", must_exist=True
    )
    gold = layout.resolve("data-prepared/development-gold.jsonl", must_exist=True)
    split = layout.resolve("data-prepared/split-manifest.json", must_exist=True)
    selection = load_json(threshold)
    if (
        not isinstance(selection, dict)
        or selection.get("selection_split") != "development"
        or selection.get("used_test_labels") is not False
        or selection.get("development_candidate_index_sha256")
        != sha256_file(candidates)
        or selection.get("development_gold_sha256") != sha256_file(gold)
        or selection.get("split_manifest_sha256") != sha256_file(split)
    ):
        raise ValueError("smoke threshold is not bound to its same-run inputs")
    manifest_path = layout.resolve(
        "predictions/smoke/threshold-selection.manifest.json"
    )
    document = {
        "schema_version": "phase-b-smoke-threshold-1.0",
        "run_id": layout.run_id,
        "inputs": {
            layout.relative_identity(path): sha256_file(path)
            for path in (candidates, gold, split)
        },
        "output": {
            "path": layout.relative_identity(threshold),
            "sha256": sha256_file(threshold),
        },
    }
    if manifest_path.exists():
        if load_json(manifest_path) != document:
            raise ValueError("smoke threshold manifest differs from the selected run")
    else:
        atomic_write_json(manifest_path, document)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepared = commands.add_parser("prepare")
    prepared.add_argument("--run-id", required=True)
    prepared.add_argument("--seed", required=True, type=int, choices=TRAINING_SEEDS)
    verdict = commands.add_parser("clone-verdicts")
    verdict.add_argument("--run-id", required=True)
    verdict.add_argument("--mode", required=True, choices=("simple", "corrective"))
    threshold = commands.add_parser("seal-threshold")
    threshold.add_argument("--run-id", required=True)
    args = parser.parse_args()
    layout = RunLayout(discover_source_root(), args.run_id)
    if args.command == "prepare":
        prepare(layout, args.seed)
    elif args.command == "clone-verdicts":
        clone_verdicts(layout, args.mode)
    else:
        seal_threshold(layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
