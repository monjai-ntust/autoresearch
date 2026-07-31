"""Small, fully hash-bound Phase B run used by C-04 contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

from records import StrictTriple, candidate_id_for


PROTOCOL = "B04-PATH-A-1.3"
WORKFLOW = "PATH-A-WORKFLOW-1.3"
MATCHER = "CODE-STRICT-1"
DATASET = "CODE-ACCORD-v1.0.0"
SEEDS = tuple(range(42, 50))


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_bytes(row) for row in rows)


def _write(root: Path, relative: str, payload: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _span(start: int, end: int, kind: str, text: str) -> dict[str, Any]:
    return {"start": start, "end": end, "type": kind, "text": text}


def _triple(
    example_id: str,
    head: dict[str, Any],
    relation: str,
    tail: dict[str, Any],
) -> StrictTriple:
    return StrictTriple.from_mapping(
        {"head": head, "relation": relation, "tail": tail},
        example_id=example_id,
        label="fixture triple",
    )


def build_phase_b_run(root: Path, run_id: str) -> dict[str, Any]:
    root.mkdir(parents=True)
    examples = (
        {
            "example_id": "example-001",
            "source_document_id": "document-alpha",
            "source_country": "UK",
            "words": ["Door", "requires", "clear", "width"],
            "content": "Door requires clear width",
            "gold": _triple(
                "example-001",
                _span(0, 0, "Object", "Door"),
                "necessity",
                _span(2, 3, "Quality", "clear width"),
            ),
        },
        {
            "example_id": "example-002",
            "source_document_id": "document-beta",
            "source_country": "US",
            "words": ["Wall", "uses", "concrete"],
            "content": "Wall uses concrete",
            "gold": _triple(
                "example-002",
                _span(0, 0, "Object", "Wall"),
                "selection",
                _span(2, 2, "Object", "concrete"),
            ),
        },
    )
    prepared_rows = [
        {
            "content": item["content"],
            "dataset_id": DATASET,
            "entities": [],
            "example_id": item["example_id"],
            "processed_content": " ".join(item["words"]),
            "protocol_id": PROTOCOL,
            "relations": [],
            "source_country": item["source_country"],
            "source_document_id": item["source_document_id"],
            "split": "test",
            "words": item["words"],
        }
        for item in examples
    ]
    gold_rows = [
        {
            "protocol_id": PROTOCOL,
            "split_id": "CODE-SPLIT-1:test",
            "example_id": item["example_id"],
            "source_document_id": item["source_document_id"],
            "gold_triples": [
                {
                    "head": item["gold"].head.to_mapping(),
                    "relation": item["gold"].relation,
                    "tail": item["gold"].tail.to_mapping(),
                }
            ],
            "input_hashes": {"annotation_bundle": "a" * 64},
        }
        for item in examples
    ]
    prepared_sha = _write(
        root, "data-prepared/test.jsonl", _jsonl(prepared_rows)
    )
    gold_sha = _write(root, "data-prepared/test-gold.jsonl", _jsonl(gold_rows))
    split_sha = _write(
        root,
        "data-prepared/split-manifest.json",
        _bytes({"split_id": "CODE-SPLIT-1", "seed": 42}),
    )
    dev_gold_sha = _write(
        root,
        "data-prepared/development-gold.jsonl",
        _jsonl([{"fixture": "development-gold"}]),
    )
    candidate_index_sha = _write(
        root,
        "predictions/dev/candidate-index.json",
        _bytes({"schema_version": "phase-b-candidate-index-1.0"}),
    )
    threshold = {
        "protocol_id": PROTOCOL,
        "selection_split": "development",
        "objective": "mean_per_seed_development_strict_triple_f1",
        "tie_rule": "higher_threshold",
        "selected_threshold": 0.5,
        "used_test_labels": False,
        "split_manifest_sha256": split_sha,
        "development_gold_sha256": dev_gold_sha,
        "development_candidate_index_sha256": candidate_index_sha,
    }
    threshold_sha = _write(
        root, "predictions/dev/threshold-selection.json", _bytes(threshold)
    )

    false = _triple(
        "example-001",
        _span(0, 0, "Object", "Door"),
        "selection",
        _span(1, 1, "Property", "requires"),
    )
    candidate_rows = []
    triples_by_candidate = {}
    for seed in SEEDS:
        for item, triple, confidence in (
            (examples[0], examples[0]["gold"], 0.9),
            (examples[0], false, 0.8),
            (examples[1], examples[1]["gold"], 0.9),
        ):
            candidate_id = candidate_id_for(seed, triple)
            triples_by_candidate[candidate_id] = (triple, seed)
            candidate_rows.append(
                {
                    "protocol_id": PROTOCOL,
                    "split_id": "CODE-SPLIT-1:test",
                    "training_seed": seed,
                    "example_id": triple.example_id,
                    "source_document_id": item["source_document_id"],
                    "candidate_id": candidate_id,
                    "head": triple.head.to_mapping(),
                    "relation": triple.relation,
                    "tail": triple.tail.to_mapping(),
                    "triple_confidence": confidence,
                    "input_hashes": {
                        "checkpoint_manifest": "b" * 64,
                        "prediction_ledger": "c" * 64,
                        "prepared_sentences": prepared_sha,
                    },
                }
            )
    candidate_rows.sort(
        key=lambda row: (
            row["training_seed"],
            row["example_id"],
            row["candidate_id"],
        )
    )
    candidate_sha = _write(
        root, "predictions/test/candidates.jsonl", _jsonl(candidate_rows)
    )

    simple_rows = []
    corrective_rows = []
    for candidate_id, (triple, seed) in sorted(
        triples_by_candidate.items(), key=lambda item: (item[1][1], item[0])
    ):
        is_false = triple == false
        is_second = triple.example_id == "example-002"
        common = {
            "attempts": 1,
            "candidate_id": candidate_id,
            "decoding_sha256": "d" * 64,
            "error_category": None,
            "model_manifest_sha256": "e" * 64,
            "protocol_id": PROTOCOL,
            "raw_response_sha256": "f" * 64,
            "reason_code": "WRONG_RELATION" if is_false else "SUPPORTED",
            "response_status": "valid_response",
            "telemetry": {},
            "training_seed": seed,
        }
        simple_rows.append(
            {
                **common,
                "condition_id": "VER-SIMPLE",
                "action": "DISCARD" if is_false else "KEEP",
                "corrected": None,
                "correction_validation_status": "not_applicable",
                "prompt_sha256": "1" * 64,
            }
        )
        corrected = examples[0]["gold"] if is_false else None
        corrective_rows.append(
            {
                **common,
                "condition_id": "VER-CORRECTIVE",
                "action": "CORRECT"
                if is_false
                else "DISCARD"
                if is_second
                else "KEEP",
                "corrected": (
                    {
                        "head": corrected.head.to_mapping(),
                        "relation": corrected.relation,
                        "tail": corrected.tail.to_mapping(),
                    }
                    if corrected is not None
                    else None
                ),
                "correction_validation_status": "valid"
                if is_false
                else "not_applicable",
                "prompt_sha256": "2" * 64,
            }
        )
    simple_sha = _write(
        root, "verifier/simple/verdicts.jsonl", _jsonl(simple_rows)
    )
    corrective_sha = _write(
        root, "verifier/corrective/verdicts.jsonl", _jsonl(corrective_rows)
    )

    emitted = {
        "VER-RAW": {
            seed: {
                "example-001": {examples[0]["gold"].key(), false.key()},
                "example-002": {examples[1]["gold"].key()},
            }
            for seed in SEEDS
        },
        "VER-CONFIDENCE": {
            seed: {
                "example-001": {examples[0]["gold"].key(), false.key()},
                "example-002": {examples[1]["gold"].key()},
            }
            for seed in SEEDS
        },
        "VER-SIMPLE": {
            seed: {
                "example-001": {examples[0]["gold"].key()},
                "example-002": {examples[1]["gold"].key()},
            }
            for seed in SEEDS
        },
        "VER-CORRECTIVE": {
            seed: {
                "example-001": {examples[0]["gold"].key()},
                "example-002": set(),
            }
            for seed in SEEDS
        },
    }
    sentence_rows = []
    counts_by_condition = {}
    for condition, by_seed in emitted.items():
        counts_by_condition[condition] = {}
        for seed, by_example in by_seed.items():
            total = {"tp": 0, "fp": 0, "fn": 0}
            for item in examples:
                example_id = item["example_id"]
                target = {item["gold"].key()}
                predicted = by_example[example_id]
                counts = {
                    "tp": len(predicted & target),
                    "fp": len(predicted - target),
                    "fn": len(target - predicted),
                }
                for key in total:
                    total[key] += counts[key]
                sentence_rows.append(
                    {
                        "protocol_id": PROTOCOL,
                        "matcher_id": MATCHER,
                        "condition_id": condition,
                        "training_seed": seed,
                        "example_id": example_id,
                        "source_document_id": item["source_document_id"],
                        "gold_strict_keys": [list(item["gold"].key())],
                        "emitted_strict_keys": [
                            list(key) for key in sorted(predicted)
                        ],
                        "duplicate_emissions_collapsed": 0,
                        "counts": counts,
                    }
                )
            counts_by_condition[condition][seed] = total
    sentence_rows.sort(
        key=lambda row: (
            row["condition_id"],
            row["training_seed"],
            row["example_id"],
        )
    )
    sentence_sha = _write(
        root, "outcomes/sentence-outcomes.jsonl", _jsonl(sentence_rows)
    )
    candidate_outcome_sha = _write(
        root,
        "outcomes/candidate-outcomes.jsonl",
        _jsonl([{"fixture": "candidate outcomes bound but not consumed"}]),
    )
    metric_conditions = {}
    for condition, by_seed in counts_by_condition.items():
        metric_conditions[condition] = {
            "per_seed": {
                str(seed): {
                    "end_to_end_strict_triple": {"counts": counts}
                }
                for seed, counts in by_seed.items()
            }
        }
    metrics = {
        "selected_confidence_threshold": 0.5,
        "observed_training_seeds": list(SEEDS),
        "conditions": metric_conditions,
    }
    metrics_sha = _write(root, "metrics/metrics.json", _bytes(metrics))
    _write(root, "metrics/publication-table.tsv", b"fixture\n")
    _write(root, "metrics/publication-summary.md", b"fixture\n")

    score = {
        "schema_version": "phase-b-score-manifest-1.0",
        "protocol_id": PROTOCOL,
        "workflow_id": WORKFLOW,
        "matcher_id": MATCHER,
        "run_id": run_id,
        "inputs": {
            "data-prepared/test-gold.jsonl": gold_sha,
            "predictions/dev/threshold-selection.json": threshold_sha,
            "predictions/test/candidates.jsonl": candidate_sha,
            "verifier/simple/verdicts.jsonl": simple_sha,
            "verifier/corrective/verdicts.jsonl": corrective_sha,
        },
        "outputs": {
            "metrics/metrics.json": metrics_sha,
            "outcomes/candidate-outcomes.jsonl": candidate_outcome_sha,
            "outcomes/sentence-outcomes.jsonl": sentence_sha,
        },
        "nonpublication_smoke": False,
    }
    _write(root, "manifests/score-manifest.json", _bytes(score))
    _write(
        root,
        "manifests/00-checkout-manifest.json",
        _bytes(
            {
                "schema_version": "phase-b-checkout-manifest-1.0",
                "status": "pass",
                "run_id": run_id,
                "protocol_id": PROTOCOL,
                "workflow_id": WORKFLOW,
            }
        ),
    )
    _write(
        root,
        "manifests/debug-full-run.json",
        _bytes(
            {
                "schema_version": "phase-b-debug-full-run-1.0",
                "run_id": run_id,
                "protocol_id": PROTOCOL,
                "workflow_id": WORKFLOW,
                "conditional_b07_approval": True,
                "training_seeds": list(SEEDS),
            }
        ),
    )
    _write(
        root,
        "manifests/03-data-preparation-manifest.json",
        _bytes(
            {
                "schema_version": "phase-b-data-preparation-manifest-2.0",
                "protocol_id": PROTOCOL,
                "dataset_id": DATASET,
                "byte_identical_independent_materializations": True,
                "artifacts": [
                    {"path": "test.jsonl", "sha256": prepared_sha},
                    {"path": "test-gold.jsonl", "sha256": gold_sha},
                    {"path": "split-manifest.json", "sha256": split_sha},
                    {
                        "path": "development-gold.jsonl",
                        "sha256": dev_gold_sha,
                    },
                ],
            }
        ),
    )
    for mode, condition, verdict_sha in (
        ("simple", "VER-SIMPLE", simple_sha),
        ("corrective", "VER-CORRECTIVE", corrective_sha),
    ):
        verdict_path = f"verifier/{mode}/verdicts.jsonl"
        _write(
            root,
            f"manifests/verifier-{mode}-live.json",
            _bytes(
                {
                    "protocol_id": PROTOCOL,
                    "condition_id": condition,
                    "execution_mode": "live",
                    "status": "completed",
                    "verdict_count": len(candidate_rows),
                    "inputs": {
                        "data-prepared/test.jsonl": prepared_sha,
                        "predictions/test/candidates.jsonl": candidate_sha,
                    },
                    "outputs": {verdict_path: verdict_sha},
                }
            ),
        )
    return {
        "examples": examples,
        "expected_counts": {
            condition: counts_by_condition[condition][42]
            for condition in (
                "VER-CONFIDENCE",
                "VER-SIMPLE",
                "VER-CORRECTIVE",
            )
        },
    }


def write_test_config(root: Path, destination: Path, *, resamples: int = 100) -> None:
    config = json.loads(
        (root / "configs/phase_b_graph_rag_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    config["statistics"]["bootstrap_resamples"] = resamples
    destination.write_bytes(_bytes(config))
