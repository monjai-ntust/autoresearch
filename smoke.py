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
from phase_b_io import atomic_write_jsonl, iter_jsonl
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


def prepare(source_dev: Path, generated_test: Path, destination: Path) -> None:
    dev = _one(source_dev, "development candidate input")
    test = _one(generated_test, "test candidate input")
    dev_clones = [_clone_candidate(dev, seed) for seed in TRAINING_SEEDS]
    test_clones = [_clone_candidate(test, seed) for seed in TRAINING_SEEDS]
    root = destination / "predictions" / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(root / "development-candidates.jsonl", dev_clones)
    atomic_write_jsonl(root / "test-candidates.jsonl", test_clones)
    atomic_write_jsonl(root / "test-live-candidate.jsonl", [_clone_candidate(test, 42)])
    atomic_write_jsonl(root / "warmup-candidate.jsonl", [_clone_candidate(dev, 42)])


def clone_verdicts(source: Path, candidates: Path, destination: Path) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepared = commands.add_parser("prepare")
    prepared.add_argument("--source-dev", required=True, type=Path)
    prepared.add_argument("--generated-test", required=True, type=Path)
    prepared.add_argument("--destination-run", required=True, type=Path)
    verdict = commands.add_parser("clone-verdicts")
    verdict.add_argument("--source", required=True, type=Path)
    verdict.add_argument("--candidates", required=True, type=Path)
    verdict.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_dev, args.generated_test, args.destination_run)
    else:
        clone_verdicts(args.source, args.candidates, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
