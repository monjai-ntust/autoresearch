"""Read-only, hash-bound ingestion of a completed canonical Phase B run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
import json
import math

from graph_rag_eval.contracts import content_sha256
from graph_rag_eval.identity import file_sha256
from records import Candidate, GoldRecord, StrictTriple, Verdict


StrictKey = tuple[str, int, int, str, str, int, int, str]


class PhaseBArtifactError(ValueError):
    """Raised before Graph RAG writes when a parent artifact contract fails."""


@dataclass(frozen=True)
class VerifiedArtifact:
    path: str
    bytes: int
    sha256: str
    role: str


@dataclass(frozen=True)
class PreparedExample:
    example_id: str
    source_document_id: str
    content: str
    words: tuple[str, ...]
    word_char_spans: tuple[tuple[int, int], ...]
    source_country: str

    def span_text(self, start: int, end: int) -> str:
        if not 0 <= start <= end < len(self.words):
            raise PhaseBArtifactError(
                f"{self.example_id}: strict span {start}:{end} escapes prepared words"
            )
        return " ".join(self.words[start : end + 1])

    def span_char_range(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start <= end < len(self.word_char_spans):
            raise PhaseBArtifactError(
                f"{self.example_id}: strict span {start}:{end} escapes prepared words"
            )
        return self.word_char_spans[start][0], self.word_char_spans[end][1]


@dataclass(frozen=True)
class PhaseBRunData:
    run_id: str
    protocol_id: str
    workflow_id: str
    matcher_id: str
    dataset_id: str
    seeds: tuple[int, ...]
    selected_threshold: float
    prepared: Mapping[str, PreparedExample]
    gold: Mapping[str, GoldRecord]
    candidates: tuple[Candidate, ...]
    simple_verdicts: Mapping[str, Verdict]
    corrective_verdicts: Mapping[str, Verdict]
    emitted: Mapping[str, Mapping[int, Mapping[str, frozenset[StrictKey]]]]
    phase_b_metrics: Mapping[str, Any]
    verified_artifacts: tuple[VerifiedArtifact, ...]
    score_manifest_sha256: str

    @property
    def artifact_set_sha256(self) -> str:
        return content_sha256(self.verified_artifacts)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PhaseBArtifactError(f"{label} is unreadable JSON: {path}") from error
    if not isinstance(value, dict):
        raise PhaseBArtifactError(f"{label} must be a JSON object: {path}")
    return value


def _jsonl(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PhaseBArtifactError(f"{label} is unreadable JSONL: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PhaseBArtifactError(
                f"{label}:{line_number} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise PhaseBArtifactError(f"{label}:{line_number} must be an object")
        rows.append(value)
    if not rows:
        raise PhaseBArtifactError(f"{label} is empty")
    return tuple(rows)


class _ParentReader:
    def __init__(self, parent_root: Path):
        self.root = parent_root.resolve()
        self._verified: dict[str, VerifiedArtifact] = {}

    def path(self, relative: str) -> Path:
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            raise PhaseBArtifactError(f"invalid parent artifact path: {relative!r}")
        unresolved = self.root.joinpath(*PurePosixPath(relative).parts)
        if unresolved.is_symlink() or any(
            item.is_symlink()
            for item in unresolved.parents
            if item != self.root.parent
        ):
            raise PhaseBArtifactError(
                f"required parent artifact traverses a symlink: {relative}"
            )
        candidate = unresolved.resolve()
        if self.root not in candidate.parents:
            raise PhaseBArtifactError(f"parent artifact escapes run root: {relative}")
        if candidate.is_symlink() or not candidate.is_file():
            raise PhaseBArtifactError(
                f"required parent artifact is not a regular file: {relative}"
            )
        return candidate

    def verify(self, relative: str, expected: str | None, role: str) -> Path:
        path = self.path(relative)
        observed = file_sha256(path)
        if expected is not None and observed != expected:
            raise PhaseBArtifactError(
                f"parent artifact hash mismatch for {relative}: "
                f"expected {expected}, observed {observed}"
            )
        previous = self._verified.get(relative)
        artifact = VerifiedArtifact(relative, path.stat().st_size, observed, role)
        if previous is not None and previous.sha256 != artifact.sha256:
            raise PhaseBArtifactError(f"parent artifact changed during ingestion: {relative}")
        self._verified[relative] = artifact
        return path

    def artifacts(self) -> tuple[VerifiedArtifact, ...]:
        artifacts = []
        for key in sorted(self._verified):
            expected = self._verified[key]
            path = self.path(key)
            if (
                path.stat().st_size != expected.bytes
                or file_sha256(path) != expected.sha256
            ):
                raise PhaseBArtifactError(
                    f"parent artifact changed during ingestion: {key}"
                )
            artifacts.append(expected)
        return tuple(artifacts)


def _require_identity(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for key, required in expected.items():
        if value.get(key) != required:
            raise PhaseBArtifactError(
                f"{label}.{key} differs from the canonical parent contract"
            )


def _manifest_hash(
    mapping: Mapping[str, Any],
    relative: str,
    *,
    label: str,
) -> str:
    value = mapping.get(relative)
    if not isinstance(value, str) or len(value) != 64:
        raise PhaseBArtifactError(f"{label} does not bind {relative}")
    return value


def _preparation_hash(
    manifest: Mapping[str, Any],
    name: str,
) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PhaseBArtifactError("preparation manifest has no artifact ledger")
    matches = [
        item
        for item in artifacts
        if isinstance(item, dict) and item.get("path") == name
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise PhaseBArtifactError(f"preparation manifest does not bind {name}")
    return matches[0]["sha256"]


def _verifier_contract(
    reader: _ParentReader,
    manifest_relative: str,
    verdict_relative: str,
    candidate_relative: str,
    prepared_relative: str,
    *,
    condition_id: str,
    expected_protocol: str,
) -> dict[str, Any]:
    path = reader.verify(manifest_relative, None, f"{condition_id} condition manifest")
    manifest = _load_json(path, f"{condition_id} condition manifest")
    _require_identity(
        manifest,
        {
            "protocol_id": expected_protocol,
            "condition_id": condition_id,
            "execution_mode": "live",
            "status": "completed",
        },
        f"{condition_id} manifest",
    )
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise PhaseBArtifactError(f"{condition_id} manifest lacks input/output hashes")
    reader.verify(
        candidate_relative,
        _manifest_hash(inputs, candidate_relative, label=f"{condition_id}.inputs"),
        f"{condition_id} candidate input",
    )
    reader.verify(
        prepared_relative,
        _manifest_hash(inputs, prepared_relative, label=f"{condition_id}.inputs"),
        f"{condition_id} prepared input",
    )
    reader.verify(
        verdict_relative,
        _manifest_hash(outputs, verdict_relative, label=f"{condition_id}.outputs"),
        f"{condition_id} verdict output",
    )
    return manifest


def _load_prepared(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_id: str,
    protocol_id: str,
) -> dict[str, PreparedExample]:
    result: dict[str, PreparedExample] = {}
    observed = []
    for index, row in enumerate(rows, start=1):
        _require_identity(
            row,
            {
                "dataset_id": dataset_id,
                "protocol_id": protocol_id,
                "split": "test",
            },
            f"prepared test row {index}",
        )
        example_id = row.get("example_id")
        source_id = row.get("source_document_id")
        content = row.get("content")
        words = row.get("words")
        country = row.get("source_country")
        if (
            not isinstance(example_id, str)
            or not example_id
            or not isinstance(source_id, str)
            or not source_id
            or not isinstance(content, str)
            or not isinstance(words, list)
            or not all(isinstance(item, str) and item for item in words)
            or not isinstance(country, str)
        ):
            raise PhaseBArtifactError(f"prepared test row {index} is malformed")
        if example_id in result:
            raise PhaseBArtifactError(f"duplicate prepared example: {example_id}")
        cursor = 0
        char_spans = []
        for word in words:
            start = content.find(word, cursor)
            if start < 0:
                raise PhaseBArtifactError(
                    f"prepared words do not align to content: {example_id}"
                )
            end = start + len(word)
            char_spans.append((start, end))
            cursor = end
        result[example_id] = PreparedExample(
            example_id=example_id,
            source_document_id=source_id,
            content=content,
            words=tuple(words),
            word_char_spans=tuple(char_spans),
            source_country=country,
        )
        observed.append(example_id)
    if observed != sorted(observed):
        raise PhaseBArtifactError("prepared test examples are not stably sorted")
    return result


def _load_gold(
    rows: Iterable[dict[str, Any]],
    prepared: Mapping[str, PreparedExample],
    *,
    split_id: str,
) -> dict[str, GoldRecord]:
    result = {}
    observed = []
    for index, row in enumerate(rows, start=1):
        record = GoldRecord.from_mapping(row, f"private gold:{index}")
        if record.split_id != split_id:
            raise PhaseBArtifactError("private gold split differs from diagnostic split")
        prepared_row = prepared.get(record.example_id)
        if prepared_row is None:
            raise PhaseBArtifactError(
                f"private gold has no prepared example: {record.example_id}"
            )
        if prepared_row.source_document_id != record.source_document_id:
            raise PhaseBArtifactError(
                f"source identity differs for gold example {record.example_id}"
            )
        result[record.example_id] = record
        observed.append(record.example_id)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise PhaseBArtifactError("private gold must be uniquely and stably sorted")
    if set(result) != set(prepared):
        raise PhaseBArtifactError("prepared test and private gold example sets differ")
    return result


def _load_candidates(
    rows: Iterable[dict[str, Any]],
    gold: Mapping[str, GoldRecord],
    *,
    split_id: str,
    seeds: tuple[int, ...],
    prepared_sha256: str,
) -> tuple[Candidate, ...]:
    result = []
    ids = set()
    for index, row in enumerate(rows, start=1):
        candidate = Candidate.from_mapping(row, f"test candidates:{index}")
        if candidate.split_id != split_id:
            raise PhaseBArtifactError("candidate split differs from diagnostic split")
        if candidate.training_seed not in seeds:
            raise PhaseBArtifactError("candidate seed differs from configured seed family")
        gold_record = gold.get(candidate.triple.example_id)
        if gold_record is None:
            raise PhaseBArtifactError("candidate example has no private gold record")
        if candidate.source_document_id != gold_record.source_document_id:
            raise PhaseBArtifactError("candidate source identity differs from private gold")
        if candidate.input_hashes.get("prepared_sentences") != prepared_sha256:
            raise PhaseBArtifactError(
                f"candidate {candidate.candidate_id} is not bound to prepared test text"
            )
        if candidate.candidate_id in ids:
            raise PhaseBArtifactError(f"duplicate candidate ID: {candidate.candidate_id}")
        ids.add(candidate.candidate_id)
        result.append(candidate)
    if [item.sort_key() for item in result] != sorted(item.sort_key() for item in result):
        raise PhaseBArtifactError("test candidate ledger is not stably sorted")
    if tuple(sorted({item.training_seed for item in result})) != seeds:
        raise PhaseBArtifactError("test candidate ledger lacks the complete seed family")
    return tuple(result)


def _load_verdicts(
    rows: Iterable[dict[str, Any]],
    candidates: Mapping[str, Candidate],
    *,
    condition_id: str,
) -> dict[str, Verdict]:
    result = {}
    order = []
    for index, row in enumerate(rows, start=1):
        candidate_id = row.get("candidate_id")
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise PhaseBArtifactError(
                f"{condition_id} verdict {index} references an unknown candidate"
            )
        verdict = Verdict.from_mapping(
            row,
            f"{condition_id} verdicts:{index}",
            candidate,
        )
        if verdict.condition_id != condition_id:
            raise PhaseBArtifactError(f"verdict condition differs from {condition_id}")
        if verdict.candidate_id in result:
            raise PhaseBArtifactError(f"duplicate {condition_id} verdict")
        result[verdict.candidate_id] = verdict
        order.append((verdict.training_seed, verdict.candidate_id))
    if order != sorted(order):
        raise PhaseBArtifactError(f"{condition_id} verdicts are not stably sorted")
    if set(result) != set(candidates):
        raise PhaseBArtifactError(f"{condition_id} verdict coverage differs from candidates")
    return result


def _selected_threshold(
    value: Mapping[str, Any],
    *,
    protocol_id: str,
    split_hash: str,
    development_gold_hash: str,
    candidate_index_hash: str,
) -> float:
    _require_identity(
        value,
        {
            "protocol_id": protocol_id,
            "selection_split": "development",
            "objective": "mean_per_seed_development_strict_triple_f1",
            "tie_rule": "higher_threshold",
            "used_test_labels": False,
            "split_manifest_sha256": split_hash,
            "development_gold_sha256": development_gold_hash,
            "development_candidate_index_sha256": candidate_index_hash,
        },
        "threshold selection",
    )
    threshold = value.get("selected_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise PhaseBArtifactError("selected confidence threshold is invalid")
    return float(threshold)


def _emitted_for(
    condition: str,
    candidate: Candidate,
    threshold: float,
    simple: Mapping[str, Verdict],
    corrective: Mapping[str, Verdict],
) -> StrictKey | None:
    if condition == "VER-CONFIDENCE":
        return candidate.triple.key() if candidate.triple_confidence >= threshold else None
    verdict = (
        simple[candidate.candidate_id]
        if condition == "VER-SIMPLE"
        else corrective[candidate.candidate_id]
    )
    if verdict.response_status != "valid_response":
        return None
    if verdict.action == "KEEP":
        return candidate.triple.key()
    if (
        condition == "VER-CORRECTIVE"
        and verdict.action == "CORRECT"
        and verdict.correction_validation_status == "valid"
        and verdict.corrected is not None
    ):
        return verdict.corrected.key()
    return None


def _derive_emitted(
    candidates: tuple[Candidate, ...],
    threshold: float,
    simple: Mapping[str, Verdict],
    corrective: Mapping[str, Verdict],
    *,
    conditions: Mapping[str, str],
    seeds: tuple[int, ...],
    example_ids: Iterable[str],
) -> dict[str, dict[int, dict[str, frozenset[StrictKey]]]]:
    mutable: dict[str, dict[int, dict[str, set[StrictKey]]]] = {
        condition: {
            seed: {example_id: set() for example_id in example_ids}
            for seed in seeds
        }
        for condition in conditions
    }
    for candidate in candidates:
        for condition in conditions:
            key = _emitted_for(
                condition,
                candidate,
                threshold,
                simple,
                corrective,
            )
            if key is not None:
                mutable[condition][candidate.training_seed][
                    candidate.triple.example_id
                ].add(key)
    return {
        condition: {
            seed: {
                example_id: frozenset(keys)
                for example_id, keys in sorted(examples.items())
            }
            for seed, examples in sorted(by_seed.items())
        }
        for condition, by_seed in sorted(mutable.items())
    }


def _validate_outcomes(
    rows: Iterable[dict[str, Any]],
    emitted: Mapping[str, Mapping[int, Mapping[str, frozenset[StrictKey]]]],
    gold: Mapping[str, GoldRecord],
    *,
    protocol_id: str,
    matcher_id: str,
) -> None:
    observed = set()
    for index, row in enumerate(rows, start=1):
        condition = row.get("condition_id")
        if condition not in emitted:
            continue
        _require_identity(
            row,
            {"protocol_id": protocol_id, "matcher_id": matcher_id},
            f"sentence outcome {index}",
        )
        seed = row.get("training_seed")
        example_id = row.get("example_id")
        key = (condition, seed, example_id)
        if key in observed:
            raise PhaseBArtifactError(f"duplicate sentence outcome: {key}")
        observed.add(key)
        try:
            expected = emitted[condition][seed][example_id]
        except (KeyError, TypeError) as error:
            raise PhaseBArtifactError(f"foreign sentence outcome identity: {key}") from error
        recorded = row.get("emitted_strict_keys")
        if not isinstance(recorded, list):
            raise PhaseBArtifactError(f"sentence outcome has no emitted keys: {key}")
        recorded_keys = frozenset(tuple(item) for item in recorded)
        if recorded_keys != expected:
            raise PhaseBArtifactError(
                f"independent condition reconstruction differs from outcome ledger: {key}"
            )
        gold_keys = frozenset(item.key() for item in gold[example_id].triples)
        counts = {
            "tp": len(expected & gold_keys),
            "fp": len(expected - gold_keys),
            "fn": len(gold_keys - expected),
        }
        if row.get("counts") != counts:
            raise PhaseBArtifactError(f"sentence outcome counts do not recompute: {key}")
    required = {
        (condition, seed, example_id)
        for condition in emitted
        for seed in emitted[condition]
        for example_id in emitted[condition][seed]
    }
    if observed != required:
        raise PhaseBArtifactError("sentence outcome ledger lacks the diagnostic matrix")


def strict_counts(
    emitted: Mapping[str, Mapping[int, Mapping[str, frozenset[StrictKey]]]],
    gold: Mapping[str, GoldRecord],
) -> dict[str, dict[int, dict[str, int]]]:
    result: dict[str, dict[int, dict[str, int]]] = {}
    gold_keys = {
        example_id: frozenset(item.key() for item in record.triples)
        for example_id, record in gold.items()
    }
    for condition, by_seed in emitted.items():
        result[condition] = {}
        for seed, examples in by_seed.items():
            counts = {"tp": 0, "fp": 0, "fn": 0}
            for example_id, predicted in examples.items():
                target = gold_keys[example_id]
                counts["tp"] += len(predicted & target)
                counts["fp"] += len(predicted - target)
                counts["fn"] += len(target - predicted)
            result[condition][seed] = counts
    return result


def _validate_phase_b_metrics(
    metrics: Mapping[str, Any],
    recomputed: Mapping[str, Mapping[int, Mapping[str, int]]],
    *,
    conditions: Mapping[str, str],
    seeds: tuple[int, ...],
    selected_threshold: float,
) -> None:
    if metrics.get("selected_confidence_threshold") != selected_threshold:
        raise PhaseBArtifactError("Phase B metric threshold differs from selection manifest")
    if tuple(metrics.get("observed_training_seeds", ())) != seeds:
        raise PhaseBArtifactError("Phase B metrics do not contain the complete seed family")
    phase_conditions = metrics.get("conditions")
    if not isinstance(phase_conditions, dict):
        raise PhaseBArtifactError("Phase B metrics have no condition mapping")
    for condition in conditions:
        per_seed = phase_conditions.get(condition, {}).get("per_seed", {})
        for seed in seeds:
            recorded = (
                per_seed.get(str(seed), {})
                .get("end_to_end_strict_triple", {})
                .get("counts")
            )
            if recorded != recomputed[condition][seed]:
                raise PhaseBArtifactError(
                    f"independent strict counts differ from Phase B metrics: "
                    f"{condition}/seed-{seed}"
                )


def load_phase_b_run(
    parent_root: str | Path,
    config: Mapping[str, Any],
    *,
    run_id: str,
) -> PhaseBRunData:
    """Validate every required parent byte before returning derived conditions."""

    root = Path(parent_root)
    if root.is_symlink() or not root.is_dir():
        raise PhaseBArtifactError("Phase B parent run root must be a real directory")
    reader = _ParentReader(root)
    artifacts = config["artifacts"]
    parent = config["parent"]
    protocol_id = parent["phase_b_protocol_id"]
    workflow_id = parent["workflow_id"]
    matcher_id = parent["matcher_id"]
    dataset_id = parent["dataset_id"]
    split_id = config["protocol"]["evaluation_split_id"]
    seeds = tuple(parent["training_seeds"])
    conditions = config["conditions"]

    checkout_path = reader.verify(
        artifacts["checkout_manifest"], None, "Phase B checkout identity"
    )
    checkout = _load_json(checkout_path, "Phase B checkout identity")
    _require_identity(
        checkout,
        {
            "schema_version": "phase-b-checkout-manifest-1.0",
            "status": "pass",
            "run_id": run_id,
            "protocol_id": protocol_id,
            "workflow_id": workflow_id,
        },
        "checkout manifest",
    )

    full_path = reader.verify(
        artifacts["full_run_manifest"], None, "Phase B full-run identity"
    )
    full = _load_json(full_path, "Phase B full-run identity")
    _require_identity(
        full,
        {
            "schema_version": "phase-b-debug-full-run-1.0",
            "run_id": run_id,
            "protocol_id": protocol_id,
            "workflow_id": workflow_id,
            "conditional_b07_approval": True,
            "training_seeds": list(seeds),
        },
        "full-run manifest",
    )

    preparation_path = reader.verify(
        artifacts["preparation_manifest"], None, "data preparation identity"
    )
    preparation = _load_json(preparation_path, "data preparation identity")
    _require_identity(
        preparation,
        {
            "schema_version": "phase-b-data-preparation-manifest-2.0",
            "protocol_id": protocol_id,
            "dataset_id": dataset_id,
            "byte_identical_independent_materializations": True,
        },
        "preparation manifest",
    )
    prepared_hash = _preparation_hash(preparation, "test.jsonl")
    gold_hash = _preparation_hash(preparation, "test-gold.jsonl")
    split_hash = _preparation_hash(preparation, "split-manifest.json")
    development_gold_hash = _preparation_hash(preparation, "development-gold.jsonl")
    prepared_path = reader.verify(
        artifacts["prepared_test"], prepared_hash, "prepared test corpus"
    )
    gold_path = reader.verify(
        artifacts["private_gold"], gold_hash, "private gold/support records"
    )
    reader.verify(artifacts["split_manifest"], split_hash, "split identity")
    reader.verify(
        artifacts["development_gold"],
        development_gold_hash,
        "development threshold gold",
    )

    candidate_index_path = reader.verify(
        artifacts["development_candidate_index"],
        None,
        "development candidate index",
    )
    candidate_index_hash = file_sha256(candidate_index_path)
    threshold_path = reader.verify(
        artifacts["threshold_selection"], None, "development threshold selection"
    )
    threshold_document = _load_json(
        threshold_path, "development threshold selection"
    )
    threshold = _selected_threshold(
        threshold_document,
        protocol_id=protocol_id,
        split_hash=split_hash,
        development_gold_hash=development_gold_hash,
        candidate_index_hash=candidate_index_hash,
    )

    score_path = reader.verify(
        artifacts["score_manifest"], None, "Phase B score identity"
    )
    score = _load_json(score_path, "Phase B score identity")
    _require_identity(
        score,
        {
            "schema_version": "phase-b-score-manifest-1.0",
            "protocol_id": protocol_id,
            "workflow_id": workflow_id,
            "matcher_id": matcher_id,
            "run_id": run_id,
            "nonpublication_smoke": False,
        },
        "score manifest",
    )
    score_inputs = score.get("inputs")
    score_outputs = score.get("outputs")
    if not isinstance(score_inputs, dict) or not isinstance(score_outputs, dict):
        raise PhaseBArtifactError("score manifest lacks input/output hash mappings")
    for key in (
        "private_gold",
        "threshold_selection",
        "candidates",
        "simple_verdicts",
        "corrective_verdicts",
    ):
        relative = artifacts[key]
        reader.verify(
            relative,
            _manifest_hash(score_inputs, relative, label="score.inputs"),
            f"score input: {key}",
        )
    for key in ("candidate_outcomes", "sentence_outcomes", "phase_b_metrics"):
        relative = artifacts[key]
        reader.verify(
            relative,
            _manifest_hash(score_outputs, relative, label="score.outputs"),
            f"score output: {key}",
        )

    simple_manifest = _verifier_contract(
        reader,
        artifacts["simple_verifier_manifest"],
        artifacts["simple_verdicts"],
        artifacts["candidates"],
        artifacts["prepared_test"],
        condition_id="VER-SIMPLE",
        expected_protocol=protocol_id,
    )
    corrective_manifest = _verifier_contract(
        reader,
        artifacts["corrective_verifier_manifest"],
        artifacts["corrective_verdicts"],
        artifacts["candidates"],
        artifacts["prepared_test"],
        condition_id="VER-CORRECTIVE",
        expected_protocol=protocol_id,
    )

    prepared = _load_prepared(
        _jsonl(prepared_path, "prepared test corpus"),
        dataset_id=dataset_id,
        protocol_id=protocol_id,
    )
    gold = _load_gold(
        _jsonl(gold_path, "private gold/support records"),
        prepared,
        split_id=split_id,
    )
    candidate_path = reader.path(artifacts["candidates"])
    candidates = _load_candidates(
        _jsonl(candidate_path, "test candidate ledger"),
        gold,
        split_id=split_id,
        seeds=seeds,
        prepared_sha256=prepared_hash,
    )
    candidate_by_id = {item.candidate_id: item for item in candidates}
    simple = _load_verdicts(
        _jsonl(reader.path(artifacts["simple_verdicts"]), "simple verdict ledger"),
        candidate_by_id,
        condition_id="VER-SIMPLE",
    )
    corrective = _load_verdicts(
        _jsonl(
            reader.path(artifacts["corrective_verdicts"]),
            "corrective verdict ledger",
        ),
        candidate_by_id,
        condition_id="VER-CORRECTIVE",
    )
    if (
        len(simple) != simple_manifest.get("verdict_count")
        or len(corrective) != corrective_manifest.get("verdict_count")
    ):
        raise PhaseBArtifactError("verifier manifest count differs from verdict ledger")

    emitted = _derive_emitted(
        candidates,
        threshold,
        simple,
        corrective,
        conditions=conditions,
        seeds=seeds,
        example_ids=tuple(sorted(gold)),
    )
    _validate_outcomes(
        _jsonl(
            reader.path(artifacts["sentence_outcomes"]),
            "sentence outcome ledger",
        ),
        emitted,
        gold,
        protocol_id=protocol_id,
        matcher_id=matcher_id,
    )
    phase_metrics = _load_json(
        reader.path(artifacts["phase_b_metrics"]),
        "Phase B metrics",
    )
    _validate_phase_b_metrics(
        phase_metrics,
        strict_counts(emitted, gold),
        conditions=conditions,
        seeds=seeds,
        selected_threshold=threshold,
    )

    return PhaseBRunData(
        run_id=run_id,
        protocol_id=protocol_id,
        workflow_id=workflow_id,
        matcher_id=matcher_id,
        dataset_id=dataset_id,
        seeds=seeds,
        selected_threshold=threshold,
        prepared=prepared,
        gold=gold,
        candidates=candidates,
        simple_verdicts=simple,
        corrective_verdicts=corrective,
        emitted=emitted,
        phase_b_metrics=phase_metrics,
        verified_artifacts=reader.artifacts(),
        score_manifest_sha256=file_sha256(score_path),
    )


def strict_key_from_sequence(value: Iterable[Any]) -> StrictKey:
    key = tuple(value)
    if len(key) != 8:
        raise PhaseBArtifactError("strict key must contain eight fields")
    return key  # type: ignore[return-value]


def strict_triple_from_key(
    key: StrictKey,
    prepared: Mapping[str, PreparedExample],
) -> StrictTriple:
    example_id, hs, he, ht, relation, ts, te, tt = key
    example = prepared.get(example_id)
    if example is None:
        raise PhaseBArtifactError(f"strict key references unknown example: {example_id}")
    return StrictTriple.from_mapping(
        {
            "head": {
                "start": hs,
                "end": he,
                "type": ht,
                "text": example.span_text(hs, he),
            },
            "relation": relation,
            "tail": {
                "start": ts,
                "end": te,
                "type": tt,
                "text": example.span_text(ts, te),
            },
        },
        example_id=example_id,
        label="derived strict triple",
    )
