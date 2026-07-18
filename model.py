"""Canonical encoder candidate-generation adapter for ``CODE-STRICT-1``.

This B-05U slice adds the ``phase_b.py model generate-candidates`` stage. It is
the canonical successor to the historical ``inference_kg.py`` script named in
``docs/historical-transition-map.md``. The historical script loaded an arbitrary
checkpoint, ran span NER plus relation extraction over a runtime-selected
dataset, and wrote a free-form ``results/kg_inference.jsonl`` outside the output
contract. This adapter keeps the same extraction *semantics* the paper draft
depends on -- the encoder proposes entity spans and relations with confidence
scores, and a triple's confidence is the encoder softmax product (draft
Section 3.1) -- while binding every emitted record to the typed strict data,
split, and checkpoint identities and confining all output beneath the ignored
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
* ``live`` runs the retained encoder (``models.bert_kg_encoder`` under the
  frozen recipe) over the prepared sentences to produce that prediction ledger,
  then applies the same deterministic ledger->candidate transform as ``replay``.
  Its forward outputs cannot be validated offline, so live execution remains
  externally gated on the accelerator + real checkpoint, and ``inference_kg.py``
  stays retained as provenance until external output parity is confirmed (see
  ``docs/historical-transition-map.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PipelineConfig
from constants import ENTITY_TYPES, MATCHER_ID, PROTOCOL_ID, RELATION_TYPES, TRAINING_SEEDS
from paths import RunLayout
from phase_b_io import (
    DataContractError,
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    iter_jsonl,
    sha256_file,
)
from records import Candidate, EntitySpan, StrictTriple, candidate_id_for
from verifier import PreparedSentence, _load_sentences, _source_text

EXECUTION_MODES = {"dry-run", "replay", "live"}
TRAIN_EXECUTION_MODES = {"dry-run"}
_SPLIT_ID = "CODE-SPLIT-1"
_CONFIDENCE_PRECISION = 6
_CHECKPOINT_MANIFEST_CONTRACT = "urn:phase-b:model-checkpoint-manifest:1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _exact_keys(
    value: dict[str, Any], required: set[str], label: str, optional: set[str] | None = None
) -> None:
    optional = optional or set()
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be an object")
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing:
        raise DataContractError(f"{label} is missing required fields: {', '.join(missing)}")
    if extra:
        raise DataContractError(f"{label} contains unsupported fields: {', '.join(extra)}")


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise DataContractError(f"{label} must be >= {minimum}")
    return value


def _unit_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataContractError(f"{label} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise DataContractError(f"{label} must be finite")
    if not 0.0 <= number <= 1.0:
        raise DataContractError(f"{label} must be in [0, 1]")
    return number


@dataclass(frozen=True)
class CheckpointIdentity:
    training_seed: int
    checkpoint_sha256: str
    checkpoint_step: int
    split_manifest_sha256: str
    max_span_width: int


@dataclass(frozen=True)
class ScoredSpan:
    start: int
    end: int
    entity_type: str
    confidence: float


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
            "protocol_id",
            "split_id",
            "training_seed",
            "base_model",
            "base_model_revision",
            "checkpoint_sha256",
            "checkpoint_step",
            "split_manifest_sha256",
            "max_span_width",
            "context_between_spans",
        },
        label,
    )
    if value["protocol_id"] != PROTOCOL_ID:
        raise DataContractError(f"{label}.protocol_id differs from {PROTOCOL_ID}")
    if value["split_id"] != _SPLIT_ID:
        raise DataContractError(f"{label}.split_id must be {_SPLIT_ID}")
    seed = _integer(value["training_seed"], f"{label}.training_seed")
    if seed not in TRAINING_SEEDS:
        raise DataContractError(f"{label}.training_seed is outside seeds 42-49")
    training = config.value["training"]
    for field in ("base_model", "base_model_revision", "max_span_width", "context_between_spans"):
        if value[field] != training[field]:
            raise DataContractError(
                f"{label}.{field} differs from the approved training recipe"
            )
    step = _integer(value["checkpoint_step"], f"{label}.checkpoint_step", minimum=1)
    return CheckpointIdentity(
        training_seed=seed,
        checkpoint_sha256=_sha256_hex(value["checkpoint_sha256"], f"{label}.checkpoint_sha256"),
        checkpoint_step=step,
        split_manifest_sha256=_sha256_hex(
            value["split_manifest_sha256"], f"{label}.split_manifest_sha256"
        ),
        max_span_width=_integer(value["max_span_width"], f"{label}.max_span_width", minimum=1),
    )


def _scored_span(value: Any, label: str, max_span_width: int) -> ScoredSpan:
    _exact_keys(value, {"start", "end", "type", "confidence"}, label)
    start = _integer(value["start"], f"{label}.start", minimum=0)
    end = _integer(value["end"], f"{label}.end", minimum=0)
    if end < start:
        raise DataContractError(f"{label}.end must be >= {label}.start")
    if end - start + 1 > max_span_width:
        raise DataContractError(f"{label} width exceeds the recipe max_span_width")
    entity_type = value["type"]
    if entity_type not in ENTITY_TYPES:
        raise DataContractError(f"{label}.type must be one of {list(ENTITY_TYPES)}")
    return ScoredSpan(
        start=start,
        end=end,
        entity_type=entity_type,
        confidence=_unit_float(value["confidence"], f"{label}.confidence"),
    )


def _endpoint(value: Any, label: str) -> tuple[int, int]:
    _exact_keys(value, {"start", "end"}, label)
    start = _integer(value["start"], f"{label}.start", minimum=0)
    end = _integer(value["end"], f"{label}.end", minimum=0)
    if end < start:
        raise DataContractError(f"{label}.end must be >= {label}.start")
    return start, end


def _select_non_overlapping(spans: list[ScoredSpan]) -> dict[tuple[int, int], ScoredSpan]:
    """Greedy highest-confidence non-overlapping selection (inference_kg parity).

    ``spans`` is required to be ordered by ``(start, end, type)`` so that the
    stable descending-confidence sort resolves ties deterministically, matching
    the historical script's stable ``sort(key=-confidence)`` over an ordered
    candidate list.
    """

    ordered = sorted(range(len(spans)), key=lambda index: -spans[index].confidence)
    taken: list[tuple[int, int]] = []
    selected: dict[tuple[int, int], ScoredSpan] = {}
    for index in ordered:
        span = spans[index]
        overlaps = any(not (span.end < start or end < span.start) for start, end in taken)
        if overlaps:
            continue
        taken.append((span.start, span.end))
        selected[(span.start, span.end)] = span
    return selected


def _sentence_candidates(
    sentence: PreparedSentence,
    scored_spans: list[ScoredSpan],
    relations: list[dict[str, Any]],
    seed: int,
    input_hashes: dict[str, str],
    label: str,
) -> list[dict[str, Any]]:
    selected = _select_non_overlapping(scored_spans)
    records: list[dict[str, Any]] = []
    for index, relation in enumerate(relations):
        relation_label = f"{label}.predicted_relations[{index}]"
        _exact_keys(relation, {"head", "tail", "relation", "re_confidence"}, relation_label)
        head_key = _endpoint(relation["head"], f"{relation_label}.head")
        tail_key = _endpoint(relation["tail"], f"{relation_label}.tail")
        if head_key == tail_key:
            raise DataContractError(f"{relation_label} head and tail are identical spans")
        relation_type = relation["relation"]
        if relation_type not in RELATION_TYPES:
            raise DataContractError(
                f"{relation_label}.relation must be one of {list(RELATION_TYPES)}"
            )
        head_span = selected.get(head_key)
        tail_span = selected.get(tail_key)
        if head_span is None or tail_span is None:
            raise DataContractError(
                f"{relation_label} references a span that was not a selected entity"
            )
        re_confidence = _unit_float(relation["re_confidence"], f"{relation_label}.re_confidence")
        confidence = round(
            min(head_span.confidence, tail_span.confidence) * re_confidence,
            _CONFIDENCE_PRECISION,
        )
        head = EntitySpan(
            head_span.start,
            head_span.end,
            head_span.entity_type,
            _source_text(
                sentence,
                EntitySpan(head_span.start, head_span.end, head_span.entity_type),
                f"{relation_label}.head",
            ),
        )
        tail = EntitySpan(
            tail_span.start,
            tail_span.end,
            tail_span.entity_type,
            _source_text(
                sentence,
                EntitySpan(tail_span.start, tail_span.end, tail_span.entity_type),
                f"{relation_label}.tail",
            ),
        )
        triple = StrictTriple(sentence.example_id, head, relation_type, tail)
        record = {
            "protocol_id": PROTOCOL_ID,
            "split_id": f"{_SPLIT_ID}:{sentence.split}",
            "training_seed": seed,
            "example_id": sentence.example_id,
            "source_document_id": sentence.source_document_id,
            "candidate_id": candidate_id_for(seed, triple),
            "head": head.to_mapping(include_text=True),
            "relation": relation_type,
            "tail": tail.to_mapping(include_text=True),
            "triple_confidence": confidence,
            "input_hashes": dict(input_hashes),
        }
        # Re-validate through the shared contract so the emitted file always
        # parses under the same rules the verifier and scorer apply.
        Candidate.from_mapping(record, relation_label)
        records.append(record)
    return records


def _load_prediction_ledger(
    path: Path,
    sentences: dict[str, PreparedSentence],
    checkpoint: CheckpointIdentity,
    input_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], int, int, int]:
    seen: set[str] = set()
    observed_order: list[str] = []
    collected: dict[str, dict[str, Any]] = {}
    duplicate_collapsed = 0
    selected_total = 0
    for line_number, value in iter_jsonl(path):
        label = f"{path.name}:{line_number}"
        _exact_keys(
            value,
            {"protocol_id", "example_id", "training_seed", "predicted_spans", "predicted_relations"},
            label,
        )
        if value["protocol_id"] != PROTOCOL_ID:
            raise DataContractError(f"{label}.protocol_id differs from {PROTOCOL_ID}")
        example_id = value["example_id"]
        if not isinstance(example_id, str) or not example_id:
            raise DataContractError(f"{label}.example_id must be a nonempty string")
        if example_id in seen:
            raise DataContractError(f"{label} duplicates example_id {example_id}")
        seen.add(example_id)
        observed_order.append(example_id)
        sentence = sentences.get(example_id)
        if sentence is None:
            raise DataContractError(f"{label} refers to an unknown prepared sentence")
        seed = _integer(value["training_seed"], f"{label}.training_seed")
        if seed != checkpoint.training_seed:
            raise DataContractError(f"{label}.training_seed differs from the checkpoint identity")
        raw_spans = value["predicted_spans"]
        if not isinstance(raw_spans, list):
            raise DataContractError(f"{label}.predicted_spans must be an array")
        scored_spans = [
            _scored_span(item, f"{label}.predicted_spans[{index}]", checkpoint.max_span_width)
            for index, item in enumerate(raw_spans)
        ]
        span_order = [(span.start, span.end, span.entity_type) for span in scored_spans]
        if span_order != sorted(span_order):
            raise DataContractError(
                f"{label}.predicted_spans must be sorted by start, end, type"
            )
        if len(set(span_order)) != len(span_order):
            raise DataContractError(f"{label}.predicted_spans contains a duplicate span")
        for span in scored_spans:
            _source_text(
                sentence,
                EntitySpan(span.start, span.end, span.entity_type),
                f"{label}.predicted_spans",
            )
        raw_relations = value["predicted_relations"]
        if not isinstance(raw_relations, list):
            raise DataContractError(f"{label}.predicted_relations must be an array")
        selected_total += len(_select_non_overlapping(scored_spans))
        for record in _sentence_candidates(
            sentence, scored_spans, raw_relations, seed, input_hashes, label
        ):
            candidate_id = record["candidate_id"]
            existing = collected.get(candidate_id)
            if existing is None:
                collected[candidate_id] = record
            else:
                duplicate_collapsed += 1
                if record["triple_confidence"] > existing["triple_confidence"]:
                    collected[candidate_id] = record
    if observed_order != sorted(observed_order):
        raise DataContractError(f"{path.name} must be sorted by example_id")
    candidates = sorted(
        collected.values(),
        key=lambda record: (record["training_seed"], record["example_id"], record["candidate_id"]),
    )
    return candidates, len(observed_order), selected_total, duplicate_collapsed


def _live_inference_records(
    config: PipelineConfig,
    checkpoint: CheckpointIdentity,
    sentences: dict[str, PreparedSentence],
    checkpoint_blob_path: Path,
    base_model: str,
    device_preference: str | None,
) -> list[dict[str, Any]]:
    """Run the retained encoder over prepared sentences and emit ledger records.

    This is the externally gated `live` execution. It reuses the retained
    ``models.bert_kg_encoder.BertKGExtractor`` and the exact
    ``inference_kg.py`` forward logic (greedy non-overlapping span selection, then
    relation extraction over the selected pairs with a softmax-product
    confidence), but binds the CODE label space and writes the canonical
    prediction-ledger schema. Torch and the model are imported lazily so the
    module remains importable without an accelerator stack.

    Assumptions (documented for external verification): the checkpoint was trained
    with the same CODE label space (``data.code_accord``), ``re_context_span``
    equal to the frozen ``context_between_spans`` recipe flag, ``bio_enrich`` off,
    and non-marker relation extraction; and each sentence fits within the recipe
    ``max_length``. These match ``model train`` / ``train_span.py`` defaults.
    """

    import torch  # lazy: only the externally gated live path needs the accelerator stack
    from transformers import AutoTokenizer

    from data.code_accord import (
        ENTITY_TYPES as CA_ENTITY_TYPES,
        ID2REL,
        NO_REL_ID,
        NUM_BIO_TAGS,
        NUM_RELATIONS,
    )
    from models.bert_kg_encoder import BertKGExtractor

    training = config.value["training"]
    max_length = training["max_length"]
    max_span_width = training["max_span_width"]
    device = torch.device(device_preference or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = BertKGExtractor(
        base_model,
        num_bio_tags=NUM_BIO_TAGS,
        num_relations=NUM_RELATIONS,
        num_entity_types=len(CA_ENTITY_TYPES),
        use_span_ner=True,
        max_span_width=max_span_width,
    )
    # Match train_span.py: enabling re_context_span (the frozen
    # context_between_spans recipe flag) requires rebuilding re_head with a 3H
    # input before loading a checkpoint trained with that head.
    if bool(training["context_between_spans"]):
        model.re_context_span = True
        hidden_size = model.backbone.hidden_size
        model.re_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 3, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, NUM_RELATIONS),
        )
    model = model.to(device)
    state = torch.load(str(checkpoint_blob_path), map_location=device)
    if isinstance(state, dict) and "encoder" in state:
        model.load_state_dict(state["encoder"])
    elif isinstance(state, dict) and "discriminator" in state:
        model.load_state_dict(state["discriminator"])
    else:
        model.load_state_dict(state)
    model.eval()

    id2entity = {index + 1: entity_type for index, entity_type in enumerate(CA_ENTITY_TYPES)}
    records: list[dict[str, Any]] = []
    for example_id in sorted(sentences):
        sentence = sentences[example_id]
        words = list(sentence.words)
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [encoding["attention_mask"]], dtype=torch.long, device=device
        )
        word_ids = encoding.word_ids()
        n_words = len(words)

        with torch.no_grad():
            hidden = model.encode(
                modality="text", input_ids=input_ids, attention_mask=attention_mask
            )
            span_logits, candidate_spans = model.forward_span_ner(
                hidden[0], word_ids, n_words, max_span_width
            )

        predicted_spans: list[dict[str, Any]] = []
        if candidate_spans:
            span_probabilities = torch.softmax(span_logits, dim=-1)
            predicted_types = span_logits.argmax(dim=-1).tolist()
            predicted_confidences = span_probabilities.max(dim=-1).values.tolist()
            for (start, end), type_id, confidence in zip(
                candidate_spans, predicted_types, predicted_confidences
            ):
                if type_id > 0 and end < n_words:
                    predicted_spans.append(
                        {
                            "start": int(start),
                            "end": int(end),
                            "type": id2entity[type_id],
                            "confidence": round(float(confidence), _CONFIDENCE_PRECISION),
                        }
                    )
        predicted_spans.sort(key=lambda span: (span["start"], span["end"], span["type"]))

        # Build the greedy-selection input from the sorted predicted spans so the
        # live tie-break order is identical to the replay path (which reads the
        # sorted ledger), guaranteeing live candidates equal a replay of the
        # produced ledger.
        scored = [
            ScoredSpan(span["start"], span["end"], span["type"], span["confidence"])
            for span in predicted_spans
        ]
        selected = _select_non_overlapping(scored)
        selected_coords = [(span.start, span.end) for span in selected.values()]
        pairs = [(head, tail) for head in selected_coords for tail in selected_coords if head != tail]
        predicted_relations: list[dict[str, Any]] = []
        if pairs:
            with torch.no_grad():
                relation_logits = model.forward_re(
                    hidden[0],
                    word_ids,
                    [((head[0], head[1]), (tail[0], tail[1])) for head, tail in pairs],
                )
            relation_probabilities = torch.softmax(relation_logits, dim=-1)
            relation_predictions = relation_logits.argmax(dim=-1).tolist()
            relation_confidences = relation_probabilities.max(dim=-1).values.tolist()
            for (head, tail), relation_id, relation_confidence in zip(
                pairs, relation_predictions, relation_confidences
            ):
                if relation_id == NO_REL_ID:
                    continue
                predicted_relations.append(
                    {
                        "head": {"start": head[0], "end": head[1]},
                        "tail": {"start": tail[0], "end": tail[1]},
                        "relation": ID2REL[relation_id],
                        "re_confidence": round(float(relation_confidence), _CONFIDENCE_PRECISION),
                    }
                )

        records.append(
            {
                "protocol_id": PROTOCOL_ID,
                "example_id": example_id,
                "training_seed": checkpoint.training_seed,
                "predicted_spans": predicted_spans,
                "predicted_relations": predicted_relations,
            }
        )
    return records


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
                "max_span_width": checkpoint.max_span_width,
                "training_seed": checkpoint.training_seed,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            }
        )
    return plan


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

    plan_path = candidates_out_path.parent / "generation-plan.jsonl"
    live_ledger_path = candidates_out_path.parent / "prediction-ledger.jsonl"
    manifest_path = layout.resolve(f"manifests/model-generate-candidates-{execution_mode}.json")
    planned_outputs = [manifest_path]
    planned_outputs.append(plan_path if execution_mode == "dry-run" else candidates_out_path)
    if execution_mode == "live":
        planned_outputs.append(live_ledger_path)
    existing = [layout.relative_identity(path) for path in planned_outputs if path.exists()]
    if existing:
        raise DataContractError(
            "model stage refuses to overwrite existing artifacts: " + ", ".join(existing)
        )

    sentences = _load_sentences(sentences_path)
    checkpoint = _load_checkpoint_manifest(checkpoint_manifest_path, config)

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
        records = _live_inference_records(
            config,
            checkpoint,
            sentences,
            checkpoint_blob_path,
            base_model or config.value["training"]["base_model"],
            device,
        )
        atomic_write_jsonl(
            live_ledger_path,
            sorted(records, key=lambda record: record["example_id"]),
        )
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
    atomic_write_jsonl(candidates_out_path, candidates)
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
    atomic_write_json(manifest_path, manifest)
    return manifest


def plan_training(
    layout: RunLayout,
    config: PipelineConfig,
    *,
    execution_mode: str,
    training_seed: int,
) -> dict[str, Any]:
    """Plan deterministic encoder training for one seed (the `model train` stage).

    Only `dry-run` is implemented in this slice: it validates the frozen recipe
    and seed and materializes a machine-readable training plan plus the
    checkpoint-manifest contract that live training must satisfy. Live training
    itself requires the pinned accelerator profile and is an externally gated
    stage; it is not exposed here, so `train_span.py` is not yet removable.
    """

    layout.require_existing()
    if execution_mode not in TRAIN_EXECUTION_MODES:
        raise DataContractError(
            f"unsupported model-train execution mode: {execution_mode!r} "
            "(only dry-run is implemented in this slice)"
        )
    if training_seed not in TRAINING_SEEDS:
        raise DataContractError(f"training_seed {training_seed} is outside seeds 42-49")

    split = config.value["split"]
    if split.get("split_id") != _SPLIT_ID:
        raise DataContractError(f"config.split.split_id must be {_SPLIT_ID}")

    manifest_path = layout.resolve(f"manifests/model-train-{execution_mode}.json")
    if manifest_path.exists():
        raise DataContractError(
            "model train refuses to overwrite existing artifact: "
            + layout.relative_identity(manifest_path)
        )

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "matcher_id": MATCHER_ID,
        "stage": "model-train",
        "execution_mode": execution_mode,
        "status": "planned",
        "created_at_utc": _utc_now(),
        "training_seed": training_seed,
        "split": dict(split),
        "recipe": dict(config.value["training"]),
        "expected_checkpoint_dir": f"checkpoints/seed-{training_seed}",
        "checkpoint_manifest_contract": _CHECKPOINT_MANIFEST_CONTRACT,
        "final_test_selection_forbidden": True,
        "live_execution_status": "gated_external_accelerator",
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
