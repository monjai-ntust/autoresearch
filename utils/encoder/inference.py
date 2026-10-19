"""encoder inference responsibilities."""
from __future__ import annotations

from utils.encoder.candidates import ScoredSpan, _select_non_overlapping, _CONFIDENCE_PRECISION
from pathlib import Path
from typing import Any, TYPE_CHECKING
from utils.common.artifact_io import DataContractError
if TYPE_CHECKING:
    from utils.encoder.model import CheckpointIdentity
from utils.common.config import PipelineConfig
from utils.common.constants import PROTOCOL_ID
from utils.verifier.requests import PreparedSentence

def _live_inference_records(
    config: PipelineConfig,
    checkpoint: CheckpointIdentity,
    sentences: dict[str, PreparedSentence],
    checkpoint_blob_path: Path,
    base_model: str,
    device_preference: str | None,
    model_cache_dir: Path,
) -> list[dict[str, Any]]:
    """Run the retained encoder over prepared sentences and emit ledger records.

    This is the externally gated `live` execution. It reuses the retained
    ``utils.encoder.network.BertKGExtractor`` and the exact
    forward logic established by the retained source history (greedy
    non-overlapping span selection, then relation extraction over the selected
    pairs with a softmax-product confidence), but binds the CODE label space and
    writes the canonical prediction-ledger schema. Torch and the model are
    imported lazily so the module remains importable without an accelerator
    stack.

    Assumptions (documented for external verification): the checkpoint was trained
    with the same CODE label space (``utils.encoder.data``), ``re_context_span``
    equal to the frozen ``context_between_spans`` recipe flag, and each sentence
    fits within the recipe ``max_length``. These are enforced by the one retained
    trainer path.
    """

    import torch  # lazy: only the externally gated live path needs the accelerator stack
    from transformers import AutoTokenizer

    from utils.encoder.data import (
        ENTITY_TYPES as CA_ENTITY_TYPES,
        ID2REL,
        NO_REL_ID,
        NUM_BIO_TAGS,
        NUM_RELATIONS,
    )
    from utils.encoder.network import BertKGExtractor

    training = config.value["training"]
    from utils.encoder.runtime import configure_numerics, record_model_runtime, record_runtime
    configure_numerics()
    record_runtime(checkpoint_blob_path.parent.parent.parent / 'data-prepared')
    max_length = training["max_length"]
    max_span_width = training["max_span_width"]
    device = torch.device(device_preference or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_revision = training["base_model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=model_revision,
        cache_dir=str(model_cache_dir),
        local_files_only=True,
    )
    model = BertKGExtractor(
        base_model,
        num_bio_tags=NUM_BIO_TAGS,
        num_relations=NUM_RELATIONS,
        num_entity_types=len(CA_ENTITY_TYPES),
        max_span_width=max_span_width,
        model_revision=model_revision,
        model_cache_dir=str(model_cache_dir),
        model_local_files_only=True,
    )
    # Match stages/encoder.py: enabling re_context_span (the frozen
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
    record_model_runtime(checkpoint_blob_path.parent.parent.parent / 'data-prepared', model)
    state = torch.load(str(checkpoint_blob_path), map_location=device)
    if not isinstance(state, dict) or set(state) != {"encoder", "step", "metrics"}:
        raise DataContractError("fresh inference requires the authenticated encoder checkpoint")
    model.load_state_dict(state["encoder"])
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
            hidden = model.encode(input_ids=input_ids, attention_mask=attention_mask)
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
