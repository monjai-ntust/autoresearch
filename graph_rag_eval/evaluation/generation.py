"""Frozen non-judge answer, support, and abstention metrics."""

from __future__ import annotations

from collections import Counter
import re

from graph_rag_eval.contracts import AnswerAlias, EvidenceSet
from graph_rag_eval.contracts import QueryView
from graph_rag_eval.retrieval.base import RetrievalResult

ABSTENTION = "INSUFFICIENT_EVIDENCE"


class FrozenTransformerGenerator:
    """Pinned local-cache causal generator for externally approved runs."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str,
        prompt_template: str,
        device: str = "cpu",
        local_files_only: bool = True,
        max_input_tokens: int = 2048,
        max_output_tokens: int = 64,
        seed: int = 42,
    ):
        if not revision or revision in {"main", "latest"}:
            raise ValueError("generator requires an immutable revision")
        if "{question}" not in prompt_template or "{evidence}" not in prompt_template:
            raise ValueError("prompt template must contain question and evidence fields")
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.prompt_template = prompt_template
        self.device = device
        self.local_files_only = local_files_only
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self._model.eval()

    def generate(self, query: QueryView, result: RetrievalResult) -> tuple[str, dict]:
        self._load()
        import torch

        evidence = "\n".join(item.content for item in result.items)
        prompt = self.prompt_template.format(question=query.text, evidence=evidence)
        encoded = self._tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        torch.manual_seed(self.seed)
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_output_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        response = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return response, {
            "prompt": prompt,
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "output_tokens": int(generated.shape[0]),
            "model_id": self.model_id,
            "revision": self.revision,
            "seed": self.seed,
        }


def normalize_answer(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def token_f1(prediction: str, target: str) -> float:
    predicted = Counter(normalize_answer(prediction).split())
    gold = Counter(normalize_answer(target).split())
    common = sum((predicted & gold).values())
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold or common == 0:
        return 0.0
    precision = common / sum(predicted.values())
    recall = common / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, answer: AnswerAlias) -> dict[str, float | bool | str]:
    abstained = prediction.strip() == ABSTENTION
    if answer.abstention_expected:
        return {
            "status": "available",
            "exact_match": float(abstained),
            "token_f1": float(abstained),
            "abstention_expected": True,
            "abstained": abstained,
        }
    normalized_prediction = normalize_answer(prediction)
    normalized_targets = [normalize_answer(item) for item in answer.normalized_answers]
    return {
        "status": "available",
        "exact_match": float(normalized_prediction in normalized_targets),
        "token_f1": max(token_f1(prediction, item) for item in answer.normalized_answers),
        "abstention_expected": False,
        "abstained": abstained,
    }


def support_metrics(
    retrieved_ids: tuple[str, ...],
    evidence_sets: tuple[EvidenceSet, ...],
    question_id: str,
) -> dict[str, float | int | str]:
    alternatives = [
        set(item.evidence_ids)
        for item in evidence_sets
        if item.question_id == question_id
    ]
    if not alternatives:
        return {
            "status": "not_applicable",
            "support_precision": 0.0,
            "support_recall": 0.0,
            "denominator": 0,
        }
    retrieved = set(retrieved_ids)
    best = max(alternatives, key=lambda item: len(item & retrieved) / len(item))
    overlap = len(retrieved & best)
    return {
        "status": "available",
        "support_precision": overlap / len(retrieved) if retrieved else 0.0,
        "support_recall": overlap / len(best),
        "denominator": len(best),
    }
