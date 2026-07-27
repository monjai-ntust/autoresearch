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

    generator_id = "frozen-transformer-v1"

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

    def build_prompt(self, query: QueryView, result: RetrievalResult) -> str:
        return self.prompt_template.format(
            question=query.text,
            evidence="\n".join(item.content for item in result.items),
        )

    def generate(self, query: QueryView, result: RetrievalResult) -> tuple[str, dict]:
        self._load()
        import torch

        prompt = self.build_prompt(query, result)
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


class RuleBasedEvidenceGenerator:
    """Offline generator whose extraction rules are supplied by configuration.

    It never reads answers, evidence labels, or gold graph fields: the prompt and
    every rule apply only to the public question text and the retrieved evidence.
    Rules live in the run configuration so no dataset-specific pattern is
    compiled into core code.
    """

    generator_id = "rule-based-evidence-fixture-v1"

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        prompt_template: str = "Question: {question}\nEvidence:\n{evidence}",
        rules: tuple[dict, ...] = (),
        seed: int = 42,
    ):
        if not revision or revision in {"main", "latest"}:
            raise ValueError("generator requires an immutable revision")
        if "{question}" not in prompt_template or "{evidence}" not in prompt_template:
            raise ValueError("prompt template must contain question and evidence fields")
        self.model_id = model_id
        self.revision = revision
        self.prompt_template = prompt_template
        self.rules = tuple(dict(item) for item in rules)
        self.seed = seed

    def build_prompt(self, query: QueryView, result: RetrievalResult) -> str:
        return self.prompt_template.format(
            question=query.text,
            evidence="\n".join(item.content for item in result.items),
        )

    def _matches(self, rule: dict, question: str) -> bool:
        keywords = rule.get("question_any", ())
        return any(keyword in question for keyword in keywords) if keywords else False

    def generate(self, query: QueryView, result: RetrievalResult) -> tuple[str, dict]:
        prompt = self.build_prompt(query, result)
        question = query.text.casefold()
        triples = []
        for item in result.items:
            match = re.fullmatch(r"\((.*?), (.*?), (.*?)\)", item.content)
            if match:
                triples.append(tuple(part.strip() for part in match.groups()))
        joined = " ".join(item.content for item in result.items).casefold()
        response = ABSTENTION
        for rule in self.rules:
            if not self._matches(rule, question):
                continue
            relation = rule.get("relation")
            if relation is not None:
                tails = [tail for _, label, tail in triples if label == relation]
                if tails:
                    response = tails[0]
                    break
            pattern = rule.get("text_pattern")
            if pattern:
                found = re.search(pattern, joined)
                if found:
                    response = found.group(1).strip()
                    break
        return response, {
            "prompt": prompt,
            "input_tokens": len(re.findall(r"\w+|[^\w\s]", prompt)),
            "output_tokens": len(re.findall(r"\w+|[^\w\s]", response)),
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
    """Score one answer, keeping abstention outcomes out of the accuracy metrics.

    An unanswerable question emits `abstention_correct` and no `exact_match` or
    `token_f1`, so a condition that abstains everywhere cannot inflate the pooled
    answer-accuracy mean. `answer_correct` is the single regime-neutral outcome
    that the coupled layer consumes.
    """

    abstained = prediction.strip() == ABSTENTION
    if answer.abstention_expected:
        return {
            "status": "available",
            "answer_correct": float(abstained),
            "abstention_correct": float(abstained),
            "abstention_expected": True,
            "abstained": abstained,
        }
    normalized_prediction = normalize_answer(prediction)
    normalized_targets = [normalize_answer(item) for item in answer.normalized_answers]
    exact_match = float(normalized_prediction in normalized_targets)
    return {
        "status": "available",
        "answer_correct": exact_match,
        "exact_match": exact_match,
        "token_f1": max(token_f1(prediction, item) for item in answer.normalized_answers),
        "abstention_expected": False,
        "abstained": abstained,
    }


def support_metrics(
    retrieved_ids: tuple[str, ...],
    evidence_sets: tuple[EvidenceSet, ...],
    question_id: str,
) -> dict[str, float | int | str | list[str]]:
    """Score retrieved evidence against the best-matching acceptable evidence set.

    A question with no frozen evidence set yields a typed unavailable record with
    no numeric fields; emitting 0.0 would silently enter aggregate means as a
    measured failure.
    """

    alternatives = [
        set(item.evidence_ids)
        for item in evidence_sets
        if item.question_id == question_id
    ]
    if not alternatives:
        return {
            "status": "not_applicable",
            "missing": ["question_evidence_set"],
            "reason": "question has no frozen acceptable evidence set",
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
