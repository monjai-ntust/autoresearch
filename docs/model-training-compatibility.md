# Phase B model-training compatibility contract

This document freezes the evidence and minimal-difference boundary for B-05C.
It describes implementation compatibility, not a completed experiment or a
claim that canonical results equal the historical statistics.

## Historical baseline and available evidence

- Fixed line-review baseline: source commit
  `9feafa4029e65ab48ecfb2f452f4b0fabbff0826`; immediate pre-B-05C source:
  `07a8918c5c6c382c3e822d7b93bafab412fc9b7c`.
- Recipe-bearing trainer changes are traceable to `63f030a` (comparison boost),
  `3485686` (context between spans), `37e7fea` (adaptive curriculum), `d6bc904`
  (A20 staircase plus A21 label smoothing), and `292b269` (dataset-directory
  control). The seed-propagation correction is `67d1eb2`.
- The frozen base model is `microsoft/deberta-large` revision
  `28c23d9eb93ea6cf11f845501ab7aeb2a497658b`. This is the first immutable
  model-repository revision that contains `vocab.json` and `merges.txt` for the
  DeBERTa v1 tokenizer. Its `pytorch_model.bin` SHA-256 is
  `79da1770e499ab894937c426e1e43194e2ede6b492945efe0ee777b19722f334`,
  identical to the previously selected `9a8befc6...` revision; only the model-
  repository metadata/tokenizer assets needed for current Transformers loading
  changed. The source config documents
  Python 3.10.20 and uv 0.11.26 without enforcing them; the current lock selects
  Torch 2.9.1 from the CUDA 13.0 index. Every live summary records the actual
  runtime, CPU/memory, GPU/memory, and deterministic-library settings.
- The 2026-05-11 report preserves historical development Triple F1 for seeds
  42–49 as `0.3969, 0.4330, 0.4257, 0.3542, 0.4000, 0.4746, 0.4061,
  0.3871` (mean `0.4097`, reported standard deviation `0.0356`). The
  2026-05-19 report records an exact eight-seed rerun of those development
  values and the historical test values. This is strong secondary evidence
  that the old recipe was reproducible in its then-current environment, but
  the present clone lacks its checkpoints, complete raw logs, full runtime
  manifest, and persisted seed-specific random split memberships.
- The only pre–Phase A CODE-ACCORD file recoverable from Git is
  `data/code_accord/entities/train.csv`, blob
  `1b74f4a7d3693a903d93690adffcc2ef4f276bea`, LF SHA-256
  `c13ad02ab72f0f3a3ddca588c02bcd5d7db1622ae81351d967431623963e4fcd`.
  It byte-matches the official downloaded entity-training member after checkout
  newline normalization. Pre–Phase A relation/test bytes are not recoverable.

Consequently, historical variability and repeated outcomes are known, but the
old seed-specific random partitions cannot be paired with the new fixed
`CODE-SPLIT-1` partition. The comparison inputs and environment are therefore
not sufficiently matched for a meaningful practical-equivalence test, even
though the old workflow reproduced itself exactly. Canonical output must be
called a new rerun. A future equivalence claim requires the missing
data/split/environment evidence and a plan amendment before an eight-seed
comparison is interpreted that way.

## Preserved implementation invariants

| Invariant | Compatibility host |
| --- | --- |
| Encoder/backbone, adapters, parameter names and shapes | `models/bert_kg_encoder.BertKGExtractor`; only an optional HF revision argument was added. |
| Span-NER and relation heads | Existing `train_span.py` construction, including the 3H context-between-spans relation head. |
| Entity/relation label order | Existing `data.code_accord` four entity types and `NO_REL` plus nine relation types. |
| Span enumeration and pair construction | Existing `forward_span_ner`, loss, and evaluation functions. |
| Losses and weights | Existing focal/label-smoothing/RE loss code with A20+A21+A12 values supplied from `phase_b_path_a.json`. |
| Optimizer and schedule | Existing AdamW, weight decay `0.01`, linear warm-up/decay, 3,500 steps, 250 warm-up steps. |
| Batch size and negative sampling | Existing batch size 16 and NER/RE sampling logic; canonical input uses the same dataset/collate classes. |
| Seed initialization | Existing Python and Torch seed path, plus CUDA `manual_seed_all`; actual runtime settings are recorded. |
| Development selection | Existing strict `>` update preserves earliest-step ties on Triple F1. |
| Inference checkpoint representation | Existing `{"encoder": state_dict, "step": ..., "metrics": ...}` payload remains loadable by candidate inference. |

## Necessary opt-in differences

| Difference | Reason and expected effect |
| --- | --- |
| Prepared `train.jsonl`/`development.jsonl` adapter | Prevents the historical per-model-seed random repartition and relation-task overlap. It deliberately changes membership from the historical experiment, so numeric equality is not assumed. |
| No test loader/evaluation in canonical mode | Prevents final-test access during training and checkpoint selection. Direct historical invocation retains its final test evaluation. |
| Immutable model revision | Removes upstream model-repository drift without changing architecture. Callers that omit it retain historical behavior. |
| Run-local frozen Hugging Face cache | Canonical mode downloads the same pinned revision beneath `output/<run-id>/inputs/huggingface`, freezes a file/tree manifest, and forces later seeds/inference to local-only loading. Historical callers that omit the cache controls retain their defaults. |
| Serializable shuffle sampler | Preserves canonical random-shuffle semantics while storing current order/cursor for exact next-batch resume with zero data-loader workers. The historical CSV path still uses PyTorch's default sampler. |
| Full atomic restart state | Adds optimizer, scheduler, RNG, sampler, adaptive-gate, and best-selection state beside the unchanged inference checkpoint. It affects recovery, not the loss objective. |
| Run-local logs/summary/manifests | Replaces `/tmp` and ad hoc checkpoint provenance only in canonical mode; all writes stay under `output/<run-id>/`. |
| Final development checkpoint promotion | Canonical mode saves a newly improved final-step development checkpoint; the historical path retains its previous behavior. This prevents the manifest from naming metrics not represented by `checkpoint.pt`. |

## Validation and remaining gate

Focused offline tests prove prepared-record order/span/relation adaptation,
split substitution rejection, sampler continuation, opt-in defaults, model-
revision forwarding, interrupted orchestration retry, no-test summary checks,
and checkpoint/data/code identity binding. They do not prove accelerator
numerics. The next evidence-producing action is a clean-clone seed-42
train/interruption/resume smoke on the external accelerator. B-07 review is
required before seeds 42–49, final-test inference, or live-verifier execution.
