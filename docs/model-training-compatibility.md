# Model-training compatibility contract

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
- The historical reference/default base model is `microsoft/deberta-large` revision
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
  `resources/data/code-accord-entities-train.csv`, blob
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
| Encoder/backbone, text adapter, parameter names and shapes | `utils.encoder.network.BertKGExtractor`; inactive experimental branches were removed without changing the canonical state-dict keyspace. |
| Span-NER and relation heads | Existing `stages/encoder.py` construction, including the 3H context-between-spans relation head. |
| Entity/relation label order | Existing `utils.encoder.data` four entity types and `NO_REL` plus nine relation types. |
| Span enumeration and pair construction | Existing `forward_span_ner`, loss, and evaluation functions. |
| Losses and weights | Existing focal/label-smoothing/RE loss code with A20+A21+A12 values supplied from `resources/configs/pipeline.json`. |
| Optimizer and schedule | Existing AdamW, weight decay `0.01`, linear warm-up/decay, 3,500 steps, 250 warm-up steps. |
| Batch size and negative sampling | Existing batch size 16 and NER/RE sampling logic; canonical input uses the same dataset/collate classes. |
| Seed initialization | Existing Python and Torch seed path, plus CUDA `manual_seed_all`; actual runtime settings are recorded. |
| Development selection | Existing strict `>` update preserves earliest-step ties on Triple F1. |
| Inference checkpoint representation | Existing `{"encoder": state_dict, "step": ..., "metrics": ...}` payload remains loadable by candidate inference. |

## Necessary opt-in differences

| Difference | Reason and expected effect |
| --- | --- |
| Prepared `train.jsonl`/`development.jsonl` adapter | Prevents the historical per-model-seed random repartition and relation-task overlap. It deliberately changes membership from the historical experiment, so numeric equality is not assumed. |
| No test loader/evaluation in canonical mode | Prevents final-test access during training and checkpoint selection. The repository exposes no public noncanonical trainer route. |
| Immutable model revision | Removes upstream model-repository drift without changing architecture. Any valid new run may select another immutable 40-hex revision; that run-local identity must remain consistent downstream. |
| Run-local frozen Hugging Face cache | The retained trainer downloads the pinned revision beneath `output/<run-id>/inputs/huggingface`, freezes a file/tree manifest, and forces later seeds/inference to local-only loading. |
| Serializable shuffle sampler | Preserves canonical random-shuffle semantics while storing current order/cursor for exact next-batch resume with zero data-loader workers. |
| Full atomic restart state | Adds optimizer, scheduler, RNG, sampler, adaptive-gate, and best-selection state beside the unchanged inference checkpoint. It affects recovery, not the loss objective. |
| Run-local logs/summary/manifests | Replaces `/tmp` and ad hoc checkpoint provenance only in canonical mode; all writes stay under `output/<run-id>/`. |
| Final development checkpoint promotion | The retained trainer saves a newly improved final-step development checkpoint. This prevents the manifest from naming metrics not represented by `checkpoint.pt`. |

## Validation and remaining gate

Focused offline tests prove prepared-record order/span/relation adaptation,
split substitution rejection, sampler continuation, the canonical-only trainer
surface, model-revision forwarding, interrupted orchestration retry, no-test
summary checks, and checkpoint/data/config identity binding. They do not prove accelerator
numerics. The publication run's encoder training/inference and verifier stages
are completed immutable prerequisites and must not be rerun. The remaining
external gate is E-07: restore the complete authenticated run tree, execute the
Table-2 dry run and live evaluator only, compare with legacy Table-2 evidence,
and archive the verified output.
