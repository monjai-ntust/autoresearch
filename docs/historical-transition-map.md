# Historical executable transition map

This is the B-05U file- and claim-level deletion gate. It expands the family
inventory in `historical-transition-inventory.md` using the README command
surface and current import/reference search. A command is not eligible for
removal merely because it is historical: every row specifies its retained
evidence, canonical successor, and the proof still required. Until that proof
is complete, this map is a **retain/demote** record, not a deletion list.

## Scope and reading rules

- The source tree has 29 non-test Python command entry points; the README
  exposes 20 historical invocations plus the canonical command and test command.
- `phase_b.py` is the only paper-facing command now. Its replacement stages
  must write only below `output/<run-id>/`, bind data/split/config identities,
  and be tested from a standalone clone.
- Historical measurements and the two TSV ledgers are immutable secondary
  provenance. A canonical rerun is a new measurement, never an in-place repair.
- “Unsupported” means the paper claim must be qualified or removed through the
  parent discrepancy process; it never means its result is discarded.

| Historical path(s) and draft claim family | Existing contract / current consumer | Canonical successor | Required proof before removal | Current disposition |
| --- | --- | --- | --- | --- |
| `train_span.py`, `models/bert_kg_encoder.py` — CODE-ACCORD encoder capacity/full-recipe and zh-Hant span results | Dataset-selected training writes checkpoints outside `output`; used by augmentation, GAN/Gumbel, and inference code; historical split and type-agnostic scorer are incompatible with `CODE-STRICT-1`. | `phase_b.py model train` (`model.py`) plus a typed-record adapter. | Live single-seed training under the pinned accelerator profile plus eligible-fixture loss/output parity; frozen `CODE-SPLIT-1`; output/checkpoint identity; no final-test selection; clean-clone dry run. | Retain pending live/parity gate. `train --execution dry-run` now plans the frozen recipe and declares the checkpoint-manifest contract; live training is not implemented, so this script is not yet removable. |
| `inference_kg.py` — encoder candidate/KG inputs and confidence ablations | Reads historical checkpoints; writes arbitrary JSONL; feeds the free-form verifier and graph builder. | `phase_b.py model generate-candidates` (`model.py`). | Output parity between the `live` adapter and the historical script on a real checkpoint under the accelerator (verifiable only externally); candidate IDs already tied to strict data/split/model hashes. | Retain pending external parity validation. `dry-run`, `replay`, and `live` executions now exist; `live` runs the retained encoder and its imports/signatures/label space are statically verified, but its outputs cannot be validated offline, so this script is retained as provenance until external parity is confirmed. |
| `verify_triples_llm.py` — SciERC verifier row and CODE audit lineage | Free-form SciERC ontology/parser; arbitrary JSONL I/O; imported parser helpers are regression-tested. | `phase_b.py verifier dry-run|live|replay`. | Transport-only parity if reused; frozen prompt/schema/model digest/cache/telemetry tests already cover canonical semantics. | Demote to immutable provenance; remove only after parser fixtures move to canonical/legacy fixtures. |
| `build_kg.py`, `eval_graph_rag.py`, `diagnose_evidence_paths.py` — diagnostic Graph RAG and evidence-coverage claims | Chain consumes historical inference/verdict files; questions/gold derive from the same source, so it is diagnostic rather than leakage-free accuracy. | `phase_b.py secondary graph-rag`. | Versioned source-control fixture, diagnostic label in output, and no gold-derived value presented as canonical task accuracy. | Retain pending a non-comparable diagnostic adapter. |
| `train_multi.py`; `data/download_scierc.py`, `data/download_conll04.py`, `data/download_ade.py` — cross-dataset comparator | Token-BIO runner/downloaders have separate schemas and mutable acquisition. | `phase_b.py secondary benchmark`. | Per-dataset immutable input manifest, type/matcher declaration, matrix row, and fixture adapter. | Retain pending benchmark adapters. |
| `generate_accord_llm_aug.py`, `generate_entigraph.py`, `generate_cycle_data.py`, `generate_paraphrase_dataset.py`, `generate_synth_dataset.py` — augmentation/coverage-gap variants | Local models or incomplete inputs; generated JSONL normally writes under `results/`; `generate_accord_llm_aug.py` imports `train_span`. | `phase_b.py secondary generate-data`. | Input/prompt/model hashes, output containment, matched gold-only control, and claim-family reconciliation. | Retain pending one matrix-bound generator contract per supported family. |
| `generate_entity_masks.py`, `train_pretrain_cooperative.py`, `data/download_arxiv_real.py` — cooperative masking/pretraining negative result | Requires mutable arXiv input and historical checkpoint; writes arbitrary checkpoints/results. | `phase_b.py secondary cooperative-pretrain` or unsupported claim. | Reacquirable licensed input manifest, adapter parity, or discrepancy entry removing/qualifying the claim. | Retain as secondary provenance. |
| `train_stage2b.py`, `train_stage2c.py`, `train_stage2d.py`, `train_stage2e.py`, `train_gan.py`, `train_gumbel.py` and transitive decoder/critic modules — closed-loop negative-result families | Multiple unavailable checkpoints/models and direct writes; GAN/Gumbel import `train_span`; ledgers do not provide a complete replayable matrix. | `phase_b.py secondary closed-loop` with blocked/unsupported matrix rows. | Each claimed intervention mapped to an evidence bundle; valid fixture parity or explicit noncomparability; no orphan README command. | Retain pending claim-by-claim disposition. |
| `zh_translate_project.py`, `dapt_zh.py` — Traditional-Chinese transfer/ablation claims | Uses incomplete CODE inputs, mutable law sources/model caches, and distinct split/normalization rules. | `phase_b.py secondary zh-data`. | Licensed acquisition/normalization manifests, fixed holdout isolation, typed record adapter, and external-machine validation. | Retain pending standalone-data contract; current paper claim constrained by DISC-013–DISC-016. |
| `bench_gpu.py`, `eval_checkpoint.py` — environment/checkpoint diagnostics | Smoke-only utility and absent historical checkpoint; neither emits a paper metric. | `phase_b.py doctor` plus optional `secondary inspect-checkpoint`. | A bounded no-download diagnostic stage or removal after README command retirement. | Retain only while documented diagnostic remains useful. |
| `results.tsv`, `results_stage2.tsv`, `run_a19_cosine_probe.sh` — Section 5 experiment/command evidence | Ledgers are generated evidence with known structural gaps; shell record is DGX-specific. | `phase_b.py reconcile`, later canonical report/archive stages. | B-10 verified byte/hash archive move; historical shell record either normalized into an auditable command manifest or removed after source-inventory review. | Preserve bytes and Git lineage; not executable canonical inputs. |

## Resolved secondary dispositions (user decision, 2026-07-18)

The user resolved every secondary-family disposition toward
retain-as-provenance + paper-owner correction (see
[`secondary-data-replacement-map.md`](secondary-data-replacement-map.md)).
Accordingly, the Graph RAG (`build_kg.py`, `eval_graph_rag.py`,
`diagnose_evidence_paths.py`), cross-dataset (`train_multi.py`, downloaders),
augmentation (`generate_*`), cooperative-pretraining, closed-loop
(`train_stage2*`, `train_gan.py`, `train_gumbel.py`), and zh-Hant
(`zh_translate_project.py`, `dapt_zh.py`) rows are **retained as provenance
evidence**: their paper claims are corrected or marked unsupported through the
discrepancy ledger, no canonical `secondary` stage is built, and none is
regenerated or deleted this cycle. `train_span.py`/`inference_kg.py` likewise stay
provenance behind the ready-but-gated `model` chain. Deletion therefore remains
out of scope until a future finalization pass satisfies the invariant below for a
specific file.

## Command-surface invariant

Before any row changes from retain/demote to delete, `README.md`, Python
entry-point enumeration, imports, tests, configs, and docs must show that the
row's supported behavior is reachable only through `python -B phase_b.py`.
The deletion commit must update this map, `source-change-inventory.md`, the
Phase B progress log, and the parent discrepancy record when a paper claim is
unsupported or non-comparable.
