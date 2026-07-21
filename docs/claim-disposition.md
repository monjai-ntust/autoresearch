# Draft claim disposition against the canonical workflow

This document dispositions every claim enumerated in the external Phase A
inventory (`A-CLAIM-001`…`A-CLAIM-029`) against the canonical `phase_b.sh`
workflow and the user's 2026-07-18 secondary-data decisions (recorded in
[`secondary-data-replacement-map.md`](secondary-data-replacement-map.md) and the
external discrepancy ledger). It is a source-artifact reading aid, not a runtime
dependency on that external research record. Paper corrections remain the paper
owner's `DISC-*` responsibility.

## Categories

- **Extends** — the canonical workflow produces a *new* statistic that did not
  previously exist, without altering the legacy record.
- **Superseded** — the canonical workflow *replaces* the claim's measurement with
  a leakage-safe CODE-STRICT-1 equivalent; the historical value is preserved
  unmodified as provenance.
- **Retained (provenance)** — the historical result/script is kept as provenance
  evidence; no canonical replacement is produced (negative result, cross-dataset
  metric-noncomparability, external dependency, or frozen-recipe capture).
- **Removed** — the claim is withdrawn from the deliverable (unsupported/invalid,
  or an external/manual study outside this software artifact); any historical
  numbers survive only in Git history, not as a carried-forward deliverable
  claim.

All output paths below are **run-relative under `output/<run-id>/`** (the ignored
runtime root); no committed run is required for the path to be defined. Live
model-backed stages remain externally gated on the accelerator + pinned Ollama.

## Disposition table

| ID | Claim (abbreviated) | Category | New-data output path (Extends/Superseded) | Rationale |
| --- | --- | --- | --- | --- |
| A-CLAIM-001 | CODE-ACCORD DeBERTa-large extractor recipe; strict Triple F1 0.410±0.036 | **Superseded** | `output/<run-id>/checkpoints/seed-<n>/`, `output/<run-id>/predictions/test/candidates.jsonl`, `output/<run-id>/metrics/metrics.json` | The `model train`→`generate-candidates`→`score` chain reproduces extraction under leakage-safe `CODE-SPLIT-1`, superseding the historical leaky split (DISC-016). Historical 0.410 retained as provenance with DISC-001/002/003 corrections. |
| A-CLAIM-002 | Encoder capacity is the dominant lever (BERT/DeBERTa-base/large paired t-test) | **Retained (provenance)** | — | The frozen config pins one backbone (`deberta-large`); the multi-backbone comparison is not a canonical stage. The DeBERTa-large operating point is regenerated via A-CLAIM-001. |
| A-CLAIM-003 | 38 generative-augmentation variants uniformly neutral-to-negative | **Retained (provenance)** | — | D-AUG: paper-owner correction (DISC-004); generators retained as provenance; no regeneration. |
| A-CLAIM-004 | CODE-ACCORD composition, ~862/~700 sentences, 4/9 schema, imbalance | **Superseded** | `output/<run-id>/data-prepared/` (`raw-relation-provenance.jsonl`, `typed-strict-eligibility.jsonl`, `gold-alignment-audit.json`, distributions) | `prepare` recomputes exact counts/distributions from the immutable v1.0.0 archive, replacing the draft's approximate figures. |
| A-CLAIM-005 | Hand-tuned per-class multipliers / adaptive schedule; not Cui | **Retained (provenance)** | — | The production schedule is captured in `configs/phase_b_path_a.json` (DISC-003 correction); the schedule/curriculum/Cui ablations remain provenance and Cui is not implemented (D-CUI). |
| A-CLAIM-006 | Joint typed span+relation confidence; softmax-product threshold; two-layer reference | **Superseded** | `output/<run-id>/predictions/test/candidates.jsonl` (confidence), `output/<run-id>/predictions/dev/threshold-selection.json`, `output/<run-id>/metrics/metrics.json` | `generate-candidates` emits the softmax-product `triple_confidence`; `select-threshold` freezes the VER-CONFIDENCE threshold; `score` reports the two-layer VER-RAW/VER-CONFIDENCE vs verifier conditions. |
| A-CLAIM-007 | Qwen3-32B KEEP/CORRECT/DISCARD verifier; CODE 1/32/7 of 40 | **Superseded** | `output/<run-id>/verifier/{simple,corrective}/verdicts.jsonl` | The frozen `CODE-VERIFIER-1` (dry/live/replay) supersedes the historical SciERC-schema verifier (DISC-005). The 1/32/7 run is retained as provenance, invalid for CODE. |
| A-CLAIM-008 | SciERC verifier table (0.380/0.372/0.375/0.348) | **Retained (provenance)** | — | SciERC is metric-noncomparable to CODE-STRICT-1 (D-BENCH; DISC-006); retained as provenance. |
| A-CLAIM-009 | Graph assembly / entity resolution; 133/53, 58/39, 48/31 | **Retained (provenance)** | — | `provenance/build_kg.py` retained as provenance; it feeds the unsupported Graph RAG claim and is not a canonical stage. |
| A-CLAIM-010 | Graph RAG diagnostic harness (cloze questions, overlap scoring) | **Retained (provenance)** | — | `provenance/eval_graph_rag.py` retained as provenance (D-RAG); no canonical Graph RAG stage is built. |
| A-CLAIM-011 | Graph RAG hit rates at n=10 (0/20/0/30/20/50–60%) | **Removed** | — | D-RAG: A-CLAIM-022 marked unsupported; the leakage-prone hit rates (DISC-007/008) are withdrawn from the deliverable and survive only as Git provenance. |
| A-CLAIM-012 | Cross-dataset SciERC/CoNLL04/ADE + BIO auxiliary gains | **Retained (provenance)** | — | Metric-noncomparable to CODE-STRICT-1 (D-BENCH); stale values are a paper-owner correction (DISC-009). |
| A-CLAIM-013 | DGX compute footprint (128 GB, ~25%, ~20 GB, latencies) | **Retained (provenance)** | — | D-FOOTPRINT: mostly estimates (DISC-011). The one auditable verifier latency is instrumented under A-CLAIM-024. |
| A-CLAIM-014 | Full on-premise locality / privacy property | **Retained (provenance)** | — | Architectural property, not a numeric result: the standalone `src` checkout and local Ollama endpoint are its static evidence. |
| A-CLAIM-015 | Deployed Streamlit review UI | **Removed** | — | DISC-010: the UI is absent from `src` and not shipped; it remains only in read-only research history. |
| A-CLAIM-016 | Traditional-Chinese silver-data engine; 84.8% keep+correct | **Retained (provenance)** | — | D-ZH: the zh engine/normalizer are absent from `src` (DISC-013); retained scripts (`provenance/zh_translate_project.py`, `provenance/dapt_zh.py`) are provenance. |
| A-CLAIM-017 | zh-Hant results/backbones (XLM-R/CKIP/silver/schedule) | **Retained (provenance)** | — | D-ZH; stale final row is a paper-owner correction (DISC-012). |
| A-CLAIM-018 | zh-Hant ablations (NLLB projection, DAPT, typed markers, scale) | **Retained (provenance)** | — | D-ZH: negative/ablation provenance; not reproduced in the canonical workflow. |
| A-CLAIM-019 | CODE verifier verified-vs-unverified gold P/R/F1 (E-1) | **Extends** | `output/<run-id>/metrics/{metrics.json,publication-table.tsv,publication-summary.md}`, `output/<run-id>/outcomes/{candidate,sentence}-outcomes.jsonl` | The primary missing statistic: `score` computes accuracy/precision/recall/F1, confusion counts, and paired outcomes for VER-RAW/CONFIDENCE/SIMPLE/CORRECTIVE, then renders publication views without altering the metrics. Requires external live-verifier verdicts. |
| A-CLAIM-020 | Local Qwen zero-/few-shot extraction baseline (E-2) | **Removed** | — | No artifact exists and it was not authorized; withdrawn from the deliverable (external if later authorized). |
| A-CLAIM-021 | External published/span-RE baseline or non-comparability (E-3) | **Retained (provenance)** | — | D-BENCH resolves to recorded metric-noncomparability; the SpERT reproduction is retained provenance. |
| A-CLAIM-022 | Graph RAG rerun at n≥100 with intervals (E-4) | **Removed** | — | D-RAG: marked unsupported; the redesigned evaluation is not implemented. |
| A-CLAIM-023 | Full-recipe paired p-value + clean Cui/focal comparisons (E-5) | **Extends** | `output/<run-id>/metrics/metrics.json` (paired hierarchical bootstrap, exact signed-rank, paired t-test, Holm — when all 8 seeds are present) | `score`/`phase_b_statistics` add the paired uncertainty/tests. The clean Cui comparison is not implemented (D-CUI). |
| A-CLAIM-024 | Per-document calls/tokens, energy/monetary/cloud cost (E-6) | **Extends** | `output/<run-id>/verifier/{simple,corrective}/run-log.jsonl`, `output/<run-id>/verifier/{simple,corrective}/environment-manifest.json` | The B-06 verifier instruments calls/tokens/latency/retries. Energy/monetary/cloud-price remain paper-owner estimates (DISC-011). |
| A-CLAIM-025 | Exact corpus class/NO_REL denominators + environment appendix (E-7) | **Extends** | `output/<run-id>/data-prepared/` (exact counts/distributions), `output/<run-id>/manifests/00-checkout-manifest.json` (environment) | `prepare` emits exact denominators; `doctor` records the pinned environment/identity manifest. |
| A-CLAIM-026 | NER-F1 and RE-given-gold-entities decomposition (E-8) | **Removed** | — | No canonical decomposition stage is built; not produced this cycle (external/deferred). |
| A-CLAIM-027 | Eighth zh seed + dual-annotated 60–100-sentence gold test (E-9) | **Removed** | — | External data-creation/annotation; outside this software artifact. |
| A-CLAIM-028 | Expert study with 2–3 domain engineers (E-10) | **Removed** | — | Human-subject study; outside the artifact (may require institutional review). |
| A-CLAIM-029 | End-to-end RASE compliance case study or title rescope (E-11) | **Removed** | — | Paper-scope decision / manual case study; owned by the paper authors. |

## Summary

- **Superseded (4):** A-CLAIM-001, 004, 006, 007 — the canonical CODE-STRICT-1
  pipeline replaces the historical extraction/verifier/corpus basis; new data
  lands under `data-prepared/`, `predictions/`, `checkpoints/`, `verifier/`, and
  `metrics/`.
- **Extends (4):** A-CLAIM-019, 023, 024, 025 — genuinely new statistics
  (verifier P/R/F1, paired tests, operational metrics, exact denominators) under
  `metrics/`, `outcomes/`, `verifier/`, `data-prepared/`, and `manifests/`.
- **Retained as provenance (13):** A-CLAIM-002, 003, 005, 008, 009, 010, 012,
  013, 014, 016, 017, 018, 021 — negative results, cross-dataset
  noncomparability, frozen-recipe capture, locality, and zh-Hant provenance.
- **Removed (8):** A-CLAIM-011, 015, 020, 022, 026, 027, 028, 029 — unsupported
  Graph RAG results, the absent UI, and external/manual studies withdrawn from
  the software deliverable.

(Counts: 4 + 4 + 13 + 8 = 29; A-CLAIM-021 is listed once under Retained.)
