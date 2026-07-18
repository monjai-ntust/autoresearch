# Secondary-data replacement map (B-05U step 4)

This is the B-05U data-item-level companion to
[`historical-transition-map.md`](historical-transition-map.md). That map covers
executable *command families*; this map covers every **secondary data item used
by the paper draft** (draft Section 5 result families) and assigns each a
disposition. It exists to make explicit which replacement sub-workflows can be
built now, which are externally blocked, and which require a paper-owner or
scientific decision before any canonical stage is implemented.

It does not authorize edits to the read-only draft; paper corrections are
tracked as `DISC-*` entries in the parent discrepancy ledger and remain the
paper owner's responsibility.

## Disposition legend (from Phase B plan B-05U step 4)

- **(a) Canonical stage** — regenerate the data through a reproducible
  `phase_b.py` stage bound to frozen identities and confined to `output/`.
- **(b) Retained raw input** — keep an immutable input/ledger with a canonical
  normalizer; do not regenerate.
- **(c) Unsupported / corrected** — the paper claim must be qualified or removed
  through the discrepancy process; the historical result is retained as
  provenance, never presented as a canonical measurement.
- **(d) External / manual** — the study is out of scope for this software
  artifact (data, people, governance, or hardware beyond ordinary execution).

A single family can require more than one disposition (e.g. retain the historical
ledger as (b)/(c) provenance while its canonical successor is a gated (a) stage).

## Map

| Family (register claim / A-CLAIM / DISC) | Secondary data item | Disposition | Canonical successor stage | Blocking gate | Decision needed |
| --- | --- | --- | --- | --- | --- |
| Encoder capacity + full imbalance recipe (`S5-1-ENCODER-RECIPE`; A-CLAIM-023; DISC-001/002/003/016/019) | 8-seed strict Triple F1 (dev 0.4097 / test 0.0766; best 0.475), DeBERTa-large `0.3797` row, full-recipe ablation | (a) regenerate under `CODE-SPLIT-1`; (c) historical values corrected per DISC-001/002/003/016; (b) `results.tsv` retained as provenance | `model train` (dry/live compatibility path implemented; external smoke gated) → `model generate-candidates` (done) → `score` (done) | External accelerator + real checkpoints; only historical entity-train bytes are recoverable, and DISC-016 leakage forces a new canonical rerun rather than reuse | **D-TRAIN**: review seed-42 smoke/restart evidence before the eight-seed train/inference run, or keep historical as provenance with DISC corrections. **D-CUI**: implement the A-CLAIM-023 Cui loss ablation only if approved. |
| Generative augmentation "38 variants" (`S5-1-AUGMENTATION`; DISC-004) | 38 augmentation runs reported uniformly neutral-to-negative | (c) claim miscounted/misclassified (DISC-004); generators retained as (b) provenance | `secondary generate-data` | Local generator models/inputs; and DISC-004 shows "38" is not defensible without a recovered row-level manifest | **D-AUG**: qualify per DISC-004 and retain generators as provenance (regeneration adds no canonical value for a negative result), or authorize regeneration. |
| Cross-dataset / model comparison (`S5-1-CROSS-DATASET`; A-CLAIM-020/021; DISC-009) | SciERC/CoNLL04/ADE BIO-auxiliary gains and comparator rows (stale per DISC-009) | (c) metric-noncomparable to `CODE-STRICT-1` (different ontologies); DISC-009 corrections | `secondary benchmark` | External dataset downloads; matcher/ontology semantics differ from CODE | **D-BENCH**: record metric-noncomparability (recommended), or authorize an external same-protocol benchmark. |
| Graph RAG diagnostic (`S5-2-GRAPH-RAG`; A-CLAIM-022; DISC-007/008) | n=10 QA hit rates, 50–60% gold ceiling, "BM25" text baseline | (c) leakage-prone + not-BM25 (DISC-007/008); historical numbers are diagnostics only | `secondary graph-rag` | A valid result needs the DISC-008 redesign: query/evidence separation, source control, frozen questions, n≥100, Wilson/Clopper–Pearson intervals | **D-RAG**: (i) approve the leakage-safe redesign and a frozen question/gold fixture so `secondary graph-rag` can be built; (ii) keep historical numbers labeled leakage-prone diagnostics only (no new stage); or (iii) mark A-CLAIM-022 unsupported. |
| LLM verifier effect (`S5-3-VERIFIER`; A-CLAIM-019; DISC-005/006) | Historical 1/32/7 CODE run; SciERC verifier table; simple-mode row | (c) historical CODE run invalid (SciERC schema, DISC-005) → provenance-only; canonical P/R/F1 is the primary B-08 workstream | `verifier` (done) → `score` (done) | Live Qwen + gold labels + B-07 go/no-go (already the primary gated workstream) | None new — governed by the existing B-07 checkpoint. |
| Computational footprint + UI + expert studies (`S5-4-INDUSTRY-RESOURCE-UI`; A-CLAIM-024/027–029; DISC-010/011) | DGX footprint table; Streamlit UI; dual-annotation/domain-engineer/RASE studies | (c) footprint mostly estimates (DISC-011; only 40/77 s verifier latency auditable); UI (d) out-of-artifact (DISC-010, done); studies (d) external | Verifier telemetry (done) covers latency; rest external | External instrumentation; human-subject studies | **D-FOOTPRINT**: mark footprint values as estimates per DISC-011 (recommended), or authorize an external instrumentation run. Studies remain external. |
| Traditional-Chinese transfer (`S5-5-ZH-HANT`; A-CLAIM-025/026; DISC-012/013/014/015/016) | zh Table 4 (stale, DISC-012), silver engine (not in `src`, DISC-013), 84.8% rate (DISC-014), portability claim (DISC-015) | (c) paper-owner corrections; (d) zh engine + normalizer absent from `src` | `secondary zh-data` | Mutable law sources, model caches, and the missing `preprocessing.zh_normalize` normalizer (DISC-013) | **D-ZH**: retain zh as constrained provenance with DISC-012–016 corrections (recommended; engine/normalizer are absent), or authorize external regeneration. |

## What this map establishes

Every secondary family's canonical regeneration path is **externally gated**
(accelerator, live Ollama, external corpora, or code absent from `src`) or
resolves to a **paper-owner correction**. No `secondary` stage can be validly
implemented offline without first resolving its disposition decision, because
building a canonical stage presupposes the underlying claim is retained rather
than corrected/removed. Implementing gated stages as empty scaffolds would add
no evidence and would risk presenting blocked measurements as results, which the
plan forbids.

The `model` chain (`train` → `generate-candidates` → `score`) is the exception:
its contracts are implemented (train dry-run and candidate replay/dry-run) and
only live execution is gated, so it needs an execution authorization (**D-TRAIN**),
not a disposition decision.

## Resolved dispositions (user decision, 2026-07-18)

The user resolved every open decision toward **retain-as-provenance + paper-owner
correction**; no secondary claim is regenerated, no `secondary` stage is built,
and no historical executable is deleted.

- **D-RAG → mark A-CLAIM-022 unsupported.** The Graph RAG claim is recorded as
  unsupported/removed for the paper owner (reinforcing DISC-007/008). Historical
  numbers are retained only as leakage-prone diagnostics; `build_kg.py`,
  `eval_graph_rag.py`, and `diagnose_evidence_paths.py` remain provenance and are
  **not** rebuilt as a canonical stage.
- **D-BENCH → record noncomparability.** SciERC/CoNLL04/ADE remain
  metric-noncomparable to `CODE-STRICT-1`; DISC-009 correction stands.
  `train_multi.py` and the downloaders stay provenance.
- **D-AUG → qualify per DISC-004, retain provenance.** The augmentation
  generators stay provenance; the "38 variants" framing is a paper-owner
  correction.
- **D-FOOTPRINT → mark as estimates per DISC-011.** Only the 40/77 s verifier
  latency is auditable; the rest are estimates for the paper owner. Verifier
  telemetry already covers latency canonically.
- **D-ZH → retain constrained provenance.** DISC-012–016 corrections stand; the
  absent zh engine/normalizer keeps the family provenance-only.
- **D-TRAIN/D-CUI → provenance-only for now.** Historical encoder results stay
  provenance with DISC-001/002/003/016 corrections. The `model` chain
  (`train` dry-run, `generate-candidates` dry-run/replay) stays ready for a later
  externally authorized live run; the Cui ablation is not implemented this cycle.

Consequently the historical executables mapped in
`historical-transition-map.md` are **retained as provenance evidence**, their
paper claims are corrected or marked unsupported through the discrepancy ledger,
and none is regenerated or deleted. The remaining Phase B execution (canonical
verifier Precision/Recall/F1, the real B-07 pilot, and any live `model` run) is
externally gated on the accelerator, pinned Ollama/Qwen runtime, and gold labels.
