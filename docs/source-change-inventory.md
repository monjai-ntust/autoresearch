# Source change inventory relative to `9feafa4`

This document explains the current source tree by comparison with the fixed
baseline commit `9feafa4029e65ab48ecfb2f452f4b0fabbff0826`. It classifies paths by
their present relationship to that baseline and records the purpose of every
difference. It is deliberately state-based and excludes development labels and
provenance timelines.

The inventory is part of the source artifact. Any edit that changes the tracked
source surface, a path's role, or the reason for retaining it must update this
document in the same source commit.

## How to verify and maintain this inventory

Run these commands from the `src` repository:

```bash
git diff --name-status 9feafa4029e65ab48ecfb2f452f4b0fabbff0826..HEAD
git ls-tree -r --name-only 9feafa4029e65ab48ecfb2f452f4b0fabbff0826
git ls-tree -r --name-only HEAD
git diff --check 9feafa4029e65ab48ecfb2f452f4b0fabbff0826..HEAD
```

The maintained classification has these invariants:

- every path reported by `git diff --name-status` appears under exactly one of
  **Modified baseline paths**, **Baseline-only paths**, or **Current-only
  paths**;
- every remaining baseline path appears under **Reused without source edits**;
- reasons describe the current implementation or evidence role, not a point in
  the development timeline; and
- a moved or renamed path is described explicitly instead of being presented as
  an unrelated deletion and creation.

With this document included, the comparison contains 101 differing paths: 59
current-only paths, 33 baseline-only paths, and 9 modified baseline paths. The
other 48 baseline paths are reused byte-for-byte. The baseline has 90 tracked
paths and the current tree has 116.

## Current architecture and rewrite boundary

The publication-facing command is `python -B phase_b.py`. The canonical root
modules are a protocol and orchestration layer, not a replacement encoder. They
centralize contracts that the historical top-level scripts do not share:
immutable input and model identities, a frozen experiment matrix, strict typed
matching, leakage-safe splits, deterministic records, replay, output
containment, and machine-readable manifests.

Keeping canonical semantics distinct from the historical scripts while placing
the active modules at the root is the current boundary for three reasons:

1. The root training, inference, verification, graph, and ablation paths are
   evidence-bearing implementations. Reworking them in place before behavioral
   parity is demonstrated could invalidate provenance or make old result rows
   impossible to interpret.
2. Evaluation protocol is cross-cutting. Putting acquisition, record schemas,
   verifier telemetry, scoring, and statistical tests into one historical
   training script would couple unrelated responsibilities and make isolated
   validation harder.
3. The remaining integration can use thin adapters around retained model code.
   That avoids a duplicate encoder while allowing the canonical runner to
   enforce its contracts at every boundary.

This is not yet a complete end-to-end replacement. Canonical training and
candidate extraction adapters still require parity evidence, and official data
preparation correctly stops on nine unresolved typed alignment cases. The
root-module layout is an intermediate reviewable surface, not a claim that the
historical model paths have been superseded.

The similarly named paths below are not one-to-one rewrites:

| Baseline/current boundary | Current interpretation and reason |
| --- | --- |
| `prepare.py` vs `preparation.py` | The baseline script prepared FineWeb bytes-per-byte training data. The canonical module validates and materializes official CODE-ACCORD records. The old script is removed as unrelated upstream code; the new module does not inherit its behavior. |
| `data/code_accord.py` vs canonical preparation | The unchanged loader preserves the historical fuzzy, type-agnostic experiment path. Canonical preparation uses strict typed alignment and hard-stop auditing because fuzzy recovery is unsuitable for publication scoring. |
| `verify_triples_llm.py` vs `verifier.py` | The unchanged root verifier preserves the SciERC free-form response path. The canonical verifier uses frozen CODE prompts, structured response schemas, model identity, caching, replay, retry policy, and telemetry. |
| `eval/triple_f1.py` vs `scoring.py` | The historical helper preserves type-agnostic exact triple scoring used by retained experiments. Canonical scoring applies directed, entity-typed `CODE-STRICT-1` matching across the frozen four-condition matrix. |
| Root training/model modules vs canonical root modules | Model behavior remains in the retained modules. The canonical modules own surrounding contracts; thin parity-tested adapters are the intended integration mechanism. |
| Historical result/checkpoint locations vs `output/<run-id>/` | Historical ledgers remain evidence pending verified archival. Every canonical runtime artifact is confined to the ignored run directory so a standalone clone has one reproducible output boundary. |

## Modified baseline paths

| Path | Current edit and reason |
| --- | --- |
| `.gitignore` | Ignores `/output/`, checkpoints, and downloaded/generated dataset directories while explicitly allowing the tracked CODE-ACCORD fixture. The obsolete ignore rule for tracked `results.tsv` is absent. This enforces the single temporary-output root without hiding retained evidence. |
| `README.md` | Defines the repository as a standalone publication artifact, separates canonical commands from historical provenance paths, documents datasets, execution gates, outputs, limitations, and links this inventory. The baseline upstream pretraining description did not describe the retained research implementation. |
| `bench_gpu.py` | Places benchmark execution behind `main()` and an import guard. Benchmark behavior remains available, while importing the module no longer downloads a model or allocates accelerator memory. |
| `eval_checkpoint.py` | Places checkpoint evaluation behind `main()` and an import guard. The evaluation behavior is retained, while tests and tooling can import the module without executing a full evaluation. |
| `eval_graph_rag.py` | Describes retrieval as unique whitespace-token overlap instead of BM25-style retrieval. Runtime behavior is unchanged; the text now states the actual algorithm and avoids overstating the evaluation method. |
| `pyproject.toml` | Names the publication artifact, removes unused upstream packages, and declares dependencies used by retained code (`peft`, `pytorch-crf`, and `safetensors`) while keeping the required Torch/Transformers/data stack. This makes installation reflect reachable code. |
| `train_gumbel.py` | Restricts process-environment side effects to command execution under the import guard. The training command retains its environment settings, while library import remains side-effect free. |
| `train_span.py` | Escapes literal percent signs in argument help text. Scientific behavior is unchanged; `--help` no longer fails through `argparse` percent interpolation. |
| `uv.lock` | Resolves the dependency graph declared by the current `pyproject.toml`. It supports repeatable installation of the current artifact and is not represented as an exact manifest of historical experiment machines. |

## Baseline-only paths

These paths are absent from the current source tree. Git retains their original
content and lineage.

| Path | Reason it is not part of the current tree |
| --- | --- |
| `Task_done.md` | Duplicated project history whose authoritative, read-only copy lives outside the standalone source artifact; implementation provenance is represented by retained code, result ledgers, Git, and the parent research state. |
| `analysis.ipynb` | Belonged to the disconnected FineWeb/bytes-per-byte upstream workflow and had no reachable role in the paper's entity/relation pipeline. |
| `data/accord_entigraph_5pairs.jsonl` | Was a generated data artifact with unresolved redistribution and input status. The generator and result evidence preserve the method without shipping an ambiguous runtime prerequisite. |
| `generate_pseudo_labels.py` | Implemented an early pseudo-label exploration not used by the retained claim-bearing configuration or the canonical evaluation design. |
| `generate_pseudo_labels_cast.py` | Implemented an obsolete cross-dataset pseudo-label variant with no current paper statistic or canonical input role. |
| `generate_pseudo_labels_multi.py` | Implemented an obsolete multi-source pseudo-label variant with no retained result row requiring execution. |
| `generate_pseudo_labels_span_cast.py` | Implemented an obsolete span-casting pseudo-label variant and is not the retained cross-dataset BIO comparator. |
| `generate_relation_replay_data.py` | Produced relation replay data for an experiment without a confirmed result row or current claim; retained synthetic controls cover the auditable comparison surface. |
| `inspect_stage2b_synth.py` | Was a qualitative inspection helper with no machine-readable metric or publication command; the associated negative-result evidence remains in source history and retained ledgers. |
| `inspect_stage2d_lora.py` | Was a qualitative LoRA inspection helper with no claim-bearing metric or required runtime role. |
| `models/decoder.py` | Formed part of the disconnected randomized-toy adversarial cluster rather than the real-data joint extraction implementation. |
| `models/encoder.py` | Formed part of that randomized-toy adversarial cluster; the evidence-bearing span encoder is retained elsewhere. |
| `models/transformer.py` | Supported the removed toy cluster and had no reachable role in the publication pipeline. |
| `prepare.py` | Prepared FineWeb data for the upstream bytes-per-byte workflow. It is unrelated to CODE-ACCORD preparation despite the generic filename. |
| `program.md` | Described a nonportable autonomous controller with unsafe assumptions and no standalone publication-runtime role. |
| `progress.png` | Illustrated the removed upstream workflow and was referenced only by its obsolete README. |
| `reports/daily/daytime_2026-05-04.md` | Duplicated an authoritative report outside the source artifact; the source tree does not carry a competing research-log copy. |
| `reports/daily/morning_2026-05-04.md` | Contained contextual report prose rather than executable input; applicable configuration and result evidence is represented by retained code, ledgers, and Git. |
| `reports/daily/morning_2026-05-05.md` | Contained contextual report prose rather than an executable dependency; applicable conclusions remain traceable through the retained implementation and external research record. |
| `sources/papers_20260501_kg_re_coverage.md` | Was literature commentary, not runtime source or an independently required artifact input. |
| `sources/papers_20260502_graphrag_kg_retrieval_low_resource_ie.md` | Was a byte-identical duplicate of the authoritative research-log copy and had no standalone execution role. |
| `sources/papers_20260504_cross_doc_re_provenance_graphrag.md` | Was literature commentary; the Graph RAG implementation and its result provenance remain in executable paths and Git. |
| `sources/papers_20260504_llm_synth_quality_diversity_low_resource_re.md` | Was literature commentary; retained augmentation generators and ledgers provide the implementation/evidence surface. |
| `sources/research_20260422_span_ner_variance.md` | Was a non-runtime literature note rather than a source dependency. |
| `train.py` | Belonged to the removed FineWeb/bytes-per-byte upstream training workflow and did not train the paper's extraction model. |
| `train_adversarial.py` | Drove the disconnected randomized-toy Stage-1 cluster and did not support auditable real-data claims. |
| `train_stage2.py` | Was superseded within the historical implementation by the retained multi/span runners and had no distinct current result row. |
| `train_stage2_curriculum.py` | Represented a historical pseudo-label curriculum exploration without a current claim-bearing data point. |
| `train_stage2_multiround.py` | Represented a historical multi-round pseudo-label exploration without a current claim-bearing data point. |
| `wiki/raw/KG_RE_Coverage_Augmentation_20260501.md` | Was research commentary rather than runtime material; claim and variant provenance remains in retained artifacts and Git. |
| `wiki/raw/KXDocRE_cross_document_RE_2024.md` | Was a literature note with no execution role. |
| `wiki/raw/LLM_synth_quality_diversity_low_resource_RE_20260504.md` | Was a literature note; the source artifact retains executable augmentation evidence instead of a second prose archive. |
| `wiki/raw/small_dataset_ere_2025.md` | Was a literature survey with no runtime role; its provenance remains recoverable from Git and the external research record. |

## Current-only paths

These paths have no tracked counterpart at the baseline commit. “Current-only”
is a set relationship between the two trees, not provenance metadata.

### Repository and protocol documentation

| Path | Current role and reason |
| --- | --- |
| `.gitattributes` | Normalizes text to LF so hashes and machine-readable artifacts remain stable across operating systems. |
| `configs/phase_b_experiment_matrix.json` | Freezes the four evaluated conditions (`VER-RAW`, `VER-CONFIDENCE`, `VER-SIMPLE`, and `VER-CORRECTIVE`) in one machine-validated authority. |
| `configs/phase_b_path_a.json` | Freezes dataset, model, seed, split, training, threshold, statistics, runtime, and tracked-resource identities for the canonical protocol. |
| `configs/phase_b_section5_evidence.json` | Registers the two historical ledgers and seven claim families as immutable secondary evidence so reconciliation cannot silently promote them to canonical results. |
| `docs/phase_b_workflow.md` | Gives standalone commands, stage gates, output layout, environment assumptions, and reproducibility limitations for the canonical runner. |
| `docs/phase_c_migration.md` | Records a design-only raw/typed migration boundary and the evidence required before model adapters can become canonical. |
| `docs/historical-transition-map.md` | B-05U file- and claim-level retention/deletion gate for every historical executable family, README command, secondary claim, and canonical successor. |
| `docs/phase_b_root_module_mapping.md` | Maps every former package module to its root host, records the two standard-library collision renames, and states the package-removal parity gates. |
| `docs/source-change-inventory.md` | Maintains this exhaustive, reasoned comparison with the fixed baseline and prevents future source edits from losing their deletion/reuse/rewrite rationale. |

### Canonical root-module workflow

| Path | Current role and reason |
| --- | --- |
| `acquisition.py` | Downloads the approved immutable archive resumably, verifies size/hash/licensing metadata, and selectively extracts it with path-safety checks. |
| `phase_b.py` | Exposes one command surface for `doctor`, `reconcile`, `fetch`, `prepare`, verifier modes, pilot audit, and scoring, keeping stage transitions explicit. |
| `config.py` | Loads and strictly validates the frozen config, experiment matrix, schemas, and tracked-resource identities so protocol drift fails early. |
| `constants.py` | Centralizes frozen entity, relation, condition, schema, and protocol identifiers to prevent spelling or ordering drift between stages. |
| `doctor.py` | Checks checkout identity, environment, ignore rules, resources, and worktree constraints and emits a machine-readable preflight manifest. |
| `phase_b_io.py` | Supplies canonical JSON/JSONL serialization, atomic writes, hashing, and explicit data-contract errors shared across stages. |
| `metrics.py` | Implements zero-safe precision, recall, and F1 primitives so edge cases have a deterministic definition. |
| `paths.py` | Discovers the standalone repository and proves every generated path remains beneath `output/<run-id>/`. |
| `pilot.py` | Implements the development-only four-capture verifier audit with predeclared subset/seeds, determinism checks, and an explicit publication-admission barrier. |
| `preparation.py` | Parses the immutable official corpus, audits BIO/relation alignment, hard-stops unresolved typed cases, and materializes deterministic records only after validation. |
| `reconciliation.py` | Verifies historical ledger identity and physical defects while keeping those ledgers secondary to canonical evidence. |
| `records.py` | Defines typed immutable sentence, gold, candidate, and verdict records so stages exchange validated objects instead of ad hoc dictionaries. |
| `scoring.py` | Scores all frozen conditions offline with directed entity-typed matching and emits paired outcomes plus provenance. |
| `split.py` | Implements the deterministic seed-42 `586/103/173` split and isolation checks needed to prevent evaluation leakage. |
| `phase_b_statistics.py` | Implements paired hierarchical bootstrap intervals, paired t-tests, exact Wilcoxon tests, and Holm correction for the predeclared paired design. |
| `verifier.py` | Builds canonical requests and prompts and supports dry, live, and replay execution with pinned model identity, warmup, retries, caching, validation, and telemetry. |

### Frozen verifier prompts

| Path | Current role and reason |
| --- | --- |
| `prompts/phase_b/code-verifier-1-system.txt` | Freezes the source-only evidence rule, ontology, and structured-response obligation shared by verifier conditions. |
| `prompts/phase_b/code-verifier-1-simple.txt` | Freezes the `KEEP`/`DISCARD` decision task without permitting relation rewriting. |
| `prompts/phase_b/code-verifier-1-corrective.txt` | Freezes the `KEEP`/`CORRECT`/`DISCARD` task and requires any correction to remain grounded in the supplied source sentence. |

### Machine-readable contracts

| Path | Current role and reason |
| --- | --- |
| `schemas/phase_b/candidate-index.schema.json` | Validates the complete development candidate index so pilot sampling is reproducible and cannot be hand-selected. |
| `schemas/phase_b/checkout-manifest.schema.json` | Validates doctor output and records checkout/environment readiness in a portable form. |
| `schemas/phase_b/config.schema.json` | Validates the canonical Path A configuration and rejects unknown or malformed protocol fields. |
| `schemas/phase_b/data-preparation-manifest.schema.json` | Validates preparation identities, counts, outputs, and audit state for deterministic corpus materialization. |
| `schemas/phase_b/experiment-matrix.schema.json` | Validates the four-condition evaluation matrix and condition-specific verifier/threshold rules. |
| `schemas/phase_b/gold-alignment-audit.schema.json` | Validates the hard-stop audit for ambiguous or missing typed BIO alignment rather than allowing silent data loss. |
| `schemas/phase_b/input-acquisition-manifest.schema.json` | Validates archive origin, licensing acknowledgment, size, hash, extraction, and cache status. |
| `schemas/phase_b/metrics.schema.json` | Validates machine-readable offline metric counts and precision/recall/F1 outputs. |
| `schemas/phase_b/outcomes.schema.json` | Validates paired candidate- and sentence-level outcomes required by confidence intervals and paired tests. |
| `schemas/phase_b/prepared-sentence.schema.json` | Validates each normalized, provenance-bearing CODE-ACCORD sentence record. |
| `schemas/phase_b/records.schema.json` | Validates gold, candidate, and verifier record variants at stage boundaries. |
| `schemas/phase_b/score-manifest.schema.json` | Validates scoring inputs, identities, condition coverage, and emitted artifact hashes. |
| `schemas/phase_b/section5-evidence-reconciliation.schema.json` | Validates the secondary-evidence audit without authorizing historical claims as canonical. |
| `schemas/phase_b/section5-evidence-register.schema.json` | Validates the frozen historical ledger and claim-family register consumed by reconciliation. |
| `schemas/phase_b/threshold-selection.schema.json` | Validates development-only confidence-threshold selection and prevents test-set threshold tuning. |
| `schemas/phase_b/verifier-corrective-response.schema.json` | Constrains corrective replies to the approved verdicts and typed corrected relation form. |
| `schemas/phase_b/verifier-environment.schema.json` | Records and validates the local verifier service/model environment needed to interpret a run. |
| `schemas/phase_b/verifier-manifest.schema.json` | Validates verifier-stage inputs, prompt/model identities, counters, cache state, and outputs. |
| `schemas/phase_b/verifier-pilot-audit.schema.json` | Validates pilot gate results and keeps development evidence separate from publication outcomes. |
| `schemas/phase_b/verifier-pilot-captures.schema.json` | Validates the four required pilot captures and their reproducible identities. |
| `schemas/phase_b/verifier-pilot-failure.schema.json` | Gives pilot failures a machine-readable, retained form rather than reducing them to console text. |
| `schemas/phase_b/verifier-pilot-selection.schema.json` | Validates the predeclared pilot subset and seed selection before any live verifier output is observed. |
| `schemas/phase_b/verifier-replay.schema.json` | Validates cached request/response replay records so offline reruns use the same model interaction evidence. |
| `schemas/phase_b/verifier-run-log.schema.json` | Validates structured warmup, retry, latency, cache, parse, and terminal events from verifier execution. |
| `schemas/phase_b/verifier-simple-response.schema.json` | Constrains simple replies to the approved non-corrective verdict form. |

### Regression and contract tests

| Path | Current role and reason |
| --- | --- |
| `tests/test_artifact_contracts.py` | Protects retained historical pure functions, import safety, graph/rule behavior, and verifier contracts from cleanup regressions. |
| `tests/test_phase_b_pilot.py` | Exercises valid and invalid pilot selections, captures, determinism, and publication-admission barriers with synthetic data. |
| `tests/test_phase_b_pipeline.py` | Tests config/path containment, typed records, metrics, split isolation, statistics, and offline scoring. |
| `tests/test_phase_b_preparation.py` | Tests acquisition safety, official CSV/BIO parsing, alignment auditing, and deterministic materialization. |
| `tests/test_phase_b_reconciliation.py` | Tests historical ledger identities, drift detection, physical-defect preservation, and secondary-evidence classification. |
| `tests/test_phase_b_verifier.py` | Tests dry/live/replay behavior, model pinning, warmup, retry/cache/error handling, and corrective-response validation. |

## Reused without source edits

These 48 baseline paths are byte-for-byte unchanged. Their presence is
intentional; “historical” means their results are provenance evidence, not that
they are canonical publication commands.

### Repository and dataset foundations

| Path | Reason for reuse |
| --- | --- |
| `.python-version` | Retains the selected Python interpreter family used by the standalone environment. |
| `data/__init__.py` | Preserves the data package boundary used by retained loaders. |
| `data/ade.py` | Preserves the historical ADE real-data loader and comparator path. |
| `data/arxiv_real.py` | Preserves the historical real-arXiv loader used by retained experiments. |
| `data/code_accord.py` | Preserves exact historical CODE-ACCORD fuzzy-loading behavior for provenance; canonical strict preparation intentionally does not reuse its matching semantics. |
| `data/code_accord/entities/train.csv` | Preserves the tracked partial historical input needed to interpret the loader and old checkout, while documentation states that it is not the complete canonical corpus. |
| `data/conll04.py` | Preserves the historical CoNLL04 loader and cross-dataset path. |
| `data/cuad.py` | Preserves the historical CUAD transfer loader. |
| `data/scier.py` | Preserves the historical SciER loader. |
| `data/scierc.py` | Preserves the historical SciERC loader central to the encoder/KG evidence chain. |
| `data/synth_loader.py` | Preserves generated-data loading required by retained augmentation comparators. |
| `data/download_ade.py` | Preserves the historical ADE acquisition path. |
| `data/download_arxiv_real.py` | Preserves the historical arXiv acquisition path. |
| `data/download_conll04.py` | Preserves the historical CoNLL04 acquisition path. |
| `data/download_scierc.py` | Preserves the historical SciERC acquisition path. |

### Historical extraction, graph, and evaluation chain

| Path | Reason for reuse |
| --- | --- |
| `build_kg.py` | Preserves provenance-bearing entity resolution, rule filtering, and graph construction behavior. |
| `inference_kg.py` | Preserves encoder inference and per-triple confidence behavior supporting historical KG results. |
| `verify_triples_llm.py` | Preserves the historical SciERC free-form local-verifier behavior; its incompatible ontology and output contract are not repurposed for canonical scoring. |
| `rule_engine.py` | Preserves the ontology constraint implementation used by retained graph construction. |
| `eval/__init__.py` | Preserves the evaluation package boundary. |
| `eval/triple_f1.py` | Preserves the historical type-agnostic exact-triple metric used to interpret old results. |
| `diagnose_evidence_paths.py` | Preserves the evidence-path diagnostic used to inspect historical Graph RAG behavior. |

### Historical model and training implementation

| Path | Reason for reuse |
| --- | --- |
| `models/__init__.py` | Preserves the model package boundary. |
| `models/bert_kg_encoder.py` | Preserves the direct encoder underlying the historical joint NER/RE evidence chain and is the intended target of a thin canonical adapter. |
| `models/critic.py` | Preserves the critic used by retained generative/cooperative negative-result paths. |
| `models/decoder_d.py` | Preserves the decoder used by retained generative comparators. |
| `models/decoder_d_lora.py` | Preserves the LoRA decoder variant supporting a retained negative-result family. |
| `models/electra_generator.py` | Preserves the ELECTRA generator used by retained pretraining comparisons. |
| `models/encoder_reward.py` | Preserves reward-encoder behavior used by retained cooperative training. |
| `models/triple_recovery.py` | Preserves triple-recovery behavior used by retained training variants. |
| `train_gan.py` | Preserves the generative-adversarial comparator. |
| `train_multi.py` | Preserves a direct historical multi-task training entry point. |
| `train_pretrain_cooperative.py` | Preserves the cooperative masking/pretraining negative-result path. |
| `train_stage2b.py` | Preserves a claim-bearing historical closed-loop variant. |
| `train_stage2c.py` | Preserves a claim-bearing historical closed-loop variant. |
| `train_stage2d.py` | Preserves a claim-bearing historical LoRA variant. |
| `train_stage2e.py` | Preserves a claim-bearing historical cooperative variant. |
| `run_a19_cosine_probe.sh` | Preserves the recorded command surface for the A19 cosine probe. |

### Historical data generation and auxiliary comparators

| Path | Reason for reuse |
| --- | --- |
| `generate_accord_llm_aug.py` | Preserves the schema-aware CODE-ACCORD augmentation implementation tied to retained multi-seed evidence. |
| `generate_cycle_data.py` | Preserves cycle-data generation required by a retained closed-loop negative-result path. |
| `generate_entigraph.py` | Preserves the EntiGraph augmentation generator and its reproducible method. |
| `generate_entity_masks.py` | Preserves entity-mask generation used by retained cooperative pretraining. |
| `generate_paraphrase_dataset.py` | Preserves paraphrase augmentation used by retained comparators. |
| `generate_synth_dataset.py` | Preserves synthetic-data generation used by retained negative-result experiments. |
| `dapt_zh.py` | Preserves the Traditional-Chinese domain-adaptive pretraining comparator. |
| `zh_translate_project.py` | Preserves the Traditional-Chinese translation comparator. |

### Historical result evidence

| Path | Reason for reuse |
| --- | --- |
| `results.tsv` | Preserves the original physical ledger, including known blank/concatenated rows, as immutable secondary evidence pending verified archival. |
| `results_stage2.tsv` | Preserves the Stage-2 ledger as immutable secondary evidence pending verified archival. |

## Checklist for subsequent source edits

Before committing a source change:

1. Re-run the baseline comparison commands above.
2. Add every newly differing path to the appropriate table and explain its
   current responsibility and rationale.
3. Move any edited byte-for-byte reuse path into **Modified baseline paths** and
   describe both behavioral and non-behavioral effects.
4. Reclassify removals, renames, restorations, and generated-output candidates
   explicitly; do not infer their role from an extension or filename.
5. Confirm that architecture wording still matches executable reachability and
   that incomplete adapters, blocked measurements, and runtime limitations are
   not described as complete.
6. Run the relevant focused tests, the broadest affordable suite, and
   `git diff --check`.
