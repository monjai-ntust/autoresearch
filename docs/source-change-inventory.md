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
  **Modified baseline paths**, **Baseline-only paths**, **Current-only paths**,
  or the explicitly relocated entries under **Reused or mechanically relocated
  historical source**;
- every remaining baseline path is reused without source edits at its original
  path;
- reasons describe the current implementation or evidence role, not a point in
  the development timeline; and
- a moved or renamed path is described explicitly instead of being presented as
  an unrelated deletion and creation.

With this document included, Git reports 81 additions, 35 deletions, seven
same-path modifications, and 22 detected renames. One additional logical move
(`bench_gpu.py` to `provenance/bench_gpu.py`) appears as an add/delete because
its earlier import-safety edit lowers similarity. Classified by role, there are
80 genuinely current-only paths, 34 genuinely baseline-only paths, 11 modified
baseline artifacts (four relocated), 19 byte-identical relocations, and 26
baseline paths reused byte-for-byte at their original locations. The baseline
has 90 tracked paths and the current tree has 136.

## Current architecture and rewrite boundary

The publication-facing command is `python -B phase_b.py`. The canonical root
modules are a protocol and orchestration layer, not a replacement encoder. They
centralize contracts that the historical scripts under `provenance/` do not share:
immutable input and model identities, a frozen experiment matrix, strict typed
matching, leakage-safe splits, deterministic records, replay, output
containment, and machine-readable manifests.

Keeping canonical semantics distinct from the historical scripts while placing
the active modules at the root is the current boundary for three reasons:

1. The historical training, inference, verification, graph, and ablation paths
   are evidence-bearing implementations. Relocating secondary entry points
   under `provenance/` preserves their code while preventing them from being
   mistaken for the canonical root workflow; `train_span.py` remains at root
   because the canonical model stage invokes that original loop directly.
2. Evaluation protocol is cross-cutting. Putting acquisition, record schemas,
   verifier telemetry, scoring, and statistical tests into one historical
   training script would couple unrelated responsibilities and make isolated
   validation harder.
3. The remaining integration can use thin adapters around retained model code.
   That avoids a duplicate encoder while allowing the canonical runner to
   enforce its contracts at every boundary.

The end-to-end command surface is implemented but has not yet produced a
completed external publication run. `model.py` supplies the canonical `model`
stage: `generate-candidates` supports dry-run/replay/live and `train`
supports dry-run plus accelerator-gated live execution through the original
`train_span.py` loop. The compatibility mode consumes the prepared split,
pins the backbone revision, preserves the historical default path, emits full
restart/checkpoint provenance, and forbids test access during selection. Actual
accelerator parity/smoke evidence and the eight-seed publication run remain
gated, so the historical scripts remain necessary provenance. Official data
preparation materializes the raw/typed-strict dual view. The root-module layout
is the publication-facing surface; it does not claim that historical results or
their implementation lineage have been superseded.

The similarly named paths below are not one-to-one rewrites:

| Baseline/current boundary | Current interpretation and reason |
| --- | --- |
| `prepare.py` vs `preparation.py` | The baseline script prepared FineWeb bytes-per-byte training data. The canonical module validates and materializes official CODE-ACCORD records. The old script is removed as unrelated upstream code; the new module does not inherit its behavior. |
| `data/code_accord.py` vs canonical preparation | The historical CSV/fuzzy path is preserved as the default. An opt-in lossless prepared-JSONL adapter and resumable sampler feed the same dataset/example classes for canonical training; canonical preparation still owns strict typed alignment. |
| `provenance/verify_triples_llm.py` vs `verifier.py` | The relocated historical verifier preserves the SciERC free-form response path. The canonical verifier uses frozen CODE prompts, structured response schemas, model identity, caching, replay, retry policy, and telemetry. |
| `eval/triple_f1.py` vs `scoring.py` | The historical helper preserves type-agnostic exact triple scoring used by retained experiments. Canonical scoring applies directed, entity-typed `CODE-STRICT-1` matching across the frozen four-condition matrix. |
| Historical training/model modules vs canonical root modules | Model behavior remains in the retained modules. The canonical runner supplies prepared-input, revision, restart, no-test, and manifest controls around the original root `train_span.py` loop; secondary historical entry points are namespaced under `provenance/`. |
| Historical result/checkpoint locations vs `output/<run-id>/` | Historical ledgers remain evidence pending verified archival. Every canonical runtime artifact is confined to the ignored run directory so a standalone clone has one reproducible output boundary. |

## Modified baseline paths

| Path | Current edit and reason |
| --- | --- |
| `.gitignore` | Ignores `/output/`, checkpoints, and downloaded/generated dataset directories while explicitly allowing the tracked CODE-ACCORD fixture. The obsolete ignore rule for tracked `results.tsv` is absent. This enforces the single temporary-output root without hiding retained evidence. |
| `README.md` | Defines the repository as a standalone publication artifact, separates canonical commands from historical provenance paths, documents fresh-clone preparation without historical checkpoints, environment-version recording, datasets, execution gates, outputs, limitations, and links this inventory. The baseline upstream pretraining description did not describe the retained research implementation. |
| `provenance/bench_gpu.py` (moved from `bench_gpu.py`) | Places benchmark execution behind `main()` and an import guard. Benchmark behavior remains available, while importing the module no longer downloads a model or allocates accelerator memory. Revision 1.9 relocates the already-modified historical entry point without changing that behavior. |
| `provenance/build_kg.py` (moved from `build_kg.py`) | Preserves provenance-bearing entity resolution, rule filtering, and graph construction behavior while mechanically changing the sibling rule import to `from provenance import rule_engine` so documented module execution continues to work after relocation. |
| `data/code_accord.py` | Preserves the historical CSV loader as the default and adds an opt-in, order-preserving adapter from prepared train/development JSONL into the same example shape. Canonical-only shuffling uses a serializable sampler cursor/order so restart does not repartition or silently reshuffle the remaining epoch. |
| `provenance/eval_graph_rag.py` (moved from `eval_graph_rag.py`) | Describes retrieval as unique whitespace-token overlap instead of BM25-style retrieval. Runtime behavior is unchanged; the text now states the actual algorithm and avoids overstating the evaluation method. Revision 1.9 relocates the already-modified historical entry point. |
| `models/bert_kg_encoder.py` | Adds an optional model-revision argument and forwards it to `AutoModel.from_pretrained`; callers that omit it retain the exact historical loading behavior. Canonical training/inference uses it to bind DeBERTa to the frozen revision. |
| `pyproject.toml` | Names the publication artifact, removes unused upstream packages, and declares dependencies used by retained code (`peft`, `pytorch-crf`, and `safetensors`) while keeping the required Torch/Transformers/data stack. Torch resolves from the CUDA 13.0 index so the locked aarch64 environment can target the external GB10 (compute capability 12.1). |
| `provenance/train_gumbel.py` (moved from `train_gumbel.py`) | Restricts process-environment side effects to command execution under the import guard. The training command retains its environment settings, while library import remains side-effect free. Revision 1.9 relocates the already-modified historical entry point. |
| `train_span.py` | Keeps all historical arguments/default behavior and adds opt-in canonical controls for prepared split input, pinned model revision, no-test selection, run-local progress/summary/checkpoint paths, and full atomic restart state. Canonical restart payloads load CPU-first so sampler/RNG byte tensors remain valid before model/optimizer state is restored to the active device. The original encoder, heads, loss loop, optimizer, scheduler, recipe, and checkpoint `encoder` representation remain the implementation host. |
| `uv.lock` | Resolves the dependency graph declared by the current `pyproject.toml`, including Torch 2.9.1 CUDA 13.0 and its CUDA-13 runtime packages. It supports repeatable installation of the current artifact and is not represented as an exact manifest of historical experiment machines. |

## Baseline-only paths

These paths are absent from the current source tree. Git retains their original
content and lineage.

| Path | Reason it is not part of the current tree |
| --- | --- |
| `Task_done.md` | Duplicated project history whose authoritative, read-only copy lives outside the standalone source artifact; implementation provenance is represented by retained code, result ledgers, Git, and the parent research state. |
| `analysis.ipynb` | Belonged to the disconnected FineWeb/bytes-per-byte upstream workflow and had no reachable role in the paper's entity/relation pipeline. |
| `data/accord_entigraph_5pairs.jsonl` | Was a generated data artifact with unresolved redistribution and input status. The generator and result evidence preserve the method without shipping an ambiguous runtime prerequisite. |
| `eval_checkpoint.py` | Removed after its historical checkpoint-evaluation function was superseded by the canonical `model generate-candidates` → `score` chain and its README command was retired; its only consumer was the import-safety test. Behavior remains recoverable from Git history. |
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
| `configs/phase_b_path_a.json` | Freezes dataset, model, seed, split, training, threshold, statistics, runtime, and tracked-resource identities for the canonical protocol, including the CUDA-13 lock index and compatibility-training source/schema/script hashes. |
| `configs/phase_b_section5_evidence.json` | Registers the two historical ledgers and seven claim families as immutable secondary evidence so reconciliation cannot silently promote them to canonical results. |
| `docs/phase_b_workflow.md` | Is the fresh-clone primary-data guide: it gives the bootstrap, non-enforcing environment-version recording, exact prepared-data boundary, canonical live/resume command and artifacts, partial pre–Phase A data comparison, legacy diagnostic distinction, GB10 runtime requirement, B-07/eight-seed/verifier gates, and an isolated non-publication real GPU/Ollama smoke lane. |
| `docs/model-training-compatibility.md` | Freezes the historical trainer/data/recipe/environment evidence, recoverable per-seed outcomes, preserved implementation invariants, necessary opt-in differences, unavailable equivalence evidence, focused validation, and external smoke gate for B-05C. |
| `scripts/phase_b_debug.sh` | Bash recovery runner for the standalone external checkout. It derives a run ID, upstream, seeds, Python environment, Ollama model, and blob identity instead of pinning machine-local values. Its `full` stage runs the independent eight-seed development/pilot/test/verifier/score chain only under conditional B-07 approval, stops at the first new error, resumes training or partial final-verifier responses, and otherwise performs validated stage-scoped cleanup recorded in a recovery manifest. The default availability sweep, isolated smoke, verified archive reuse, seed-specific completion checks, and acknowledged legacy CSV diagnostic remain available. |
| `provenance/README.md` | Defines the relocated secondary/historical command boundary and the module-style invocation contract without promoting any retained script into the publication workflow. |
| `docs/phase_c_migration.md` | Records a design-only raw/typed migration boundary and the evidence required before model adapters can become canonical. |
| `docs/historical-transition-map.md` | B-05U file- and claim-level retention/deletion gate for every historical executable family, README command, secondary claim, and canonical successor. |
| `docs/historical-transition-inventory.md` | B-05U family-level record of why each legacy executable family cannot remain independent, its canonical successor stage, and its raw-evidence disposition. |
| `docs/secondary-data-replacement-map.md` | B-05U data-item-level map assigning every draft Section 5 secondary result family a disposition (canonical stage, retained input, corrected/unsupported, or external) and recording the open user decisions that gate each `secondary` sub-workflow. |
| `docs/claim-disposition.md` | Dispositions every Phase A inventory claim (`A-CLAIM-001`…`029`) as Extends / Superseded / Retained-provenance / Removed against the canonical workflow, with the `output/<run-id>/` path where each extended or superseded claim's new data is produced. |
| `docs/phase_b_root_module_mapping.md` | Maps every former package module to its root host, records the two standard-library collision renames, and states the package-removal parity gates. |
| `docs/source-change-inventory.md` | Maintains this exhaustive, reasoned comparison with the fixed baseline and prevents future source edits from losing their deletion/reuse/rewrite rationale. |

### Canonical root-module workflow

| Path | Current role and reason |
| --- | --- |
| `acquisition.py` | Downloads the approved immutable archive resumably, verifies size/hash/licensing metadata, and selectively extracts it with path-safety checks. |
| `phase_b.py` | Exposes one command surface for `doctor`, `reconcile`, `fetch`, `prepare`, dry/live model training, eight-seed candidate assembly, label-blind pilot preparation, verifier modes, pilot audit, and scoring, keeping stage transitions explicit. |
| `config.py` | Loads and strictly validates the frozen config, experiment matrix, schemas, and tracked-resource identities so protocol drift fails early. |
| `constants.py` | Centralizes frozen entity, relation, condition, schema, and protocol identifiers to prevent spelling or ordering drift between stages. |
| `doctor.py` | Checks checkout identity, ignore rules, resources, and worktree constraints; records configured-reference and actual Python/uv versions without enforcing an exact version; and emits a machine-readable preflight manifest. |
| `phase_b_io.py` | Supplies canonical JSON/JSONL serialization, atomic writes, hashing, and explicit data-contract errors shared across stages. |
| `metrics.py` | Implements zero-safe precision, recall, and F1 primitives so edge cases have a deterministic definition. |
| `model.py` | Canonical `model` stage. Candidate dry/replay/live paths use seed-specific manifests and verify checkpoint plus acquisition/preparation/split identities. Training dry-run plans a seed; live validates bootstrap/data compatibility, invokes canonical mode in the original trainer, resumes run-local state, and emits checkpoint/stage manifests without implementing a second training loop. |
| `paths.py` | Discovers the standalone repository and proves every generated path remains beneath `output/<run-id>/`. |
| `pilot.py` | Implements the development-only four-capture verifier audit with predeclared subset/seeds, determinism checks, and an explicit publication-admission barrier. |
| `preparation.py` | Parses the immutable official corpus, preserves raw relation-coverage statistics separately from typed-strict eligibility, audits BIO/relation alignment, and materializes deterministic records only after validation. |
| `publication.py` | Validates and assembles seed-specific development/test candidates, writes the full development candidate index, and freezes one label-blind pilot candidate per seed plus a warm-up before any live call. |
| `reconciliation.py` | Verifies historical ledger identity and physical defects while keeping those ledgers secondary to canonical evidence. |
| `records.py` | Defines typed immutable sentence, gold, candidate, and verdict records so stages exchange validated objects instead of ad hoc dictionaries. |
| `scoring.py` | Scores all frozen conditions offline with directed entity-typed matching and emits paired outcomes plus provenance; an explicit namespaced nonpublication-smoke output mode suppresses publication coverage/statistics. |
| `smoke.py` | Creates deterministic one-candidate, pseudo-seed diagnostic inputs and verdict clones solely for the isolated nonpublication end-to-end smoke lane. |
| `split.py` | Implements the deterministic seed-42 `586/103/173` split and isolation checks needed to prevent evaluation leakage. |
| `threshold.py` | Canonical `select-threshold` producer of the frozen development confidence threshold (VER-CONFIDENCE): computes per-seed development strict Triple F1 over the grid, averages across eight seeds, selects the argmax with the higher-threshold tie rule, and binds the emitted selection to the full candidate-index hash used by B-07. It never inspects test labels. |
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
| `schemas/phase_b/model-checkpoint-manifest.schema.json` | Validates the encoder checkpoint identity and selected metric plus archive, acquisition, annotation, prepared tree/files/split, config/code, restart, source-commit, and historical-comparability bindings consumed by candidate generation. |
| `schemas/phase_b/model-generation-manifest.schema.json` | Validates the `model generate-candidates` stage manifest for both dry-run and replay executions. |
| `schemas/phase_b/model-generation-plan.schema.json` | Validates the per-sentence dry-run inference plan emitted without contacting a model. |
| `schemas/phase_b/model-prediction-ledger.schema.json` | Validates the frozen per-sentence encoder span/relation scores consumed by replay candidate generation. |
| `schemas/phase_b/model-train-manifest.schema.json` | Validates dry-run plans and completed compatibility-hosted live seed manifests, including run-local input/output, resume, environment, and historical-comparability evidence. |
| `schemas/phase_b/outcomes.schema.json` | Validates paired candidate- and sentence-level outcomes required by confidence intervals and paired tests. |
| `schemas/phase_b/prepared-sentence.schema.json` | Validates each normalized, provenance-bearing CODE-ACCORD sentence record. |
| `schemas/phase_b/records.schema.json` | Validates gold, candidate, and verifier record variants at stage boundaries. |
| `schemas/phase_b/score-manifest.schema.json` | Validates scoring inputs, identities, condition coverage, emitted artifact hashes, and the explicit nonpublication-smoke designation. |
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
| `tests/test_phase_b_debug_script.py` | Statically protects the external Bash launcher's config-derived defaults, full-stage ordering/B-07 gate, run-scoped recovery safeguards, exact-resume contract, and canonical-root/provenance layout when Bash/GPU/Ollama execution is unavailable locally. |
| `tests/test_phase_b_model.py` | Tests candidate dry/replay identity and confidence contracts plus training dry-run, missing-bootstrap rejection, interrupted-live retry/resume command construction, partial legacy-data audit, no-test summary, and schema-bound checkpoint output with a fake trainer process. |
| `tests/test_phase_b_publication.py` | Tests eight-seed candidate assembly, index identity, threshold binding, label-blind pilot selection, and wrong-seed rejection. |
| `tests/test_phase_b_training_compatibility.py` | Tests prepared-record order/span/relation adaptation, split-substitution rejection, sampler cursor/order resume, CPU-first restart loading, opt-in historical defaults, and optional model-revision forwarding without downloading a model. |
| `tests/test_phase_b_threshold.py` | Tests development threshold selection: hand-derived argmax/tie-rule and false-positive curves, all-eight-seed coverage, overwrite refusal, and that the emitted file is consumable by the scorer's threshold loader. |
| `tests/test_phase_b_pilot.py` | Exercises valid and invalid pilot selections, captures, determinism, and publication-admission barriers with synthetic data. |
| `tests/test_phase_b_pipeline.py` | Tests config/path containment, typed records, metrics, split isolation, statistics, and offline scoring. |
| `tests/test_phase_b_preparation.py` | Tests acquisition safety, official CSV/BIO parsing, alignment auditing, and deterministic materialization. |
| `tests/test_phase_b_reconciliation.py` | Tests historical ledger identities, drift detection, physical-defect preservation, and secondary-evidence classification. |
| `tests/test_phase_b_smoke.py` | Tests the isolated nonpublication smoke-input selection and pseudo-seed cloning contracts without performing accelerator or Ollama execution. |
| `tests/test_phase_b_verifier.py` | Tests dry/live/replay behavior, model pinning, warmup, retry/cache/error handling, and corrective-response validation. |

## Reused or mechanically relocated historical source

Dataset/model helpers below remain at their original paths. Inventory-confirmed
secondary entry points now live under `provenance/`; their moves are
organizational, with the one mechanical `provenance.build_kg` import adjustment
required by module execution. Historical results remain provenance evidence,
not canonical publication commands.

### Repository and dataset foundations

| Path | Reason for reuse |
| --- | --- |
| `.python-version` | Retains the selected Python interpreter family used by the standalone environment. |
| `data/__init__.py` | Preserves the data package boundary used by retained loaders. |
| `data/ade.py` | Preserves the historical ADE real-data loader and comparator path. |
| `data/arxiv_real.py` | Preserves the historical real-arXiv loader used by retained experiments. |
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
| `provenance/inference_kg.py` (moved from `inference_kg.py`) | Preserves encoder inference and per-triple confidence behavior supporting historical KG results. |
| `provenance/verify_triples_llm.py` (moved from `verify_triples_llm.py`) | Preserves the historical SciERC free-form local-verifier behavior; its incompatible ontology and output contract are not repurposed for canonical scoring. |
| `provenance/rule_engine.py` (moved from `rule_engine.py`) | Preserves the ontology constraint implementation used by retained graph construction. |
| `eval/__init__.py` | Preserves the evaluation package boundary. |
| `eval/triple_f1.py` | Preserves the historical type-agnostic exact-triple metric used to interpret old results. |
| `provenance/diagnose_evidence_paths.py` (moved from `diagnose_evidence_paths.py`) | Preserves the evidence-path diagnostic used to inspect historical Graph RAG behavior. |

### Historical model and training implementation

| Path | Reason for reuse |
| --- | --- |
| `models/__init__.py` | Preserves the model package boundary. |
| `models/critic.py` | Preserves the critic used by retained generative/cooperative negative-result paths. |
| `models/decoder_d.py` | Preserves the decoder used by retained generative comparators. |
| `models/decoder_d_lora.py` | Preserves the LoRA decoder variant supporting a retained negative-result family. |
| `models/electra_generator.py` | Preserves the ELECTRA generator used by retained pretraining comparisons. |
| `models/encoder_reward.py` | Preserves reward-encoder behavior used by retained cooperative training. |
| `models/triple_recovery.py` | Preserves triple-recovery behavior used by retained training variants. |
| `provenance/train_gan.py` (moved from `train_gan.py`) | Preserves the generative-adversarial comparator. |
| `provenance/train_multi.py` (moved from `train_multi.py`) | Preserves a direct historical multi-task training entry point. |
| `provenance/train_pretrain_cooperative.py` (moved from `train_pretrain_cooperative.py`) | Preserves the cooperative masking/pretraining negative-result path. |
| `provenance/train_stage2b.py` (moved from `train_stage2b.py`) | Preserves a claim-bearing historical closed-loop variant. |
| `provenance/train_stage2c.py` (moved from `train_stage2c.py`) | Preserves a claim-bearing historical closed-loop variant. |
| `provenance/train_stage2d.py` (moved from `train_stage2d.py`) | Preserves a claim-bearing historical LoRA variant. |
| `provenance/train_stage2e.py` (moved from `train_stage2e.py`) | Preserves a claim-bearing historical cooperative variant. |
| `run_a19_cosine_probe.sh` | Preserves the recorded command surface for the A19 cosine probe. |

### Historical data generation and auxiliary comparators

| Path | Reason for reuse |
| --- | --- |
| `provenance/generate_accord_llm_aug.py` (moved from `generate_accord_llm_aug.py`) | Preserves the schema-aware CODE-ACCORD augmentation implementation tied to retained multi-seed evidence. |
| `provenance/generate_cycle_data.py` (moved from `generate_cycle_data.py`) | Preserves cycle-data generation required by a retained closed-loop negative-result path. |
| `provenance/generate_entigraph.py` (moved from `generate_entigraph.py`) | Preserves the EntiGraph augmentation generator and its reproducible method. |
| `provenance/generate_entity_masks.py` (moved from `generate_entity_masks.py`) | Preserves entity-mask generation used by retained cooperative pretraining. |
| `provenance/generate_paraphrase_dataset.py` (moved from `generate_paraphrase_dataset.py`) | Preserves paraphrase augmentation used by retained comparators. |
| `provenance/generate_synth_dataset.py` (moved from `generate_synth_dataset.py`) | Preserves synthetic-data generation used by retained negative-result experiments. |
| `provenance/dapt_zh.py` (moved from `dapt_zh.py`) | Preserves the Traditional-Chinese domain-adaptive pretraining comparator. |
| `provenance/zh_translate_project.py` (moved from `zh_translate_project.py`) | Preserves the Traditional-Chinese translation comparator. |

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
