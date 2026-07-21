# Low-resource joint entity and relation extraction artifact

This repository contains the research implementation for a low-resource knowledge-graph construction pipeline. It trains a joint named-entity recognition (NER) and relation extraction (RE) encoder, extracts confidence-bearing triples, optionally verifies or corrects them with an Ollama-hosted LLM, and builds a provenance-bearing graph.

The canonical publication-facing Phase B entry point is:

```bash
bash phase_b.sh --stage full
```

The launcher derives the frozen seeds and model identities from configuration,
resumes verified completed stages, and preserves the mandatory passing B-07
pilot audit before final-test access. Live/full invocation is authorized by
default; there is no separate approval flag. Its internal `phase_b.py`
dispatcher implements
checkout/environment `doctor`, frozen Section 5 ledger `reconcile`, immutable
CODE-ACCORD `fetch`, raw-provenance/typed-strict `prepare`, canonical
`model train` (`dry-run|live`) and `model generate-candidates` (`dry-run|replay|live`), frozen verifier
`dry-run|live|replay`, development `pilot-verifier` auditing, development
`select-threshold`, and offline `CODE-STRICT-1` `score` stages. Official v1.0.0 preparation
preserves all official relation rows in a raw-provenance view and writes a
complete audit because nine relation-marker arguments do not map to unique typed
BIO spans. Those nine rows remain auditable but are ineligible for typed-strict
gold; no endpoint is silently repaired, projected, expanded, or omitted. See
[`docs/phase_b_workflow.md`](docs/phase_b_workflow.md) for exact commands, input
schemas, output layout, reproducibility boundaries, and the still-gated stages.
The [`model-training compatibility contract`](docs/model-training-compatibility.md)
records preserved trainer invariants, necessary opt-in differences, and the
limit on historical statistical comparison.
Every workflow-created file is confined to ignored `output/<run-id>/`.

The repository also retains the historical `train_span.py ->
provenance/inference_kg.py -> provenance/verify_triples_llm.py ->
provenance/build_kg.py` chain, reported baselines, ablations,
closed-loop attempts, cross-dataset transfer, and negative results. Those paths
are provenance evidence; they are not publication commands for approved Path A
until their claim-bearing behavior is migrated behind the canonical runner.
All secondary historical entry points are grouped under `provenance/` and run
as modules from the repository root; `train_span.py` remains at root because it
is also the compatibility-hosted canonical encoder trainer.

`phase_b.py` and its root-module collaborators are the launcher's internal
protocol and orchestration layer, not a second encoder
implementation. It owns the contracts that the historical root scripts do not
provide: immutable identities, one experiment matrix, typed matching, split
isolation, output containment, replay, and machine-readable manifests. The
`model generate-candidates` adapter transforms a frozen encoder prediction
ledger into typed candidates under those contracts, and its `live` execution
runs the retained encoder to produce that ledger. `model train --execution
live` invokes an opt-in compatibility mode in the original `train_span.py`
loop: prepared train/development input, revision pinning, full restart state,
no test access, and schema-bound provenance are added without a second encoder
or loss loop. The first external eight-seed full run completed through its
passing pilot, final inference, both verifier modes, and scoring; the original
trainer and inference script remain retained for compatibility/provenance review.
The new verifier caller is necessarily canonical-specific because the retained
historical verifier hard-codes the SciERC ontology, parses free-form line
responses, and lacks the approved prompt/schema/digest/cache/telemetry contract.
That script remains under `provenance/` rather than being silently repurposed.

This creates an explicit two-tier authority boundary. Canonical Path A data,
manifests, typed matching, splits, checkpoints, and scores are authoritative for
the main evaluation. Historical commands are secondary execution/provenance
interfaces, and the draft Section 5 data and ledgers they preserve are secondary
evidence. Those existing results remain priority reconciliation targets, but
they do not override canonical data or metric contracts. Historical paths may
be adapted for portability or migrated behind canonical adapters, but original
result records remain immutable and every rerun receives a new identity. Any
newly added measurement must reuse the canonical framework. Similar trend
direction is a useful discrepancy check, not an acceptance criterion and never
a reason to rewrite measurements.

The `reconcile` stage makes that boundary machine-checkable. After `doctor`, it
validates the frozen hashes, physical-structure facts, and claim references for
the two historical ledgers now held in external archival storage, then writes
`audit/section5-evidence-reconciliation.json`. The standalone source does not
read that archive or retain the generated TSVs. The register preserves their
verified pre-removal facts and reports zero claim families as canonical-ready;
it neither promotes archived evidence nor authorizes final-test access.

[`docs/phase_c_migration.md`](docs/phase_c_migration.md) records the raw/strict
design lineage, frozen assumptions, and evidence conditions that require a
revision. The first full eight-seed run passed B-07 and completed final
inference, both verifier modes, and scoring. Final-test tuning remains
prohibited; publication claims still require the retained-run and release
audits.

The maintained [`source change inventory`](docs/source-change-inventory.md)
classifies every deletion, reuse, modification, and current-only path relative
to the fixed baseline `9feafa4`, including the rationale for the separate
canonical pipeline boundary. Update that inventory with every subsequent source
edit that changes the comparison or a path's role.

## Repository status

- Python 3.10 is the selected and minimum supported Python version. The Phase B
  config records Python 3.10.20 and uv 0.11.26 as reference versions; `doctor`
  records the actual versions without requiring an exact match.
- The committed lockfile describes the current source environment; it is not an exact manifest of every historical run.
- Historical raw logs and model checkpoints are not included.
- The canonical CODE-ACCORD downloader is included, but the archive is not.
  SciERC, CoNLL04, ADE, and arXiv historical acquisition scripts are also
  included. Complete CODE/ACCORD, SciER, CUAD, and zh-Hant datasets are not
  committed.
- The first full external run produced the verifier Precision, Recall, F1,
  confusion-count, paired-outcome, and uncertainty artifacts after a passing
  B-07 pilot. The scorer now renders deterministic publication TSV/Markdown
  products from those same metrics. Release remains pending external
  clean-clone/offline-cache validation.

## Installation

Install [uv](https://docs.astral.sh/uv/). For the canonical Phase B workflow,
use the fresh-clone sequence in `docs/phase_b_workflow.md`; it records, rather
than enforces, the Python and uv versions. The shorter command below is retained
for historical-script development:

```bash
uv sync
```

The tracked support files are part of the reproducibility contract:

| Path | Role |
|---|---|
| `.python-version` | Selects Python 3.10; the Path A config records 3.10.20 as its reference patch version. |
| `pyproject.toml` | Declares the package metadata and direct dependencies used by `uv sync`. |
| `uv.lock` | Locks the resolved environment used for validation of this checkout. |
| `.gitignore` | Excludes local datasets, checkpoints, caches, and generated outputs; it intentionally does not add a log rule. |

PyTorch is resolved from the CUDA 13.0 wheel index. This is required for the
external NVIDIA GB10 (compute capability 12.1): the prior CUDA 12.8 wheel
reports support only through capability 12.0 and fails when NVRTC compiles
DeBERTa relative-position kernels. After pulling a revision with this lockfile,
run `uv sync --frozen` again before any GPU stage. CPU execution is possible for
small checks, but reported training runs require a suitable CUDA GPU and
substantially more time and memory.

Canonical model backbones are loaded through Hugging Face Transformers into
`output/<run-id>/inputs/huggingface` and may require network access on the first
seed only. The resulting file inventory/tree hash is frozen; later seeds and
live inference use that run-local cache with `local_files_only=True`. Historical
direct commands retain their original cache behavior. The verifier additionally
requires `curl`, a reachable Ollama-compatible `/api/chat` endpoint, and the
requested local model (historically `qwen3:32b`). `--ollama-url` is configurable,
so on-premise execution is a deployment choice rather than a code-enforced invariant.

`sentencepiece` and `protobuf` are direct locked dependencies for tokenizer
conversion paths used by the retained model family. The frozen DeBERTa v1
revision itself includes its BPE `vocab.json` and `merges.txt`; installing
conversion packages cannot compensate for selecting an older model revision
that lacks those files. A fresh clone should use `uv sync --frozen`; do not
repair dependency or model identities ad hoc on the execution machine.

## Historical data acquisition paths

The following downloaders support retained non-Path-A experiments. They write to
historical dataset locations and are not the canonical CODE-ACCORD acquisition
stage:

```bash
uv run python data/download_scierc.py
uv run python data/download_conll04.py
uv run python data/download_ade.py
uv run python data/download_arxiv_real.py
```

Expected default locations are:

| Dataset key | Default location | Included? |
|---|---|---|
| `scierc` | `data/scierc/` | No; downloader included |
| `conll04` | `data/conll04/` | No; downloader included |
| `ade` | `data/ade/` | No; downloader included |
| arXiv critic corpus | `data/arxiv_real/cs_validation.jsonl` | No; downloader included |
| `accord` | `data/code_accord/` | Incomplete: only `entities/train.csv` is tracked |
| `scier` | `data/scier_dataset/SciER/PLM/` | No; acquire separately |
| `cuad` | loader-specific repository/home path | No; acquire separately |

The downloaders identify upstream sources but do not grant redistribution rights. Review each dataset's license and terms before acquisition or publication. Do not commit downloaded corpora or generated labels blindly.

## Historical direct encoder runner

`train_span.py` supports `scierc`, `scier`, `conll04`, `ade`, `accord`, and
`cuad`. This minimal SciERC command preserves its historical interface. It is
not the frozen Path A recipe and its output path is outside the canonical
`output/<run-id>/` contract:

```bash
uv run python train_span.py \
  --dataset scierc \
  --max-steps 1500 \
  --save-best-to checkpoints/scierc_span_best.pt
```

Inspect the complete experimental interface without starting training:

```bash
uv run python train_span.py --help
```

The CLI retains reported controls for externally supplied synthetic/CAST/CycleGT data, checkpoint initialization, evidence-graph fusion, contrastive and focal losses, BIO enrichment, class weighting, marker representations, and zh-Hant experiments. Defaults and matching behavior are intentionally preserved for provenance.

## Historical direct extract/verify/build chain

The commands below preserve the pre-Phase-B interface and output schemas. They
must not be used to generate Path A publication results. First extract triples
from a compatible checkpoint:

```bash
uv run python -m provenance.inference_kg \
  --checkpoint checkpoints/scierc_span_best.pt \
  --dataset scierc \
  --split test \
  --out-jsonl results/scierc_inference.jsonl
```

Optionally verify and correct the predicted triples with Ollama:

```bash
uv run python -m provenance.verify_triples_llm \
  --input results/scierc_inference.jsonl \
  --output results/scierc_verified.jsonl \
  --ollama-url http://localhost:11434 \
  --ollama-model qwen3:32b \
  --mode correct
```

Build the graph, optionally applying the retained neuro-symbolic rules:

```bash
uv run python -m provenance.build_kg \
  --input results/scierc_verified.jsonl \
  --output results/scierc_kg.json \
  --filter-mode verified \
  --use-rules
```

`provenance/verify_triples_llm.py` requests `temperature=0` with thinking disabled and has
timeout/retry controls, but it does not implement the approved B-06 JSON-schema,
model-digest, cache, telemetry, seed, or prompt-hash contract. Temperature zero
alone is not proof of deterministic replay.

The canonical verifier stage instead uses the tracked `CODE-VERIFIER-1` prompt
files and response schemas under `prompts/phase_b/` and `schemas/phase_b/`.
It can materialize exact requests without a model, verify and call the pinned
Ollama artifact when live execution is authorized, or turn a frozen raw-response
ledger into verdict records without another model call. See
[`docs/phase_b_workflow.md`](docs/phase_b_workflow.md) for the commands and gates.

## Evaluation semantics

`eval/triple_f1.py` provides the shared evaluation functions used by retained training paths:

- NER spans require exact boundary and entity-type agreement.
- Gold-span RE evaluates relation labels over gold entity spans.
- End-to-end triple matching requires exact head span, tail span, and relation label; for historical compatibility, it does not include entity type in the triple key.
- `train_span.py` selects checkpoints using `triple_f1` by default and can use `ner_f1` for NER-oriented pretraining.

The canonical `phase_b.py` does not reuse that type-agnostic historical
key. Its approved `CODE-STRICT-1` matcher requires example ID, typed head span,
directed relation, and typed tail span to match exactly.

The historical `provenance/eval_graph_rag.py` script is a diagnostic, not a leakage-free downstream benchmark: it derives questions and reference answers from the same gold-bearing source. `provenance/diagnose_evidence_paths.py` reports structural evidence-path coverage and likewise should not be interpreted as task accuracy.

## Retained evaluation and historical reproduction steps

These paths are retained because they produce a draft-reported comparison, a counted negative-result family, or a user-approved diagnostic. Historical commands require the datasets, checkpoints, model caches, and services named below; those artifacts are not bundled. Start with small limits and a separate output path before attempting a recorded full run.

Revisions to these secondary paths must preserve the original command/result
record and create new run/configuration/hash lineage. The corresponding Section
5 data and ledgers remain secondary evidence even when they are the first items
reconciled against the draft. They are validated by their declared semantics and
reproducibility checks, not by whether a rerun recreates a preferred trend. A
material reversal or non-comparable metric is a finding to retain and
investigate; it must not be tuned away or promoted into canonical Path A
evidence without an approved migration.

### Hardware and evaluation diagnostics

Run the installation smoke benchmark after model download/cache setup:

```bash
uv run python -m provenance.bench_gpu
```

This fixed BERT forward/backward smoke is useful for detecting CPU fallback and gross environment problems. Its timings are not the paper's encoder, verifier, or full-pipeline measurements.

For the retained Graph RAG diagnostic, first build a KG and retain inference JSONL with `gold_triples`, then run:

```bash
uv run python -m provenance.eval_graph_rag \
  --kg results/scierc_kg.json \
  --gold-jsonl results/scierc_inference.jsonl \
  --max-questions 10 \
  --output results/scierc_graph_rag_diagnostic.json
```

This reproduces the diagnostic protocol behind the draft table, not a leakage-free compliance benchmark. The question sample and exact inference/KG inputs must be recorded with any reported result.

### Cross-dataset BIO comparator

The older token-BIO runner is retained for the SciERC/CoNLL04/ADE comparator and NER/RE/full-Triple decomposition lineage:

```bash
uv run python -m provenance.train_multi \
  --dataset scierc \
  --max-steps 1500 \
  --seed 42 \
  --save-best-to checkpoints/scierc_token_bio_s42.pt
```

Repeat with `--dataset conll04` or `--dataset ade` only after running the corresponding downloader. The current primary span runner remains `train_span.py`.

### Generative augmentation comparators

Generate bounded schema-aware CODE-ACCORD examples with the local Ollama service:

```bash
uv run python -m provenance.generate_accord_llm_aug \
  --dataset accord \
  --max-examples 120 \
  --out-jsonl results/accord_llm_aug_s42.jsonl
```

Generate the EntiGraph continued-pretraining corpus from the included, incomplete entity CSV:

```bash
uv run python -m provenance.generate_entigraph \
  --input data/code_accord/entities/train.csv \
  --output results/accord_entigraph.jsonl \
  --max-pairs-per-doc 5
```

Generate CycleGT data from SciERC, using either the local Hugging Face decoder or Ollama:

```bash
uv run python -m provenance.generate_cycle_data \
  --backend ollama \
  --max-triples 20 \
  --out-jsonl results/scierc_cycle_s42.jsonl
```

The generated JSONL is consumed by the primary runner's `--synth-jsonl` and `--synth-weight` controls. Record a matched gold-only command when comparing it.

The adjacent Stage 2e paraphrase experiment is retained separately from the closed-loop count. Generate base-Qwen paraphrases and run matched synthetic/gold-only training:

```bash
uv run python -m provenance.generate_paraphrase_dataset \
  --out-jsonl results/stage2e_paraphrase_s42.jsonl

uv run python -m provenance.train_stage2e \
  --synth-jsonl results/stage2e_paraphrase_s42.jsonl \
  --max-steps 1500 \
  --gold-only-steps 250 \
  --seed 42

uv run python -m provenance.train_stage2e \
  --synth-jsonl "" \
  --max-steps 1500 \
  --seed 42
```

To reproduce the LoRA-decoder version instead, supply the separately generated Stage 2c/d adapter:

```bash
uv run python -m provenance.generate_synth_dataset \
  --lora-dir checkpoints/stage2_009_lora_final \
  --max-triples 20 \
  --out-jsonl results/stage2e_lora_synth.jsonl
```

### Cooperative masking and pretraining

After acquiring a JSONL text corpus and a compatible span checkpoint, generate entity masks and run the cooperative ELECTRA treatment:

```bash
uv run python -m provenance.generate_entity_masks \
  --checkpoint checkpoints/scierc_span_best.pt \
  --input data/arxiv_real/cs_validation.jsonl \
  --output results/arxiv_entity_masks.jsonl

uv run python -m provenance.train_pretrain_cooperative \
  --entity-mask-jsonl results/arxiv_entity_masks.jsonl \
  --max-steps 1000 \
  --save-to checkpoints/electra_entity_mask_s42.pt
```

`data/arxiv_real.py`, `models/electra_generator.py`, and the primary encoder are transitive prerequisites. These commands preserve the treatment surface; historical settings/results must be read from the ledgers and Phase A provenance because the original checkpoints are absent.

### Real-data closed-loop negative-result families

Acquire `data/arxiv_real/cs_validation.jsonl` first. The retained runners expose the final configurable surfaces for Stage 2b/c/d, GAN, and Gumbel:

```bash
uv run python -m provenance.train_stage2b \
  --arxiv-jsonl data/arxiv_real/cs_validation.jsonl \
  --max-steps 1500 \
  --save-best-to checkpoints/stage2_007_best.pt

uv run python -m provenance.train_stage2c \
  --stage2b-ckpt checkpoints/stage2_007_best.pt \
  --arxiv-jsonl data/arxiv_real/cs_validation.jsonl \
  --max-steps 3000 \
  --save-adapters-to checkpoints/stage2_008_lora

uv run python -m provenance.train_stage2d \
  --stage2b-ckpt checkpoints/stage2_007_best.pt \
  --arxiv-jsonl data/arxiv_real/cs_validation.jsonl \
  --variant-tag v4

uv run python -m provenance.train_gan \
  --dataset scierc \
  --max-steps 2500 \
  --n-critic 3 \
  --save-best-to checkpoints/stage2_gan_s42.pt

uv run python -m provenance.train_gumbel \
  --dataset scierc \
  --max-steps 2500 \
  --tau 1.0 \
  --n-critic 1 \
  --save-best-to checkpoints/stage2_gumbel_s42.pt
```

These are evidence-bearing negative experiments, not recommended production training. They require large model downloads, `peft`, substantial GPU memory, and absent historical inputs/checkpoints. `models/critic.py`, `decoder_d.py`, `decoder_d_lora.py`, `encoder_reward.py`, and `triple_recovery.py` are retained only as transitive dependencies.

The paper's "38 variants" wording is not validated by file retention: the audit finds at most 34 unique interventions because some reported slots are controls, duplicates, or reruns. Stage 2e is adjacent and must not be counted as one of the coherent 38 slots.

### Traditional-Chinese negative ablations

Translation projection requires the complete CODE-ACCORD entity/relation CSVs. A bounded marker-survival check is:

```bash
uv run python -m provenance.zh_translate_project \
  --stage1 20 \
  --rel-path data/code_accord/relations/train.csv
```

The full projection uses `--project`, `--ent-path`, `--out-dir`, and optionally `--mix-native`. This path implements the retained 1,921/2,663 projection survival and negative translate-train comparison; the source datasets are not included.

DAPT can download and normalize a bounded sample, then perform its built-in 50-step smoke:

```bash
uv run python -m provenance.dapt_zh --stage1 20
```

For the reported full treatment, first run `--prep-data data/dapt_zh_laws`, then `--train --corpus data/dapt_zh_laws/corpus.txt --steps 3000` with the recorded starting checkpoint. Network access, model caches, law-source availability, and a suitable GPU are required.

### Archived result and retained command records

The former `results.tsv` and `results_stage2.tsv` are historical generated
experiment ledgers, not clean benchmark summaries. Their exact bytes and Git
identities are held in external archival storage; the
standalone source retains only their frozen secondary-evidence register.
`run_a19_cosine_probe.sh` is the exact nonportable DGX command record for the
A19 probe; inspect it before adapting paths, shell, CUDA, or environment assumptions.

## Offline regression tests

Run the publication-critical pure-function tests without downloading a model or dataset:

```bash
uv run --frozen python -B -m unittest discover -s tests -v
```

The suite covers immutable resumable acquisition, safe selective ZIP extraction,
strict CSV/BIO decoding, exhaustive gold-alignment hard stops, deterministic
dataset materialization fixtures, and historical helpers plus canonical
output-path containment, configuration/matrix invariants, typed directed matching, stable
candidate identities, deterministic `CODE-SPLIT-1`, leakage rejection,
zero-denominator behavior, correction/error handling, development-only pilot
pairing and two-call determinism, exact Wilcoxon/Holm and
paired-t calculations, paired hierarchical bootstrap replay, and manually
derived four-condition scoring. It does not substitute for checkpoint, dataset,
GPU, or Ollama evaluation.

## Outputs and reproducibility

Canonical Path A stages may write only beneath `output/<run-id>/`, including
downloads, caches, checkpoints, predictions, verifier records, metrics, tables,
manifests, and logs. The historical commands above still name `results/` and
`checkpoints/` solely to preserve their recorded interfaces; those locations
must not be used by the publication workflow. Existing confirmed generated
ledgers are moved to external archival storage only during the approved,
hash-verified finalization step.

For a reproducible run, record at minimum:

- source commit and command;
- dataset version, license, and split/hash;
- random seed and model identifier/revision;
- Python, PyTorch, CUDA, GPU, and dependency-lock versions;
- Ollama/model version and verifier settings when used;
- produced checkpoint/result hashes and exclusions.

The historical DGX notes describe a PyTorch 2.11/CUDA 13.0 environment, while the current lock targets a different PyTorch/CUDA combination. Do not represent either as the exact environment of every reported row without a per-run manifest.

## License boundary

The inherited repository declared the source code MIT, but this checkout does not yet include a standalone `LICENSE` file. Publication owners must verify the intended code license and add the corresponding license text before release. Third-party datasets, model weights, and generated data remain subject to their own licenses and terms.
