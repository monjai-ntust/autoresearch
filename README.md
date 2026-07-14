# Low-resource joint entity and relation extraction artifact

This repository contains the research implementation for a low-resource knowledge-graph construction pipeline. It trains a joint named-entity recognition (NER) and relation extraction (RE) encoder, extracts confidence-bearing triples, optionally verifies or corrects them with an Ollama-hosted LLM, and builds a provenance-bearing graph.

The recommended artifact surface is:

```text
train_span.py
    -> inference_kg.py
    -> verify_triples_llm.py
    -> build_kg.py
```

The repository also retains scripts for reported baselines, ablations, closed-loop attempts, cross-dataset transfer, and negative results. Those historical paths are evidence, not all recommended starting points.

## Repository status

- Python 3.10 is the selected and minimum supported Python version.
- The committed lockfile describes the current source environment; it is not an exact manifest of every historical run.
- Historical raw logs and model checkpoints are not included.
- SciERC, CoNLL04, ADE, and arXiv acquisition scripts are included. Complete CODE/ACCORD, SciER, CUAD, and zh-Hant datasets are not included.
- Missing verifier Precision, Recall, and F1 evaluation is Phase B work and is not claimed as complete here.

## Installation

Install [uv](https://docs.astral.sh/uv/), then create the locked environment:

```bash
uv sync
```

The tracked support files are part of the reproducibility contract:

| Path | Role |
|---|---|
| `.python-version` | Selects Python 3.10 for compatible environment managers. |
| `pyproject.toml` | Declares the package metadata and direct dependencies used by `uv sync`. |
| `uv.lock` | Locks the resolved environment used for validation of this checkout. |
| `.gitignore` | Excludes local datasets, checkpoints, caches, and generated outputs; it intentionally does not add a log rule. |

PyTorch is resolved from the CUDA 12.8 wheel index. CPU execution is possible for small checks, but reported training runs require a suitable CUDA GPU and substantially more time and memory.

Model backbones are loaded through Hugging Face Transformers and may require network access on first use. The verifier and generation scripts additionally require `curl`, a reachable Ollama-compatible `/api/chat` endpoint, and the requested local model (historically `qwen3:32b`). `--ollama-url` is configurable, so on-premise execution is a deployment choice rather than a code-enforced invariant.

## Data

Run acquisition scripts from the repository root:

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

## Train the primary encoder

The primary runner supports `scierc`, `scier`, `conll04`, `ade`, `accord`, and `cuad`. This minimal SciERC command trains the gold-only span model and saves its best checkpoint:

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

## Extract, verify, and build a graph

First extract triples from a compatible checkpoint:

```bash
uv run python inference_kg.py \
  --checkpoint checkpoints/scierc_span_best.pt \
  --dataset scierc \
  --split test \
  --out-jsonl results/scierc_inference.jsonl
```

Optionally verify and correct the predicted triples with Ollama:

```bash
uv run python verify_triples_llm.py \
  --input results/scierc_inference.jsonl \
  --output results/scierc_verified.jsonl \
  --ollama-url http://localhost:11434 \
  --ollama-model qwen3:32b \
  --mode correct
```

Build the graph, optionally applying the retained neuro-symbolic rules:

```bash
uv run python build_kg.py \
  --input results/scierc_verified.jsonl \
  --output results/scierc_kg.json \
  --filter-mode verified \
  --use-rules
```

`verify_triples_llm.py` uses deterministic decoding (`temperature=0`, thinking disabled), per-request timeout/retry controls, and append-style JSONL output. It still requires a separately designed, leakage-safe labeled evaluation before verifier performance can be reported.

## Evaluation semantics

`eval/triple_f1.py` provides the shared evaluation functions used by retained training paths:

- NER spans require exact boundary and entity-type agreement.
- Gold-span RE evaluates relation labels over gold entity spans.
- End-to-end triple matching requires exact head span, tail span, and relation label; for historical compatibility, it does not include entity type in the triple key.
- `train_span.py` selects checkpoints using `triple_f1` by default and can use `ner_f1` for NER-oriented pretraining.

The historical `eval_graph_rag.py` script is a diagnostic, not a leakage-free downstream benchmark: it derives questions and reference answers from the same gold-bearing source. `diagnose_evidence_paths.py` reports structural evidence-path coverage and likewise should not be interpreted as task accuracy.

## Retained evaluation and historical reproduction steps

These paths are retained because they produce a draft-reported comparison, a counted negative-result family, or a user-approved diagnostic. Historical commands require the datasets, checkpoints, model caches, and services named below; those artifacts are not bundled. Start with small limits and a separate output path before attempting a recorded full run.

### Hardware and evaluation diagnostics

Run the installation smoke benchmark after model download/cache setup:

```bash
uv run python bench_gpu.py
```

This fixed BERT forward/backward smoke is useful for detecting CPU fallback and gross environment problems. Its timings are not the paper's encoder, verifier, or full-pipeline measurements.

Evaluate a legacy SciERC token-model checkpoint whose dictionary contains an `encoder` state:

```bash
uv run python eval_checkpoint.py checkpoints/cast2500_seed44_best.pt
```

The named checkpoint is historical and not included. For the retained Graph RAG diagnostic, first build a KG and retain inference JSONL with `gold_triples`, then run:

```bash
uv run python eval_graph_rag.py \
  --kg results/scierc_kg.json \
  --gold-jsonl results/scierc_inference.jsonl \
  --max-questions 10 \
  --output results/scierc_graph_rag_diagnostic.json
```

This reproduces the diagnostic protocol behind the draft table, not a leakage-free compliance benchmark. The question sample and exact inference/KG inputs must be recorded with any reported result.

### Cross-dataset BIO comparator

The older token-BIO runner is retained for the SciERC/CoNLL04/ADE comparator and NER/RE/full-Triple decomposition lineage:

```bash
uv run python train_multi.py \
  --dataset scierc \
  --max-steps 1500 \
  --seed 42 \
  --save-best-to checkpoints/scierc_token_bio_s42.pt
```

Repeat with `--dataset conll04` or `--dataset ade` only after running the corresponding downloader. The current primary span runner remains `train_span.py`.

### Generative augmentation comparators

Generate bounded schema-aware CODE-ACCORD examples with the local Ollama service:

```bash
uv run python generate_accord_llm_aug.py \
  --dataset accord \
  --max-examples 120 \
  --out-jsonl results/accord_llm_aug_s42.jsonl
```

Generate the EntiGraph continued-pretraining corpus from the included, incomplete entity CSV:

```bash
uv run python generate_entigraph.py \
  --input data/code_accord/entities/train.csv \
  --output results/accord_entigraph.jsonl \
  --max-pairs-per-doc 5
```

Generate CycleGT data from SciERC, using either the local Hugging Face decoder or Ollama:

```bash
uv run python generate_cycle_data.py \
  --backend ollama \
  --max-triples 20 \
  --out-jsonl results/scierc_cycle_s42.jsonl
```

The generated JSONL is consumed by the primary runner's `--synth-jsonl` and `--synth-weight` controls. Record a matched gold-only command when comparing it.

The adjacent Stage 2e paraphrase experiment is retained separately from the closed-loop count. Generate base-Qwen paraphrases and run matched synthetic/gold-only training:

```bash
uv run python generate_paraphrase_dataset.py \
  --out-jsonl results/stage2e_paraphrase_s42.jsonl

uv run python train_stage2e.py \
  --synth-jsonl results/stage2e_paraphrase_s42.jsonl \
  --max-steps 1500 \
  --gold-only-steps 250 \
  --seed 42

uv run python train_stage2e.py \
  --synth-jsonl "" \
  --max-steps 1500 \
  --seed 42
```

To reproduce the LoRA-decoder version instead, supply the separately generated Stage 2c/d adapter:

```bash
uv run python generate_synth_dataset.py \
  --lora-dir checkpoints/stage2_009_lora_final \
  --max-triples 20 \
  --out-jsonl results/stage2e_lora_synth.jsonl
```

### Cooperative masking and pretraining

After acquiring a JSONL text corpus and a compatible span checkpoint, generate entity masks and run the cooperative ELECTRA treatment:

```bash
uv run python generate_entity_masks.py \
  --checkpoint checkpoints/scierc_span_best.pt \
  --input data/arxiv_real/cs_validation.jsonl \
  --output results/arxiv_entity_masks.jsonl

uv run python train_pretrain_cooperative.py \
  --entity-mask-jsonl results/arxiv_entity_masks.jsonl \
  --max-steps 1000 \
  --save-to checkpoints/electra_entity_mask_s42.pt
```

`data/arxiv_real.py`, `models/electra_generator.py`, and the primary encoder are transitive prerequisites. These commands preserve the treatment surface; historical settings/results must be read from the ledgers and Phase A provenance because the original checkpoints are absent.

### Real-data closed-loop negative-result families

Acquire `data/arxiv_real/cs_validation.jsonl` first. The retained runners expose the final configurable surfaces for Stage 2b/c/d, GAN, and Gumbel:

```bash
uv run python train_stage2b.py \
  --arxiv-jsonl data/arxiv_real/cs_validation.jsonl \
  --max-steps 1500 \
  --save-best-to checkpoints/stage2_007_best.pt

uv run python train_stage2c.py \
  --stage2b-ckpt checkpoints/stage2_007_best.pt \
  --arxiv-jsonl data/arxiv_real/cs_validation.jsonl \
  --max-steps 3000 \
  --save-adapters-to checkpoints/stage2_008_lora

uv run python train_stage2d.py \
  --stage2b-ckpt checkpoints/stage2_007_best.pt \
  --arxiv-jsonl data/arxiv_real/cs_validation.jsonl \
  --variant-tag v4

uv run python train_gan.py \
  --dataset scierc \
  --max-steps 2500 \
  --n-critic 3 \
  --save-best-to checkpoints/stage2_gan_s42.pt

uv run python train_gumbel.py \
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
uv run python zh_translate_project.py \
  --stage1 20 \
  --rel-path data/code_accord/relations/train.csv
```

The full projection uses `--project`, `--ent-path`, `--out-dir`, and optionally `--mix-native`. This path implements the retained 1,921/2,663 projection survival and negative translate-train comparison; the source datasets are not included.

DAPT can download and normalize a bounded sample, then perform its built-in 50-step smoke:

```bash
uv run python dapt_zh.py --stage1 20
```

For the reported full treatment, first run `--prep-data data/dapt_zh_laws`, then `--train --corpus data/dapt_zh_laws/corpus.txt --steps 3000` with the recorded starting checkpoint. Network access, model caches, law-source availability, and a suitable GPU are required.

### Retained result and command records

`results.tsv` and `results_stage2.tsv` are historical experiment ledgers, not newly generated benchmark summaries. They preserve negative as well as positive rows and must not be treated as a clean 38-row manifest. `run_a19_cosine_probe.sh` is the exact nonportable DGX command record for the A19 probe; inspect it before adapting paths, shell, CUDA, or environment assumptions.

## Offline regression tests

Run the publication-critical pure-function tests without downloading a model or dataset:

```bash
uv run --locked python -m unittest discover -s tests -v
```

The suite covers BIO decoding and Precision/Recall/F1 arithmetic, ontology-rule filtering, graph entity normalization/clustering/filter modes, bounded evidence paths, and simple/corrective LLM verdict parsing. It does not substitute for checkpoint, dataset, GPU, or Ollama evaluation.

## Outputs and reproducibility

Use `results/` for generated metrics, inference JSONL, verified JSONL, graph JSON, and diagnostics; use `checkpoints/` for model weights. Both locations are local outputs and should not be committed unless a specific small artifact is necessary, redistributable, and documented.

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
