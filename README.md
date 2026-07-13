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

The CLI retains reported controls for synthetic/CAST/CycleGT data, relation replay, checkpoint initialization, evidence-graph fusion, contrastive and focal losses, BIO enrichment, class weighting, marker representations, and zh-Hant experiments. Defaults and matching behavior are intentionally preserved for provenance.

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

## Retained experimental families

The following paths remain because they implement or explain reported experiments:

| Family | Main paths |
|---|---|
| Initial and cross-dataset baselines | `train_stage2.py`, `train_multi.py`, `eval_checkpoint.py` |
| Pseudo-label and CAST | `generate_pseudo_labels*.py`, `train_stage2_curriculum.py`, `train_stage2_multiround.py` |
| Real-data closed-loop attempts | `train_stage2b.py`, `train_stage2c.py`, `train_stage2d.py`, `train_stage2e.py`, `train_gan.py`, `train_gumbel.py` |
| Cooperative masking/pretraining | `train_pretrain_cooperative.py`, `generate_entity_masks.py` |
| CycleGT and augmentation | `generate_cycle_data.py`, `generate_paraphrase_dataset.py`, `generate_synth_dataset.py`, `generate_accord_llm_aug.py`, `generate_entigraph.py` |
| zh-Hant transfer | `zh_translate_project.py`, `dapt_zh.py`, zh-Hant options in `train_span.py` |

`results.tsv` and `results_stage2.tsv` are retained experiment ledgers rather than newly generated benchmark summaries. Many historical scripts expect checkpoints, generated JSONL, raw data, external models, or DGX/POSIX paths that are not present in a clean checkout.

`program.md` is retained as the exact project-modified autonomous experiment-controller prompt. It is historical provenance, not a safe supported runner: it includes obsolete assumptions, environment-specific paths, and destructive keep/discard instructions. `run_a19_cosine_probe.sh` is similarly an exact nonportable DGX command record. `bench_gpu.py` is a BERT forward/backward smoke check, not evidence for the paper's resource or verifier timing claims.

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
