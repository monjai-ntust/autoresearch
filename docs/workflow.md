# Table-production workflow

`stages/pipeline.py` is the repository's only executable workflow entry point. Run it
from the root of a clean standalone source checkout. All runtime writes are
confined to the ignored `output/<run-id>/` tree.

## Run identity and same-run lineage

A run ID is a namespace, not a scientific identity. It contains 1-64 letters,
digits, `.`, `_`, or `-`, and begins with a letter or digit. Every stage uses
the same `RunLayout` and derives all paths beneath that run root.

Manifests bind the selected run to its source checkout, immutable input,
prepared split, checkpoint, candidates, development threshold, verifier
environment and verdicts, strict metrics, graphs, and Table-2 results. A valid
hash from another run is not reusable. Immutable upstream inputs are copied
into the selected run and authenticated there before consumption.

The clean source commit is the complete tracked-code identity. Checkout and
checkpoint manifests therefore do not duplicate per-Python-file or lockfile
hashes. The selected config retains its own digest because downstream Table-2
validation reads that exact historical config blob from the recorded commit.

## Environment

Requirements are declared in `pyproject.toml` and locked in `uv.lock`.

```bash
uv sync --frozen
```

The current environment uses Python 3.10 or newer, Torch 2.9.1 from the CUDA
13.0 index, Git, `curl`, and a loopback Ollama service. Checkout manifests log
actual versions and hardware. Hardware identity is comparison metadata, not a
startup gate.

The reference profile in `resources/contracts/table2.json` records the historical
hardware, encoder, and Qwen identities for reporting only. The default
configuration selects that encoder revision, while accepting any immutable
40-hex revision as the actual input for a new run. Its explicitly named Qwen
digests are references; the actual Qwen tag/blob digests are discovered at the
first verifier boundary. Within a run, the encoder revision must be identical
across training, checkpoints, candidate generation, and downstream consumers.
The Qwen tag/blob identity must be identical across both verifier modes and
Table 2. Internal disagreement fails closed.

## Static validation

```bash
uv run --frozen --no-sync python -I -B stages/pipeline.py validate-table2
uv run --frozen --no-sync python -B -m unittest discover -s tests -v
```

Validation makes no model call. Table-2 dry-run additionally authenticates one
completed run and deterministically constructs its prospective graph and record
projections without writing them:

```bash
uv run --frozen --no-sync python -I -B stages/pipeline.py table2 \
  --run-id research-run-001 \
  --dry-run
```

## Full canonical CODE-ACCORD pipeline

Start the configured Qwen model in Ollama. Then execute:

```bash
uv run --frozen --no-sync python -I -B stages/pipeline.py full \
  --run-id research-run-001 \
  --ollama-url http://localhost:11434
```

If the blob path reported by Ollama cannot be opened by the pipeline process,
add `--model-blob-source /absolute/path/to/blob`. That source cannot be located
inside this checkout's `output/` tree. Its bytes must match the blob reference
in Ollama's Modelfile.

The command runs or resumes these stages in order:

1. Checkout and environment doctor.
2. Secondary-evidence reconciliation.
3. Immutable CODE-ACCORD acquisition.
4. Leakage-safe CODE-SPLIT-1 preparation.
5. Eight encoder training runs and development candidate ledgers.
6. Eight-seed development assembly and threshold selection.
7. Predeclared development pilot selection and four same-run captures.
8. Pilot audit; final-test work remains blocked unless this passes.
9. Eight final-test candidate ledgers and assembly.
10. Simple and corrective verifier execution.
11. Strict scoring and publication metrics.
12. Same-run graph construction/projection and frozen Table-2 evaluation.

This route produces the canonical CODE-ACCORD successors to the encoder and
verifier analyses plus a new same-run Table 2. It does not regenerate the
paper's historical Table 1 or SciERC Table 3; those remain legacy comparison
evidence. Table 4 is outside this artifact.

Encoder training remains hosted by `stages/encoder.py`, but the main process starts
it through the private `stages/pipeline.py _train-encoder` route. `stages/encoder.py` is an
internal scientific implementation module and is not an executable entry point.

## Table-2-only route

For a run that already completed stages 1–11, execute only downstream Table 2:

```bash
uv run --frozen --no-sync python -I -B stages/pipeline.py table2 \
  --run-id research-run-001 \
  --ollama-url http://localhost:11434
```

This command has no encoder-training, candidate-generation, or verifier route.
It authenticates those existing outputs and derives the Qwen identity from the
same run's verifier environment. The live Ollama tag, registry manifest, model
blob, family, parameter size, and quantization must match that run-local
identity.

Three graphs are used: seed-42 confidence-filtered candidates at the
development-selected threshold, seed-42 correction-capable verifier emissions,
and the private-gold oracle. Missing graphs are constructed from the selected
run's authenticated parents. Existing graph manifests are reused only after
exact canonical equality with that reconstruction. Representation-only graph
and record projections feed the frozen evaluator.

## Resume and failure behavior

Completed upstream stages are reused only when their expected manifests,
identities, stage seals, and transitive input/output hashes still validate.
Training restart state includes optimizer, scheduler, random generators,
sampler, adaptive gate, and selection
state. Interrupted inference ledgers and verifier responses are first
validated, preserved under `inputs/recovery/`, and bound to the selected run's
checkout and stage inputs before reuse. Invalid partial files are quarantined
and never used as caches. Missing seals may be reconstructed only after the
complete producer manifest and every declared output validate.
Development-pilot calls never use recovery caches.

Table-2 stages record a run identity before their first write. A validated
completed stage is reusable. An unverified interrupted result is moved into
`output/<run-id>/table2-q105/recovery/` before a new attempt. Cross-run paths,
symlinks/junctions, changed inputs, mismatched model identities, malformed or
empty model responses, and output collisions fail closed.

## Primary artifacts

Important run-relative paths include:

- `manifests/00-checkout-manifest.json`
- `manifests/03-data-preparation-manifest.json`
- `data-prepared/{train,development,test}.jsonl`
- `data-prepared/{development,test}-gold.jsonl`
- `checkpoints/seed-<seed>/checkpoint-manifest.json`
- `predictions/dev/threshold-selection.json`
- `predictions/test/candidates.jsonl`
- `verifier/{simple,corrective}/environment-manifest.json`
- `verifier/{simple,corrective}/verdicts.jsonl`
- `metrics/metrics.json`
- `manifests/score-manifest.json`
- `table2-q105/projections/`
- `table2-q105/rag-results/`
- `table2-q105/table2-results.json`
- `table2-q105/artifact-hashes.json`

Generated output is temporary working storage. The parent research repository's
verified archival process must move a completed run out of `src/output/` before
final release.
