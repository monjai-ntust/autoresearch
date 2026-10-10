# Matched Table-1 encoder comparison

The fresh command coordinates four isolated scientific arms:

| Arm | Backbone | Recipe | Claim role |
|---|---|---|---|
| `bert-base-common` | BERT-base uncased | Historical common | Backbone sensitivity |
| `deberta-base-common` | DeBERTa-base | Historical common | Backbone sensitivity |
| `deberta-large-common` | DeBERTa-large | Historical common | Backbone/recipe reference |
| `deberta-large-a20-a21-a12` | DeBERTa-large | A20+A21+A12 | Same-backbone recipe comparison |

The bounded historical trainer/network remains separate from the canonical
trainer. Both consume canonical preparation and compatible shared utilities.
Frozen profile revisions, exact weight sizes/SHA-256, tokenizer files, recipes,
seeds and statistical declarations live once in
[`encoder-comparison.json`](../resources/configs/encoder-comparison.json).
Historical development statistics are not new fixed-test results.

```bash
uv run --frozen --no-sync python -I -B encoder_comparison.py validate
uv run --frozen --no-sync python -I -B encoder_comparison.py run --matrix-id comparison-001
```

The second command creates `output/comparison-001/` and four member roots named
`output/comparison-001-<arm>/`. Choose a matrix ID short enough that every derived
run ID satisfies the 64-character rule. Each arm acquires its own dataset/model
inputs; imported checkpoints or another run's cache are forbidden. Repeat the
command unchanged to continue validated durable progress.

## Automatic stage barriers

First all four arms complete authenticated seed-42 smokes: one CUDA step,
development evaluation, restart-state reload, selected-checkpoint round-trip,
one-record candidate ledger, frozen cache identity and exact base-parameter
count. Smoke and full training use separate processes; smoke weights never
initialize full training. The fresh lifecycle has no manual admission action,
pause or replacement admission artifact.

Then the four arms train seeds 42–49. All 32 development runs and selected
checkpoints must validate before any member performs final-test inference or
inspects test outcomes. Preparation may materialize the fixed held-out split.
The coordinator and private workers enforce identical gates. Matrix membership,
actual clean source/configuration identity and model hashes cannot change.

Each full training/inference invocation requires 200,000,000,000 free storage
bytes. The matrix enforces the frozen cumulative 48-hour full-execution bound
using elapsed accelerator-job wall time. Failed and resumed invocations are
charged; abandoned starts without finishes are conservatively charged through
reinvocation. This is a policy bound, not measured GPU utilization. Exhaustion
fails without replacing arms, resetting the budget or deleting evidence.
Bounded smoke retains its separate frozen policy.

## Deliberate scientific details

CODE-SPLIT-1, seeds 42–49, typed strict triples and development-only selection
are common. `NO_REL=1.0` belongs to historical CODE behavior. A20 uses linear
5×→2× decay and a one-time adaptive gate; its 3.5× middle value lasts one batch.
A21 applies span-NER label smoothing; A12 uses inter-span context. Initialized
heads unused by a recipe remain because changing initialization changes RNG.

Parameter counting verifies full pinned weight bytes before loading and sums
tensor elements. Only the pinned DeBERTa-large embedded `config` dictionary is
excluded as known metadata; unexpected metadata or tensor-free files fail.

## Reports and execution boundary

Each arm writes authenticated per-seed strict P/R/F1/counts, mean, sample SD,
tied best seeds and recorded training cost under `reports/comparison/`. The
coordinator references isolated reports under `reports/matrix/`. Predeclared but
unimplemented/unrecorded inferential measurements are unavailable; Phase I adds
no experiment or statistical procedure. Other-backbone rows support sensitivity,
not method-superiority claims.

Offline tests use source-grounded bytes and synthetic CPU fault fixtures. Live
CUDA behavior, exact parameter count and fresh full outcomes require external
execution and review. Existing partial G-05 runs remain on the pinned repair/
review workflow in [workflow.md](workflow.md); this command cannot adopt them.
