# Phase B standalone Path A workflow

This document is the source-only execution handoff for approved protocol
`B04-PATH-A-1.3` and workflow `PATH-A-WORKFLOW-1.3`. It does not depend on the
parent repository or `research-logs`.

## Current implementation boundary

The B-05 core slice implements one public module entry point with two tested
stages:

- `doctor` validates the direct source checkout, exact Python/uv requirements,
  Git cleanliness, root-anchored output ignore behavior, configuration/matrix
  identities, and tracked artifact hashes. It creates only
  `output/<run-id>/...` and records `manifests/00-checkout-manifest.json`.
- `score` validates frozen gold, candidate, simple-verdict, corrective-verdict,
  and development-threshold records; applies typed directed matcher
  `CODE-STRICT-1`; and writes candidate outcomes, sentence outcomes, confusion
  counts, and micro metrics beneath the same run.

The generic `CODE-SPLIT-1` iterative multilabel assignment primitive is also
implemented and tested for exact size, disjointness, repeatability, and input
order independence. Acquisition and full CODE-ACCORD parsing/materialization,
deterministic checkpoint rebuilding, inference, threshold selection, live
verifier calls, publication rendering, and parent-side archival are not
implemented by this slice. The historical standalone scripts remain provenance
paths and are not publication
commands for Path A. Expensive training and final-test Qwen execution remain
blocked until the approved B-07 go/no-go checkpoint.

## 1. Prepare the standalone checkout

Clone the source repository directly, check out the release commit, and run from
its root. Install exactly uv 0.11.26; uv must then provision Python 3.10.20 from
the lockfile:

```bash
uv sync --frozen --python 3.10.20
```

Do not copy a parent-repository path into the configuration. The sole writable
runtime root is `output/`, which is root-anchored in `.gitignore`.

## 2. Validate the checkout and open one run

Choose a new stable run ID. A run ID is 1-64 characters, begins with an
alphanumeric character, and otherwise uses only letters, digits, `.`, `_`, and
`-`.

```bash
uv run --frozen --python 3.10.20 python -B -m phase_b_pipeline \
  doctor \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example
```

The command exits 0 only when all implemented preflight checks pass. It exits 2
and writes a blocked checkout manifest when the environment or checkout is not
compliant. It refuses an existing run ID and rejects symlink, junction,
absolute-path, and `..` escapes.

The checkout manifest always reports that publication execution is not yet
admitted because B-07 approval has not occurred. A passing B-05 doctor is not
permission to run the gated training or final-test verifier work.

## 3. Supply immutable offline-scoring inputs

Until the acquisition/inference/verifier stages are implemented, place the
following externally produced, schema-valid files at their configured paths
inside the selected run:

```text
output/path-a-example/
├── data-prepared/test-gold.jsonl
├── predictions/dev/threshold-selection.json
├── predictions/test/candidates.jsonl
├── verifier/simple/verdicts.jsonl
└── verifier/corrective/verdicts.jsonl
```

Every JSONL record must end with a newline. Files must use stable ordering:

- gold by `example_id`;
- candidates by `training_seed`, `example_id`, `candidate_id`; and
- each verdict file by `training_seed`, `candidate_id`.

The machine contracts are in `schemas/phase_b/records.schema.json`. In addition
to schema validation, the runner recomputes every candidate ID from protocol,
seed, and its typed directed strict key. SHA-256 identities are mandatory.

The threshold file must conform to
`schemas/phase_b/threshold-selection.schema.json`. It retains hashes for the
development candidate index, development gold, and split manifest, plus one
ordered row for every threshold `0.00, 0.05, ..., 0.95`. Every row contains the
eight per-seed development strict-Triple F1 values and their arithmetic mean.
The scorer recomputes every mean and selects the largest; an exact tie must use
the higher threshold. It rejects an unproven selected value or any assertion
that test labels were used.

## 4. Reproduce offline strict scoring

```bash
uv run --frozen --python 3.10.20 python -B -m phase_b_pipeline \
  score \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example
```

The scorer writes, without overwriting an existing artifact:

```text
outcomes/candidate-outcomes.jsonl
outcomes/sentence-outcomes.jsonl
metrics/metrics.json
manifests/score-manifest.json
```

All four conditions consume the identical candidate universe. `VER-RAW` keeps
every candidate. `VER-CONFIDENCE` applies the development-selected threshold.
`VER-SIMPLE` keeps only `KEEP`. In `VER-CORRECTIVE`, `KEEP` judges and emits the
original candidate; `CORRECT` judges the original invalid and emits only one
already source-mapped, schema-valid corrected triple. Missing, malformed,
uncertain, timeout, and exhausted-transport records predict original-invalid
and emit no triple.

Candidate scoring retains raw candidate identities. End-to-end scoring collapses
duplicate strict keys within a seed/sentence/condition, records the duplicate
count, intersects typed directed sets per sentence, and only then aggregates
micro counts. A zero denominator produces JSON `null` with an explicit status;
it never silently becomes zero.

When all eight seeds are present, scoring also performs the predeclared 10,000
paired hierarchical seed/document bootstrap, exact signed-rank tests, paired
t-test sensitivity checks, and separate Holm adjustments. Incomplete seed
coverage is retained as a development result with statistics explicitly blocked.

The manifest stores only run-relative paths plus hashes. A second invocation on
the same run refuses to replace the first result. To compare a rerun, use a new
run ID and compare its input/output hashes.

The `-B` flag is mandatory for canonical commands: it prevents Python from
creating `__pycache__` files in the tracked source area. `doctor` checks this
before declaring the environment compliant.

## 5. Reproducibility interpretation

This core slice supports deterministic offline result replay once frozen inputs
exist. It does not yet establish exact same-seed checkpoint rebuilding. The
final workflow must additionally fetch and verify CODE-ACCORD and DeBERTa,
reconstruct `CODE-SPLIT-1` byte-identically, serialize full restart state,
duplicate seed-42 training/inference under the pinned accelerator profile,
rebuild seeds 43-49, freeze development/test inference, materialize and hash the
approved prompts, pass the development verifier pilot, score uncertainty/tests,
and render/archive the final output. Those stages will be added to this same
module entry point as their B-05/B-06 gates are completed.
