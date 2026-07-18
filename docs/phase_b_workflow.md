# Phase B standalone Path A workflow

This document is the source-only execution handoff for approved protocol
`B04-PATH-A-1.3` and workflow `PATH-A-WORKFLOW-1.3`. It does not depend on the
parent repository or `research-logs`.

## Current implementation boundary

The B-05/B-06, development-only B-07, and B-05U slice implements one public
module entry point with nine stages and focused contract tests:

- `doctor` validates the direct source checkout, exact Python/uv requirements,
  Git cleanliness, root-anchored output ignore behavior, configuration/matrix
  identities, and tracked artifact hashes. It creates only
  `output/<run-id>/...` and records `manifests/00-checkout-manifest.json`.
- `fetch` downloads the immutable Zenodo CODE-ACCORD v1.0.0 archive with safe
  partial-file resume, then requires exactly 101,265,616 bytes and MD5
  `57e2efa465f41e2f582db62810fb50f5` before promotion. It also records SHA-256
  and license provenance in `manifests/02-input-acquisition-manifest.json`.
- `reconcile` verifies the two frozen historical result ledgers and their known
  physical structure, checks stable evidence/absence anchors for seven draft
  Section 5 claim families, and emits a secondary-only audit without repairing
  or promoting any historical result.
- `prepare` safely extracts only the six required annotation CSVs, verifies
  their exact byte/SHA-256 identities, parses all rows, audits the one approved
  UUID repair, and attempts typed directed `CODE-STRICT-1` reconstruction plus
  byte-identical `CODE-SPLIT-1` materialization.
- `model train` (`--execution dry-run`) is the canonical successor to
  `train_span.py` for training orchestration. It validates the frozen recipe and
  seed and writes a machine-readable training plan plus the checkpoint-manifest
  contract that gated live training must satisfy; live training under the pinned
  accelerator profile is not exposed by this slice.
- `model generate-candidates` is the canonical successor to `inference_kg.py`.
  Its `dry-run` execution validates prepared sentences plus an encoder
  checkpoint identity and writes a per-sentence inference plan without calling a
  model; its `replay` execution transforms a frozen encoder prediction ledger
  into typed `CODE-STRICT-1` candidates, reproducing the greedy non-overlapping
  entity selection and `min(head, tail) * re` softmax-product confidence on
  typed CODE spans. Live encoder inference remains an externally gated stage and
  is not exposed by this slice.
- `select-threshold` is the canonical producer of the frozen development
  confidence threshold (VER-CONFIDENCE). It computes the per-seed development
  strict Triple F1 over the frozen grid from the eight-seed development candidate
  universe and development gold, averages across seeds, and selects the argmax
  with the higher-threshold tie rule, never inspecting the test labels. It emits
  the `threshold-selection.json` the `score` stage consumes.
- `verifier` materializes the exact `CODE-VERIFIER-1` request universe, supports
  explicit `dry-run`, optional hash-gated `live`, and model-free `replay`
  execution, and retains environment, raw-response, retry, cache, token, and
  latency evidence under the selected run.
- `pilot-verifier` audits two independently captured response ledgers for each
  verifier mode against development-only sentences, typed gold, candidates,
  and the frozen development threshold. It checks exact coverage, source-only
  prompt evidence, response schemas, cache exclusion, source-grounded
  corrections, and normalized two-call determinism. It never admits publication
  execution.
- `score` validates frozen gold, candidate, simple-verdict, corrective-verdict,
  and development-threshold records; applies typed directed matcher
  `CODE-STRICT-1`; and writes candidate outcomes, sentence outcomes, confusion
  counts, and micro metrics beneath the same run.

The generic `CODE-SPLIT-1` iterative multilabel assignment primitive is tested
for exact size, disjointness, repeatability, and input-order independence.
Official-data preparation preserves a raw-provenance view before producing
typed-strict gold: nine of 6,658 positive-relation marker arguments cannot
resolve to one authoritative typed BIO span. `prepare` inventories all nine in
`data-prepared/gold-alignment-audit.json`, preserves every raw row, and makes
only those nine rows ineligible for typed strict gold; it does not project,
expand, or manually type an endpoint. Deterministic checkpoint
rebuilding, inference, threshold selection, the actual model-backed development
pilot, publication rendering, and parent-side archival are not completed by
this slice. Historical standalone scripts remain provenance paths and are not
publication commands for Path A. Expensive publication training and final-test
Qwen execution remain blocked until the B-07 evidence exists and the user makes
the required go/no-go decision.

[`Phase C migration plan`](phase_c_migration.md) records the raw/typed design
lineage and its remaining execution gates. B-05U approves its materialization,
not training, final-test access, live verifier calls, or reporting a result.

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
uv run --frozen --python 3.10.20 python -B phase_b.py \
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

## 3. Reconcile the frozen Section 5 ledgers

This stage is read-only with respect to the ledgers and does not require the
CODE-ACCORD download:

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  reconcile \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example
```

The register at `configs/phase_b_section5_evidence.json` binds both ledgers by
UTF-8 content normalized to LF, so clean Linux and Windows checkouts reconcile
the same historical text. The audit separately records the raw checkout hash,
byte count, and newline style. It validates and retains the two blank physical
lines and two concatenated two-record lines in `results.tsv`; normalization is
never used to rewrite the source files.

The output is `audit/section5-evidence-reconciliation.json`. Its authority is
always `secondary_only`, its canonical-ready count is zero, and publication
execution remains unadmitted. The command exits 2 on identity, structure,
anchor, or overwrite drift. Missing final-recipe, cross-dataset, verifier, and
zh-Hant evidence must be supplied through separately hashed provenance or
regenerated under the canonical framework; it is not inferred from these
ledgers.

## 4. Fetch the immutable CODE-ACCORD prerequisite

The fetch stage requires the passing checkout manifest created by `doctor` in
the same run:

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  fetch \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example
```

Network failures retain a run-local `.partial` file. Repeating `fetch` with the
same run ID resumes when the server honors byte ranges; it restarts safely when
the server sends a complete response. Neither a partial nor a complete archive
is accepted until its size and upstream MD5 match. The archive remains beneath
`output/<run-id>/inputs/cache/` and is not committed.

## 5. Audit and prepare CODE-ACCORD

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  prepare \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example
```

Under protocol `B04-PATH-A-1.3`, the official v1.0.0 data deterministically
materializes a raw-provenance and typed-strict dataset. The known immutable
audit contains 6,644 exact marker-to-BIO alignments, five markers contained in
one typed span, and nine unresolved arguments across nine rows/eight sentences;
all nine rows belong to the official entity-training partition. Three markers
overlap multiple typed spans, three partially overlap one span, and three
overlap none. `raw-relation-provenance.jsonl` preserves all 4,329 rows;
`typed-strict-eligibility.jsonl` records eligibility for the 3,329 positive
rows; the typed view contains 3,320 eligible rows and 3,319 unique directed
typed triples. Do not edit the downloaded CSVs or treat the audit as permission
to omit raw evidence.

The approved dual-view stage emits deterministic `data-prepared/`, the repair
ledger, split/data manifests, test gold, distributions, attribution, and two
independently generated trees whose inventories and tree hashes must be
identical before atomic promotion.

## 6. Plan, execute, or replay the frozen verifier

The tracked UTF-8 prompt files and response schemas define
`CODE-VERIFIER-1`. A dry run validates prepared sentences and candidates and
writes the exact request payloads, prompt/model/decoding hashes, environment
manifest, append-only run log, and stage manifest without contacting Ollama or
writing a verdict:

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  verifier \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example \
  --mode simple \
  --execution dry-run
```

Use a distinct run ID to replay a frozen live-response ledger. The ledger must
be copied beneath that run first; its cache keys must match every newly
materialized request exactly. Replay performs no HTTP request and emits the
schema-valid verdict file consumed by `score`:

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  verifier \
  --config configs/phase_b_path_a.json \
  --run-id path-a-simple-replay \
  --mode simple \
  --execution replay \
  --response-ledger inputs/frozen-simple-responses.jsonl
```

Run the corrective condition separately with `--mode corrective`; it uses a
different prompt hash and response schema. Corrected entity text is admitted
only when it maps to one exact contiguous token sequence in the supplied
sentence. Absent, repeated, same-as-original, malformed, and schema-invalid
rewrites are retained with distinct failure status and emit no corrected triple.

Live execution is implemented but remains scientifically gated. It additionally
requires the complete 20,201,240,160-byte model blob beneath the run and exactly
one predeclared development warm-up candidate. It hashes the full blob, verifies
the Ollama tag manifest, Qwen3/32.8B/Q4_K_M details, and CLI modelfile reference,
then retains the warm-up separately and excludes its latency from observations:

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  verifier \
  --config configs/phase_b_path_a.json \
  --run-id path-a-simple-live \
  --mode simple \
  --execution live \
  --model-blob inputs/ollama/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312
```

The configured warm-up paths are
`data-prepared/development.jsonl` and
`predictions/dev/verifier-warmup-candidate.jsonl`. Live calls use at most three
identical attempts, 300 seconds per attempt, and deterministic 2/4-second
backoff. Cache hits are marked and never counted as new latency observations.
The JSON-schema model responses, raw-body hashes, attempt telemetry, verdicts,
environment manifest, and run log remain under `verifier/<mode>/`. The run-log
contract is `schemas/phase_b/verifier-run-log.schema.json`; request/response
replay and stage-manifest contracts are `verifier-replay.schema.json` and
`verifier-manifest.schema.json` in the same schema directory.

These commands do not override the remaining B-07/B-08
execution gates. In the current checkout they are code-path validation and
external handoff surfaces, not permission to inspect or call the final test set.

## 7. Audit the development-only verifier pilot

This stage consumes, but does not create, the B-07 pilot evidence. The full
eight-file development candidate index is distinct from the small pilot
candidate subset. Freeze `candidate-index.json`, select the pilot subset without
test labels, and write `pilot-selection.json` before any live call. The selection
manifest binds the authoritative split, full index, exact subset hash, candidate
IDs, examples, seed coverage, and selection rule.

Use four distinct clean run IDs to execute two live calls per mode. Each call
must receive the same predeclared selection manifest as an explicit input:

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  verifier \
  --config configs/phase_b_path_a.json \
  --run-id path-a-pilot-simple-1 \
  --mode simple \
  --execution live \
  --sentences data-prepared/development.jsonl \
  --candidates predictions/dev/pilot-candidates.jsonl \
  --pilot-selection predictions/dev/pilot-selection.json \
  --model-blob inputs/ollama/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312
```

Repeat as `path-a-pilot-simple-2`, then use two more run IDs for `corrective`.
Do not resume from a response cache. Preserve each complete run. In the audit
run, copy its checkout/stage manifests and every verifier-stage output beneath
`inputs/pilot/captures/<mode>-repeat-<n>/` while preserving its original
run-relative paths. The 20 GB model-blob input remains content-addressed in its
original run and is proven by the live preflight/environment/stage identities;
do not duplicate it four times merely to assemble the audit. Then create the
four-entry index defined by
`schemas/phase_b/verifier-pilot-captures.schema.json`. The auditor verifies the
copied checkout, stage, environment, request, response, run-log, warm-up, model,
and verdict evidence against the source manifests; a response ledger alone is
not pilot evidence.

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  pilot-verifier \
  --config configs/phase_b_path_a.json \
  --run-id path-a-development-pilot \
  --evidence-class development-pilot \
  --capture-index inputs/pilot/capture-index.json
```

The command proves every sentence belongs to the hashed authoritative
development split, the threshold binds development gold plus the full
eight-file candidate index, and the pilot subset is a pre-call subset of that
index with source candidates from seeds 42–49. It reconstructs the canonical
requests; opens and hash-verifies every indexed seed file; verifies four
distinct clean, completed, pinned-Qwen live runs under one identical runtime;
and proves that each capture used the fixed warm-up input and no cache ledger.
It replays the captured responses and requires record-equivalent verdicts,
validates retry timing against exact run-log events and source-grounded
corrections, and compares only normalized scientific decision fields between
repeats. It emits development-only class balance, gold/verifier disagreement
diagnostics for all four conditions, strict-gold correction and transition
diagnostics, four normalized verdict ledgers, and
`audit/verifier-pilot/pilot-audit.json`, conforming to
`schemas/phase_b/verifier-pilot-audit.schema.json`.

A passing real pilot is reported only as `eligible_for_user_review`; the audit
always records `publication_execution_admitted: false` and requires a user
go/no-go decision. `synthetic-fixture` is available only for contract testing
and is always reported as `blocked_synthetic_fixture`, even when every check
passes. Completed captures with failed determinism, invalid responses, cache
reuse, or ungrounded corrections produce a no-go audit instead of silently
dropping candidates. After the CLI has resolved the configured run-relative
paths, structural input/provenance failures raised inside the auditor retain
`audit/verifier-pilot/pilot-failure.json` under a separate failure schema.
Missing or escaping CLI paths fail before the audit stage starts and therefore
do not claim an in-stage failure artifact.

The current official workflow has no regenerated model inputs for this command:
the dual view is materialized by code but no official run has yet fetched the
archive, regenerated checkpoints/development candidates, or selected a
threshold. This machine also lacks the pinned Ollama model/runtime.
The implemented auditor therefore validates the handoff contract; it is not a
claim that the actual B-07 pilot has run.

## 8. Supply immutable offline-scoring inputs

`predictions/test/candidates.jsonl` can now be produced inside the run by
`model generate-candidates --execution replay` from a frozen encoder prediction
ledger and checkpoint manifest, rather than supplied externally:

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  model generate-candidates \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example \
  --execution replay \
  --checkpoint-manifest checkpoints/seed-42/checkpoint-manifest.json \
  --prediction-ledger predictions/test/prediction-ledger.jsonl
```

The generated candidates are publication inputs only when the prediction ledger
and checkpoint identity were themselves produced by the gated live encoder
inference stage; a synthetic or externally supplied ledger yields development
evidence only. Until live inference exists, place the remaining
externally produced, schema-valid files at their configured
paths inside a separate development run only. Such files are not publication
inputs unless their provenance and hashes satisfy the frozen protocol:

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

The `predictions/dev/threshold-selection.json` file is now produced inside the
run by the `select-threshold` stage from the eight-seed development candidate
universe and development gold, rather than supplied externally:

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
  select-threshold \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example \
  --candidates predictions/dev/development-candidates.jsonl
```

The threshold file must conform to
`schemas/phase_b/threshold-selection.schema.json`. It retains hashes for the
development candidate index, development gold, and split manifest, plus one
ordered row for every threshold `0.00, 0.05, ..., 0.95`. Every row contains the
eight per-seed development strict-Triple F1 values and their arithmetic mean.
The scorer recomputes every mean and selects the largest; an exact tie must use
the higher threshold. It rejects an unproven selected value or any assertion
that test labels were used.

Verdict records must carry the exact configured prompt-bundle, registry
manifest, and decoding hashes. The scorer rejects internally consistent but
noncanonical identities rather than accepting an unrelated model run.

## 9. Reproduce offline strict scoring

```bash
uv run --frozen --python 3.10.20 python -B phase_b.py \
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

## 10. Reproducibility interpretation

This core slice supports immutable CODE-ACCORD acquisition, exhaustive
preparation auditing, exact verifier request planning/live instrumentation,
development-pilot evidence auditing, and deterministic offline response/result
replay once valid frozen inputs exist. It
does not yet establish an executed prepared corpus or exact
same-seed checkpoint rebuilding. The final workflow must resolve the official
raw/strict corpus identities, fetch and verify DeBERTa, reconstruct
`CODE-SPLIT-1` byte-identically, serialize full restart state,
duplicate seed-42 training/inference under the pinned accelerator profile,
rebuild seeds 43-49, freeze development/test inference, produce and pass the
development verifier determinism/label-quality pilot, receive the user go/no-go
decision, score uncertainty/tests, and
render/archive the final output. Those stages will be added to this same module
entry point as their remaining gates are completed.
