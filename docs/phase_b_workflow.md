# Phase B fresh-clone primary-data workflow

This is the source-only guide for generating the Path A primary data and its
required derived artifacts from a fresh clone under approved protocol
`B04-PATH-A-1.3` and workflow `PATH-A-WORKFLOW-1.3`. It does not depend on the
anything outside the standalone checkout, including `research-logs`. Historical checkpoints and ledgers are
provenance only: do not copy them into a fresh run or treat them as primary
data.

## Current implementation boundary

The primary public entry point is `phase_b.sh --stage full`.
It owns the canonical lifecycle, assembly, pilot, scoring, and recovery flow;
`phase_b.py` is its internal stage dispatcher. Focused contract tests cover:

- `doctor` validates the direct source checkout, Git cleanliness, root-anchored
  output ignore behavior, configuration/matrix identities, and tracked artifact
  hashes. It records the configured reference and actual Python/uv versions but
  does not require an exact Python or uv version. It creates only
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
- `model train` records the frozen recipe for one seed with `--execution
  dry-run`, or invokes the original `train_span.py` loop in opt-in canonical
  mode with `--execution live`. Live mode reads only the prepared train and
  development JSONL, pins the DeBERTa revision, writes a resumable full state,
  never loads test data, and emits a schema-bound checkpoint under the run.
- `model generate-candidates` is the canonical successor to `provenance/inference_kg.py`.
  Its `dry-run` execution validates prepared sentences plus an encoder
  checkpoint identity and writes a per-sentence inference plan without calling a
  model; its `replay` execution transforms a frozen encoder prediction ledger
  into typed `CODE-STRICT-1` candidates, reproducing the greedy non-overlapping
  entity selection and `min(head, tail) * re` softmax-product confidence on
  typed CODE spans. Its `live` execution runs the retained encoder
  (`models.bert_kg_encoder`) over the prepared sentences to produce that
  prediction ledger and then applies the identical transform, so a live run and a
  replay of its produced ledger yield the same candidates; it hash-verifies the
  checkpoint blob against the checkpoint manifest and is externally gated on the
  accelerator and a trained checkpoint.
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
expand, or manually type an endpoint. The canonical checkpoint adapter, live
candidate inference, eight-seed assembly, threshold selection, pilot
preparation, and recovery orchestration are implemented. The first external
eight-seed run completed a passing model-backed development pilot, final-test
inference, both verifier modes, and scoring; publication rendering and the
remaining parent-side archival/release audit are still pending. Historical standalone scripts remain provenance paths and
are not publication commands for Path A. Direct `train_span.py` execution still
reads legacy annotation CSVs and creates its old random development split; only
the internal `phase_b.py model train --execution live` stage opts it into the prepared
`CODE-SPLIT-1` contract.
Each new full run still blocks final-test inference and full Qwen execution
until its newly captured B-07 evidence passes. Invocation authorization is
implicit; a failed or absent pilot audit cannot be bypassed.

The [`typed-strict migration plan`](migration.md) records the Phase B raw/typed design
lineage and its remaining execution gates. B-05C approves the compatibility
trainer implementation; it does not waive the B-07 gate for the full eight-seed
run, final-test access, live verifier calls, or reporting a result.

## 1. Prepare the standalone checkout

Clone the source repository directly, check out the release commit, and run from
its root. Install uv and synchronize the locked environment:

```bash
uv sync --frozen
```

Do not copy a parent-repository path into the configuration. The sole writable
runtime root is `output/`, which is root-anchored in `.gitignore`. The current
reference environment records Python 3.10.20 and uv 0.11.26, but those are
documented in the checkout manifest rather than enforced by `doctor`.

## 2. Validate the checkout and open one run

Choose a new stable run ID. A run ID is 1-64 characters, begins with an
alphanumeric character, and otherwise uses only letters, digits, `.`, `_`, and
`-`.

```bash
uv run --frozen python -B phase_b.py \
  doctor \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example
```

The command exits 0 only when all implemented preflight checks pass. It exits 2
and writes a blocked checkout manifest when the checkout is not compliant. Its
manifest records both the configured reference and actual Python/uv versions,
without treating version differences as a failure. It refuses an existing run ID and rejects symlink, junction,
absolute-path, and `..` escapes.

The checkout manifest always reports that publication execution is not yet
admitted because B-07 approval has not occurred. A passing B-05 doctor is not
permission to run the gated training or final-test verifier work.

### Fresh clone: run this bootstrap block, then stop

On a fresh external machine, clone the maintained source branch (or update an
existing clone with `git pull --ff-only origin refactor`) and run the following
commands from the repository root. The one `RUN_ID` is deliberately reused for
every bootstrap stage; do not substitute a new live-verifier run ID.

```bash
git clone --branch refactor --single-branch \
  https://github.com/monjai-ntust/autoresearch.git autoresearch-max
cd autoresearch-max
uv sync --frozen

RUN_ID="path-a-bootstrap-$(date -u +%Y%m%dT%H%M%SZ)"

uv run --frozen python -B phase_b.py doctor \
  --config configs/phase_b_path_a.json --run-id "$RUN_ID"
uv run --frozen python -B phase_b.py reconcile \
  --config configs/phase_b_path_a.json --run-id "$RUN_ID"
uv run --frozen python -B phase_b.py fetch \
  --config configs/phase_b_path_a.json --run-id "$RUN_ID"
uv run --frozen python -B phase_b.py prepare \
  --config configs/phase_b_path_a.json --run-id "$RUN_ID"
```

Expected terminal statuses are `pass`, `reconciled_secondary_evidence`,
`fetched`, and `prepared`. The final result must report `byte_identical: true`.
This bootstrap needs no encoder checkpoint, prediction ledger, verifier record,
Ollama model blob, or historical result: `fetch` downloads the immutable
CODE-ACCORD archive, and `prepare` derives the raw-provenance/typed-strict data
under `output/$RUN_ID/`.

Stop after `prepare`. Do **not** run `verifier --execution live` from a fresh
clone, including with a new ID such as `path-a-simple-live`: opening that ID
with `doctor` creates only the empty run layout, not its prepared sentences,
candidates, warm-up input, pilot selection, or run-local model blob. Those
files are produced only after the externally gated checkpoint/candidate and
development-pilot workflow. The absent historical checkpoints are provenance,
not a prerequisite for this bootstrap.

The next section describes the required checkpoint boundary. It is deliberately
explicit about what this revision can and cannot generate, so a fresh clone
does not silently train on a different split and label it as Path A primary
data.

### Primary Bash launcher and recovery behavior

On the external Linux machine, use the tracked runner to pull the current
branch's configured upstream, synchronize the locked environment, and run every
implemented canonical prerequisite from its first incomplete run-local artifact:

```bash
phase_b.sh
```

It treats a passing checkout manifest, reconciliation audit, verified
acquisition manifest/archive, and byte-identical preparation manifest plus
prepared split as completed. The default `available` target parser-checks every
documented command, runs every stage whose required run-local artifacts are
present, and exits on a new command error; after fixing the error, rerun the
exact command it prints with the generated or supplied `RUN_ID`. It reports missing external inputs and
ungranted publication gates as `BLOCKED` and continues to test independent
available stages, then exits nonzero when any blocks remain; it never reports a
blocked sweep as complete. `--stage publishable` instead stops at the first missing
canonical publication gate. `--help` lists explicit stages for every later
documented command (checkpoint planning, candidate generation, thresholding,
verifier/pilot, score, and the retained legacy diagnostic). These stages do not
fabricate or copy missing artifacts.

Every invoked stage is fail-fast and transactional. Training resumes from its
CPU-loaded full restart state. Full live verification preserves each completely
written response as a run-local recovery cache. A nonresumable or repeatedly
failing resume is removed only from that stage's declared subtree and rerun
from the latest validated upstream artifacts; the runner validates the resolved
target beneath the selected run before deletion and records failure, cleanup,
and completion events in `manifests/debug-recovery.jsonl`. It never deletes an
upstream completed stage, another run, or an undeclared path.

For a new run ID on the same checkout, the runner avoids another network
download when any earlier `output/<other-run-id>/` contains the immutable
CODE-ACCORD archive. It checks that candidate's configured byte size and MD5,
hard-links it into the new run where the filesystem permits (otherwise copies
it), and then invokes `fetch` so the new run still receives an independently
verified acquisition manifest. It downloads only when no valid local cache is
available. The reused archive is an immutable input, not a checkpoint,
prediction, or verifier result; every other artifact remains run-local.

## 3. Reconcile the archived Section 5 ledgers

This stage does not require the external archive or the CODE-ACCORD download:

```bash
uv run --frozen python -B phase_b.py \
  reconcile \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example
```

The register at `configs/phase_b_section5_evidence.json` records both former
source ledgers' verified archive path, checkout SHA-256, Git blob, LF-normalized
identity, line counts, and known physical defects. The generated TSVs themselves
have moved to external archival storage and are not runtime inputs. A
standalone clone validates the frozen register without reading that archive.

The output is `audit/section5-evidence-reconciliation.json`. Its authority is
always `secondary_only`, its canonical-ready count is zero, and publication
execution remains unadmitted. The command exits 2 on register identity,
structure/reference, or overwrite drift. Missing final-recipe, cross-dataset, verifier, and
zh-Hant evidence must be supplied through separately hashed provenance or
regenerated under the canonical framework; it is not inferred from these
ledgers.

## 4. Fetch the immutable CODE-ACCORD prerequisite

The fetch stage requires the passing checkout manifest created by `doctor` in
the same run:

```bash
uv run --frozen python -B phase_b.py \
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
uv run --frozen python -B phase_b.py \
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
to omit raw evidence. Source-identity checks retain the immutable raw counts of
857 relation-covered and five entity-only sentences; typed-strict eligibility is
reported separately and can cover fewer sentences when every relation marker in
a source sentence is ineligible.

The approved dual-view stage emits deterministic `data-prepared/`, the repair
ledger, split/data manifests, test gold, distributions, attribution, and two
independently generated trees whose inventories and tree hashes must be
identical before atomic promotion.

## 6. Generate the required encoder checkpoints

The primary workflow requires eight independently trained encoder checkpoints,
one for each seed `42` through `49`. Each checkpoint must be trained only on
`output/<run-id>/data-prepared/train.jsonl`, selected only against the matching
`development.jsonl`, and kept away from `test.jsonl`
and `test-gold.jsonl` until the protocol's final evaluation stage. The prepared
split manifest is the authoritative source of those memberships; do not let a
trainer resample or repartition them.

First record the intended seed and frozen recipe in the bootstrap run. This is
an auditable preflight and does **not** create model weights:

```bash
uv run --frozen python -B phase_b.py \
  model train \
  --config configs/phase_b_path_a.json \
  --run-id "$RUN_ID" \
  --execution dry-run \
  --seed 42
```

It writes `manifests/model-train-dry-run-seed-42.json`. At the approved external
seed-42 smoke checkpoint, run the canonical live stage:

```bash
uv run --frozen python -B phase_b.py \
  model train \
  --config configs/phase_b_path_a.json \
  --run-id "$RUN_ID" \
  --execution live \
  --seed 42
```

This thin orchestration call invokes the original `train_span.py` loop; it does
not add a parallel encoder or replacement loss loop. Canonical mode pins the
DeBERTa revision, adapts the prepared records into the trainer's existing
example representation, uses the frozen recipe, selects by development Triple
F1 with the earliest-step tie rule, and never loads the test split.

Every completed development evaluation atomically refreshes
`checkpoints/seed-42/restart-state.pt`, including model, optimizer, scheduler,
Python/Torch/CUDA RNG, and train-sampler order/cursor. After a machine or command
failure, rerun the identical live command; it resumes from that full state.
Restart payloads load CPU-first before model/optimizer restoration so the
serialized CPU sampler-generator and RNG byte tensors cannot be relocated to
CUDA and rejected by `torch.Generator.set_state`.

Completion creates the selected checkpoint, restart state, training summary,
checkpoint manifest, seed-specific stage manifest, progress log, and dataset
compatibility audit beneath the same run. The checkpoint manifest binds the
model to the fetched archive, acquisition/annotation/prepared/split identities,
train/development files, code/config/model revision, the run-local Hugging Face
cache manifest/tree, selected metric, restart state, and source commit. The
first seed may fetch the pinned revision; every later seed and live inference
uses the verified cache with local-only loading. Candidate generation rechecks those identities and
rejects a copied, legacy-split, stale, or renamed checkpoint.

### Historical-data comparability limit

The pre–Phase A Git tree recovers only
`data/code_accord/entities/train.csv`. Its Git blob and LF-normalized SHA-256
match the official freshly downloaded entity-training CSV. The historical
relation/test bytes and its seed-specific random train/development memberships
were never committed. The audit therefore records
`partial_match_full_legacy_equivalence_unavailable`: this is a new reproducible
canonical rerun, not proof that the old aggregate statistics are unchanged.
Matching the dataset name is not accepted as stronger evidence.
The complete invariant/difference table and historical per-seed evidence are in
[`model-training compatibility contract`](model-training-compatibility.md).

### Complete conditional publication run

The complete external-machine command is:

```bash
bash phase_b.sh --stage full
```

No run ID, branch, seed list, Python patch version, model digest, or Ollama blob
path is hard-coded by this command. The runner generates a UTC run ID, pulls the
checked-out branch's upstream, reads seeds/model/blob identity from the tracked
config, and resolves the local blob with
`ollama show --modelfile <configured-model>`. Supply `--run-id` only to resume a printed
run identity or deliberately name a new run.

`full` performs dry planning, independent training and development inference
for every configured seed, candidate assembly/indexing, development-only
threshold selection, a label-blind one-candidate-per-seed pilot selection, four
fresh live pilot captures, and the B-07 audit. Applicable live/full invocation
is authorized by default, but the launcher continues into test inference and
the two complete live verifier passes only when that newly produced pilot
reports `pilot_status=pass` and `material_protocol_review_required=false`; any
other pilot result stops the run. It then independently infers the test split for all seeds, assembles the
test universe, runs simple and corrective Qwen verification, and scores the
four frozen conditions. The full command writes an append-only console log to
`output/<run-id>/logs/phase-b-debug.log` and records all derived settings in
`manifests/debug-full-run.json`.

The command stays attached to the invoking shell. Run it in an existing
`tmux`/`screen` session if the SSH connection may close. On any new error it
stops and prints an exact resume command. Do not start a second concurrent
launcher for the same run ID.

### Debug runner and B-07 boundary

Run or resume the selected seed with:

```bash
bash phase_b.sh --run-id "$RUN_ID" --stage train-live --seed 42
```

`--stage publishable --seed 42` remains a diagnostic route that runs the dry
plan, canonical training, and development candidate generation, then stops
before the full-run pilot and final-test stages.
The default `available` sweep parser-checks live training but reports it as
blocked rather than unexpectedly starting a 3,500-step job. The primary `full`
stage handles seeds `42` through `49`; final-test inference and live verification
remain gated by its mandatory passing pilot audit.

### Non-publication end-to-end smoke lane

After `--stage publishable` has completed seed 42 for a run such as
`path-a-simple-live-8`, the following opt-in diagnostic checks the remaining
model-inference, both live verifier modes, threshold-selection, and scoring
code paths without overwriting the seed-42 checkpoint, development candidates,
or any canonical verifier/score location:

```bash
bash phase_b.sh --run-id path-a-simple-live-8 --stage smoke --seed 42 \
  --allow-live-smoke --ollama-model qwen3:32b
```

It performs real GPU inference over the prepared test split, chooses one
deterministically ordered real test candidate, and sends that candidate plus a
development warm-up through the pinned Ollama model in both `simple` and
`corrective` modes: four live calls total. The runner then copies only the
resulting *smoke candidate and verdict identities* to pseudo-seeds 43--49 so
that the strict eight-seed threshold and scoring contracts execute. All such
artifacts live under `predictions/smoke/`, `verifier/smoke/`, and
`smoke/score/`; the normal `verifier/<mode>/`, `metrics/`, and `outcomes/`
locations remain untouched for a later real run.

When `--model-blob-source` is omitted, the runner discovers it from the frozen
Ollama tag. It materializes the immutable blob beneath the selected run by hard
link when possible (copy fallback otherwise); the live verifier still
hash-verifies it before making a request. An explicit source path remains
available for nonstandard Ollama installations.

This is a code-path and environment smoke test, not scientific evidence. Its
score manifest and metrics contain `nonpublication_smoke: true`, suppress
publication seed coverage and uncertainty/statistical output, and are never
eligible for B-07 approval, a paper table, or an eight-seed claim. The fake
seed copies are permitted only in this explicitly namespaced diagnostic lane;
the publishable workflow must independently train and infer every seed 42--49.

Unlike normal stages, `smoke` deliberately reuses the existing run's passing
checkout manifest rather than requiring a new `doctor` manifest for the latest
source commit. A completed checkpoint is bound to that original manifest; a
new `doctor` cannot overwrite it. This narrow exception is available only to
the nonpublication smoke stage, still requires a clean current checkout, and
does not loosen ordinary bootstrap or publishable-run provenance checks.

### Legacy trainer: debug only, never publishable

The runner also exposes the retained `train_span.py` command for diagnosing the
historical CSV trainer with the closest frozen recipe settings:

```bash
bash phase_b.sh --run-id "$RUN_ID" \
  --stage legacy-train --seed 42 --allow-legacy-diagnostic
```

To have the default `available` sweep make this same diagnostic checkpoint when
the canonical seed checkpoint is absent, retain the acknowledgement while using
the normal target:

```bash
bash phase_b.sh --run-id "$RUN_ID" --allow-legacy-diagnostic
```

It uses only the run-local extracted annotation CSVs and writes its checkpoint
plus a `completed_noncanonical_diagnostic` marker under
`output/$RUN_ID/checkpoints/legacy-train-span/`. The explicit acknowledgement
is required because this direct command constructs its own legacy development
split and is outside the canonical manifest chain. Its marker records
`publishable_primary_data: false`; the primary launcher will not use that checkpoint
for canonical candidate generation or let it satisfy `--stage publishable`.

#### GB10 CUDA runtime requirement

The locked environment uses PyTorch `2.9.1+cu130`, not CUDA 12.8. NVIDIA GB10
reports compute capability 12.1; the former CUDA 12.8 wheel supports only up to
12.0 and fails during NVRTC compilation of DeBERTa kernels with
`invalid value for --gpu-architecture`. After pulling the CUDA-13 lockfile,
run `uv sync --frozen` before retrying training. The diagnostic runner checks
the active device capability and bundled CUDA major version before model load;
canonical training additionally records Python, Torch, CUDA runtime, CUDA
availability, and device identity in its training summary.

### Canonical versus direct training

Do not substitute a direct `python -B train_span.py --dataset accord ...`
command for the canonical stage. Direct invocation deliberately preserves the
historical CSV/random-split defaults. Only `phase_b.py model train --execution
live` supplies the prepared split, revision pin, restart paths, no-test guard,
and manifest validation required for Path A.

### Live candidate handoff

After the live training manifest is complete, generate development candidates
from that run's checkpoint. The checkpoint manifest is created by the trainer;
do not hand-author or trim its schema-bound provenance fields.

```bash
uv run --frozen python -B phase_b.py \
  model generate-candidates \
  --config configs/phase_b_path_a.json \
  --run-id "$RUN_ID" \
  --execution live \
  --sentences data-prepared/development.jsonl \
  --checkpoint-manifest checkpoints/seed-42/checkpoint-manifest.json \
  --checkpoint-blob checkpoints/seed-42/checkpoint.pt \
  --candidates-out predictions/dev/seed-42-candidates.jsonl
```

Run the same seed-specific process for development only after the smoke review,
and run final-test inference only after threshold/pilot approval. The candidate
stage verifies both the checkpoint blob hash and every fetched/prepared identity
recorded by training. Missing external execution remains a disclosed block; do
not create placeholder checkpoints, candidates, verdicts, or metrics.

## 7. Plan, execute, or replay the frozen verifier

This section is a later-stage interface reference, not the next command after
fresh-clone preparation. A live verifier call requires prepared sentences and
canonical candidates in its **same** run plus the pinned run-local model blob;
the development-pilot rules below impose further prerequisites. Do not create
or copy placeholder inputs merely to make this command start.

The tracked UTF-8 prompt files and response schemas define
`CODE-VERIFIER-1`. A dry run validates prepared sentences and candidates and
writes the exact request payloads, prompt/model/decoding hashes, environment
manifest, append-only run log, and stage manifest without contacting Ollama or
writing a verdict:

```bash
uv run --frozen python -B phase_b.py \
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
uv run --frozen python -B phase_b.py \
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
uv run --frozen python -B phase_b.py \
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

## 8. Audit the development-only verifier pilot

After all eight development ledgers exist, assemble and freeze the B-07 inputs
before any live call:

```bash
uv run --frozen python -B phase_b.py assemble-candidates \
  --config configs/phase_b_path_a.json \
  --run-id "$RUN_ID" \
  --split development

uv run --frozen python -B phase_b.py select-threshold \
  --config configs/phase_b_path_a.json \
  --run-id "$RUN_ID" \
  --candidates predictions/dev/development-candidates.jsonl \
  --candidate-index predictions/dev/candidate-index.json

uv run --frozen python -B phase_b.py prepare-pilot \
  --config configs/phase_b_path_a.json \
  --run-id "$RUN_ID"
```

The full eight-file development candidate index is distinct from the small
pilot subset. `prepare-pilot` deterministically selects the lexicographically
smallest canonical candidate ID independently within each seed, without reading
test data or development labels, and freezes one warm-up candidate. The
selection manifest binds the authoritative split, full index, exact subset
hash, candidate IDs, examples, seed coverage, and selection rule.

Use four distinct clean run IDs to execute two live calls per mode. Each call
must receive the same predeclared selection manifest as an explicit input:

```bash
uv run --frozen python -B phase_b.py \
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
uv run --frozen python -B phase_b.py \
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

## 9. Supply immutable offline-scoring inputs

`predictions/test/candidates.jsonl` can now be produced inside the run by
`model generate-candidates --execution replay` from a frozen encoder prediction
ledger and checkpoint manifest, rather than supplied externally:

```bash
uv run --frozen python -B phase_b.py \
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
uv run --frozen python -B phase_b.py \
  select-threshold \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example \
  --candidates predictions/dev/development-candidates.jsonl \
  --candidate-index predictions/dev/candidate-index.json
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

## 10. Reproduce offline strict scoring

```bash
uv run --frozen python -B phase_b.py \
  score \
  --config configs/phase_b_path_a.json \
  --run-id path-a-example
```

The scorer writes, without overwriting an existing artifact:

```text
outcomes/candidate-outcomes.jsonl
outcomes/sentence-outcomes.jsonl
metrics/metrics.json
metrics/publication-table.tsv
metrics/publication-summary.md
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
The publication table and Markdown summary are deterministic views of the same
`metrics.json` counts/statistics; they do not recompute, select, or tune results.

The manifest stores only run-relative paths plus hashes. A second invocation on
the same run refuses to replace the first result. To compare a rerun, use a new
run ID and compare its input/output hashes.

The `-B` flag is mandatory for canonical commands: it prevents Python from
creating `__pycache__` files in the tracked source area. `doctor` checks this
before declaring the environment compliant.

## 11. Reproducibility interpretation

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
