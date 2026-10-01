# CODE-ACCORD publication artifact

This branch contains the canonical CODE-ACCORD publication workflow and the
same-run reproduction path for the paper's Table 2. `pipeline.py` is the
single supported canonical table-production entry point. A separate
`encoder_comparison.py` entry point hosts the Phase G historical-derived
Table-1 comparison path without changing the canonical pipeline.

The pipeline retains the established CODE-ACCORD encoder, threshold, local
Qwen verifier, strict scoring, and original Table-2 Graph RAG methods. It does
not claim to regenerate every historical table. Table 4,
Traditional-Chinese work, unrelated experiments, diagnostics, and standalone
artifact-path entry points are excluded.

## Source layout

- Root `pipeline.py` is the thin canonical table-production entry point; it parses the CLI and calls the internal preparation, encoder, verifier, evaluation, and RAG functions in `stages/`.
- Root `encoder_comparison.py` is a separate thin Phase G entry point. Its historical-derived model/trainer live under `utils/encoder_comparison/`, while its controller reuses canonical preparation, data, cache, path, and artifact utilities.
- Each `stages/<stage>.py` owns that stage's validation, resume/recovery, and execution coordination; only genuinely shared artifact primitives remain under `utils/common/`.
- `utils/` groups implementation by preparation, encoder, verifier, evaluation, and RAG stage; `utils/common/` holds cross-stage contracts.
- `resources/` contains the tracked configurations, prompts, schemas, contracts, and compatibility data consumed by those stages.
- `tests/` mirrors the pipeline contracts without exposing additional runtime entry points.

## Scientific boundary

The full route is:

```text
immutable CODE-ACCORD input
  -> deterministic CODE-SPLIT-1 preparation
  -> eight DeBERTa-large training and candidate runs
  -> development-only threshold and verifier pilot gate
  -> simple and corrective Qwen verification
  -> canonical CODE-ACCORD encoder/verifier metrics
  -> same-run confidence, corrective, and gold graphs
  -> frozen five-mode answer evaluation for Table 2
```

| Paper artifact | Implemented route and data contract | Comparison evidence and missing work |
| --- | --- | --- |
| Historical Table 1 | Not regenerated. Its canonical successor is the eight-seed CODE-ACCORD `CODE-SPLIT-1` held-out encoder evaluation under `CODE-STRICT-1`. | Legacy displayed aggregates are comparison evidence only; the historical BERT-base/DeBERTa-base table, split memberships, and checkpoints are not reproduced. |
| Table 2 | Newly evaluated from one run's CODE-ACCORD test records and confidence/corrective/gold graphs, projected into the frozen five-mode table-era evaluator. | Legacy displayed cells are comparison references only. External E-07 execution and comparison remain pending. |
| Historical Table 3 | Not regenerated. Its canonical successor is the CODE-ACCORD `CODE-SPLIT-1` simple/corrective Qwen evaluation under `CODE-STRICT-1`. | Legacy SciERC text-tuple aggregates are comparison evidence only; their source data/checkpoints/verdict ledgers are not reproduced. |
| Table 4 | No route. | Excluded from this artifact. |

Every generated or downloaded artifact lives under one
`output/<run-id>/` tree. Downstream stages derive their inputs from that tree
and validate producer hashes and manifests. A run cannot import an output from
another run.

Tracked source is authenticated once by a clean full Git commit. Manifests do
not repeat hashes for individual Python files or the lockfile. Content hashes
remain mandatory for runtime inputs, model blobs, prompts, prepared data, and
stage outputs. The only retained JSON Schemas are the two response formats sent
to Ollama; other record contracts are enforced directly by their consumers.

The encoder revision and Qwen registry/blob digests must each remain internally
consistent wherever they are consumed within one run. They do not have to equal
the historical reference digests recorded by this project. Hardware and
historical digest comparisons are logged separately and do not admit or reject
a run. Matching statistics are expected only when the method, data, split,
seed, threshold, hardware, encoder, and Qwen determinants match, and equality
must still be confirmed from the generated outputs.

## Install and validate

Use Python 3.10 or newer from a clean clone of branch
`publication-refactored-rag`:

```bash
uv sync --frozen
uv run --frozen --no-sync python -I -B pipeline.py validate-table2
uv run --frozen --no-sync python -B -m unittest discover -s tests -v
```

The tracked configuration is `resources/configs/pipeline.json`. Its explicitly named
Qwen reference digests are comparison metadata. The default encoder revision
matches the historical reference, but configuration validation accepts any
immutable 40-hex revision; that configured value becomes the new run's actual
encoder input. A full run discovers and hashes the Qwen model that is actually
served, copies its bytes into the run's content-addressed input area, and binds
all verifier consumers to that run-local identity.

## Generate the canonical upstream artifacts and expanded Table 2

Choose a new run ID containing 1-64 letters, digits, `.`, `_`, or `-`, beginning
with a letter or digit. Start Ollama with the configured Qwen model, then run:

```bash
uv run --frozen --no-sync python -I -B pipeline.py full \
  --run-id research-run-001 \
  --ollama-url http://localhost:11434
```

If Ollama's `FROM` path is not directly accessible, add
`--model-blob-source /absolute/path/to/the/model/blob`. The source must be an
external immutable input, not a file under any `output/<run-id>/` tree.

The Python entry point performs and resumes the complete ordered workflow. Its
RAG stage independently materializes same-run projections and freshly evaluates
the 467-question original-CODE panel. It emits the new 105-question `part-of`
subset for later comparison with a separately archived historical oracle. It
does not replace a different existing artifact, bypass the development pilot,
or reuse a stage whose manifest/hash binding has changed.

## Phase G encoder comparisons

`encoder_comparison.py` validates a finite registry of immutable BERT-base,
DeBERTa-base, and DeBERTa-large profiles. The comparison implementation is a
bounded transplant of the historical result-producing trainer/network, not a
generalization of the canonical encoder. It consumes the same canonical
prepared split and compatible current utilities through a distinct one-profile-
per-run namespace.

Offline profile validation is available without acquiring a model:

```bash
uv run --frozen --no-sync python -I -B encoder_comparison.py \
  validate-profile --profile deberta-large-v1

uv run --frozen --no-sync python -I -B encoder_comparison.py \
  validate-arm --arm deberta-large-a20-a21-a12
```

The approved matrix contains only the four populated historical Table 1 rows:
BERT-base, DeBERTa-base, and DeBERTa-large under the common base recipe, plus
DeBERTa-large under A20+A21+A12. Historical values are provenance only; eligible
new results use the matched canonical protocol. Live use remains gated on
run-local model acquisition, storage and accelerator checks, and G-05 smoke
admission. The `smoke` action runs one training step, reloads its restart state,
and produces a one-record candidate ledger on CUDA. Full `train` actions fail
closed until a separately recorded post-review admission exists. See
`docs/encoder-comparison.md` for the source/reuse boundary,
statistics contract, and command contract.

### Detached full-training and prediction launcher

Run this Bash block from the standalone source checkout, with dependencies
already synchronized. Set `comparison_prefix` to the **unchanged** prefix of
your four existing, prepared, smoke-reviewed and admitted runs; the example
uses `g05-<arm>`. Do not update source during an authenticated run. This block
does not prepare, admit, delete or copy runs, or rerun the canonical pipeline.
Final-test access must also have received the review required by the protocol.
It completes all four arms' seeds 42-49 before generating test predictions.
Missing run directories are reported and skipped; if any arm is missing, it
trains the available arms but skips final-test generation for the whole matrix.
Other validation/training errors still stop the launcher.

```bash
comparison_prefix="${COMPARISON_PREFIX:-g05}"
job_dir="output/comparison-launch-$(date -u +%Y%m%dT%H%M%S%N)-$$"
mkdir -p "$job_dir/logs"
log="$job_dir/logs/runner.log"

nohup bash -s -- "$job_dir" "$comparison_prefix" >"$log" 2>&1 <<'BASH' &
set -euo pipefail
job_dir=$1
prefix=$2
export PYTHONUNBUFFERED=1 DISABLE_SAFETENSORS_CONVERSION=1
trap 'rc=$?; printf "\nFINISHED %s exit_code=%s\n" "$(date -u +%FT%TZ)" "$rc"' EXIT
[[ -f encoder_comparison.py && -f uv.lock ]] ||
  { printf 'Run this from the source checkout.\n'; exit 1; }
mkdir -p output/comparison-launcher
exec 9>output/comparison-launcher/full.lock
flock -n 9 || { printf 'Another comparison launcher holds the lock.\n'; exit 1; }

arms=(bert-base-common deberta-base-common deberta-large-common deberta-large-a20-a21-a12)
seeds=(42 43 44 45 46 47 48 49)
available=()
for arm in "${arms[@]}"; do
  run_dir="output/$prefix-$arm"
  if [[ ! -d "$run_dir" ]]; then
    printf 'SKIP missing directory: %s\n' "$run_dir"
    continue
  fi
  [[ -s "$run_dir/manifests/encoder-comparison-admission.json" ]] ||
    { printf 'Missing admission: %s\n' "$run_dir"; exit 1; }
  available+=("$arm")
done

for arm in "${available[@]}"; do
  for seed in "${seeds[@]}"; do
    printf '\nTRAIN arm=%s seed=%s\n' "$arm" "$seed"
    uv run --frozen --no-sync python -I -B -u encoder_comparison.py \
      train --arm "$arm" --run-id "$prefix-$arm" --training-seed "$seed"
  done
done

if (( ${#available[@]} == ${#arms[@]} )); then
  printf '\nALL 32 TRAINING RUNS COMPLETE\n'
  for arm in "${arms[@]}"; do
    for seed in "${seeds[@]}"; do
      printf '\nGENERATE arm=%s seed=%s\n' "$arm" "$seed"
      uv run --frozen --no-sync python -I -B -u encoder_comparison.py \
        generate --arm "$arm" --run-id "$prefix-$arm" --training-seed "$seed" --split test
    done
  done
  printf '\nALL TRAINING AND TEST PREDICTION GENERATION COMPLETE\n'
else
  printf '\nSKIP final-test generation: the four-arm matrix is incomplete.\n'
fi
BASH

printf 'PID: %s\nLog: %s\nFollow: tail -f %q\n' "$!" "$log" "$log"
```

Requires Linux Bash, `nohup`, `flock` and `uv`. The generated `job_dir` holds
launcher logs only; scientific inputs/results remain in their original
arm-specific runs. The shared lock prevents overlapping copies of this
launcher, not independently launched comparison commands. `nohup` lets the
job continue after SSH disconnect. Check the log for the final completion
message and `exit_code=0`; a missing-directory skip is not a complete matrix.
Training summaries contain development-selected metrics. `generate` writes
test prediction/candidate JSONL, not scored, aggregated publication statistics.
Those require subsequent strict scoring under the declared statistics contract.

## Run only downstream Table 2

Use this route when a complete authenticated run already contains the encoder,
candidate, threshold, verifier, and score outputs. It never trains the encoder
or invokes the verifier. It neither requires nor reads `table2-q105/`, and it
does not insert stored answers into the replacement output.

First validate the same-run plan without writes or model calls:

```bash
uv run --frozen --no-sync python -I -B pipeline.py table2 \
  --run-id research-run-001 \
  --dry-run
```

Then execute the replacement answer evaluator against the same Qwen identity
that the selected run's corrective verifier recorded:

```bash
uv run --frozen --no-sync python -I -B pipeline.py table2 \
  --run-id research-run-001 \
  --ollama-url http://localhost:11434
```

The replacement output is confined to
`output/<run-id>/table2-code-all/`. It derives its own authenticated projections
from same-run parents, freshly executes all 467 questions (7,005 calls across
five modes and three graph conditions), and writes the 105 new `part-of` rows
under `part-of-results/`. Historical results are archived and compared only
after execution; they are never runtime inputs. Missing canonical graphs are
deterministically reconstructed only from authenticated prepared data, gold,
seed-42 candidates, the development-selected threshold, and corrective verdicts
in that same run. If a canonical graph already exists, its canonical JSON must
exactly equal the reconstruction before it is reused.

## Frozen Table-2 evaluator

Historical source blob `964893546b28f04e34e8c546bcbbfd4cfbc27354`
remains the prompt and method authority. The protected generator retains its 13
case-folded templates and 105-question panel. The replacement adds only the
seven literal original-CODE templates from the bounded historical source and
removes sampling, yielding 467 ordered questions on the reference input. Both
paths preserve five-mode order, retrieval behavior, exact prompt construction,
`think=false`, temperature `0.0`, `num_predict=50`, lenient answer matching,
aggregation, output semantics, and the fail-closed empty-panel guard.

The text condition is historical unique-word overlap, not BM25. The
answer-derived retrieval query, unequal evidence access, and lenient matching
are retained limitations rather than corrected methodology. The former
100-question sampling limitation is replaced only by complete ordered coverage
of the original CODE question generator.

## Failure and output handling

The workflow fails closed on cross-run paths, manifest or hash drift, mixed
encoder/Qwen identities, graph mismatch, unsafe links, model-call errors,
empty responses, or an output collision. Interrupted Table-2 results stay in a
run-local recovery namespace and are never silently accepted. Legacy displayed
values are comparison metadata only; they are not inputs or tuning targets.

See `docs/workflow.md` for the artifact and resume contract and
`docs/model-training-compatibility.md` for the canonical encoder compatibility
boundary. See `docs/encoder-comparison.md` for the isolated Phase G path.

### ZIP the comparison or pipeline text outputs

After the launchers exit with zero, run this Bash block from the source checkout.
It uses `file` to identify text by content (including extensionless and empty
files), preserves paths, and omits binary weights/checkpoints. Missing paths
print a message and are skipped; no archive is created when none exist or none
contain text. Requires `file`, `find` and `zip`. Keep `job_dir` and
`comparison_prefix` from the launcher above, or set them to the existing job
directory and run prefix in a new SSH session. Set `PIPELINE_RUN_ID` to the
actual pipeline run ID; do not create or rename a run to package it.

```bash
archive_dir="output/text-handoff-$(date -u +%Y%m%dT%H%M%S%N)-$$"
pipeline_dir="${PIPELINE_RUN_ID:+output/$PIPELINE_RUN_ID}"

zip_text() (
  set -euo pipefail
  archive=$1; shift
  paths=()
  for path in "$@"; do
    if [[ -e "$path" ]]; then paths+=("$path")
    else printf 'SKIP missing path: %s\n' "${path:-<unset>}"; fi
  done
  (( ${#paths[@]} )) || { printf 'SKIP archive: no input paths.\n'; return 0; }
  mkdir -p "$archive_dir"
  list=$(mktemp "$archive_dir/.text-list-XXXXXX")
  trap 'rm -f -- "$list"' EXIT
  find "${paths[@]}" -type f \( -empty -o -exec sh -c \
    '[ "$(file -b --mime-encoding "$1")" != binary ]' sh {} \; \) -print >"$list"
  [[ -s "$list" ]] || { printf 'SKIP archive: no text files.\n'; return 0; }
  zip -q "$archive" -@ <"$list"
  printf 'Archive: %s\n' "$archive"
)

comparison_prefix="${comparison_prefix:-${COMPARISON_PREFIX:-g05}}"
zip_text "$archive_dir/comparison-text.zip" \
  output/"$comparison_prefix"-{bert-base-common,deberta-base-common,deberta-large-common,deberta-large-a20-a21-a12} \
  "${job_dir:-}"
zip_text "$archive_dir/pipeline-text.zip" "$pipeline_dir"
```

The generated `archive_dir` is a text-only handoff namespace, not a new
scientific run. These ZIPs are diagnostic exports, not complete authenticated
run archives: keep the original run trees and binary artifacts for full lineage
validation and final retention. The ZIP list supports spaces in names but not
embedded newlines. A zero launcher exit is assumed here, not checked by the ZIP
helper; inspect any reported skips before treating an export as complete.
