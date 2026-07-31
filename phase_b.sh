#!/usr/bin/env bash
# Primary publication launcher and recovery interface for the Phase B workflow.
#
# The script is intentionally conservative: it never creates placeholder
# checkpoint, candidate, pilot, or verifier inputs. A selected stage either
# finds its documented completion artifact, runs once, or exits at the first
# new error. Re-run the same command with the same --run-id after fixing that
# error; completed earlier commands are skipped from their run-local outputs.

set -Eeuo pipefail
IFS=$'\n\t'
# Keep argparse/help and provenance strings UTF-8 even on a minimally configured
# external shell; several retained historical help texts contain Unicode.
export PYTHONUTF8=1

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_ROOT="$SCRIPT_DIR"

ORIGINAL_ARGS=("$@")
RUN_ID=""
RUN_ID_PROVIDED=false
STAGE="available"
SEED=""
MODE="simple"
ALLOW_LEGACY_DIAGNOSTIC=false
ALLOW_LIVE_SMOKE=false
BLOCKED_COUNT=0
CONFIG="configs/phase_b_path_a.json"
RESPONSE_LEDGER="inputs/frozen-simple-responses.jsonl"
CAPTURE_INDEX="inputs/pilot/capture-index.json"
CHECKPOINT_MANIFEST=""
CHECKPOINT_BLOB=""
PREDICTION_LEDGER=""
SENTENCES=""
CANDIDATES_OUT=""
CANDIDATES="predictions/dev/development-candidates.jsonl"
MODEL_BLOB=""
MODEL_BLOB_SOURCE=""
OLLAMA_MODEL=""
PILOT_CANDIDATES="predictions/dev/pilot-candidates.jsonl"
PILOT_SELECTION="predictions/dev/pilot-selection.json"
ACTIVE_STAGE_ID="launcher-bootstrap"
ACTIVE_RECOVERY_MODE="fresh"
RECOVERY_MANIFEST=""
TRAINING_SEEDS=()
PROTOCOL_ID=""
WORKFLOW_ID=""
MODEL_BLOB_SHA256=""
ALLOW_FULL_RUN_RECOVERY_DOCTOR=false

usage() {
  cat <<'EOF'
Usage:
  ./phase_b.sh [--run-id RUN_ID] [options]

Always runs `git pull --ff-only` against the checked-out branch's configured
upstream and then `uv sync --frozen` first.
Then it skips only stages whose run-local completion artifacts are present and
runs from the first incomplete selected stage. It exits immediately on a new
error; rerun the same command after fixing the cause.

For a new run ID, `fetch` first searches other local `output/<run-id>/` trees
for the configured immutable CODE-ACCORD archive. A matching size-and-MD5
archive is hard-linked into the new run when possible (copied only when the
filesystem does not support hard links), then `fetch` verifies it and writes
the new run's own acquisition manifest. A network download occurs only when
no verified local run cache exists.

Stages:
  available          parser-check every command and run every available stage (default)
  smoke              non-publication one-candidate real GPU/Ollama end-to-end smoke
  full               resumable Phase B run through score, then C-04 Graph RAG
  graph-rag          recovery-only Regime-D diagnostic for an already scored full run
  publishable        run the canonical chain until its first publication gate
  bootstrap          doctor -> reconcile -> fetch -> prepare only
  plan               record the implemented seed-specific model-train dry-run
  train-live         run/resume canonical CODE-SPLIT-1 training for one seed
  legacy-train       run the retained CSV trainer as a noncanonical diagnostic
  generate-live      run documented live candidate generation
  generate-replay    run documented replay candidate generation
  threshold          run documented development threshold selection
  verifier-dry       materialize simple/corrective verifier requests
  verifier-replay    replay a simple/corrective response ledger
  verifier-live      run the simple/corrective live verifier
  pilot-live         run one development-pilot capture
  pilot-audit        audit the four separately captured pilot runs
  score              run the documented strict scorer

Options:
  --run-id ID                 Existing or new Phase B run ID (generated when omitted)
  --stage NAME                One stage from the list above
  --seed N                    Checkpoint seed (default: 42)
  --allow-legacy-diagnostic   Required acknowledgement for legacy-train
  --allow-live-smoke          Required acknowledgement before real Ollama smoke calls
  --ollama-model NAME         Local Ollama tag (default: frozen config model)
  --config PATH               Tracked Phase B config (default: configs/phase_b_path_a.json)
  --mode simple|corrective    Verifier mode (default: simple)
  --checkpoint-manifest PATH  Run-relative checkpoint identity manifest
  --checkpoint-blob PATH      Run-relative checkpoint weights for generate-live
  --prediction-ledger PATH    Run-relative ledger for generate-replay
  --sentences PATH            Run-relative prepared sentence JSONL
  --candidates-out PATH       Run-relative generated-candidate JSONL
  --candidates PATH           Run-relative development candidate JSONL for threshold
  --response-ledger PATH      Run-relative response ledger for verifier-replay
  --model-blob PATH           Run-relative Ollama blob for verifier-live
  --model-blob-source PATH    Existing local blob to hard-link/copy into --model-blob
  --pilot-candidates PATH     Run-relative pilot candidate JSONL for pilot-live
  --pilot-selection PATH      Run-relative pilot-selection JSON for pilot-live
  --capture-index PATH        Run-relative four-capture index for pilot-audit
  -h, --help                  Show this help

Examples:
  ./phase_b.sh --stage bootstrap
  ./phase_b.sh --run-id path-a-bootstrap-20260718T120000Z --stage plan --seed 42
  ./phase_b.sh --run-id path-a-simple-live-8 --stage smoke --seed 42 \
    --allow-live-smoke --model-blob-source ~/.ollama/models/blobs/<locked-blob>
  ./phase_b.sh --run-id path-a-bootstrap-20260718T120000Z --stage train-live --seed 42
  ./phase_b.sh --run-id path-a-bootstrap-20260718T120000Z \
    --stage legacy-train --seed 42 --allow-legacy-diagnostic
  ./phase_b.sh --run-id path-a-bootstrap-20260718T120000Z --stage generate-live \
    --checkpoint-manifest checkpoints/seed-42/checkpoint-manifest.json \
    --checkpoint-blob checkpoints/seed-42/checkpoint.pt
  ./phase_b.sh --stage full
  ./phase_b.sh --run-id path-a-full-20260719T153901Z --stage graph-rag

The train-live stage is the canonical checkpoint producer. It consumes only
the run-local prepared train/development split, writes a full restart state,
and resumes it automatically after an interrupted evaluation checkpoint. The
legacy-train stage remains diagnostic-only and is never fed into publication.

The default available sweep treats missing checkpoint/candidate/verifier/pilot
inputs and ungranted publication gates as BLOCKED, not as errors. Any command
whose prerequisites are present is run and still stops the script on failure.

The smoke stage reuses a completed seed checkpoint and development candidates
in this same run, creates real test candidates, and sends one real candidate
plus one development warm-up through each verifier mode. It clones only those
smoke records into pseudo-seeds 43-49 to exercise the strict downstream
threshold/scoring contracts. Its output is marked non-publication and must
never be used for a paper result or B-07 approval.

The full stage derives seeds, Python, verifier model, and blob digest from the
tracked config. Applicable live work is authorized by default, but final-test
and full live-verifier work begins only when the newly captured development
pilot passes and reports that no material protocol review is required. Every command
stops on its first new error. Re-running the printed command resumes a valid
training or verifier response state; an invalid/nonresumable partial stage is
cleaned only within that stage's declared output paths and rerun from the last
verified upstream artifacts.
EOF
}

die() {
  printf 'phase_b: error: %s\n' "$*" >&2
  if declare -F print_resume_command >/dev/null 2>&1; then
    print_resume_command
  fi
  exit 2
}

note() {
  printf '\n==> %s\n' "$*"
}

while (($#)); do
  case "$1" in
    --run-id) RUN_ID="${2:-}"; RUN_ID_PROVIDED=true; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --allow-legacy-diagnostic) ALLOW_LEGACY_DIAGNOSTIC=true; shift ;;
    --allow-live-smoke) ALLOW_LIVE_SMOKE=true; shift ;;
    --ollama-model) OLLAMA_MODEL="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --checkpoint-manifest) CHECKPOINT_MANIFEST="${2:-}"; shift 2 ;;
    --checkpoint-blob) CHECKPOINT_BLOB="${2:-}"; shift 2 ;;
    --prediction-ledger) PREDICTION_LEDGER="${2:-}"; shift 2 ;;
    --sentences) SENTENCES="${2:-}"; shift 2 ;;
    --candidates-out) CANDIDATES_OUT="${2:-}"; shift 2 ;;
    --candidates) CANDIDATES="${2:-}"; shift 2 ;;
    --response-ledger) RESPONSE_LEDGER="${2:-}"; shift 2 ;;
    --model-blob) MODEL_BLOB="${2:-}"; shift 2 ;;
    --model-blob-source) MODEL_BLOB_SOURCE="${2:-}"; shift 2 ;;
    --pilot-candidates) PILOT_CANDIDATES="${2:-}"; shift 2 ;;
    --pilot-selection) PILOT_SELECTION="${2:-}"; shift 2 ;;
    --capture-index) CAPTURE_INDEX="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="path-a-$STAGE-$(date -u +%Y%m%dT%H%M%SZ)"
  note "generated run ID: $RUN_ID"
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "invalid --run-id: $RUN_ID"
[[ "$MODE" == "simple" || "$MODE" == "corrective" ]] || die "--mode must be simple or corrective"

cd "$SOURCE_ROOT"
readonly RUN_ROOT="output/$RUN_ID"
if [[ "$STAGE" == "graph-rag" ]]; then
  ACTIVE_STAGE_ID="graph-rag-diagnostic"
  ACTIVE_RECOVERY_MODE="resume-or-fresh-child"
fi

[[ -n "$(git branch --show-current)" ]] \
  || die "the Phase B launcher requires a checked-out branch with an upstream"
git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1 \
  || die "the checked-out branch has no upstream; configure it before running"

print_resume_command() {
  printf 'Resume with:' >&2
  printf ' %q' bash phase_b.sh "${ORIGINAL_ARGS[@]}" >&2
  if [[ "$RUN_ID_PROVIDED" == false ]]; then
    printf ' --run-id %q' "$RUN_ID" >&2
  fi
  printf '\n' >&2
}

record_recovery_event() {
  local status="$1"
  local detail="$2"
  # C-04 treats the completed Phase B parent as immutable evidence. Its
  # dedicated runner owns all child trace/completion records, so the launcher
  # must not append its generic recovery ledger under the Phase B parent.
  if [[ "$ACTIVE_STAGE_ID" == "graph-rag-diagnostic" ]]; then
    return 0
  fi
  mkdir -p -- "$RUN_ROOT/manifests"
  uv run --frozen --no-sync python -B - \
    "$RUN_ROOT/manifests/debug-recovery.jsonl" "$ACTIVE_STAGE_ID" \
    "$ACTIVE_RECOVERY_MODE" "$status" "$detail" <<'PY'
import datetime
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "stage_id": sys.argv[2],
    "recovery_mode": sys.argv[3],
    "status": sys.argv[4],
    "detail": sys.argv[5],
}
with path.open("a", encoding="utf-8", newline="\n") as handle:
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
PY
}

last_failed_mode() {
  local stage_id="$1"
  local manifest="$RUN_ROOT/manifests/debug-recovery.jsonl"
  [[ -f "$manifest" ]] || return 1
  uv run --frozen --no-sync python -B - "$manifest" "$stage_id" <<'PY'
import json
import sys
from pathlib import Path

records = []
try:
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
        if line:
            records.append(json.loads(line))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1)
for record in reversed(records):
    if record.get("stage_id") == sys.argv[2]:
        if record.get("status") == "failed":
            print(record.get("recovery_mode", ""))
            raise SystemExit(0)
        raise SystemExit(1)
raise SystemExit(1)
PY
}

safe_cleanup_stage() {
  local stage_id="$1"
  shift
  local run_abs
  local relative
  local target
  local target_abs
  run_abs="$(realpath -m -- "$RUN_ROOT")"
  [[ "$run_abs" == "$SOURCE_ROOT"/output/* ]] \
    || die "refusing recovery cleanup outside the selected run: $run_abs"
  for relative in "$@"; do
    [[ -n "$relative" && "$relative" != /* && "$relative" != *'..'* ]] \
      || die "unsafe stage recovery target: $relative"
    target="$RUN_ROOT/$relative"
    target_abs="$(realpath -m -- "$target")"
    [[ "$target_abs" == "$run_abs"/* ]] \
      || die "stage recovery target escapes the selected run: $relative"
    [[ ! -L "$target" ]] || die "stage recovery refuses symlink target: $relative"
  done
  ACTIVE_STAGE_ID="$stage_id"
  ACTIVE_RECOVERY_MODE="cleanup"
  record_recovery_event "cleanup" "removing only declared incomplete stage outputs: $*"
  note "removing incomplete outputs for stage $stage_id; upstream artifacts are preserved"
  for relative in "$@"; do
    target="$RUN_ROOT/$relative"
    [[ ! -e "$target" ]] || rm -rf -- "$target"
  done
}

begin_stage() {
  ACTIVE_STAGE_ID="$1"
  ACTIVE_RECOVERY_MODE="$2"
}

finish_stage() {
  record_recovery_event "completed" "$1"
  ACTIVE_STAGE_ID="idle"
  ACTIVE_RECOVERY_MODE="none"
}

on_stage_error() {
  local status=$?
  trap - ERR
  set +e
  record_recovery_event "failed" "command exited with status $status"
  printf '\nphase_b: stopped at first new error in stage %q (mode %q).\n' \
    "$ACTIVE_STAGE_ID" "$ACTIVE_RECOVERY_MODE" >&2
  print_resume_command
  exit "$status"
}

link_or_copy() {
  local source="$1"
  local destination="$2"
  [[ -f "$source" && ! -L "$source" ]] || die "source is not a regular file: $source"
  if [[ -e "$destination" ]]; then
    [[ -f "$destination" && ! -L "$destination" ]] \
      || die "existing destination is not a regular file: $destination"
    [[ "$(stat -c '%s' -- "$source")" == "$(stat -c '%s' -- "$destination")" ]] \
      || die "existing destination size differs from its source: $destination"
    return 0
  fi
  mkdir -p -- "$(dirname -- "$destination")"
  if ! ln -- "$source" "$destination" 2>/dev/null; then
    cp -p -- "$source" "$destination"
  fi
}

json_equals() {
  local path="$1"
  local dotted_key="$2"
  local expected_json="$3"
  [[ -f "$path" ]] || return 1
  uv run --frozen --no-sync python -B - "$path" "$dotted_key" "$expected_json" <<'PY'
import json
import sys
from pathlib import Path

path, dotted_key, expected_json = sys.argv[1:]
try:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    for part in dotted_key.split("."):
        value = value[part]
    expected = json.loads(expected_json)
except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
    raise SystemExit(1)
raise SystemExit(0 if value == expected else 1)
PY
}

archive_ready() {
  local manifest="$RUN_ROOT/manifests/02-input-acquisition-manifest.json"
  [[ -f "$manifest" ]] || return 1
  uv run --frozen --no-sync python -B - "$RUN_ROOT" "$manifest" <<'PY'
import json
import sys
from pathlib import Path

run_root, manifest = map(Path, sys.argv[1:])
try:
    archive = json.loads(manifest.read_text(encoding="utf-8"))["archive"]
    path = archive["path"]
    good = isinstance(path, str) and (run_root / path).is_file()
except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
    good = False
raise SystemExit(0 if good else 1)
PY
}

find_verified_archive_cache() {
  uv run --frozen --no-sync python -B - "$RUN_ROOT" "$CONFIG" <<'PY'
import hashlib
import sys
from pathlib import Path

from acquisition import archive_contract_from_config
from config import load_pipeline_config

run_root = Path(sys.argv[1]).resolve(strict=False)
source_root = Path.cwd().resolve()
contract = archive_contract_from_config(
    load_pipeline_config(source_root, sys.argv[2])
)
relative = Path("inputs") / "cache" / f"md5-{contract.expected_md5}" / contract.name

for candidate_run in sorted((source_root / "output").glob("*")):
    candidate = candidate_run / relative
    if (
        candidate_run.resolve(strict=False) == run_root
        or candidate.is_symlink()
        or not candidate.is_file()
        or candidate.stat().st_size != contract.expected_bytes
    ):
        continue
    digest = hashlib.md5(usedforsecurity=False)
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() == contract.expected_md5:
        print(f"{candidate}\t{relative.as_posix()}")
        break
PY
}

seed_archive_from_local_cache() {
  local cache_entry
  local cached_archive
  local cache_relative
  local cached_display
  local destination
  cache_entry="$(find_verified_archive_cache)"
  [[ -n "$cache_entry" ]] || return 0
  IFS=$'\t' read -r cached_archive cache_relative <<<"$cache_entry"

  destination="$RUN_ROOT/$cache_relative"
  [[ ! -e "$destination" ]] || return 0
  cached_display="${cached_archive#"$SOURCE_ROOT/"}"
  mkdir -p -- "$(dirname -- "$destination")"
  if ln -- "$cached_archive" "$destination"; then
    note "reusing verified CODE-ACCORD archive from $cached_display (hard link)"
  elif cp --reflink=auto -- "$cached_archive" "$destination" 2>/dev/null \
      || cp -- "$cached_archive" "$destination"; then
    note "reusing verified CODE-ACCORD archive from $cached_display (copy fallback)"
  else
    die "could not materialize verified local archive cache from $cached_archive"
  fi
}

skip_or_run() {
  local label="$1"
  local check="$2"
  shift 2
  if "$check"; then
    note "$label already has its completion artifact; skipping"
    return 0
  fi
  note "running $label"
  "$@"
}

command_check_marker() {
  printf '%s\n' "$RUN_ROOT/manifests/debug-command-check.json"
}

command_check_complete() {
  local marker
  local commit
  marker="$(command_check_marker)"
  commit="$(git rev-parse HEAD)"
  json_equals "$marker" status '"passed"' \
    && json_equals "$marker" source_commit "\"$commit\""
}

write_command_check_marker() {
  local marker
  local commit
  marker="$(command_check_marker)"
  commit="$(git rev-parse HEAD)"
  uv run --frozen --no-sync python -B - "$marker" "$commit" <<'PY'
import json
import sys
from pathlib import Path

marker = Path(sys.argv[1])
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(
    json.dumps(
        {"status": "passed", "source_commit": sys.argv[2]},
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n",
    encoding="utf-8",
)
PY
}

run_command_check() {
  uv run --frozen --no-sync python -B phase_b.py --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py doctor --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py reconcile --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py fetch --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py prepare --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py assemble-candidates --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py prepare-pilot --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py model --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py model train --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py model generate-candidates --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py select-threshold --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py verifier --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py pilot-verifier --help >/dev/null
  uv run --frozen --no-sync python -B phase_b.py score --help >/dev/null
  uv run --frozen --no-sync python -B graph_rag.py phase-b-diagnostic --help >/dev/null
  uv run --frozen --no-sync python -B train_span.py --help >/dev/null
  uv run --frozen --no-sync python -B smoke.py --help >/dev/null
  uv run --frozen --no-sync python -B -c \
    'import provenance.build_kg, provenance.diagnose_evidence_paths, provenance.rule_engine, provenance.verify_triples_llm'
  write_command_check_marker
}

ensure_command_check() {
  skip_or_run "Phase B command-parser check" command_check_complete run_command_check
}

blocked() {
  BLOCKED_COUNT=$((BLOCKED_COUNT + 1))
  note "BLOCKED: $*"
}

doctor_complete() {
  local commit
  commit="$(git rev-parse HEAD)"
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    && json_equals "$RUN_ROOT/manifests/00-checkout-manifest.json" status '"pass"' \
    && json_equals "$RUN_ROOT/manifests/00-checkout-manifest.json" \
      source.commit "\"$commit\""
}

reconcile_complete() {
  json_equals "$RUN_ROOT/audit/section5-evidence-reconciliation.json" status '"reconciled_secondary_evidence"'
}

fetch_complete() {
  archive_ready
}

prepare_complete() {
  local artifact
  json_equals "$RUN_ROOT/manifests/03-data-preparation-manifest.json" \
    byte_identical_independent_materializations true || return 1
  for artifact in sentences.jsonl train.jsonl development.jsonl test.jsonl \
    development-gold.jsonl test-gold.jsonl split-manifest.json; do
    [[ -f "$RUN_ROOT/data-prepared/$artifact" ]] || return 1
  done
}

plan_complete() {
  json_equals "$RUN_ROOT/manifests/model-train-dry-run-seed-$SEED.json" status '"planned"' \
    && json_equals "$RUN_ROOT/manifests/model-train-dry-run-seed-$SEED.json" training_seed "$SEED"
}

train_live_complete() {
  json_equals "$RUN_ROOT/manifests/model-train-live-seed-$SEED.json" status '"completed"' \
    && json_equals "$RUN_ROOT/checkpoints/seed-$SEED/checkpoint-manifest.json" \
      training_seed "$SEED" \
    && [[ -s "$RUN_ROOT/checkpoints/seed-$SEED/checkpoint.pt" ]] \
    && [[ -s "$RUN_ROOT/checkpoints/seed-$SEED/restart-state.pt" ]]
}

legacy_checkpoint_path() {
  printf '%s\n' "$RUN_ROOT/checkpoints/legacy-train-span/seed-$SEED/checkpoint.pt"
}

legacy_completion_marker() {
  printf '%s\n' "$RUN_ROOT/manifests/legacy-train-span-seed-$SEED.json"
}

legacy_train_complete() {
  local checkpoint
  local marker
  checkpoint="$(legacy_checkpoint_path)"
  marker="$(legacy_completion_marker)"
  json_equals "$marker" status '"completed_noncanonical_diagnostic"' \
    && json_equals "$marker" training_seed "$SEED" \
    && [[ -s "$checkpoint" ]]
}

candidate_complete() {
  local execution="$1"
  local output="$2"
  local split
  if json_equals "$RUN_ROOT/manifests/model-generate-candidates-$execution-seed-$SEED.json" status '"completed"' \
    && [[ -f "$RUN_ROOT/$output" ]]; then
    return 0
  fi
  for split in development test; do
    if json_equals "$RUN_ROOT/manifests/model-generate-candidates-$execution-seed-$SEED-$split.json" status '"completed"' \
      && [[ -f "$RUN_ROOT/$output" ]]; then
      return 0
    fi
  done
  return 1
}

threshold_complete() {
  json_equals "$RUN_ROOT/predictions/dev/threshold-selection.json" used_test_labels false \
    && [[ -f "$RUN_ROOT/predictions/dev/threshold-selection.json" ]]
}

verifier_complete() {
  local execution="$1"
  local expected_status="$2"
  local manifest="$RUN_ROOT/manifests/verifier-$MODE-$execution.json"
  json_equals "$manifest" status "\"$expected_status\"" && [[ -f "$manifest" ]]
}

pilot_audit_complete() {
  [[ -f "$RUN_ROOT/audit/verifier-pilot/pilot-audit.json" ]]
}

score_complete() {
  [[ -f "$RUN_ROOT/metrics/metrics.json" ]] \
    && [[ -f "$RUN_ROOT/metrics/publication-table.tsv" ]] \
    && [[ -f "$RUN_ROOT/metrics/publication-summary.md" ]] \
    && [[ -f "$RUN_ROOT/manifests/score-manifest.json" ]]
}

graph_rag_complete() {
  json_equals "$RUN_ROOT/graph-rag/manifests/complete.json" \
    status '"diagnostic_complete"' \
    && json_equals "$RUN_ROOT/graph-rag/manifests/complete.json" \
      regime '"diagnostic_fact_probe"'
}

ensure_doctor() {
  if [[ "$ALLOW_FULL_RUN_RECOVERY_DOCTOR" == true ]] && smoke_doctor_complete; then
    note "reusing the existing passing checkout manifest for full-run recovery"
    return 0
  fi
  skip_or_run doctor doctor_complete \
    uv run --frozen --no-sync python -B phase_b.py doctor \
      --config "$CONFIG" --run-id "$RUN_ID"
}

ensure_reconcile() {
  skip_or_run reconcile reconcile_complete \
    uv run --frozen --no-sync python -B phase_b.py reconcile \
      --config "$CONFIG" --run-id "$RUN_ID"
}

ensure_fetch() {
  if ! fetch_complete; then
    seed_archive_from_local_cache
  fi
  skip_or_run fetch fetch_complete \
    uv run --frozen --no-sync python -B phase_b.py fetch \
      --config "$CONFIG" --run-id "$RUN_ID"
}

ensure_prepare() {
  if [[ -f "$RUN_ROOT/manifests/03-data-preparation-manifest.json" ]] && ! prepare_complete; then
    die "prepared tree is incomplete or from an older artifact contract; do not mix it with this run. Start a new --run-id so prepare can materialize a complete fresh tree."
  fi
  skip_or_run prepare prepare_complete \
    uv run --frozen --no-sync python -B phase_b.py prepare \
      --config "$CONFIG" --run-id "$RUN_ID"
}

ensure_bootstrap() {
  ensure_doctor
  ensure_reconcile
  ensure_fetch
  ensure_prepare
}

smoke_doctor_complete() {
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] \
    && json_equals "$RUN_ROOT/manifests/00-checkout-manifest.json" status '"pass"'
}

ensure_smoke_bootstrap() {
  smoke_doctor_complete \
    || die "smoke can reuse only an existing clean run with a passing checkout manifest; run --stage publishable first or choose a completed canonical seed run"
  note "reusing the existing source-locked checkout manifest for non-publication smoke"
  ensure_reconcile
  ensure_fetch
  ensure_prepare
}

ensure_plan() {
  ensure_bootstrap
  skip_or_run "model train dry-run (seed $SEED)" plan_complete \
    uv run --frozen --no-sync python -B phase_b.py model train \
      --config "$CONFIG" --run-id "$RUN_ID" \
      --execution dry-run --seed "$SEED"
}

ensure_train_live() {
  ensure_bootstrap
  local stage_id="train-seed-$SEED"
  local failed_mode=""
  if train_live_complete; then
    note "canonical model training (seed $SEED) already has its completion artifacts; skipping"
    return 0
  fi
  failed_mode="$(last_failed_mode "$stage_id" || true)"
  if [[ -s "$RUN_ROOT/checkpoints/seed-$SEED/restart-state.pt" \
      && "$failed_mode" != "resume" ]]; then
    begin_stage "$stage_id" "resume"
  elif [[ -e "$RUN_ROOT/checkpoints/seed-$SEED" \
      || -e "$RUN_ROOT/logs/model-train-seed-$SEED.log" \
      || -e "$RUN_ROOT/manifests/model-train-live-seed-$SEED.json" ]]; then
    safe_cleanup_stage "$stage_id" \
      "checkpoints/seed-$SEED" \
      "logs/model-train-seed-$SEED.log" \
      "manifests/model-train-live-seed-$SEED.json"
    begin_stage "$stage_id" "restart-after-cleanup"
  else
    begin_stage "$stage_id" "fresh"
  fi
  note "running canonical model training (seed $SEED; $ACTIVE_RECOVERY_MODE)"
  uv run --frozen --no-sync python -B phase_b.py model train \
    --config "$CONFIG" --run-id "$RUN_ID" \
    --execution live --seed "$SEED"
  finish_stage "canonical seed-$SEED training completed"
}

write_legacy_completion_marker() {
  local checkpoint
  local marker
  checkpoint="$(legacy_checkpoint_path)"
  marker="$(legacy_completion_marker)"
  uv run --frozen --no-sync python -B - "$checkpoint" "$marker" "$SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
marker = Path(sys.argv[2])
seed = int(sys.argv[3])
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(
    json.dumps(
        {
            "status": "completed_noncanonical_diagnostic",
            "training_seed": seed,
            "checkpoint_path": str(checkpoint).replace("\\", "/"),
            "checkpoint_sha256": digest,
            "publishable_primary_data": False,
            "reason": "train_span.py uses legacy CSV inputs and resamples development data",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n",
    encoding="utf-8",
)
PY
}

run_legacy_train() {
  local entities_train
  local data_dir
  local checkpoint
  entities_train="$(find "$RUN_ROOT/inputs/extracted" -type f -path '*/entities/train.csv' -print -quit)"
  [[ -n "$entities_train" ]] \
    || die "legacy-train could not locate entities/train.csv beneath $RUN_ROOT/inputs/extracted"
  data_dir="$(dirname "$(dirname "$entities_train")")"
  checkpoint="$(legacy_checkpoint_path)"
  [[ -d "$data_dir/entities" && -d "$data_dir/relations" ]] \
    || die "legacy-train requires sibling entities/ and relations/ directories under $data_dir"
  uv run --frozen --no-sync python -B - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "legacy-train requires a CUDA-capable PyTorch runtime; no CUDA device is available"
    )

device = torch.cuda.get_device_properties(0)
capability = (device.major, device.minor)
cuda_version = tuple(int(part) for part in (torch.version.cuda or "0").split(".")[:2])
if capability >= (12, 1) and cuda_version < (13, 0):
    raise SystemExit(
        f"CUDA device {device.name!r} has compute capability "
        f"{device.major}.{device.minor}, but torch {torch.__version__} bundles CUDA "
        f"{torch.version.cuda}; install the locked CUDA 13.0 environment with "
        "`uv sync --frozen` after updating this checkout."
    )
print(
    f"legacy-train CUDA preflight: {device.name}; "
    f"compute capability {capability[0]}.{capability[1]}; "
    f"torch {torch.__version__}; CUDA {torch.version.cuda}"
)
PY
  uv run --frozen --no-sync python -B train_span.py \
    --dataset accord --data-dir "$data_dir" \
    --model-name microsoft/deberta-large \
    --batch-size 16 --max-length 128 --lr 3e-5 \
    --max-steps 3500 --warmup-steps 250 --max-span-width 8 \
    --re-weight 1.0 --re-no-rel-weight 1.0 \
    --neg-sample-ratio 3.0 --focal-gamma 2.0 \
    --eval-every 100 --seed "$SEED" --primary-metric triple_f1 \
    --label-smoothing 0.1 --re-focal-gamma 0.0 --re-neg-subsample 0.0 \
    --re-context-span --doc-window-size 1 \
    --re-comparison-boost 5.0 --re-boost-adaptive-steps 1000 \
    --re-boost-adaptive-threshold 0.35 --re-boost-adaptive-threshold2 0.40 \
    --re-boost-mid 3.5 --re-boost-end 2.0 \
    --save-best-to "$checkpoint"
  write_legacy_completion_marker
}

ensure_legacy_train() {
  ensure_bootstrap
  [[ "$ALLOW_LEGACY_DIAGNOSTIC" == true ]] \
    || die "legacy-train is noncanonical; pass --allow-legacy-diagnostic to run it"
  skip_or_run "legacy train_span.py diagnostic (seed $SEED)" legacy_train_complete \
    run_legacy_train
}

ensure_generate_live() {
  ensure_bootstrap
  local checkpoint_manifest="${CHECKPOINT_MANIFEST:-checkpoints/seed-$SEED/checkpoint-manifest.json}"
  local checkpoint_blob="${CHECKPOINT_BLOB:-checkpoints/seed-$SEED/checkpoint.pt}"
  [[ -f "$RUN_ROOT/$checkpoint_manifest" ]] \
    || die "generate-live requires $checkpoint_manifest (or --checkpoint-manifest)"
  [[ -f "$RUN_ROOT/$checkpoint_blob" ]] \
    || die "generate-live requires $checkpoint_blob (or --checkpoint-blob)"
  local sentences="${SENTENCES:-data-prepared/development.jsonl}"
  local output="${CANDIDATES_OUT:-predictions/dev/seed-$SEED-candidates.jsonl}"
  local split="${sentences##*/}"
  split="${split%.jsonl}"
  local stage_id="generate-$split-seed-$SEED"
  local output_dir="${output%/*}"
  if candidate_live_complete; then
    note "model generate-candidates live ($split, seed $SEED) already has its completion artifact; skipping"
    return 0
  fi
  if [[ -e "$RUN_ROOT/$output" \
      || -e "$RUN_ROOT/$output_dir/seed-$SEED-prediction-ledger.jsonl" \
      || -e "$RUN_ROOT/manifests/model-generate-candidates-live-seed-$SEED-$split.json" ]]; then
    safe_cleanup_stage "$stage_id" \
      "$output" \
      "$output_dir/seed-$SEED-prediction-ledger.jsonl" \
      "manifests/model-generate-candidates-live-seed-$SEED-$split.json"
  fi
  begin_stage "$stage_id" "fresh"
  note "running model generate-candidates live ($split, seed $SEED)"
  uv run --frozen --no-sync python -B phase_b.py model generate-candidates \
    --config "$CONFIG" --run-id "$RUN_ID" --execution live \
    --sentences "$sentences" --checkpoint-manifest "$checkpoint_manifest" \
    --checkpoint-blob "$checkpoint_blob" --candidates-out "$output"
  finish_stage "$split candidates for seed $SEED completed"
}

candidate_live_complete() {
  local output="${CANDIDATES_OUT:-predictions/dev/seed-$SEED-candidates.jsonl}"
  candidate_complete live "$output"
}

assembled_candidates_complete() {
  local split="$1"
  if [[ "$split" == "development" ]]; then
    [[ -s "$RUN_ROOT/predictions/dev/development-candidates.jsonl" ]] \
      && json_equals "$RUN_ROOT/predictions/dev/candidate-index.json" \
        schema_version '"phase-b-candidate-index-1.0"'
  else
    [[ -s "$RUN_ROOT/predictions/test/candidates.jsonl" ]]
  fi
}

ensure_assemble_candidates() {
  local split="$1"
  local stage_id="assemble-$split-candidates"
  local outputs=()
  if [[ "$split" == "development" ]]; then
    outputs=(
      "predictions/dev/development-candidates.jsonl"
      "predictions/dev/candidate-index.json"
    )
  else
    outputs=("predictions/test/candidates.jsonl")
  fi
  if assembled_candidates_complete "$split"; then
    note "$split candidate assembly already has its completion artifacts; skipping"
    return 0
  fi
  local output
  for output in "${outputs[@]}"; do
    if [[ -e "$RUN_ROOT/$output" ]]; then
      safe_cleanup_stage "$stage_id" "${outputs[@]}"
      break
    fi
  done
  begin_stage "$stage_id" "fresh"
  note "assembling validated $split candidates across configured seeds"
  uv run --frozen --no-sync python -B phase_b.py assemble-candidates \
    --config "$CONFIG" --run-id "$RUN_ID" --split "$split"
  finish_stage "$split candidate assembly completed"
}

pilot_inputs_complete() {
  json_equals "$RUN_ROOT/predictions/dev/pilot-selection.json" \
    selected_before_live_calls true \
    && [[ -s "$RUN_ROOT/predictions/dev/pilot-candidates.jsonl" ]] \
    && [[ -s "$RUN_ROOT/predictions/dev/verifier-warmup-candidate.jsonl" ]]
}

ensure_prepare_pilot() {
  local stage_id="prepare-development-pilot"
  local outputs=(
    "predictions/dev/pilot-candidates.jsonl"
    "predictions/dev/pilot-selection.json"
    "predictions/dev/verifier-warmup-candidate.jsonl"
  )
  if pilot_inputs_complete; then
    note "development pilot inputs already have their completion artifacts; skipping"
    return 0
  fi
  local output
  for output in "${outputs[@]}"; do
    if [[ -e "$RUN_ROOT/$output" ]]; then
      safe_cleanup_stage "$stage_id" "${outputs[@]}"
      break
    fi
  done
  begin_stage "$stage_id" "fresh"
  note "freezing label-blind development pilot inputs"
  uv run --frozen --no-sync python -B phase_b.py prepare-pilot \
    --config "$CONFIG" --run-id "$RUN_ID"
  finish_stage "development pilot inputs completed"
}

ensure_generate_replay() {
  ensure_bootstrap
  local checkpoint_manifest="${CHECKPOINT_MANIFEST:-checkpoints/seed-$SEED/checkpoint-manifest.json}"
  local prediction_ledger="${PREDICTION_LEDGER:-predictions/test/prediction-ledger.jsonl}"
  [[ -f "$RUN_ROOT/$checkpoint_manifest" ]] \
    || die "generate-replay requires $checkpoint_manifest (or --checkpoint-manifest)"
  [[ -f "$RUN_ROOT/$prediction_ledger" ]] \
    || die "generate-replay requires $prediction_ledger (or --prediction-ledger)"
  local sentences="${SENTENCES:-data-prepared/test.jsonl}"
  local output="${CANDIDATES_OUT:-predictions/test/candidates.jsonl}"
  skip_or_run "model generate-candidates replay" \
    "candidate_replay_complete" \
    uv run --frozen --no-sync python -B phase_b.py model generate-candidates \
      --config "$CONFIG" --run-id "$RUN_ID" --execution replay \
      --sentences "$sentences" --checkpoint-manifest "$checkpoint_manifest" \
      --prediction-ledger "$prediction_ledger" --candidates-out "$output"
}

candidate_replay_complete() {
  local output="${CANDIDATES_OUT:-predictions/test/candidates.jsonl}"
  candidate_complete replay "$output"
}

ensure_threshold() {
  ensure_bootstrap
  local stage_id="select-development-threshold"
  if threshold_complete; then
    note "select-threshold already has its completion artifact; skipping"
    return 0
  fi
  if [[ -e "$RUN_ROOT/predictions/dev/threshold-selection.json" ]]; then
    safe_cleanup_stage "$stage_id" "predictions/dev/threshold-selection.json"
  fi
  begin_stage "$stage_id" "fresh"
  note "running select-threshold"
  local candidate_index_args=()
  if [[ -f "$RUN_ROOT/predictions/dev/candidate-index.json" ]]; then
    candidate_index_args=(--candidate-index predictions/dev/candidate-index.json)
  fi
  uv run --frozen --no-sync python -B phase_b.py select-threshold \
    --config "$CONFIG" --run-id "$RUN_ID" --candidates "$CANDIDATES" \
    "${candidate_index_args[@]}"
  finish_stage "development threshold selected"
}

ensure_verifier_dry() {
  ensure_bootstrap
  skip_or_run "verifier dry-run ($MODE)" verifier_dry_complete \
    uv run --frozen --no-sync python -B phase_b.py verifier \
      --config "$CONFIG" --run-id "$RUN_ID" \
      --mode "$MODE" --execution dry-run
}

verifier_dry_complete() { verifier_complete dry-run planned; }

ensure_verifier_replay() {
  ensure_bootstrap
  skip_or_run "verifier replay ($MODE)" verifier_replay_complete \
    uv run --frozen --no-sync python -B phase_b.py verifier \
      --config "$CONFIG" --run-id "$RUN_ID" \
      --mode "$MODE" --execution replay --response-ledger "$RESPONSE_LEDGER"
}

verifier_replay_complete() { verifier_complete replay completed; }

ensure_verifier_live() {
  ensure_bootstrap
  local stage_id="final-verifier-$MODE"
  local cache_relative="inputs/recovery/verifier-$MODE-responses.jsonl"
  local cache_args=()
  local failed_mode=""
  if verifier_live_complete; then
    note "verifier live ($MODE) already has its completion artifact; skipping"
    return 0
  fi
  failed_mode="$(last_failed_mode "$stage_id" || true)"
  begin_stage "$stage_id" "recovery-check"
  if [[ "$failed_mode" == "resume-from-response-cache" \
      && -e "$RUN_ROOT/$cache_relative" ]]; then
    safe_cleanup_stage "$stage_id" "$cache_relative"
  elif [[ -s "$RUN_ROOT/verifier/$MODE/responses.jsonl" ]]; then
    mkdir -p -- "$RUN_ROOT/inputs/recovery"
    cp -p -- "$RUN_ROOT/verifier/$MODE/responses.jsonl" "$RUN_ROOT/$cache_relative"
    record_recovery_event "checkpointed" \
      "preserved completed live responses as $cache_relative before stage cleanup"
  fi
  if [[ -e "$RUN_ROOT/verifier/$MODE" \
      || -e "$RUN_ROOT/manifests/verifier-$MODE-live.json" ]]; then
    safe_cleanup_stage "$stage_id" \
      "verifier/$MODE" "manifests/verifier-$MODE-live.json"
  fi
  if [[ -s "$RUN_ROOT/$cache_relative" ]]; then
    cache_args=(--cache-ledger "$cache_relative")
    begin_stage "$stage_id" "resume-from-response-cache"
  else
    begin_stage "$stage_id" "fresh"
  fi
  note "running verifier live ($MODE; $ACTIVE_RECOVERY_MODE)"
  uv run --frozen --no-sync python -B phase_b.py verifier \
    --config "$CONFIG" --run-id "$RUN_ID" \
    --mode "$MODE" --execution live --model-blob "$MODEL_BLOB" \
    "${cache_args[@]}"
  finish_stage "final $MODE verifier completed"
}

verifier_live_complete() { verifier_complete live completed; }

smoke_verifier_live_complete() {
  json_equals "$RUN_ROOT/manifests/verifier-smoke-$MODE-live.json" status '"completed"' \
    && [[ -f "$RUN_ROOT/verifier/smoke/$MODE/verdicts.jsonl" ]]
}

materialize_model_blob() {
  local destination="$RUN_ROOT/$MODEL_BLOB"
  local observed_digest=""
  if [[ -f "$destination" && ! -L "$destination" ]]; then
    observed_digest="$(sha256sum -- "$destination" | awk '{print $1}')"
    if [[ "$observed_digest" == "$MODEL_BLOB_SHA256" ]]; then
      return 0
    fi
    safe_cleanup_stage "materialize-model-blob" "$MODEL_BLOB"
  elif [[ -e "$destination" ]]; then
    die "run-local model blob is not a regular file: $destination"
  fi
  if [[ -z "$MODEL_BLOB_SOURCE" ]]; then
    command -v ollama >/dev/null 2>&1 \
      || die "Ollama is unavailable; install it or provide --model-blob-source"
    MODEL_BLOB_SOURCE="$(ollama show --modelfile "$OLLAMA_MODEL" \
      | awk '$1=="FROM"{print $2; exit}')"
  fi
  [[ -n "$MODEL_BLOB_SOURCE" ]] \
    || die "ollama show --modelfile $OLLAMA_MODEL did not report a FROM blob"
  [[ -f "$MODEL_BLOB_SOURCE" && ! -L "$MODEL_BLOB_SOURCE" ]] \
    || die "--model-blob-source must name a regular existing file: $MODEL_BLOB_SOURCE"
  begin_stage "materialize-model-blob" "fresh"
  mkdir -p "$(dirname "$destination")"
  if ln "$MODEL_BLOB_SOURCE" "$destination" 2>/dev/null; then
    note "hard-linked locked model blob into this run"
  else
    note "copying locked model blob into this run (hard link unavailable)"
    cp -p -- "$MODEL_BLOB_SOURCE" "$destination"
  fi
  observed_digest="$(sha256sum -- "$destination" | awk '{print $1}')"
  [[ "$observed_digest" == "$MODEL_BLOB_SHA256" ]] \
    || die "Ollama blob digest differs from the tracked config: expected $MODEL_BLOB_SHA256, got $observed_digest"
  finish_stage "locked Ollama model blob materialized"
}

ensure_pilot_live() {
  ensure_doctor
  [[ -n "$PILOT_CANDIDATES" ]] || die "pilot-live requires --pilot-candidates"
  [[ -n "$PILOT_SELECTION" ]] || die "pilot-live requires --pilot-selection"
  local sentences="${SENTENCES:-data-prepared/development.jsonl}"
  local stage_id="pilot-verifier-$MODE"
  if verifier_live_complete; then
    note "development-pilot live verifier ($MODE) already has its completion artifact; skipping"
    return 0
  fi
  if [[ -e "$RUN_ROOT/verifier/$MODE" \
      || -e "$RUN_ROOT/manifests/verifier-$MODE-live.json" ]]; then
    safe_cleanup_stage "$stage_id" \
      "verifier/$MODE" "manifests/verifier-$MODE-live.json"
  fi
  begin_stage "$stage_id" "fresh-no-cache"
  note "running development-pilot live verifier ($MODE)"
  uv run --frozen --no-sync python -B phase_b.py verifier \
    --config "$CONFIG" --run-id "$RUN_ID" \
    --mode "$MODE" --execution live --sentences "$sentences" \
    --candidates "$PILOT_CANDIDATES" --pilot-selection "$PILOT_SELECTION" \
    --model-blob "$MODEL_BLOB"
  finish_stage "development pilot $MODE capture completed"
}

ensure_pilot_audit() {
  ensure_bootstrap
  local stage_id="audit-development-pilot"
  if pilot_audit_complete; then
    note "pilot-verifier audit already has its completion artifact; skipping"
    return 0
  fi
  if [[ -e "$RUN_ROOT/audit/verifier-pilot" ]]; then
    safe_cleanup_stage "$stage_id" "audit/verifier-pilot"
  fi
  begin_stage "$stage_id" "fresh"
  note "running pilot-verifier audit"
  uv run --frozen --no-sync python -B phase_b.py pilot-verifier \
    --config "$CONFIG" --run-id "$RUN_ID" \
    --evidence-class development-pilot --capture-index "$CAPTURE_INDEX"
  finish_stage "development pilot audit completed"
}

ensure_score() {
  ensure_bootstrap
  local stage_id="score-publication"
  if score_complete; then
    note "score already has its completion artifacts; skipping"
    return 0
  fi
  if [[ -e "$RUN_ROOT/metrics" || -e "$RUN_ROOT/outcomes" \
      || -e "$RUN_ROOT/manifests/score-manifest.json" ]]; then
    safe_cleanup_stage "$stage_id" \
      "metrics" "outcomes" "publication" "manifests/score-manifest.json"
  fi
  begin_stage "$stage_id" "fresh"
  note "running score"
  uv run --frozen --no-sync python -B phase_b.py score \
    --config "$CONFIG" --run-id "$RUN_ID"
  finish_stage "publication scoring completed"
}

ensure_graph_rag() {
  score_complete \
    || die "graph-rag requires a completed Phase B score stage in this run"
  local stage_id="graph-rag-diagnostic"
  begin_stage "$stage_id" "resume-or-fresh-child"
  note "running Phase B-output Regime-D Graph RAG diagnostic"
  uv run --frozen --no-sync python -B graph_rag.py phase-b-diagnostic \
    --config configs/phase_b_graph_rag_diagnostic.json --run-id "$RUN_ID"
  graph_rag_complete \
    || die "Graph RAG diagnostic returned without a valid completion manifest"
  finish_stage "Phase B-output Graph RAG diagnostic completed"
}

maybe_generate_live() {
  local checkpoint_manifest="${CHECKPOINT_MANIFEST:-checkpoints/seed-$SEED/checkpoint-manifest.json}"
  local checkpoint_blob="${CHECKPOINT_BLOB:-checkpoints/seed-$SEED/checkpoint.pt}"
  local sentences="${SENTENCES:-data-prepared/development.jsonl}"
  if candidate_live_complete; then
    note "model generate-candidates live already has its completion artifact; skipping"
  elif [[ -f "$RUN_ROOT/$checkpoint_manifest" && -f "$RUN_ROOT/$checkpoint_blob" \
      && -f "$RUN_ROOT/$sentences" ]]; then
    ensure_generate_live
  else
    blocked "model generate-candidates live needs $checkpoint_manifest, $checkpoint_blob, and $sentences"
  fi
}

maybe_legacy_checkpoint_diagnostic() {
  local checkpoint_manifest="${CHECKPOINT_MANIFEST:-checkpoints/seed-$SEED/checkpoint-manifest.json}"
  local checkpoint_blob="${CHECKPOINT_BLOB:-checkpoints/seed-$SEED/checkpoint.pt}"
  if [[ -f "$RUN_ROOT/$checkpoint_manifest" && -f "$RUN_ROOT/$checkpoint_blob" ]]; then
    note "canonical checkpoint inputs are present; legacy diagnostic is not needed"
  elif [[ "$ALLOW_LEGACY_DIAGNOSTIC" == true ]]; then
    note "canonical checkpoint inputs are missing; running acknowledged legacy diagnostic checkpoint command"
    ensure_legacy_train
  else
    blocked "canonical checkpoint inputs are missing; for a diagnostic-only checkpoint rerun with: bash phase_b.sh --run-id $RUN_ID --stage legacy-train --seed $SEED --allow-legacy-diagnostic"
  fi
}

maybe_train_live() {
  if train_live_complete; then
    note "canonical model training (seed $SEED) already has its completion artifacts; skipping"
  else
    blocked "canonical model training is an external accelerator stage; run --stage train-live --seed $SEED to create/resume its checkpoint"
  fi
}

maybe_generate_replay() {
  local checkpoint_manifest="${CHECKPOINT_MANIFEST:-checkpoints/seed-$SEED/checkpoint-manifest.json}"
  local prediction_ledger="${PREDICTION_LEDGER:-predictions/test/prediction-ledger.jsonl}"
  local sentences="${SENTENCES:-data-prepared/test.jsonl}"
  if candidate_replay_complete; then
    note "model generate-candidates replay already has its completion artifact; skipping"
  elif [[ -f "$RUN_ROOT/$checkpoint_manifest" && -f "$RUN_ROOT/$prediction_ledger" \
      && -f "$RUN_ROOT/$sentences" ]]; then
    ensure_generate_replay
  else
    blocked "model generate-candidates replay needs $checkpoint_manifest, $prediction_ledger, and $sentences"
  fi
}

maybe_threshold() {
  if threshold_complete; then
    note "select-threshold already has its completion artifact; skipping"
  elif [[ -f "$RUN_ROOT/$CANDIDATES" ]]; then
    ensure_threshold
  else
    blocked "select-threshold needs $CANDIDATES"
  fi
}

maybe_verifier_dry() {
  local original_mode="$MODE"
  local mode
  for mode in simple corrective; do
    MODE="$mode"
    if verifier_dry_complete; then
      note "verifier dry-run ($MODE) already has its completion artifact; skipping"
    elif [[ -f "$RUN_ROOT/data-prepared/test.jsonl" \
        && -f "$RUN_ROOT/predictions/test/candidates.jsonl" ]]; then
      ensure_verifier_dry
    else
      blocked "verifier dry-run ($MODE) needs data-prepared/test.jsonl and predictions/test/candidates.jsonl"
    fi
  done
  MODE="$original_mode"
}

maybe_verifier_replay() {
  if verifier_replay_complete; then
    note "verifier replay ($MODE) already has its completion artifact; skipping"
  elif [[ -f "$RUN_ROOT/data-prepared/test.jsonl" \
      && -f "$RUN_ROOT/predictions/test/candidates.jsonl" \
      && -f "$RUN_ROOT/$RESPONSE_LEDGER" ]]; then
    ensure_verifier_replay
  else
    blocked "verifier replay ($MODE) needs test sentences, test candidates, and $RESPONSE_LEDGER"
  fi
}

maybe_pilot_audit() {
  if pilot_audit_complete; then
    note "pilot-verifier audit already has its completion artifact; skipping"
  elif [[ -f "$RUN_ROOT/$CAPTURE_INDEX" ]]; then
    ensure_pilot_audit
  else
    blocked "pilot-verifier audit needs $CAPTURE_INDEX from four separately completed pilot-live runs"
  fi
}

maybe_score() {
  if score_complete; then
    note "score already has its completion artifact; skipping"
  elif [[ -f "$RUN_ROOT/data-prepared/test-gold.jsonl" \
      && -f "$RUN_ROOT/predictions/test/candidates.jsonl" \
      && -f "$RUN_ROOT/verifier/simple/verdicts.jsonl" \
      && -f "$RUN_ROOT/verifier/corrective/verdicts.jsonl" \
      && -f "$RUN_ROOT/predictions/dev/threshold-selection.json" ]]; then
    ensure_score
  else
    blocked "score needs test gold/candidates, both verdict ledgers, and the development threshold"
  fi
}

maybe_graph_rag() {
  if score_complete; then
    ensure_graph_rag
  else
    blocked "Graph RAG diagnostic needs a completed Phase B score stage"
  fi
}

ensure_available() {
  ensure_bootstrap
  ensure_command_check
  ensure_plan
  maybe_train_live
  maybe_legacy_checkpoint_diagnostic
  maybe_generate_live
  maybe_generate_replay
  maybe_threshold
  maybe_verifier_dry
  maybe_verifier_replay
  blocked "verifier live and pilot-live require separately selected/captured inputs; final-test access still requires a passing B-07 pilot audit"
  maybe_pilot_audit
  maybe_score
  maybe_graph_rag
}

ensure_publishable() {
  ensure_plan
  ensure_train_live
  ensure_generate_live
  die "seed-$SEED canonical training and development candidate generation are complete. The seed-42 smoke/restart evidence must be reviewed at B-07 before running and aggregating seeds 42-49 or executing the live verifier. Re-run completed stages with the same run ID after that decision."
}

write_full_run_manifest() {
  local path="$RUN_ROOT/manifests/debug-full-run.json"
  local commit
  local branch
  local upstream
  local uv_version
  commit="$(git rev-parse HEAD)"
  branch="$(git branch --show-current)"
  upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')"
  uv_version="$(uv --version)"
  uv run --frozen --no-sync python -B - \
    "$path" "$RUN_ID" "$CONFIG" "$commit" "$OLLAMA_MODEL" "$MODEL_BLOB" \
    true "${TRAINING_SEEDS[*]}" "$PROTOCOL_ID" "$WORKFLOW_ID" \
    "$MODEL_BLOB_SHA256" "$branch" "$upstream" "$uv_version" <<'PY'
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

path = Path(sys.argv[1])
config_path = Path(sys.argv[3])
document = {
    "schema_version": "phase-b-debug-full-run-1.0",
    "run_id": sys.argv[2],
    "config": sys.argv[3],
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "source_commit": sys.argv[4],
    "ollama_model": sys.argv[5],
    "run_local_model_blob": sys.argv[6],
    "conditional_b07_approval": sys.argv[7] == "true",
    "training_seeds": [int(item) for item in sys.argv[8].split()],
    "protocol_id": sys.argv[9],
    "workflow_id": sys.argv[10],
    "model_blob_sha256": sys.argv[11],
    "source_branch": sys.argv[12],
    "source_upstream": sys.argv[13],
    "python_version": platform.python_version(),
    "uv_version": sys.argv[14],
    "ollama_blob_discovery": "ollama show --modelfile <configured-model>",
    "failure_policy": "stop_first_error_then_resume_or_stage_scoped_cleanup",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "run_id",
        "config",
        "config_sha256",
        "ollama_model",
        "run_local_model_blob",
        "conditional_b07_approval",
        "training_seeds",
        "protocol_id",
        "workflow_id",
        "model_blob_sha256",
        "source_branch",
        "source_upstream",
        "python_version",
        "uv_version",
        "ollama_blob_discovery",
        "failure_policy",
    ):
        if previous.get(field) != document[field]:
            raise SystemExit(f"existing full-run manifest differs at {field}")
    if previous.get("source_commit") != document["source_commit"]:
        recovery_commits = previous.get("recovery_source_commits", [])
        if (
            not isinstance(recovery_commits, list)
            or any(
                not isinstance(commit, str) or len(commit) != 40
                for commit in recovery_commits
            )
        ):
            raise SystemExit("existing full-run recovery_source_commits is invalid")
        if document["source_commit"] not in recovery_commits:
            recovery_commits.append(document["source_commit"])
            previous["recovery_source_commits"] = recovery_commits
            path.write_text(
                json.dumps(previous, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
else:
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
PY
}

pilot_run_id() {
  local mode="$1"
  local repeat="$2"
  local short_mode="${mode:0:1}"
  printf '%s\n' "${RUN_ID:0:48}-pilot-${short_mode}${repeat}"
}

prepare_pilot_run_inputs() {
  local pilot_id="$1"
  local pilot_root="output/$pilot_id"
  local relative
  bash phase_b.sh \
    --run-id "$pilot_id" --stage bootstrap --config "$CONFIG"
  for relative in \
    predictions/dev/development-candidates.jsonl \
    predictions/dev/candidate-index.json \
    predictions/dev/pilot-candidates.jsonl \
    predictions/dev/pilot-selection.json \
    predictions/dev/verifier-warmup-candidate.jsonl \
    predictions/dev/threshold-selection.json; do
    link_or_copy "$RUN_ROOT/$relative" "$pilot_root/$relative"
  done
  local seed
  for seed in "${TRAINING_SEEDS[@]}"; do
    relative="predictions/dev/seed-$seed-candidates.jsonl"
    link_or_copy "$RUN_ROOT/$relative" "$pilot_root/$relative"
  done
  link_or_copy "$RUN_ROOT/$MODEL_BLOB" "$pilot_root/$MODEL_BLOB"
}

capture_bundle_complete() {
  local capture_id="$1"
  local mode="$2"
  local root="$RUN_ROOT/inputs/pilot/captures/$capture_id"
  [[ -f "$root/manifests/00-checkout-manifest.json" ]] \
    && json_equals "$root/manifests/verifier-$mode-live.json" status '"completed"' \
    && [[ -s "$root/verifier/$mode/responses.jsonl" ]] \
    && [[ -s "$root/verifier/$mode/verdicts.jsonl" ]]
}

copy_pilot_capture() {
  local pilot_id="$1"
  local capture_id="$2"
  local mode="$3"
  local source="output/$pilot_id"
  local relative_root="inputs/pilot/captures/$capture_id"
  local destination="$RUN_ROOT/$relative_root"
  if capture_bundle_complete "$capture_id" "$mode"; then
    note "pilot capture $capture_id already copied; skipping"
    return 0
  fi
  if [[ -e "$destination" ]]; then
    safe_cleanup_stage "copy-$capture_id" "$relative_root"
  fi
  mkdir -p -- "$destination/manifests"
  cp -p -- "$source/manifests/00-checkout-manifest.json" "$destination/manifests/"
  cp -p -- "$source/manifests/verifier-$mode-live.json" "$destination/manifests/"
  cp -a -- "$source/verifier" "$destination/verifier"
}

write_capture_index() {
  local path="$RUN_ROOT/$CAPTURE_INDEX"
  local simple1 simple2 corrective1 corrective2
  simple1="$(pilot_run_id simple 1)"
  simple2="$(pilot_run_id simple 2)"
  corrective1="$(pilot_run_id corrective 1)"
  corrective2="$(pilot_run_id corrective 2)"
  if json_equals "$path" schema_version '"phase-b-verifier-pilot-captures-1.0"'; then
    return 0
  fi
  if [[ -e "$path" ]]; then
    safe_cleanup_stage "index-pilot-captures" "$CAPTURE_INDEX"
  fi
  uv run --frozen --no-sync python -B - "$path" "$PROTOCOL_ID" "$WORKFLOW_ID" \
    "$simple1" "$simple2" "$corrective1" "$corrective2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
identities = (
    ("simple-repeat-1", "simple", 1, sys.argv[4]),
    ("simple-repeat-2", "simple", 2, sys.argv[5]),
    ("corrective-repeat-1", "corrective", 1, sys.argv[6]),
    ("corrective-repeat-2", "corrective", 2, sys.argv[7]),
)
document = {
    "schema_version": "phase-b-verifier-pilot-captures-1.0",
    "protocol_id": sys.argv[2],
    "workflow_id": sys.argv[3],
    "captures": [
        {
            "capture_id": capture_id,
            "mode": mode,
            "repeat": repeat,
            "source_run_id": source_run_id,
            "capture_root": f"inputs/pilot/captures/{capture_id}",
        }
        for capture_id, mode, repeat, source_run_id in identities
    ],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
    newline="\n",
)
PY
}

ensure_pilot_captures() {
  local mode
  local repeat
  local capture_id
  local pilot_id
  for mode in simple corrective; do
    for repeat in 1 2; do
      capture_id="$mode-repeat-$repeat"
      pilot_id="$(pilot_run_id "$mode" "$repeat")"
      if ! capture_bundle_complete "$capture_id" "$mode"; then
        prepare_pilot_run_inputs "$pilot_id"
        bash phase_b.sh \
          --run-id "$pilot_id" --stage pilot-live --mode "$mode" \
          --config "$CONFIG"
        copy_pilot_capture "$pilot_id" "$capture_id" "$mode"
      fi
    done
  done
  write_capture_index
}

ensure_full() {
  ALLOW_FULL_RUN_RECOVERY_DOCTOR=true
  ensure_bootstrap
  mkdir -p -- "$RUN_ROOT/logs"
  exec > >(tee -a "$RUN_ROOT/logs/phase-b-debug.log") 2>&1
  write_full_run_manifest
  ensure_command_check

  local original_seed="$SEED"
  local seed
  for seed in "${TRAINING_SEEDS[@]}"; do
    SEED="$seed"
    ensure_plan
    ensure_train_live
    SENTENCES="data-prepared/development.jsonl"
    CANDIDATES_OUT="predictions/dev/seed-$seed-candidates.jsonl"
    ensure_generate_live
  done
  SEED="$original_seed"
  SENTENCES=""
  CANDIDATES_OUT=""

  ensure_assemble_candidates development
  CANDIDATES="predictions/dev/development-candidates.jsonl"
  ensure_threshold
  ensure_prepare_pilot
  materialize_model_blob
  ensure_pilot_captures
  ensure_pilot_audit
  json_equals "$RUN_ROOT/audit/verifier-pilot/pilot-audit.json" pilot_status '"pass"' \
    || die "B-07 pilot did not pass; final-test execution remains blocked"
  json_equals "$RUN_ROOT/audit/verifier-pilot/pilot-audit.json" \
    material_protocol_review_required false \
    || die "B-07 pilot requires material protocol review; final-test execution remains blocked"

  for seed in "${TRAINING_SEEDS[@]}"; do
    SEED="$seed"
    SENTENCES="data-prepared/test.jsonl"
    CANDIDATES_OUT="predictions/test/seed-$seed-candidates.jsonl"
    ensure_generate_live
  done
  SEED="$original_seed"
  SENTENCES=""
  CANDIDATES_OUT=""
  ensure_assemble_candidates test

  for MODE in simple corrective; do
    ensure_verifier_live
  done
  MODE="simple"
  ensure_score
  ensure_graph_rag
}

ensure_smoke() {
  ensure_smoke_bootstrap
  [[ "$ALLOW_LIVE_SMOKE" == true ]] \
    || die "smoke sends four real Ollama calls; pass --allow-live-smoke to acknowledge"
  [[ -f "$RUN_ROOT/checkpoints/seed-$SEED/checkpoint-manifest.json" ]] \
    || die "smoke requires completed canonical seed-$SEED checkpoint inputs in this run"
  [[ -f "$RUN_ROOT/checkpoints/seed-$SEED/checkpoint.pt" ]] \
    || die "smoke requires completed canonical seed-$SEED checkpoint inputs in this run"
  [[ -f "$RUN_ROOT/predictions/dev/seed-$SEED-candidates.jsonl" ]] \
    || die "smoke requires seed-$SEED development candidates from publishable smoke training"
  materialize_model_blob
  if [[ ! -f "$RUN_ROOT/predictions/smoke/test-seed-$SEED-candidates.jsonl" ]]; then
    uv run --frozen --no-sync python -B phase_b.py model generate-candidates \
      --config "$CONFIG" --run-id "$RUN_ID" --execution live \
      --sentences data-prepared/test.jsonl \
      --checkpoint-manifest "checkpoints/seed-$SEED/checkpoint-manifest.json" \
      --checkpoint-blob "checkpoints/seed-$SEED/checkpoint.pt" \
      --candidates-out "predictions/smoke/test-seed-$SEED-candidates.jsonl"
  fi
  if [[ ! -f "$RUN_ROOT/predictions/smoke/test-candidates.jsonl" ]]; then
    uv run --frozen --no-sync python -B smoke.py prepare \
      --source-dev "$RUN_ROOT/predictions/dev/seed-$SEED-candidates.jsonl" \
      --generated-test "$RUN_ROOT/predictions/smoke/test-seed-$SEED-candidates.jsonl" \
      --destination-run "$RUN_ROOT"
  fi
  for MODE in simple corrective; do
    if ! smoke_verifier_live_complete; then
      uv run --frozen --no-sync python -B phase_b.py verifier \
        --config "$CONFIG" --run-id "$RUN_ID" --mode "$MODE" --execution live \
        --artifact-prefix smoke \
        --sentences data-prepared/test.jsonl \
        --candidates predictions/smoke/test-live-candidate.jsonl \
        --warmup-sentences data-prepared/development.jsonl \
        --warmup-candidates predictions/smoke/warmup-candidate.jsonl \
        --model-blob "$MODEL_BLOB"
    fi
    if [[ ! -f "$RUN_ROOT/verifier/smoke/$MODE/pseudo-seed-verdicts.jsonl" ]]; then
      uv run --frozen --no-sync python -B smoke.py clone-verdicts \
        --source "$RUN_ROOT/verifier/smoke/$MODE/verdicts.jsonl" \
        --candidates "$RUN_ROOT/predictions/smoke/test-candidates.jsonl" \
        --destination "$RUN_ROOT/verifier/smoke/$MODE/pseudo-seed-verdicts.jsonl"
    fi
  done
  if [[ ! -f "$RUN_ROOT/predictions/smoke/threshold-selection.json" ]]; then
    uv run --frozen --no-sync python -B phase_b.py select-threshold \
      --config "$CONFIG" --run-id "$RUN_ID" \
      --candidates predictions/smoke/development-candidates.jsonl \
      --out predictions/smoke/threshold-selection.json
  fi
  if ! json_equals "$RUN_ROOT/smoke/score/metrics/metrics.json" nonpublication_smoke true; then
    uv run --frozen --no-sync python -B phase_b.py score \
      --config "$CONFIG" --run-id "$RUN_ID" \
      --candidates predictions/smoke/test-candidates.jsonl \
      --simple-verdicts verifier/smoke/simple/pseudo-seed-verdicts.jsonl \
      --corrective-verdicts verifier/smoke/corrective/pseudo-seed-verdicts.jsonl \
      --threshold-selection predictions/smoke/threshold-selection.json \
      --nonpublication-smoke --output-prefix smoke/score
  fi
}

trap on_stage_error ERR

note "updating source checkout from its configured upstream"
git pull --ff-only
note "synchronizing the locked environment"
uv sync --frozen

CONFIG_RECORD=""
while IFS= read -r config_line; do
  case "$config_line" in
    PHASE_B_CONFIG=*) CONFIG_RECORD="${config_line#PHASE_B_CONFIG=}" ;;
  esac
done < <(
  uv run --frozen --no-sync python -B - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
fields = (
    " ".join(str(seed) for seed in value["training_seeds"]),
    value["verifier"]["model"],
    value["verifier"]["model_blob_sha256"],
    value["protocol_id"],
    value["workflow_id"],
)
print("PHASE_B_CONFIG=" + "\t".join(fields))
PY
)
[[ -n "$CONFIG_RECORD" ]] || die "could not read the tagged Phase B configuration record"
IFS=$'\t' read -r CONFIG_SEEDS OLLAMA_MODEL_CONFIG MODEL_BLOB_SHA256 PROTOCOL_ID WORKFLOW_ID \
  <<<"$CONFIG_RECORD"
IFS=' ' read -r -a TRAINING_SEEDS <<<"$CONFIG_SEEDS"
[[ "${#TRAINING_SEEDS[@]}" -gt 0 ]] || die "config has no training seeds"
SEED="${SEED:-${TRAINING_SEEDS[0]}}"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be an integer"
OLLAMA_MODEL="${OLLAMA_MODEL:-${OLLAMA_MODEL_CONFIG}}"
MODEL_BLOB="${MODEL_BLOB:-inputs/ollama/blobs/sha256-$MODEL_BLOB_SHA256}"

case "$STAGE" in
  available) ensure_available ;;
  smoke) ensure_smoke ;;
  full) ensure_full ;;
  bootstrap) ensure_bootstrap ;;
  plan) ensure_plan ;;
  train-live) ensure_train_live ;;
  legacy-train) ensure_legacy_train ;;
  generate-live) ensure_generate_live ;;
  generate-replay) ensure_generate_replay ;;
  threshold) ensure_threshold ;;
  verifier-dry) ensure_verifier_dry ;;
  verifier-replay) ensure_verifier_replay ;;
  verifier-live) ensure_verifier_live ;;
  pilot-live) ensure_pilot_live ;;
  pilot-audit) ensure_pilot_audit ;;
  score) ensure_score ;;
  graph-rag) ensure_graph_rag ;;
  publishable) ensure_publishable ;;
  *) die "unknown --stage: $STAGE (run with --help for the supported list)" ;;
esac

if [[ "$STAGE" == "available" && "$BLOCKED_COUNT" -gt 0 ]]; then
  printf '\nphase_b: %s stage(s) remain blocked for run %q; see the BLOCKED lines above.\n' \
    "$BLOCKED_COUNT" "$RUN_ID" >&2
  exit 2
fi

note "stage '$STAGE' is complete for run '$RUN_ID'"
