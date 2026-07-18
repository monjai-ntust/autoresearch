#!/usr/bin/env bash
# Resume one Phase B command chain from its first incomplete output artifact.
#
# The script is intentionally conservative: it never creates placeholder
# checkpoint, candidate, pilot, or verifier inputs. A selected stage either
# finds its documented completion artifact, runs once, or exits at the first
# new error. Re-run the same command with the same --run-id after fixing that
# error; completed earlier commands are skipped from their run-local outputs.

set -Eeuo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

RUN_ID=""
STAGE="bootstrap"
SEED=42
BRANCH="refactor"
MODE="simple"
RESPONSE_LEDGER="inputs/frozen-simple-responses.jsonl"
CAPTURE_INDEX="inputs/pilot/capture-index.json"
CHECKPOINT_MANIFEST=""
CHECKPOINT_BLOB=""
PREDICTION_LEDGER=""
SENTENCES=""
CANDIDATES_OUT=""
CANDIDATES="predictions/dev/development-candidates.jsonl"
MODEL_BLOB="inputs/ollama/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312"
PILOT_CANDIDATES="predictions/dev/pilot-candidates.jsonl"
PILOT_SELECTION="predictions/dev/pilot-selection.json"

usage() {
  cat <<'EOF'
Usage:
  scripts/phase_b_debug.sh --run-id RUN_ID [options]

Always runs `git pull --ff-only origin refactor` and `uv sync --frozen` first.
Then it skips only stages whose run-local completion artifacts are present and
runs from the first incomplete selected stage. It exits immediately on a new
error; rerun the same command after fixing the cause.

Stages:
  bootstrap          doctor -> reconcile -> fetch -> prepare (default)
  plan               record the implemented seed-specific model-train dry-run
  generate-live      run documented live candidate generation
  generate-replay    run documented replay candidate generation
  threshold          run documented development threshold selection
  verifier-dry       materialize simple/corrective verifier requests
  verifier-replay    replay a simple/corrective response ledger
  verifier-live      run the gated simple/corrective live verifier
  pilot-live         run one gated development-pilot capture
  pilot-audit        audit the four separately captured pilot runs
  score              run the documented strict scorer

Options:
  --run-id ID                 Existing or new Phase B run ID (required)
  --stage NAME                One stage from the list above
  --seed N                    Checkpoint seed (default: 42)
  --mode simple|corrective    Verifier mode (default: simple)
  --checkpoint-manifest PATH  Run-relative checkpoint identity manifest
  --checkpoint-blob PATH      Run-relative checkpoint weights for generate-live
  --prediction-ledger PATH    Run-relative ledger for generate-replay
  --sentences PATH            Run-relative prepared sentence JSONL
  --candidates-out PATH       Run-relative generated-candidate JSONL
  --candidates PATH           Run-relative development candidate JSONL for threshold
  --response-ledger PATH      Run-relative response ledger for verifier-replay
  --model-blob PATH           Run-relative Ollama blob for verifier-live
  --pilot-candidates PATH     Run-relative pilot candidate JSONL for pilot-live
  --pilot-selection PATH      Run-relative pilot-selection JSON for pilot-live
  --capture-index PATH        Run-relative four-capture index for pilot-audit
  --branch NAME               Remote branch to pull (default: refactor)
  -h, --help                  Show this help

Examples:
  scripts/phase_b_debug.sh --run-id path-a-bootstrap-20260718T120000Z
  scripts/phase_b_debug.sh --run-id path-a-bootstrap-20260718T120000Z --stage plan --seed 42
  scripts/phase_b_debug.sh --run-id path-a-bootstrap-20260718T120000Z --stage generate-live \
    --checkpoint-manifest checkpoints/seed-42/checkpoint-manifest.json \
    --checkpoint-blob checkpoints/seed-42/checkpoint.pt

`model train --execution live` is intentionally not a stage: this revision
does not implement a canonical CODE-SPLIT-1 JSONL training adapter. Do not use
the legacy train_span.py CSV workflow as a substitute.
EOF
}

die() {
  printf 'phase_b_debug: error: %s\n' "$*" >&2
  exit 2
}

note() {
  printf '\n==> %s\n' "$*"
}

while (($#)); do
  case "$1" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --checkpoint-manifest) CHECKPOINT_MANIFEST="${2:-}"; shift 2 ;;
    --checkpoint-blob) CHECKPOINT_BLOB="${2:-}"; shift 2 ;;
    --prediction-ledger) PREDICTION_LEDGER="${2:-}"; shift 2 ;;
    --sentences) SENTENCES="${2:-}"; shift 2 ;;
    --candidates-out) CANDIDATES_OUT="${2:-}"; shift 2 ;;
    --candidates) CANDIDATES="${2:-}"; shift 2 ;;
    --response-ledger) RESPONSE_LEDGER="${2:-}"; shift 2 ;;
    --model-blob) MODEL_BLOB="${2:-}"; shift 2 ;;
    --pilot-candidates) PILOT_CANDIDATES="${2:-}"; shift 2 ;;
    --pilot-selection) PILOT_SELECTION="${2:-}"; shift 2 ;;
    --capture-index) CAPTURE_INDEX="${2:-}"; shift 2 ;;
    --branch) BRANCH="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$RUN_ID" ]] || die "--run-id is required so a retry can inspect the same output tree"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "invalid --run-id: $RUN_ID"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be an integer"
[[ "$MODE" == "simple" || "$MODE" == "corrective" ]] || die "--mode must be simple or corrective"

cd "$SOURCE_ROOT"
readonly RUN_ROOT="output/$RUN_ID"

[[ "$(git branch --show-current)" == "$BRANCH" ]] \
  || die "checkout branch must be '$BRANCH' before this script can pull it"

json_equals() {
  local path="$1"
  local dotted_key="$2"
  local expected_json="$3"
  [[ -f "$path" ]] || return 1
  uv run --frozen python -B - "$path" "$dotted_key" "$expected_json" <<'PY'
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
  uv run --frozen python -B - "$RUN_ROOT" "$manifest" <<'PY'
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

doctor_complete() {
  json_equals "$RUN_ROOT/manifests/00-checkout-manifest.json" status '"pass"'
}

reconcile_complete() {
  json_equals "$RUN_ROOT/audit/section5-evidence-reconciliation.json" status '"reconciled_secondary_evidence"'
}

fetch_complete() {
  archive_ready
}

prepare_complete() {
  json_equals "$RUN_ROOT/manifests/03-data-preparation-manifest.json" \
    byte_identical_independent_materializations true \
    && [[ -f "$RUN_ROOT/data-prepared/split-manifest.json" ]] \
    && [[ -f "$RUN_ROOT/data-prepared/train.jsonl" ]]
}

plan_complete() {
  json_equals "$RUN_ROOT/manifests/model-train-dry-run.json" status '"planned"' \
    && json_equals "$RUN_ROOT/manifests/model-train-dry-run.json" training_seed "$SEED"
}

candidate_complete() {
  local execution="$1"
  local output="$2"
  json_equals "$RUN_ROOT/manifests/model-generate-candidates-$execution.json" status '"completed"' \
    && [[ -f "$RUN_ROOT/$output" ]]
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
    && [[ -f "$RUN_ROOT/manifests/score-manifest.json" ]]
}

ensure_doctor() {
  skip_or_run doctor doctor_complete \
    uv run --frozen python -B phase_b.py doctor \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID"
}

ensure_reconcile() {
  skip_or_run reconcile reconcile_complete \
    uv run --frozen python -B phase_b.py reconcile \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID"
}

ensure_fetch() {
  skip_or_run fetch fetch_complete \
    uv run --frozen python -B phase_b.py fetch \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID"
}

ensure_prepare() {
  skip_or_run prepare prepare_complete \
    uv run --frozen python -B phase_b.py prepare \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID"
}

ensure_bootstrap() {
  ensure_doctor
  ensure_reconcile
  ensure_fetch
  ensure_prepare
}

ensure_plan() {
  ensure_bootstrap
  skip_or_run "model train dry-run (seed $SEED)" plan_complete \
    uv run --frozen python -B phase_b.py model train \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" \
      --execution dry-run --seed "$SEED"
}

ensure_generate_live() {
  ensure_bootstrap
  [[ -n "$CHECKPOINT_MANIFEST" ]] || die "generate-live requires --checkpoint-manifest"
  [[ -n "$CHECKPOINT_BLOB" ]] || die "generate-live requires --checkpoint-blob"
  local sentences="${SENTENCES:-data-prepared/development.jsonl}"
  local output="${CANDIDATES_OUT:-predictions/dev/seed-$SEED-candidates.jsonl}"
  skip_or_run "model generate-candidates live" \
    "candidate_live_complete" \
    uv run --frozen --python 3.10.20 python -B phase_b.py model generate-candidates \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" --execution live \
      --sentences "$sentences" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
      --checkpoint-blob "$CHECKPOINT_BLOB" --candidates-out "$output"
}

candidate_live_complete() {
  local output="${CANDIDATES_OUT:-predictions/dev/seed-$SEED-candidates.jsonl}"
  candidate_complete live "$output"
}

ensure_generate_replay() {
  ensure_bootstrap
  [[ -n "$CHECKPOINT_MANIFEST" ]] || die "generate-replay requires --checkpoint-manifest"
  [[ -n "$PREDICTION_LEDGER" ]] || die "generate-replay requires --prediction-ledger"
  local sentences="${SENTENCES:-data-prepared/test.jsonl}"
  local output="${CANDIDATES_OUT:-predictions/test/candidates.jsonl}"
  skip_or_run "model generate-candidates replay" \
    "candidate_replay_complete" \
    uv run --frozen --python 3.10.20 python -B phase_b.py model generate-candidates \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" --execution replay \
      --sentences "$sentences" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
      --prediction-ledger "$PREDICTION_LEDGER" --candidates-out "$output"
}

candidate_replay_complete() {
  local output="${CANDIDATES_OUT:-predictions/test/candidates.jsonl}"
  candidate_complete replay "$output"
}

ensure_threshold() {
  ensure_bootstrap
  skip_or_run "select-threshold" threshold_complete \
    uv run --frozen --python 3.10.20 python -B phase_b.py select-threshold \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" --candidates "$CANDIDATES"
}

ensure_verifier_dry() {
  ensure_bootstrap
  skip_or_run "verifier dry-run ($MODE)" verifier_dry_complete \
    uv run --frozen --python 3.10.20 python -B phase_b.py verifier \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" \
      --mode "$MODE" --execution dry-run
}

verifier_dry_complete() { verifier_complete dry-run planned; }

ensure_verifier_replay() {
  ensure_bootstrap
  skip_or_run "verifier replay ($MODE)" verifier_replay_complete \
    uv run --frozen --python 3.10.20 python -B phase_b.py verifier \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" \
      --mode "$MODE" --execution replay --response-ledger "$RESPONSE_LEDGER"
}

verifier_replay_complete() { verifier_complete replay completed; }

ensure_verifier_live() {
  ensure_bootstrap
  skip_or_run "verifier live ($MODE)" verifier_live_complete \
    uv run --frozen --python 3.10.20 python -B phase_b.py verifier \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" \
      --mode "$MODE" --execution live --model-blob "$MODEL_BLOB"
}

verifier_live_complete() { verifier_complete live completed; }

ensure_pilot_live() {
  ensure_doctor
  [[ -n "$PILOT_CANDIDATES" ]] || die "pilot-live requires --pilot-candidates"
  [[ -n "$PILOT_SELECTION" ]] || die "pilot-live requires --pilot-selection"
  local sentences="${SENTENCES:-data-prepared/development.jsonl}"
  skip_or_run "development-pilot live verifier ($MODE)" verifier_live_complete \
    uv run --frozen --python 3.10.20 python -B phase_b.py verifier \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" \
      --mode "$MODE" --execution live --sentences "$sentences" \
      --candidates "$PILOT_CANDIDATES" --pilot-selection "$PILOT_SELECTION" \
      --model-blob "$MODEL_BLOB"
}

ensure_pilot_audit() {
  ensure_bootstrap
  skip_or_run "pilot-verifier audit" pilot_audit_complete \
    uv run --frozen --python 3.10.20 python -B phase_b.py pilot-verifier \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" \
      --evidence-class development-pilot --capture-index "$CAPTURE_INDEX"
}

ensure_score() {
  ensure_bootstrap
  skip_or_run score score_complete \
    uv run --frozen --python 3.10.20 python -B phase_b.py score \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID"
}

note "updating source checkout from origin/$BRANCH"
git pull --ff-only origin "$BRANCH"
note "synchronizing the locked environment"
uv sync --frozen

case "$STAGE" in
  bootstrap) ensure_bootstrap ;;
  plan) ensure_plan ;;
  generate-live) ensure_generate_live ;;
  generate-replay) ensure_generate_replay ;;
  threshold) ensure_threshold ;;
  verifier-dry) ensure_verifier_dry ;;
  verifier-replay) ensure_verifier_replay ;;
  verifier-live) ensure_verifier_live ;;
  pilot-live) ensure_pilot_live ;;
  pilot-audit) ensure_pilot_audit ;;
  score) ensure_score ;;
  *) die "unknown --stage: $STAGE (run with --help for the supported list)" ;;
esac

note "stage '$STAGE' is complete for run '$RUN_ID'"
