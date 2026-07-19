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
# Keep argparse/help and provenance strings UTF-8 even on a minimally configured
# external shell; several retained historical help texts contain Unicode.
export PYTHONUTF8=1

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

RUN_ID=""
STAGE="available"
SEED=42
BRANCH="refactor"
MODE="simple"
ALLOW_LEGACY_DIAGNOSTIC=false
ALLOW_LIVE_SMOKE=false
BLOCKED_COUNT=0
RESPONSE_LEDGER="inputs/frozen-simple-responses.jsonl"
CAPTURE_INDEX="inputs/pilot/capture-index.json"
CHECKPOINT_MANIFEST=""
CHECKPOINT_BLOB=""
PREDICTION_LEDGER=""
SENTENCES=""
CANDIDATES_OUT=""
CANDIDATES="predictions/dev/development-candidates.jsonl"
MODEL_BLOB="inputs/ollama/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312"
MODEL_BLOB_SOURCE=""
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

For a new run ID, `fetch` first searches other local `output/<run-id>/` trees
for the configured immutable CODE-ACCORD archive. A matching size-and-MD5
archive is hard-linked into the new run when possible (copied only when the
filesystem does not support hard links), then `fetch` verifies it and writes
the new run's own acquisition manifest. A network download occurs only when
no verified local run cache exists.

Stages:
  available          parser-check every command and run every available stage (default)
  smoke              non-publication one-candidate real GPU/Ollama end-to-end smoke
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
  verifier-live      run the gated simple/corrective live verifier
  pilot-live         run one gated development-pilot capture
  pilot-audit        audit the four separately captured pilot runs
  score              run the documented strict scorer

Options:
  --run-id ID                 Existing or new Phase B run ID (required)
  --stage NAME                One stage from the list above
  --seed N                    Checkpoint seed (default: 42)
  --allow-legacy-diagnostic   Required acknowledgement for legacy-train
  --allow-live-smoke          Required acknowledgement before real Ollama smoke calls
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
  --branch NAME               Remote branch to pull (default: refactor)
  -h, --help                  Show this help

Examples:
  scripts/phase_b_debug.sh --run-id path-a-bootstrap-20260718T120000Z
  scripts/phase_b_debug.sh --run-id path-a-bootstrap-20260718T120000Z --stage plan --seed 42
  scripts/phase_b_debug.sh --run-id path-a-simple-live-8 --stage smoke --seed 42 \
    --allow-live-smoke --model-blob-source ~/.ollama/models/blobs/<locked-blob>
  scripts/phase_b_debug.sh --run-id path-a-bootstrap-20260718T120000Z --stage train-live --seed 42
  scripts/phase_b_debug.sh --run-id path-a-bootstrap-20260718T120000Z \
    --stage legacy-train --seed 42 --allow-legacy-diagnostic
  scripts/phase_b_debug.sh --run-id path-a-bootstrap-20260718T120000Z --stage generate-live \
    --checkpoint-manifest checkpoints/seed-42/checkpoint-manifest.json \
    --checkpoint-blob checkpoints/seed-42/checkpoint.pt

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
    --allow-legacy-diagnostic) ALLOW_LEGACY_DIAGNOSTIC=true; shift ;;
    --allow-live-smoke) ALLOW_LIVE_SMOKE=true; shift ;;
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

find_verified_archive_cache() {
  uv run --frozen python -B - "$RUN_ROOT" <<'PY'
import hashlib
import sys
from pathlib import Path

from acquisition import archive_contract_from_config
from config import load_pipeline_config

run_root = Path(sys.argv[1]).resolve(strict=False)
source_root = Path.cwd().resolve()
contract = archive_contract_from_config(
    load_pipeline_config(source_root, "configs/phase_b_path_a.json")
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
  uv run --frozen python -B - "$marker" "$commit" <<'PY'
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
  uv run --frozen python -B phase_b.py --help >/dev/null
  uv run --frozen python -B phase_b.py doctor --help >/dev/null
  uv run --frozen python -B phase_b.py reconcile --help >/dev/null
  uv run --frozen python -B phase_b.py fetch --help >/dev/null
  uv run --frozen python -B phase_b.py prepare --help >/dev/null
  uv run --frozen python -B phase_b.py model --help >/dev/null
  uv run --frozen python -B phase_b.py model train --help >/dev/null
  uv run --frozen python -B phase_b.py model generate-candidates --help >/dev/null
  uv run --frozen python -B phase_b.py select-threshold --help >/dev/null
  uv run --frozen python -B phase_b.py verifier --help >/dev/null
  uv run --frozen python -B phase_b.py pilot-verifier --help >/dev/null
  uv run --frozen python -B phase_b.py score --help >/dev/null
  uv run --frozen --python 3.10.20 python -B train_span.py --help >/dev/null
  uv run --frozen python -B smoke.py --help >/dev/null
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
  if ! fetch_complete; then
    seed_archive_from_local_cache
  fi
  skip_or_run fetch fetch_complete \
    uv run --frozen python -B phase_b.py fetch \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID"
}

ensure_prepare() {
  if [[ -f "$RUN_ROOT/manifests/03-data-preparation-manifest.json" ]] && ! prepare_complete; then
    die "prepared tree is incomplete or from an older artifact contract; do not mix it with this run. Start a new --run-id so prepare can materialize a complete fresh tree."
  fi
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

ensure_train_live() {
  ensure_bootstrap
  skip_or_run "canonical model training (seed $SEED)" train_live_complete \
    uv run --frozen python -B phase_b.py model train \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" \
      --execution live --seed "$SEED"
}

write_legacy_completion_marker() {
  local checkpoint
  local marker
  checkpoint="$(legacy_checkpoint_path)"
  marker="$(legacy_completion_marker)"
  uv run --frozen python -B - "$checkpoint" "$marker" "$SEED" <<'PY'
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
  uv run --frozen --python 3.10.20 python -B - <<'PY'
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
  uv run --frozen --python 3.10.20 python -B train_span.py \
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
  skip_or_run "model generate-candidates live" \
    "candidate_live_complete" \
    uv run --frozen --python 3.10.20 python -B phase_b.py model generate-candidates \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" --execution live \
      --sentences "$sentences" --checkpoint-manifest "$checkpoint_manifest" \
      --checkpoint-blob "$checkpoint_blob" --candidates-out "$output"
}

candidate_live_complete() {
  local output="${CANDIDATES_OUT:-predictions/dev/seed-$SEED-candidates.jsonl}"
  candidate_complete live "$output"
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
    uv run --frozen --python 3.10.20 python -B phase_b.py model generate-candidates \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" --execution replay \
      --sentences "$sentences" --checkpoint-manifest "$checkpoint_manifest" \
      --prediction-ledger "$prediction_ledger" --candidates-out "$output"
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

smoke_verifier_live_complete() {
  json_equals "$RUN_ROOT/manifests/verifier-smoke-$MODE-live.json" status '"completed"' \
    && [[ -f "$RUN_ROOT/verifier/smoke/$MODE/verdicts.jsonl" ]]
}

materialize_model_blob() {
  [[ -f "$RUN_ROOT/$MODEL_BLOB" ]] && return 0
  [[ -n "$MODEL_BLOB_SOURCE" ]] \
    || die "smoke requires the locked run-local --model-blob: $MODEL_BLOB; provide --model-blob-source /path/to/the-existing-Ollama-blob to materialize it"
  [[ -f "$MODEL_BLOB_SOURCE" && ! -L "$MODEL_BLOB_SOURCE" ]] \
    || die "--model-blob-source must name a regular existing file: $MODEL_BLOB_SOURCE"
  mkdir -p "$(dirname "$RUN_ROOT/$MODEL_BLOB")"
  if ln "$MODEL_BLOB_SOURCE" "$RUN_ROOT/$MODEL_BLOB" 2>/dev/null; then
    note "hard-linked locked model blob into this run"
  else
    note "copying locked model blob into this run (hard link unavailable)"
    cp -p -- "$MODEL_BLOB_SOURCE" "$RUN_ROOT/$MODEL_BLOB"
  fi
}

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
    blocked "canonical checkpoint inputs are missing; for a diagnostic-only checkpoint rerun with: bash scripts/phase_b_debug.sh --run-id $RUN_ID --stage legacy-train --seed $SEED --allow-legacy-diagnostic"
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
  blocked "verifier live and pilot-live require the recorded B-07 user go/no-go approval and separately selected/captured inputs"
  maybe_pilot_audit
  maybe_score
}

ensure_publishable() {
  ensure_plan
  ensure_train_live
  ensure_generate_live
  die "seed-$SEED canonical training and development candidate generation are complete. The seed-42 smoke/restart evidence must be reviewed at B-07 before running and aggregating seeds 42-49 or executing the live verifier. Re-run completed stages with the same run ID after that decision."
}

ensure_smoke() {
  ensure_bootstrap
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
    uv run --frozen --python 3.10.20 python -B phase_b.py model generate-candidates \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" --execution live \
      --sentences data-prepared/test.jsonl \
      --checkpoint-manifest "checkpoints/seed-$SEED/checkpoint-manifest.json" \
      --checkpoint-blob "checkpoints/seed-$SEED/checkpoint.pt" \
      --candidates-out "predictions/smoke/test-seed-$SEED-candidates.jsonl"
  fi
  if [[ ! -f "$RUN_ROOT/predictions/smoke/test-candidates.jsonl" ]]; then
    uv run --frozen python -B smoke.py prepare \
      --source-dev "$RUN_ROOT/predictions/dev/seed-$SEED-candidates.jsonl" \
      --generated-test "$RUN_ROOT/predictions/smoke/test-seed-$SEED-candidates.jsonl" \
      --destination-run "$RUN_ROOT"
  fi
  for MODE in simple corrective; do
    if ! smoke_verifier_live_complete; then
      uv run --frozen --python 3.10.20 python -B phase_b.py verifier \
        --config configs/phase_b_path_a.json --run-id "$RUN_ID" --mode "$MODE" --execution live \
        --artifact-prefix smoke \
        --sentences data-prepared/test.jsonl \
        --candidates predictions/smoke/test-live-candidate.jsonl \
        --warmup-sentences data-prepared/development.jsonl \
        --warmup-candidates predictions/smoke/warmup-candidate.jsonl \
        --model-blob "$MODEL_BLOB"
    fi
    if [[ ! -f "$RUN_ROOT/verifier/smoke/$MODE/pseudo-seed-verdicts.jsonl" ]]; then
      uv run --frozen python -B smoke.py clone-verdicts \
        --source "$RUN_ROOT/verifier/smoke/$MODE/verdicts.jsonl" \
        --candidates "$RUN_ROOT/predictions/smoke/test-candidates.jsonl" \
        --destination "$RUN_ROOT/verifier/smoke/$MODE/pseudo-seed-verdicts.jsonl"
    fi
  done
  if [[ ! -f "$RUN_ROOT/predictions/smoke/threshold-selection.json" ]]; then
    uv run --frozen --python 3.10.20 python -B phase_b.py select-threshold \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" \
      --candidates predictions/smoke/development-candidates.jsonl \
      --out predictions/smoke/threshold-selection.json
  fi
  if ! json_equals "$RUN_ROOT/smoke/score/metrics/metrics.json" nonpublication_smoke true; then
    uv run --frozen --python 3.10.20 python -B phase_b.py score \
      --config configs/phase_b_path_a.json --run-id "$RUN_ID" \
      --candidates predictions/smoke/test-candidates.jsonl \
      --simple-verdicts verifier/smoke/simple/pseudo-seed-verdicts.jsonl \
      --corrective-verdicts verifier/smoke/corrective/pseudo-seed-verdicts.jsonl \
      --threshold-selection predictions/smoke/threshold-selection.json \
      --nonpublication-smoke --output-prefix smoke/score
  fi
}

note "updating source checkout from origin/$BRANCH"
git pull --ff-only origin "$BRANCH"
note "synchronizing the locked environment"
uv sync --frozen

case "$STAGE" in
  available) ensure_available ;;
  smoke) ensure_smoke ;;
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
  publishable) ensure_publishable ;;
  *) die "unknown --stage: $STAGE (run with --help for the supported list)" ;;
esac

if [[ "$STAGE" == "available" && "$BLOCKED_COUNT" -gt 0 ]]; then
  printf '\nphase_b_debug: %s stage(s) remain blocked for run %q; see the BLOCKED lines above.\n' \
    "$BLOCKED_COUNT" "$RUN_ID" >&2
  exit 2
fi

note "stage '$STAGE' is complete for run '$RUN_ID'"
