"""Internal stage dispatcher used by the primary Phase B launcher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from acquisition import fetch_run
from config import load_pipeline_config
from doctor import run_doctor
from model import generate_candidates, plan_training
from threshold import select_threshold
from phase_b_io import DataContractError, atomic_write_json, sha256_file
from paths import PathContractError, RunLayout, discover_source_root
from pilot import PilotInputs, run_verifier_pilot
from preparation import prepare_run
from publication import assemble_seed_candidates, prepare_verifier_pilot
from reconciliation import reconcile_section5_evidence
from scoring import ScoreInputs, score_run
from verifier import run_verifier


DEFAULT_CONFIG = "configs/phase_b_path_a.json"


def _load_manifest(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be a JSON object: {path}")
    return value


def _manifest_outputs(layout: RunLayout, manifest: dict) -> dict[str, str]:
    declared = manifest.get("outputs")
    if isinstance(declared, dict):
        outputs = declared
    else:
        outputs = {}
        for field in ("generation_plan_output", "candidates_output"):
            relative = manifest.get(field)
            if isinstance(relative, str):
                path = layout.resolve(relative, must_exist=True)
                outputs[relative] = sha256_file(path)
        if (
            manifest.get("stage") == "model-generate-candidates"
            and manifest.get("execution_mode") == "live"
        ):
            ledger = manifest.get("inputs", {}).get("prediction_ledger")
            if isinstance(ledger, dict) and isinstance(ledger.get("path"), str):
                outputs[ledger["path"]] = ledger.get("sha256")
    if not outputs:
        raise DataContractError("stage producer manifest does not declare an output")
    for relative, expected in outputs.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise DataContractError("stage producer manifest has an invalid output binding")
        path = layout.resolve(relative, must_exist=True)
        if not path.is_file() or sha256_file(path) != expected:
            raise DataContractError(f"stage output differs from its producer: {relative}")
    return outputs


def _same_run_seal_path(producer_manifest: Path) -> Path:
    return producer_manifest.with_name(f"same-run-{producer_manifest.name}")


def _write_same_run_seal(
    layout: RunLayout, producer_manifest: Path, manifest: dict
) -> Path:
    outputs = _manifest_outputs(layout, manifest)
    seal_path = _same_run_seal_path(producer_manifest)
    document = {
        "schema_version": "phase-b-same-run-stage-seal-1.0",
        "run_id": layout.run_id,
        "producer_manifest": {
            "path": layout.relative_identity(producer_manifest),
            "sha256": sha256_file(producer_manifest),
        },
        "outputs": outputs,
    }
    atomic_write_json(seal_path, document)
    return seal_path


def _validate_same_run_seal(
    layout: RunLayout, producer_manifest: Path, manifest: dict
) -> None:
    outputs = _manifest_outputs(layout, manifest)
    seal = _load_manifest(
        _same_run_seal_path(producer_manifest), "same-run stage seal"
    )
    expected = {
        "schema_version": "phase-b-same-run-stage-seal-1.0",
        "run_id": layout.run_id,
        "producer_manifest": {
            "path": layout.relative_identity(producer_manifest),
            "sha256": sha256_file(producer_manifest),
        },
        "outputs": outputs,
    }
    if seal != expected:
        raise DataContractError("stage seal does not authenticate the selected run")


def _find_written_manifest(
    layout: RunLayout, pattern: str, manifest: dict
) -> Path:
    matches = [
        path
        for path in layout.resolve("manifests").glob(pattern)
        if _load_manifest(path, "stage producer manifest") == manifest
    ]
    if len(matches) != 1:
        raise DataContractError(
            "could not identify exactly one newly written stage producer manifest"
        )
    return matches[0]


def _same_run_prediction_replay_input(
    layout: RunLayout,
    checkpoint_manifest: Path,
    sentences: Path,
    candidates_out: Path,
) -> Path:
    checkpoint = _load_manifest(checkpoint_manifest, "checkpoint manifest")
    seed = checkpoint.get("training_seed")
    if not isinstance(seed, int):
        raise DataContractError("checkpoint manifest lacks an integer training_seed")
    ledger = layout.resolve(
        candidates_out.parent.relative_to(layout.run_root)
        / f"seed-{seed}-prediction-ledger.jsonl",
        must_exist=True,
    )
    live_manifest_path = layout.resolve(
        f"manifests/model-generate-candidates-live-seed-{seed}-{sentences.stem}.json",
        must_exist=True,
    )
    live = _load_manifest(live_manifest_path, "same-run live prediction manifest")
    ledger_binding = live.get("inputs", {}).get("prediction_ledger", {})
    if (
        live.get("stage") != "model-generate-candidates"
        or live.get("execution_mode") != "live"
        or live.get("status") != "completed"
        or live.get("training_seed") != seed
        or ledger_binding.get("path") != layout.relative_identity(ledger)
        or ledger_binding.get("sha256") != sha256_file(ledger)
        or live.get("inputs", {}).get("checkpoint_manifest", {}).get("path")
        != layout.relative_identity(checkpoint_manifest)
        or live.get("inputs", {}).get("checkpoint_manifest", {}).get("sha256")
        != sha256_file(checkpoint_manifest)
        or live.get("inputs", {}).get("prepared_sentences", {}).get("path")
        != layout.relative_identity(sentences)
        or live.get("inputs", {}).get("prepared_sentences", {}).get("sha256")
        != sha256_file(sentences)
    ):
        raise DataContractError(
            "prediction replay input is not authenticated by its same-run live producer"
        )
    _validate_same_run_seal(layout, live_manifest_path, live)
    return ledger


def _same_run_verifier_replay_input(layout: RunLayout, mode: str) -> Path:
    responses = layout.resolve(f"verifier/{mode}/responses.jsonl", must_exist=True)
    live_manifest_path = layout.resolve(
        f"manifests/verifier-{mode}-live.json", must_exist=True
    )
    live = _load_manifest(live_manifest_path, "same-run live verifier manifest")
    relative = layout.relative_identity(responses)
    if (
        live.get("condition_id") != f"VER-{mode.upper()}"
        or live.get("execution_mode") != "live"
        or live.get("status") != "completed"
        or live.get("outputs", {}).get(relative) != sha256_file(responses)
    ):
        raise DataContractError(
            "verifier replay input is not authenticated by its same-run live producer"
        )
    _validate_same_run_seal(layout, live_manifest_path, live)
    return responses


def _same_run_verifier_cache_input(layout: RunLayout, mode: str) -> Path:
    cache = layout.resolve(
        f"inputs/recovery/verifier-{mode}-responses.jsonl", must_exist=True
    )
    provenance_path = layout.resolve(
        f"inputs/recovery/verifier-{mode}-responses.manifest.json", must_exist=True
    )
    checkout_path = layout.resolve(
        "manifests/00-checkout-manifest.json", must_exist=True
    )
    provenance = _load_manifest(provenance_path, "verifier recovery provenance")
    expected = {
        "schema_version": "phase-b-same-run-verifier-recovery-1.0",
        "run_id": layout.run_id,
        "mode": mode,
        "response_cache": {
            "path": layout.relative_identity(cache),
            "sha256": sha256_file(cache),
        },
        "source_response": {
            "path": f"verifier/{mode}/responses.jsonl",
            "sha256": sha256_file(cache),
        },
        "checkout_manifest": {
            "path": layout.relative_identity(checkout_path),
            "sha256": sha256_file(checkout_path),
        },
    }
    if provenance != expected:
        raise DataContractError(
            "verifier recovery cache is not authenticated to the selected run"
        )
    return cache


def _validate_pilot_capture_seals(layout: RunLayout, capture_index: Path) -> None:
    index = _load_manifest(capture_index, "pilot capture index")
    if index.get("run_id") != layout.run_id:
        raise DataContractError("pilot capture index carries another run identity")
    captures = index.get("captures")
    if not isinstance(captures, list) or len(captures) != 4:
        raise DataContractError("pilot capture index must contain four same-run captures")
    for capture in captures:
        if not isinstance(capture, dict):
            raise DataContractError("pilot capture index contains a malformed capture")
        mode = capture.get("mode")
        prefix = capture.get("artifact_prefix")
        if mode not in {"simple", "corrective"} or not isinstance(prefix, str):
            raise DataContractError("pilot capture index contains an invalid namespace")
        producer = layout.resolve(
            f"manifests/verifier-{prefix.replace('/', '-')}-{mode}-live.json",
            must_exist=True,
        )
        manifest = _load_manifest(producer, "pilot verifier producer manifest")
        _validate_same_run_seal(layout, producer, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python phase_b.py",
        description=(
            "Internal dispatcher for phase_b.sh. This B-05/B-06/B-07 and B-05U "
            "development slice implements "
            "doctor, secondary Section 5 evidence reconciliation, immutable "
            "CODE-ACCORD fetch/preparation, model train planning and candidate "
            "generation (dry-run/replay/live), eight-seed candidate assembly, "
            "development threshold and label-blind pilot selection, frozen "
            "verifier request/replay instrumentation, development-pilot auditing, "
            "and offline CODE-STRICT-1 scoring."
        ),
    )
    parser.add_argument(
        "--source-root",
        help="Direct src checkout root; normally discovered from the current directory.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    doctor = subparsers.add_parser("doctor", help="Validate checkout/environment and create a run")
    doctor.add_argument("--config", default=DEFAULT_CONFIG)
    doctor.add_argument("--run-id", required=True)

    fetch = subparsers.add_parser(
        "fetch", help="Download and verify the immutable CODE-ACCORD archive"
    )
    fetch.add_argument("--config", default=DEFAULT_CONFIG)
    fetch.add_argument("--run-id", required=True)

    reconcile = subparsers.add_parser(
        "reconcile",
        help="Audit frozen Section 5 ledgers without promoting them to canonical evidence",
    )
    reconcile.add_argument("--config", default=DEFAULT_CONFIG)
    reconcile.add_argument("--run-id", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Audit then reconstruct leakage-safe CODE-ACCORD and CODE-SPLIT-1"
    )
    prepare.add_argument("--config", default=DEFAULT_CONFIG)
    prepare.add_argument("--run-id", required=True)

    verifier = subparsers.add_parser(
        "verifier",
        help="Dry-run, execute, or replay the frozen CODE verifier contract",
    )
    verifier.add_argument("--config", default=DEFAULT_CONFIG)
    verifier.add_argument("--run-id", required=True)
    verifier.add_argument("--mode", required=True, choices=["simple", "corrective"])
    verifier.add_argument(
        "--execution", required=True, choices=["dry-run", "live", "replay"]
    )
    verifier.add_argument("--sentences", help="Run-relative prepared-sentence JSONL")
    verifier.add_argument("--candidates", help="Run-relative candidate JSONL")
    verifier.add_argument(
        "--warmup-sentences",
        help="Run-relative development sentences required by live execution",
    )
    verifier.add_argument(
        "--warmup-candidates",
        help="Run-relative single development warm-up candidate required by live execution",
    )
    verifier.add_argument(
        "--resume-from-cache",
        action="store_true",
        help="Resume live execution from its canonical cache in the selected run",
    )
    verifier.add_argument("--ollama-url", default="http://localhost:11434")
    verifier.add_argument(
        "--model-blob",
        help="Run-relative frozen Ollama model blob required by live execution",
    )
    verifier.add_argument(
        "--pilot-selection",
        help="Run-relative predeclared development-pilot selection bound by live execution",
    )
    verifier.add_argument(
        "--artifact-prefix",
        help="Optional run-relative namespace for noncanonical verifier artifacts",
    )

    pilot = subparsers.add_parser(
        "pilot-verifier",
        help="Audit two independent development-only response ledgers per verifier mode",
    )
    pilot.add_argument("--config", default=DEFAULT_CONFIG)
    pilot.add_argument("--run-id", required=True)
    pilot.add_argument(
        "--evidence-class",
        required=True,
        choices=["development-pilot", "synthetic-fixture"],
    )
    pilot.add_argument("--sentences", help="Run-relative development sentences JSONL")
    pilot.add_argument("--gold", help="Run-relative development gold JSONL")
    pilot.add_argument("--candidates", help="Run-relative predeclared pilot candidates JSONL")
    pilot.add_argument(
        "--warmup-candidates",
        help="Run-relative fixed single development warm-up candidate JSONL",
    )
    pilot.add_argument("--split-manifest", help="Run-relative authoritative split manifest")
    pilot.add_argument(
        "--candidate-index", help="Run-relative full development candidate index"
    )
    pilot.add_argument(
        "--pilot-selection", help="Run-relative pre-call pilot-selection manifest"
    )
    pilot.add_argument(
        "--threshold-selection", help="Run-relative development threshold-selection JSON"
    )
    pilot.add_argument(
        "--capture-index",
        required=True,
        help="Run-relative index of four complete same-run pilot namespaces",
    )

    assemble = subparsers.add_parser(
        "assemble-candidates",
        help="Validate and combine the eight seed-specific candidate ledgers",
    )
    assemble.add_argument("--config", default=DEFAULT_CONFIG)
    assemble.add_argument("--run-id", required=True)
    assemble.add_argument("--split", required=True, choices=["development", "test"])

    prepare_pilot = subparsers.add_parser(
        "prepare-pilot",
        help="Freeze the label-blind B-07 development pilot inputs",
    )
    prepare_pilot.add_argument("--config", default=DEFAULT_CONFIG)
    prepare_pilot.add_argument("--run-id", required=True)

    model = subparsers.add_parser(
        "model", help="Canonical encoder candidate-generation adapter"
    )
    model_actions = model.add_subparsers(dest="model_action", required=True)
    generate = model_actions.add_parser(
        "generate-candidates",
        help="Plan or replay CODE-STRICT-1 candidate generation for one training seed",
    )
    generate.add_argument("--config", default=DEFAULT_CONFIG)
    generate.add_argument("--run-id", required=True)
    generate.add_argument(
        "--execution", required=True, choices=["dry-run", "replay", "live"]
    )
    generate.add_argument("--sentences", help="Run-relative prepared-sentence JSONL")
    generate.add_argument(
        "--checkpoint-manifest",
        required=True,
        help="Run-relative encoder checkpoint identity manifest",
    )
    generate.add_argument(
        "--checkpoint-blob",
        help="Run-relative encoder checkpoint weights required by live execution",
    )
    generate.add_argument(
        "--base-model",
        help="Base model id or path for live execution (default: frozen recipe base model)",
    )
    generate.add_argument(
        "--device", help="Torch device for live execution (default: cuda if available else cpu)"
    )
    generate.add_argument("--candidates-out", help="Run-relative candidate output JSONL")

    train = model_actions.add_parser(
        "train", help="Plan or run deterministic canonical encoder training for one seed"
    )
    train.add_argument("--config", default=DEFAULT_CONFIG)
    train.add_argument("--run-id", required=True)
    train.add_argument("--execution", required=True, choices=["dry-run", "live"])
    train.add_argument("--seed", required=True, type=int)

    threshold = subparsers.add_parser(
        "select-threshold",
        help="Select the development confidence threshold (VER-CONFIDENCE)",
    )
    threshold.add_argument("--config", default=DEFAULT_CONFIG)
    threshold.add_argument("--run-id", required=True)
    threshold.add_argument(
        "--candidates",
        required=True,
        help="Run-relative eight-seed development candidates JSONL",
    )
    threshold.add_argument("--gold", help="Run-relative development gold JSONL")
    threshold.add_argument("--split-manifest", help="Run-relative split manifest")
    threshold.add_argument(
        "--candidate-index",
        help="Run-relative full development candidate index bound into the selection",
    )
    threshold.add_argument("--out", help="Run-relative threshold-selection.json output")

    score = subparsers.add_parser("score", help="Offline score frozen candidates and verdicts")
    score.add_argument("--config", default=DEFAULT_CONFIG)
    score.add_argument("--run-id", required=True)
    score.add_argument("--gold", help="Run-relative gold JSONL path")
    score.add_argument("--candidates", help="Run-relative candidate JSONL path")
    score.add_argument("--simple-verdicts", help="Run-relative simple-verdict JSONL path")
    score.add_argument("--corrective-verdicts", help="Run-relative corrective-verdict JSONL path")
    score.add_argument(
        "--threshold-selection", help="Run-relative development threshold JSON path"
    )
    score.add_argument(
        "--nonpublication-smoke",
        action="store_true",
        help="Mark a pseudo-seed smoke score as non-publication evidence",
    )
    score.add_argument(
        "--output-prefix",
        help="Optional run-relative namespace for noncanonical score artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source_root = discover_source_root(Path(args.source_root) if args.source_root else None)
        config = load_pipeline_config(source_root, args.config)
        layout = RunLayout(source_root=source_root, run_id=args.run_id)
        if args.stage == "doctor":
            manifest, passed = run_doctor(layout, config)
            print(
                json.dumps(
                    {"run_id": args.run_id, "status": manifest["status"]},
                    sort_keys=True,
                )
            )
            return 0 if passed else 2

        if args.stage == "fetch":
            manifest = fetch_run(layout, config)
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "status": "fetched",
                        "archive_sha256": manifest["archive"]["sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.stage == "reconcile":
            audit = reconcile_section5_evidence(layout, config)
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "status": audit["status"],
                        "claim_family_count": audit["summary"]["claim_family_count"],
                        "canonical_ready_claim_family_count": 0,
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.stage == "prepare":
            manifest = prepare_run(layout, config)
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "status": "prepared",
                        "dataset_tree_sha256": manifest["dataset_tree_sha256"],
                        "byte_identical": manifest[
                            "byte_identical_independent_materializations"
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.stage == "assemble-candidates":
            result = assemble_seed_candidates(layout, config, split=args.split)
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "status": "assembled",
                        **result,
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.stage == "prepare-pilot":
            result = prepare_verifier_pilot(layout, config)
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "status": "prepared_pilot",
                        **result,
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.stage == "verifier":
            configured = config.paths
            if args.resume_from_cache and args.execution != "live":
                raise DataContractError("--resume-from-cache is valid only for live verifier execution")
            response_ledger = (
                _same_run_verifier_replay_input(layout, args.mode)
                if args.execution == "replay"
                else None
            )
            manifest = run_verifier(
                layout,
                config,
                mode=args.mode,
                execution_mode=args.execution,
                sentences_path=layout.resolve(
                    args.sentences or configured["sentences"], must_exist=True
                ),
                candidates_path=layout.resolve(
                    args.candidates or configured["candidates"], must_exist=True
                ),
                warmup_sentences_path=(
                    layout.resolve(
                        args.warmup_sentences or configured["warmup_sentences"],
                        must_exist=True,
                    )
                    if args.execution == "live"
                    else None
                ),
                warmup_candidates_path=(
                    layout.resolve(
                        args.warmup_candidates or configured["warmup_candidates"],
                        must_exist=True,
                    )
                    if args.execution == "live"
                    else None
                ),
                response_ledger_path=response_ledger,
                cache_ledger_path=(
                    _same_run_verifier_cache_input(layout, args.mode)
                    if args.resume_from_cache
                    else None
                ),
                ollama_url=args.ollama_url,
                model_blob_path=(
                    layout.resolve(args.model_blob, must_exist=True) if args.model_blob else None
                ),
                pilot_selection_path=(
                    layout.resolve(args.pilot_selection, must_exist=True)
                    if args.pilot_selection
                    else None
                ),
                artifact_prefix=args.artifact_prefix or (
                    "replay" if args.execution == "replay" else ""
                ),
            )
            producer_manifest = _find_written_manifest(
                layout,
                f"verifier-*{args.mode}-{args.execution}.json",
                manifest,
            )
            _write_same_run_seal(layout, producer_manifest, manifest)
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "status": manifest["status"],
                        "condition_id": manifest["condition_id"],
                        "execution_mode": manifest["execution_mode"],
                        "candidate_count": manifest["candidate_count"],
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.stage == "pilot-verifier":
            configured = config.paths
            capture_index = layout.resolve(
                args.capture_index, must_exist=True
            )
            _validate_pilot_capture_seals(layout, capture_index)
            audit = run_verifier_pilot(
                layout,
                config,
                PilotInputs(
                    sentences=layout.resolve(
                        args.sentences or configured["warmup_sentences"],
                        must_exist=True,
                    ),
                    gold=layout.resolve(
                        args.gold or configured["development_gold"], must_exist=True
                    ),
                    candidates=layout.resolve(
                        args.candidates or configured["development_pilot_candidates"],
                        must_exist=True,
                    ),
                    warmup_candidates=layout.resolve(
                        args.warmup_candidates or configured["warmup_candidates"],
                        must_exist=True,
                    ),
                    split_manifest=layout.resolve(
                        args.split_manifest or configured["split_manifest"],
                        must_exist=True,
                    ),
                    candidate_index=layout.resolve(
                        args.candidate_index or configured["development_candidate_index"],
                        must_exist=True,
                    ),
                    pilot_selection=layout.resolve(
                        args.pilot_selection or configured["verifier_pilot_selection"],
                        must_exist=True,
                    ),
                    threshold_selection=layout.resolve(
                        args.threshold_selection or configured["threshold_selection"],
                        must_exist=True,
                    ),
                    capture_index=capture_index,
                ),
                evidence_class=args.evidence_class,
            )
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "pilot_status": audit["pilot_status"],
                        "go_no_go_status": audit["go_no_go_status"],
                        "publication_execution_admitted": False,
                    },
                    sort_keys=True,
                )
            )
            return 0 if audit["pilot_status"] == "pass" else 2

        if args.stage == "model":
            if args.model_action == "train":
                manifest = plan_training(
                    layout,
                    config,
                    execution_mode=args.execution,
                    training_seed=args.seed,
                )
                print(
                    json.dumps(
                        {
                            "run_id": args.run_id,
                            "status": manifest["status"],
                            "stage": "model-train",
                            "training_seed": manifest["training_seed"],
                        },
                        sort_keys=True,
                    )
                )
                return 0
            if args.model_action != "generate-candidates":
                raise DataContractError(f"unsupported model action: {args.model_action!r}")
            configured = config.paths
            sentences = layout.resolve(
                args.sentences or configured["sentences"], must_exist=True
            )
            checkpoint_manifest = layout.resolve(
                args.checkpoint_manifest, must_exist=True
            )
            candidates_out = layout.resolve(args.candidates_out or configured["candidates"])
            prediction_ledger = (
                _same_run_prediction_replay_input(
                    layout,
                    checkpoint_manifest,
                    sentences,
                    candidates_out,
                )
                if args.execution == "replay"
                else None
            )
            manifest = generate_candidates(
                layout,
                config,
                execution_mode=args.execution,
                sentences_path=sentences,
                checkpoint_manifest_path=checkpoint_manifest,
                candidates_out_path=candidates_out,
                prediction_ledger_path=prediction_ledger,
                checkpoint_blob_path=(
                    layout.resolve(args.checkpoint_blob, must_exist=True)
                    if args.checkpoint_blob
                    else None
                ),
                base_model=args.base_model,
                device=args.device,
            )
            producer_manifest = _find_written_manifest(
                layout,
                (
                    "model-generate-candidates-"
                    f"{args.execution}-seed-{manifest['training_seed']}-*.json"
                ),
                manifest,
            )
            _write_same_run_seal(layout, producer_manifest, manifest)
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "status": manifest["status"],
                        "execution_mode": manifest["execution_mode"],
                        "candidate_count": manifest["candidate_count"],
                    },
                    sort_keys=True,
                )
            )
            return 0

        if args.stage == "select-threshold":
            configured = config.paths
            document = select_threshold(
                layout,
                config,
                candidates_path=layout.resolve(args.candidates, must_exist=True),
                gold_path=layout.resolve(
                    args.gold or configured["development_gold"], must_exist=True
                ),
                split_manifest_path=layout.resolve(
                    args.split_manifest or configured["split_manifest"], must_exist=True
                ),
                candidate_index_path=(
                    layout.resolve(args.candidate_index, must_exist=True)
                    if args.candidate_index
                    else None
                ),
                out_path=layout.resolve(args.out or configured["threshold_selection"]),
            )
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "status": "selected",
                        "selected_threshold": document["selected_threshold"],
                    },
                    sort_keys=True,
                )
            )
            return 0

        configured = config.paths
        inputs = ScoreInputs(
            gold=layout.resolve(args.gold or configured["gold"], must_exist=True),
            candidates=layout.resolve(
                args.candidates or configured["candidates"], must_exist=True
            ),
            simple_verdicts=layout.resolve(
                args.simple_verdicts or configured["simple_verdicts"], must_exist=True
            ),
            corrective_verdicts=layout.resolve(
                args.corrective_verdicts or configured["corrective_verdicts"],
                must_exist=True,
            ),
            threshold_selection=layout.resolve(
                args.threshold_selection or configured["threshold_selection"],
                must_exist=True,
            ),
            nonpublication_smoke=args.nonpublication_smoke,
            output_prefix=args.output_prefix or "",
        )
        metrics = score_run(layout, config, inputs)
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "scored",
                    "n_raw_candidates": metrics["n_raw_candidates"],
                    "publication_seed_coverage_complete": metrics[
                        "publication_seed_coverage_complete"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    except (DataContractError, PathContractError, OSError, ValueError) as exc:
        print(f"phase_b: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
