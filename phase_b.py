"""One public command surface for implemented Phase B lifecycle stages."""

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
from phase_b_io import DataContractError
from paths import PathContractError, RunLayout, discover_source_root
from pilot import PilotInputs, run_verifier_pilot
from preparation import prepare_run
from reconciliation import reconcile_section5_evidence
from scoring import ScoreInputs, score_run
from verifier import run_verifier


DEFAULT_CONFIG = "configs/phase_b_path_a.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python phase_b.py",
        description=(
            "Canonical standalone Phase B workflow. This B-05/B-06/B-07 and B-05U "
            "development slice implements "
            "doctor, secondary Section 5 evidence reconciliation, immutable "
            "CODE-ACCORD fetch/preparation, model train planning and candidate "
            "generation (dry-run/replay/live), development threshold selection, "
            "frozen verifier request/replay instrumentation, development-pilot "
            "auditing, and offline CODE-STRICT-1 scoring."
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
        "--response-ledger", help="Run-relative frozen response JSONL required by replay"
    )
    verifier.add_argument(
        "--cache-ledger", help="Optional run-relative response cache used only by live execution"
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
        help="Run-relative index of four copied complete live-run evidence bundles",
    )

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
        "--prediction-ledger",
        help="Run-relative frozen encoder prediction ledger required by replay",
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

        if args.stage == "verifier":
            configured = config.paths
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
                response_ledger_path=(
                    layout.resolve(args.response_ledger, must_exist=True)
                    if args.response_ledger
                    else None
                ),
                cache_ledger_path=(
                    layout.resolve(args.cache_ledger, must_exist=True)
                    if args.cache_ledger
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
            )
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
                    capture_index=layout.resolve(args.capture_index, must_exist=True),
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
            manifest = generate_candidates(
                layout,
                config,
                execution_mode=args.execution,
                sentences_path=layout.resolve(
                    args.sentences or configured["sentences"], must_exist=True
                ),
                checkpoint_manifest_path=layout.resolve(
                    args.checkpoint_manifest, must_exist=True
                ),
                candidates_out_path=layout.resolve(
                    args.candidates_out or configured["candidates"]
                ),
                prediction_ledger_path=(
                    layout.resolve(args.prediction_ledger, must_exist=True)
                    if args.prediction_ledger
                    else None
                ),
                checkpoint_blob_path=(
                    layout.resolve(args.checkpoint_blob, must_exist=True)
                    if args.checkpoint_blob
                    else None
                ),
                base_model=args.base_model,
                device=args.device,
            )
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
