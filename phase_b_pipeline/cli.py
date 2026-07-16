"""One public command surface for implemented Phase B lifecycle stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_pipeline_config
from .doctor import run_doctor
from .io import DataContractError
from .paths import PathContractError, RunLayout, discover_source_root
from .scoring import ScoreInputs, score_run


DEFAULT_CONFIG = "configs/phase_b_path_a.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m phase_b_pipeline",
        description=(
            "Canonical standalone Phase B workflow. This B-05 slice implements "
            "doctor and offline CODE-STRICT-1 scoring."
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

    score = subparsers.add_parser("score", help="Offline score frozen candidates and verdicts")
    score.add_argument("--config", default=DEFAULT_CONFIG)
    score.add_argument("--run-id", required=True)
    score.add_argument("--gold", help="Run-relative gold JSONL path")
    score.add_argument("--candidates", help="Run-relative candidate JSONL path")
    score.add_argument("--simple-verdicts", help="Run-relative simple-verdict JSONL path")
    score.add_argument("--corrective-verdicts", help="Run-relative corrective-verdict JSONL path")
    score.add_argument("--threshold-selection", help="Run-relative development threshold JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source_root = discover_source_root(Path(args.source_root) if args.source_root else None)
        config = load_pipeline_config(source_root, args.config)
        layout = RunLayout(source_root=source_root, run_id=args.run_id)
        if args.stage == "doctor":
            manifest, passed = run_doctor(layout, config)
            print(json.dumps({"run_id": args.run_id, "status": manifest["status"]}, sort_keys=True))
            return 0 if passed else 2

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
        print(f"phase_b_pipeline: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
