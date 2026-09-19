"""Separate root entry point for Phase G historical encoder comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ENTRY_ROOT = Path(__file__).resolve().parent
if str(_ENTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENTRY_ROOT))

from stages.encoder_comparison import (
    describe,
    generate,
    plan_training,
    prepare,
    run_private,
    run_training,
)
from utils.common.artifact_io import DataContractError
from utils.common.paths import PathContractError, RunLayout, discover_source_root
from utils.encoder_comparison.config import DEFAULT_CONFIG, load_comparison_config


DEFAULT_RECIPE = "historical-a20-a21-a12"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python encoder_comparison.py",
        description="Historical-derived, canonical-data encoder comparison path.",
    )
    parser.add_argument("--source-root")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="action", required=True)
    validate = subparsers.add_parser("validate-profile")
    prepare_parser = subparsers.add_parser("prepare")
    plan = subparsers.add_parser("plan-training")
    train = subparsers.add_parser("train")
    predict = subparsers.add_parser("generate")
    for command in (validate, prepare_parser, plan, train, predict):
        command.add_argument("--profile", required=True)
        command.add_argument("--recipe", default=DEFAULT_RECIPE)
    for command in (prepare_parser, plan, train, predict):
        command.add_argument("--run-id", required=True)
    for command in (plan, train, predict):
        command.add_argument("--training-seed", required=True, type=int)
    predict.add_argument("--split", choices=("development", "test"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"_train-encoder", "_predict-encoder"}:
        action = "train" if arguments[0] == "_train-encoder" else "predict"
        try:
            return run_private(
                arguments[1:], action=action, default_config=DEFAULT_CONFIG
            )
        except (DataContractError, PathContractError, OSError, ValueError) as exc:
            print(f"encoder_comparison: error: {exc}", file=sys.stderr)
            return 2
    args = _parser().parse_args(arguments)
    try:
        source_root = discover_source_root(
            Path(args.source_root) if args.source_root else None
        )
        config = load_comparison_config(source_root, args.config)
        selection = config.select(args.profile, args.recipe)
        if args.action == "validate-profile":
            print(json.dumps(describe(selection), indent=2))
            return 0
        layout = RunLayout(source_root=source_root, run_id=args.run_id)
        if args.action == "prepare":
            prepare(layout, selection)
        elif args.action == "plan-training":
            print(json.dumps(plan_training(layout, selection, args.training_seed), indent=2))
        elif args.action == "train":
            print(json.dumps(run_training(layout, selection, args.training_seed), indent=2))
        elif args.action == "generate":
            print(
                json.dumps(
                    generate(
                        layout,
                        selection,
                        args.training_seed,
                        args.split,
                    ),
                    indent=2,
                )
            )
        return 0
    except (DataContractError, PathContractError, OSError, ValueError) as exc:
        print(f"encoder_comparison: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
