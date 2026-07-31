"""Command-line entry point for Graph RAG contracts and Phase B diagnostics."""

from __future__ import annotations

import argparse
import json

from graph_rag_eval.runner import create_context, doctor, evaluate, prepare
from graph_rag_eval.phase_b_diagnostic import (
    create_phase_b_diagnostic_context,
    run_phase_b_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generated-graph evaluation and canonical Phase B diagnostics"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "prepare", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument(
            "--run-id",
            help="safe output namespace override for repeat/standalone validation",
        )
        if command == "evaluate":
            child.add_argument(
                "--smoke",
                action="store_true",
                help="record a bounded smoke request; scientific gates still apply",
            )
    phase_b = subparsers.add_parser(
        "phase-b-diagnostic",
        help="run/recover the canonical Regime-D diagnostic for a scored Phase B run",
    )
    phase_b.add_argument("--config", required=True)
    phase_b.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "phase-b-diagnostic":
        result = run_phase_b_diagnostic(
            create_phase_b_diagnostic_context(
                args.config,
                run_id=args.run_id,
            )
        )
    else:
        context = create_context(args.config, run_id=args.run_id)
        if args.command == "doctor":
            result = doctor(context)
        elif args.command == "prepare":
            result = prepare(context)
        else:
            result = evaluate(context, smoke=args.smoke)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 2 if result["status"] in {"blocked"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
