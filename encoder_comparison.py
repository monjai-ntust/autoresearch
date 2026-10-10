"""Run or continue the four-arm matched encoder comparison."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common.artifact_io import DataContractError
from utils.common.paths import PathContractError, RunLayout, discover_source_root
from utils.encoder_comparison.config import DEFAULT_CONFIG, load_comparison_config


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] in {"_train-encoder", "_predict-encoder", "_smoke-encoder"}:
            from stages.encoder_comparison import run_private
            return run_private(arguments[1:], action=arguments[0][1:-8], default_config=DEFAULT_CONFIG)
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="action", required=True)
        run = commands.add_parser("run")
        run.add_argument("--matrix-id", required=True)
        commands.add_parser("validate", help="Inspect the frozen matrix without model access")
        args = parser.parse_args(arguments)
        root = discover_source_root(_ROOT)
        config = load_comparison_config(root)
        if args.action == "validate":
            print(json.dumps({arm: config.select_arm(arm).manifest() for arm in config.arms}, indent=2))
            return 0
        from stages.comparison_matrix import run_matrix
        return run_matrix(RunLayout(root, args.matrix_id), config)
    except (DataContractError, PathContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"encoder_comparison: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
