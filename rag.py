"""Expanded Graph RAG from one completed, authenticated canonical run."""
from __future__ import annotations

import argparse
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common.process_identity import capture_process_start
if __name__ == "__main__":
    capture_process_start("rag.py")

from utils.common.artifact_io import DataContractError
from utils.common.execution import exclusive_run
from utils.common.paths import PathContractError, RunLayout
from utils.rag import runner
from stages.rag import run_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="Run or continue the expanded same-run panel")
    run.add_argument("--run-id", required=True)
    run.add_argument("--ollama-url", default="http://localhost:11434")
    run.add_argument("--dry-run", action="store_true")
    commands.add_parser("validate", help="Validate the frozen method without model access")
    args = parser.parse_args(argv)
    try:
        contract = runner.load_json(runner.CONTRACT_PATH)
        if args.action == "validate":
            from utils.rag.code_questions import replacement_method
            print(json.dumps({"source": runner.validate_source_contract(contract, False),
                              "method": replacement_method()}, indent=2))
            return 0
        layout = RunLayout(_ROOT, args.run_id)
        layout.require_existing()
        with exclusive_run(layout):
            return run_command(args, contract)
    except (DataContractError, PathContractError, runner.PhaseEError, OSError, ValueError) as exc:
        print(f"rag: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
