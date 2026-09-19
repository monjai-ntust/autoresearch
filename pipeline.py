"""Primary CODE-ACCORD publication and same-run Table-2 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Python isolated mode intentionally omits the script directory from sys.path
# on some platforms. The entry point restores only its own resolved checkout
# root so local modules remain importable without accepting ambient paths.
_ENTRY_ROOT = Path(__file__).resolve().parent
if str(_ENTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENTRY_ROOT))

from utils.common.artifact_io import DataContractError
from utils.common.config import load_pipeline_config
from utils.common.paths import PathContractError, RunLayout, discover_source_root
from utils.rag import runner as table2_runner


DEFAULT_CONFIG = "resources/configs/pipeline.json"


def _run_full_pipeline(
    layout: RunLayout,
    config,
    *,
    ollama_url: str,
    model_blob_source: str | None,
) -> int:
    """Run the sole supported canonical CODE-ACCORD and Table-2 workflow."""

    from stages.encoder import (
        generate_test_candidates,
        train_and_generate_development,
    )
    from stages.evaluation import run as run_evaluation
    from stages.preparation import run as run_preparation
    from stages.rag import run as run_rag
    from stages.verifier import (
        prepare_development_gate,
        run_pilot as run_verifier_pilot,
        run_test as run_verifier_test,
    )

    run_preparation(layout, config)
    encoder_identity, training_hardware, development_candidates = (
        train_and_generate_development(layout, config)
    )
    threshold_path, pilot_selection = prepare_development_gate(
        layout, config, development_candidates
    )
    verifier_context = run_verifier_pilot(
        layout,
        config,
        threshold_path=threshold_path,
        pilot_selection=pilot_selection,
        ollama_url=ollama_url,
        model_blob_source=model_blob_source,
        encoder_identity=encoder_identity,
        training_hardware=training_hardware,
    )
    test_candidates = generate_test_candidates(layout, config)
    run_verifier_test(
        layout,
        config,
        verifier_context,
        test_candidates,
        ollama_url=ollama_url,
    )
    run_evaluation(
        layout,
        config,
        test_candidates=test_candidates,
        threshold_path=threshold_path,
    )
    return run_rag(layout, ollama_url=ollama_url)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python pipeline.py",
        description=(
            "Canonical CODE-ACCORD pipeline with a complete-run route and "
            "a same-run downstream Table-2 route."
        ),
    )
    parser.add_argument(
        "--source-root",
        help="Direct src checkout root; normally discovered from the current directory.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    full = subparsers.add_parser(
        "full", help="Run or resume the canonical CODE-ACCORD and Table-2 pipeline"
    )
    full.add_argument("--config", default=DEFAULT_CONFIG)
    full.add_argument("--run-id", required=True)
    full.add_argument("--ollama-url", default="http://localhost:11434")
    full.add_argument("--model-blob-source")

    table2 = subparsers.add_parser(
        "table2", help="Run only downstream Table-2 RAG from a completed same run"
    )
    table2.add_argument("--run-id", required=True)
    table2.add_argument("--ollama-url", default="http://localhost:11434")
    table2.add_argument("--dry-run", action="store_true")

    subparsers.add_parser(
        "validate-table2", help="Validate the frozen Table-2 source contract"
    )
    return parser


def _run_private_trainer(arguments: list[str]) -> int:
    from stages.encoder import run_private_trainer

    return run_private_trainer(arguments, default_config=DEFAULT_CONFIG)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_train-encoder":
        try:
            return _run_private_trainer(arguments[1:])
        except (DataContractError, PathContractError, OSError, ValueError) as exc:
            print(f"pipeline: error: {exc}", file=sys.stderr)
            return 2
    args = _parser().parse_args(arguments)
    try:
        if args.stage in {"table2", "validate-table2"}:
            contract = table2_runner.load_json(table2_runner.CONTRACT_PATH)
            if args.stage == "validate-table2":
                print(
                    json.dumps(
                        table2_runner.validate_source_contract(contract, False),
                        indent=2,
                    )
                )
                return 0
            return table2_runner.run_command(args, contract)
        source_root = discover_source_root(
            Path(args.source_root) if args.source_root else None
        )
        config = load_pipeline_config(source_root, args.config)
        layout = RunLayout(source_root=source_root, run_id=args.run_id)
        if args.stage == "full":
            return _run_full_pipeline(
                layout,
                config,
                ollama_url=args.ollama_url,
                model_blob_source=args.model_blob_source,
            )
    except (
        DataContractError,
        PathContractError,
        table2_runner.PhaseEError,
        OSError,
        ValueError,
    ) as exc:
        print(f"pipeline: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
