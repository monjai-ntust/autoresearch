"""Canonical CODE-ACCORD encoder, verifier and paper reporting workflow."""
from __future__ import annotations

import argparse
import subprocess
import sys
sys.dont_write_bytecode = True
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common.artifact_io import DataContractError
from utils.common.config import load_pipeline_config
from utils.common.execution import exclusive_run
from utils.common.paths import PathContractError, RunLayout, discover_source_root

DEFAULT_CONFIG = "resources/configs/pipeline.json"
STORAGE_BUDGET_BYTES = 100_000_000_000


def _run_full_pipeline(layout, config, *, ollama_url, model_blob_source) -> int:
    from stages.encoder import train_and_generate_development, generate_test_candidates
    from stages.evaluation import run as evaluate
    from stages.preparation import run as prepare
    from stages.verifier import prepare_development_gate, run_pilot, run_test
    from utils.common.storage import warn_storage

    warn_storage(layout.run_root, budget=STORAGE_BUDGET_BYTES, roots=[layout.run_root],
        label="pipeline", excluded=(layout.resolve("table2-code-all"),))
    prepare(layout, config)
    encoder, hardware, development = train_and_generate_development(layout, config)
    threshold, pilot = prepare_development_gate(layout, config, development)
    context = run_pilot(layout, config, threshold_path=threshold, pilot_selection=pilot,
        ollama_url=ollama_url, model_blob_source=model_blob_source,
        encoder_identity=encoder, training_hardware=hardware)
    candidates = generate_test_candidates(layout, config)
    run_test(layout, config, context, candidates, ollama_url=ollama_url)
    evaluate(layout, config, test_candidates=candidates, threshold_path=threshold)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] == "_train-encoder":
            from stages.encoder import run_private_trainer
            return run_private_trainer(arguments[1:], default_config=DEFAULT_CONFIG)
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="action", required=True)
        run = commands.add_parser("run", help="Run or continue the canonical workflow")
        run.add_argument("--run-id", required=True)
        run.add_argument("--ollama-url", default="http://localhost:11434")
        run.add_argument("--model-blob-source")
        commands.add_parser("validate", help="Validate configuration without model access")
        args = parser.parse_args(arguments)
        root = discover_source_root(_ROOT)
        config = load_pipeline_config(root, DEFAULT_CONFIG)
        if args.action == "validate":
            from utils.verifier.requests import _prompt_bundle
            for mode in ("simple", "corrective"):
                _prompt_bundle(config, mode)
            print("Canonical configuration and prompt/schema resource selectors valid.")
            return 0
        layout = RunLayout(root, args.run_id)
        from stages.preparation import require_canonical_namespace
        require_canonical_namespace(layout)
        layout.run_root.mkdir(parents=True, exist_ok=True)
        with exclusive_run(layout):
            require_canonical_namespace(layout)
            return _run_full_pipeline(layout, config, ollama_url=args.ollama_url,
                                      model_blob_source=args.model_blob_source)
    except (DataContractError, PathContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"pipeline: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
