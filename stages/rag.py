"""Same-run Table-2 RAG controller for the canonical workflow."""

from __future__ import annotations

import argparse

from utils.common.paths import RunLayout
from utils.rag import runner


def run(layout: RunLayout, *, ollama_url: str) -> int:
    """Validate parent lineage and execute the frozen Table-2 stage."""

    contract = runner.load_json(runner.CONTRACT_PATH)
    runner.validate_parent_lineage(layout.run_root, layout.run_id, contract)
    arguments = argparse.Namespace(
        run_id=layout.run_id,
        ollama_url=ollama_url,
        dry_run=False,
    )
    return runner.run_command(arguments, contract)
