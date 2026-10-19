# CODE-ACCORD publication artifact

This standalone checkout prepares CODE-ACCORD, trains the canonical encoder,
verifies extracted triples with local Qwen, and reports typed strict quality.
Separate commands run the matched encoder comparison and downstream Graph RAG.
Start with the [code-reading guide](docs/code-reading-guide.md), then the
[workflow contract](docs/workflow.md) or [comparison protocol](docs/encoder-comparison.md).

Install the pinned environment and validate without contacting a model:

```bash
uv sync --frozen
uv run --frozen --no-sync python -I -B pipeline.py validate
uv run --frozen --no-sync python -I -B encoder_comparison.py validate
uv run --frozen --no-sync python -I -B rag.py validate
uv run --frozen --no-sync python -B -m unittest discover -s tests -v
```

Run from the root of a clean Git checkout after `uv sync --frozen`.
`uv run --frozen --no-sync` selects the project environment without requiring
manual activation. Choose fresh descriptive IDs; repeat the same command to
continue validated durable progress in that namespace.

| Workflow | Command | Automatically written reports |
|---|---|---|
| Canonical encoder and verifier | `uv run --frozen --no-sync python -I -B pipeline.py run --run-id research-001` | `output/research-001/metrics/`, including `paper-report/paper-statistics.{json,tsv,md}` |
| Four-arm Table-1 comparison | `uv run --frozen --no-sync python -I -B encoder_comparison.py run --matrix-id comparison-001` | Each arm's `reports/comparison/` and the matrix's `reports/matrix/` |
| Expanded same-run Graph RAG | `uv run --frozen --no-sync python -I -B rag.py run --run-id research-001` | `output/research-001/table2-code-all/`, including `reports/paper-statistics.{json,tsv,md}` |

Canonical and RAG commands accept `--ollama-url http://localhost:11434`.
The canonical command also accepts `--model-blob-source /absolute/path/to/blob`
when Ollama's weight file is inaccessible. That immutable external input must
be outside every output run. The pipeline copies and verifies its actual bytes
inside the selected run. RAG supports a no-model preflight with `--dry-run`.

Every dataset, model cache, checkpoint, prediction, verdict, graph and report
belongs to one physical `output/<run-id>/` tree. Consumers validate same-run
hash lineage; model tags alone do not identify weights. Actual source, encoder,
Qwen and numerical runtime identities are retained. No historical Git object or
parent repository is needed for a fresh run or the offline tests.

Stage controllers live in `stages/`; scientific implementation lives in the
corresponding `utils/` directories. `utils/common/` contains shared artifact
primitives, `resources/` contains live configurations/prompts/schemas, and
`tests/` contains source-grounded characterization and explicitly synthetic
fault fixtures. Python 3.10 or newer is required; accelerator compatibility is
determined by the locked environment and the recorded actual runtime.

The scientific behavior remains frozen: CODE-SPLIT-1, seeds 42–49, typed strict
scoring, development-only selection, literal prompts and the five-mode RAG
evaluator. Historical Tables 1/3 are not regenerated. Expanded Graph RAG is a
diagnostic/oracle evaluation with all eligible CODE questions and a fresh
`part-of` subset; it is not the paper's original ten-question Table 2. Table 4,
zh, SciERC and CUAD experiments are outside this artifact.

Local fixtures verify scientific bytes, initialization/RNG behavior, restart,
lineage and continuation. They do not establish live CUDA/Qwen equality across
machines. Existing partial G-05 runs remain on their pinned pre-refactor repair
workflow at `fb18028cd21239c0bf913e370149a643081958fe`; preserve their original
IDs and manifests. Do not adopt them into a fresh comparison matrix or rerun a
completed scientific result. The returned expanded RAG parity discrepancy and
missing external binary lineage remain unresolved external evidence limits.
