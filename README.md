# CODE-ACCORD publication artifact

This branch contains the canonical CODE-ACCORD publication workflow and the
same-run reproduction path for the paper's Table 2. `pipeline.py` is the
single supported entry point.

The pipeline retains the established CODE-ACCORD encoder, threshold, local
Qwen verifier, strict scoring, and original Table-2 Graph RAG methods. It does
not claim to regenerate every historical table. Table 4,
Traditional-Chinese work, unrelated experiments, diagnostics, and standalone
artifact-path entry points are excluded.

## Scientific boundary

The full route is:

```text
immutable CODE-ACCORD input
  -> deterministic CODE-SPLIT-1 preparation
  -> eight DeBERTa-large training and candidate runs
  -> development-only threshold and verifier pilot gate
  -> simple and corrective Qwen verification
  -> canonical CODE-ACCORD encoder/verifier metrics
  -> same-run confidence, corrective, and gold graphs
  -> frozen five-mode answer evaluation for Table 2
```

| Paper artifact | Implemented route and data contract | Comparison evidence and missing work |
| --- | --- | --- |
| Historical Table 1 | Not regenerated. Its canonical successor is the eight-seed CODE-ACCORD `CODE-SPLIT-1` held-out encoder evaluation under `CODE-STRICT-1`. | Legacy displayed aggregates are comparison evidence only; the historical BERT-base/DeBERTa-base table, split memberships, and checkpoints are not reproduced. |
| Table 2 | Newly evaluated from one run's CODE-ACCORD test records and confidence/corrective/gold graphs, projected into the frozen five-mode table-era evaluator. | Legacy displayed cells are comparison references only. External E-07 execution and comparison remain pending. |
| Historical Table 3 | Not regenerated. Its canonical successor is the CODE-ACCORD `CODE-SPLIT-1` simple/corrective Qwen evaluation under `CODE-STRICT-1`. | Legacy SciERC text-tuple aggregates are comparison evidence only; their source data/checkpoints/verdict ledgers are not reproduced. |
| Table 4 | No route. | Excluded from this artifact. |

Every generated or downloaded artifact lives under one
`output/<run-id>/` tree. Downstream stages derive their inputs from that tree
and validate producer hashes and manifests. A run cannot import an output from
another run.

Tracked source is authenticated once by a clean full Git commit. Manifests do
not repeat hashes for individual Python files or the lockfile. Content hashes
remain mandatory for runtime inputs, model blobs, prompts, prepared data, and
stage outputs. The only retained JSON Schemas are the two response formats sent
to Ollama; other record contracts are enforced directly by their consumers.

The encoder revision and Qwen registry/blob digests must each remain internally
consistent wherever they are consumed within one run. They do not have to equal
the historical reference digests recorded by this project. Hardware and
historical digest comparisons are logged separately and do not admit or reject
a run. Matching statistics are expected only when the method, data, split,
seed, threshold, hardware, encoder, and Qwen determinants match, and equality
must still be confirmed from the generated outputs.

## Install and validate

Use Python 3.10 or newer from a clean clone of branch
`publication-refactored-rag`:

```bash
uv sync --frozen
uv run --frozen --no-sync python -I -B pipeline.py validate-table2
uv run --frozen --no-sync python -B -m unittest discover -s tests -v
```

The tracked configuration is `configs/pipeline.json`. Its explicitly named
Qwen reference digests are comparison metadata. The default encoder revision
matches the historical reference, but configuration validation accepts any
immutable 40-hex revision; that configured value becomes the new run's actual
encoder input. A full run discovers and hashes the Qwen model that is actually
served, copies its bytes into the run's content-addressed input area, and binds
all verifier consumers to that run-local identity.

## Generate the canonical upstream artifacts and Table 2

Choose a new run ID containing 1-64 letters, digits, `.`, `_`, or `-`, beginning
with a letter or digit. Start Ollama with the configured Qwen model, then run:

```bash
uv run --frozen --no-sync python -I -B pipeline.py full \
  --run-id research-run-001 \
  --ollama-url http://localhost:11434
```

If Ollama's `FROM` path is not directly accessible, add
`--model-blob-source /absolute/path/to/the/model/blob`. The source must be an
external immutable input, not a file under any `output/<run-id>/` tree.

The Python entry point performs and resumes the complete ordered workflow. It
does not replace a different existing artifact, bypass the development pilot,
or reuse a stage whose manifest/hash binding has changed.

## Run only downstream Table 2

Use this route when a complete authenticated run already contains the encoder,
candidate, threshold, verifier, and score outputs. It never trains the encoder
or invokes the verifier.

First validate the same-run plan without writes or model calls:

```bash
uv run --frozen --no-sync python -I -B pipeline.py table2 \
  --run-id research-run-001 \
  --dry-run
```

Then execute the frozen answer evaluator against the same Qwen identity that
the selected run's corrective verifier recorded:

```bash
uv run --frozen --no-sync python -I -B pipeline.py table2 \
  --run-id research-run-001 \
  --ollama-url http://localhost:11434
```

The approved full-panel Table-2 output is confined to
`output/<run-id>/table2-q105/`. This separate same-run namespace preserves any
prior 10-question Table-2 child as historical evidence. Missing canonical
graphs are deterministically reconstructed only from authenticated prepared
data, gold, seed-42 candidates, the development-selected threshold, and
corrective verdicts in that same run. If a canonical graph already exists, its
canonical JSON must exactly equal the reconstruction before it is reused.

## Frozen Table-2 evaluator

Historical source blob `964893546b28f04e34e8c546bcbbfd4cfbc27354`
is the prompt and method authority. The internal evaluator preserves its 13
case-folded relation templates, the approved full 105-question panel, five mode order,
retrieval behavior, exact prompt construction, `think=false`, temperature
`0.0`, `num_predict=50`, lenient answer matcher, aggregation, and output
semantics. The only added behavior is a fail-closed guard for an empty supported
question set.

The text condition is historical unique-word overlap, not BM25. The
answer-derived retrieval query, unequal evidence access, and lenient matching
are retained limitations rather than corrected methodology. The former
10-question sample-size limitation is replaced only by the approved full
105-question panel.

## Failure and output handling

The workflow fails closed on cross-run paths, manifest or hash drift, mixed
encoder/Qwen identities, graph mismatch, unsafe links, model-call errors,
empty responses, or an output collision. Interrupted Table-2 results stay in a
run-local recovery namespace and are never silently accepted. Legacy displayed
values are comparison metadata only; they are not inputs or tuning targets.

See `docs/workflow.md` for the artifact and resume contract and
`docs/model-training-compatibility.md` for the encoder compatibility boundary.
