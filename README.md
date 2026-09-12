# Refactored Table Artifact and Original Graph RAG Reproduction

This branch is the publication artifact for the paper's in-scope Results and
Discussion Tables 1-3. It starts from the completed refactored Phase B
encoder/LLM-verifier commit and adds only the original downstream Graph RAG
workflow used for Table 2.

Only Table-2 graph construction and RAG answer generation are active in Phase E.
The repository retains the table-reachable encoder/verifier source for audit, but
the Phase E command cannot train or run an encoder, regenerate inference, or call
the LLM verifier. Table 4, all zh work, later dataset-neutral Graph RAG code, and
non-table experiment families are excluded.

## Release lineage

- Base: `c01003a875bbb4c36393647c6d5fa1cc84d71d2c`
  (`Finalize standalone Phase B launcher and cleanup`).
- Branch: `publication-refactored-rag`.
- Bounded RAG patch evidence: `6e28882fc3b7328895ac89a9165d2e57c8bdd2c8`
  and `607f9f3672175c42f4b76c6145c654c5ed5cf577` on the preserved
  wrong-base historical branch.
- The later Phase C Graph RAG line beginning at `240805b…` is not merged or
  copied into this artifact.

## Included result routes

Tables 1 and 3 are completed upstream evidence. Their canonical refactored route
is retained under `phase_b.sh`, `phase_b.py`, and the supporting modules,
configuration, prompts, and schemas. Phase E does not execute that route.
Historical SciERC inference/verifier producers remain under `provenance/` only
to audit the draft's displayed Table-3 aggregates.

The only newly executable route is:

```text
completed 173-record inference JSONL
  -> exact original build_kg.py (confidence graph only)
  -> exact table-era eval_graph_rag.py + one empty-run guard
  -> five Table-2 RAG conditions and legacy comparison
```

A completed compatible verified graph is consumed directly. The runner has no
path that can reconstruct it by calling the verifier.

## Frozen Table-2 method

The prompt authority is historical `eval_graph_rag.py` blob
`964893546b28f04e34e8c546bcbbfd4cfbc27354`, established from the
result-producing dates in project history rather than the paper draft's
incomplete one-line description. The evaluator preserves:

- 13 generic case-folded relation-question templates;
- modes in order: `llm_only`, `text_retrieval`, `kg_1hop`, `kg_2hop`,
  and `hybrid`;
- the exact five prompt constructors, user role, labels, whitespace,
  punctuation, and context assembly;
- `think=false`, temperature `0.0`, `num_predict=50`, seed 42, and ten
  questions; and
- historical retrieval, matching, metric, aggregation, and output semantics.

The later seven explicit CODE templates are excluded. The sole added evaluator
behavior is a fail-closed guard that refuses to serialize an empty question run.

The historical text condition is whitespace-token set overlap, not BM25. The
answer-derived query, unequal evidence access, lenient score, and small
diagnostic sample remain disclosed frozen limitations.

## Runtime requirements

- A clean clone on branch `publication-refactored-rag`, with its Git history.
- Python 3.10 or newer.
- `curl`.
- A loopback Ollama endpoint containing the same model identity recorded by a
  completed Phase B verifier environment manifest.
- The environment manifest and its independently retained SHA-256.
- The completed 173-record inference JSONL, SHA-256
  `4d587f77832968a8e28b2e399d528432764567125d43ac5ca7d7fe1761ec4922`.
- Optionally, the retained confidence graph, SHA-256
  `e1deec394e9f15e5d98670a83034bb4357f023212076e69c7bb6dbee5e0725fb`.
  If absent, the runner rebuilds only this graph at threshold `0.7`.
- Either a validated completed 48-node/31-edge verified graph and its SHA-256,
  or an explicit `--block-verified` disposition.

The RAG model value is not hardcoded. The runner derives its tag, registry
manifest digest, model blob SHA-256, family, parameter size, and quantization
from the supplied completed-verifier manifest and checks live Ollama before any
RAG subprocess.

## Validate without model calls

```bash
python -B run_phase_e_rag.py validate-contract
python -B -m unittest tests.test_phase_e_rag_contract tests.test_phase_e_runner -v
```

The complete retained-source regression suite can also be run with:

```bash
python -B -m unittest discover -s tests -v
```

## Dry-run an artifact plan

With a completed verified graph:

```bash
python -B run_phase_e_rag.py run \
  --run-id <run-id> \
  --inference <completed-inference.jsonl> \
  --verifier-environment-manifest <environment-manifest.json> \
  --verifier-environment-sha256 <sha256> \
  --verified-graph <completed-verified-graph.json> \
  --verified-graph-sha256 <sha256> \
  --dry-run
```

Or preserve that condition as blocked:

```bash
python -B run_phase_e_rag.py run \
  --run-id <run-id> \
  --inference <completed-inference.jsonl> \
  --verifier-environment-manifest <environment-manifest.json> \
  --verifier-environment-sha256 <sha256> \
  --block-verified \
  --dry-run
```

Add `--confidence-graph <retained-confidence-graph.json>` to reuse rather than
rebuild the confidence graph. Dry-run validates source ancestry/blobs, artifact
hashes and schemas, graph shapes, model-manifest identity, and the ten-question
contract. It does not contact Ollama or create a run directory.

A live run removes `--dry-run` and may set
`--ollama-url http://localhost:11434`. The parent research workflow supplies
one fully resolved external command only after the verified-graph disposition
and external artifact locations are fixed.

## Output and failure behavior

Every runtime file is confined to ignored `output/<run-id>/`: copied immutable
inputs, generated graphs, condition results, model-identity responses, logs,
method/environment/status manifests, hashes, and the Table-2 comparison.

Existing stages are reused only when run identity and stored hashes still match.
A changed source blob, unexpected input hash/schema/count, model mismatch,
failed or empty model response, corrupt output, output collision, or identity
change fails closed. Failed/interrupted attempts remain in the run tree for
audit. New results are compared with legacy values without tuning.

At parent-repository finalization, every run is hash-verified and moved to the
durable result archive; no runtime output remains in `src`.

## Completed upstream evidence

The retained Phase B source and frozen `configs/phase_b_path_a.json` describe
the completed canonical CODE-ACCORD encoder/verifier evaluation. Its archived
machine-readable metrics and publication views are indexed by the parent
research repository. They are distinct from, and are not silently substituted
for, the legacy development-only Table-1 or SciERC Table-3 aggregates.

See `docs/phase_b_workflow.md` and `docs/model-training-compatibility.md` for
the upstream protocol and its known comparability limits. These documents are
audit evidence, not authorization to execute upstream work in Phase E.
