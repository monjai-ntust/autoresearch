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
one authenticated completed refactored run
  -> same-run prepared test/private gold and selected 0.25 threshold
  -> same-run seed-42 confidence, corrective-verifier, and gold snapshots
  -> deterministic representation-only projections
  -> exact table-era eval_graph_rag.py + one empty-run guard
  -> five Table-2 RAG conditions and legacy comparison
```

The runner derives every scientific input path from one selected run root. It
has no path that can train or invoke the encoder, invoke the verifier, rebuild a
canonical graph, accept an arbitrary stage-output path, or import another run.

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
- The complete authenticated `path-a-full-20260719T153901Z` tree retained at
  `output/path-a-full-20260719T153901Z/` under its unchanged run ID, including
  the 20,201,240,160-byte content-addressed Ollama blob bound by the corrective
  verifier. The compact parent archive omits this oversized blob and is not by
  itself a complete executable restore.
- A loopback Ollama endpoint containing the same stable model identity recorded
  by that run's corrective-verifier environment manifest.

The RAG model value is not hardcoded. The runner derives its tag, registry
manifest digest, model blob SHA-256, family, parameter size, and quantization
from the selected run's hash-bound corrective-verifier manifest and checks live
Ollama before creating or modifying `phase-e-rag/`.

## Validate without model calls

```bash
python -I -B run_phase_e_rag.py validate-contract
python -B -m unittest tests.test_phase_e_rag_contract tests.test_phase_e_runner -v
```

The complete retained-source regression suite can also be run with:

```bash
python -B -m unittest discover -s tests -v
```

## Dry-run the same-run plan

```bash
python -I -B run_phase_e_rag.py run \
  --run-id path-a-full-20260719T153901Z \
  --dry-run
```

Dry-run validates source ancestry/blobs; the parent and graph-child run
identities; the complete parent artifact set; preparation, candidate, threshold,
verifier, model-environment, and graph bindings; exact graph IDs/shapes; stable
projection hashes; and the ten-question contract. It does not contact Ollama or
write any output.

After the dry run passes, the exact live command is:

```bash
python -I -B run_phase_e_rag.py run \
  --run-id path-a-full-20260719T153901Z \
  --ollama-url http://localhost:11434
```

## Output and failure behavior

Every new runtime file is confined to
`output/<run-id>/phase-e-rag/`: deterministic projections, condition results,
model-identity responses, logs, method/environment/status manifests, hashes,
and the Table-2 comparison. Earlier stages remain unchanged in the same tree.

Existing stages are reused only when run identity and stored hashes still match.
A same-run stage seal binds each prospectively generated encoder-candidate or
verifier manifest and all of its outputs to the selected `run_id` without
altering the frozen producer modules. Recovery caches, pilot captures, and
smoke-derived candidate/verdict/threshold artifacts carry equivalent run-local
provenance records.
A changed source blob, run ID, seed, threshold, candidate/verdict/environment
binding, graph ID/hash/schema/count, model identity, projection hash, failed or
empty model response, corrupt output, output collision, or child identity fails
closed. Before final completion, the runner re-hashes the complete parent and
graph-child lineages and verifies every projection and raw model-identity
capture against its manifest. Failed/interrupted attempts remain in the run
tree for audit. New
results are compared with legacy values without tuning.

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
