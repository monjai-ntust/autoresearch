# Phase E: Methodology-Preserving Graph RAG Reproduction

This branch publishes the remaining downstream CODE-ACCORD Graph RAG workflow used for the paper's Table 2 diagnostic. It deliberately does not retrain an encoder, regenerate encoder inference, invoke the LLM verifier, produce Tables 1/3/4, or run any zh workflow.

The workflow consumes compatible completed upstream artifacts, constructs only the permitted confidence graph when needed, and runs the five original RAG answer conditions. Every runtime file is written beneath ignored `output/<run-id>/` for later hash-verified archival by the parent research repository.

## Frozen method

The prompt authority is historical `eval_graph_rag.py` blob `964893546b28f04e34e8c546bcbbfd4cfbc27354`, not the paper draft's incomplete one-line description. The evaluator preserves:

- 13 generic, case-folded relation-question templates;
- five answer modes in order: `llm_only`, `text_retrieval`, `kg_1hop`, `kg_2hop`, and `hybrid`;
- one user-role message, no system message, `think=false`, temperature `0.0`, and `num_predict=50`;
- exact prompt labels, whitespace, punctuation, context assembly, retrieval behavior, scoring, and JSON output semantics; and
- the original ten-question seed-42 sampling behavior.

The later May 7 evaluator and its seven CODE-specific templates are excluded. The only post-restoration behavior change is a fail-closed guard that refuses to write a misleading result when no supported question is generated.

The text condition is historical whitespace-token overlap, despite being labeled BM25 in historical material. Answer-derived queries, unequal evidence access, the lenient overlap score, small sample, and gold-graph oracle remain unchanged methodological limitations.

Machine-readable method and source identities are in `phase_e_rag_contract.json`.

## Immutable boundary

`train_span.py`, `inference_kg.py`, `verify_triples_llm.py`, and all existing `data/`, `models/`, and `eval/` files are completed upstream evidence. Phase E validates their Git blobs but does not import, execute, edit, or delete them.

The retained `build_kg.py` can reproduce the confidence-only graph. The exact transient code that produced the corrected verified graph was not committed during the Table-2 window; its first committed form is bundled with excluded later behavior. Consequently, the verified condition must receive a compatible completed graph with 48 nodes and 31 edges or be explicitly recorded as blocked. The runner never reconstructs it from verifier output.

## Requirements

- A clean clone of branch `publication-legacy-methodology` with its Git history.
- Python 3.10 or newer. The runner and active graph code use only the standard library.
- `curl`, used by the frozen evaluator for Ollama chat requests.
- A running Ollama endpoint containing the exact `qwen3:32b` model instance selected for reproduction.
- The completed 173-record CODE inference JSONL with SHA-256 `4d587f77832968a8e28b2e399d528432764567125d43ac5ca7d7fe1761ec4922`.
- Optionally, the retained confidence graph with SHA-256 `e1deec394e9f15e5d98670a83034bb4357f023212076e69c7bb6dbee5e0725fb`. If omitted, the runner builds it with the frozen confidence-only graph producer at threshold `0.7`.
- Either a user-approved completed 48-node/31-edge verified graph plus its exact SHA-256, or an explicit decision to keep that condition blocked.

Inputs may reside outside the clone. A live run verifies them and copies their exact bytes beneath its own output tree before use.

## Validate the source contract

These commands do not call a model:

```bash
python -B run_phase_e_rag.py validate-contract
python -B -m unittest discover -s tests -v
```

The tests check every frozen upstream blob, the active graph-builder blob, the exact evaluator-plus-guard blob, all 13 relation templates, all five runtime prompts, request payload fields, rejection of the later seven templates, the real seed-42 question projection, and the empty-run failure behavior.

## Pin the Ollama model identity

Before a live run, record the canonical hash of the local Ollama `/api/show` response:

```bash
python -B run_phase_e_rag.py inspect-model --ollama-url http://localhost:11434
```

The live command requires the printed `canonical_show_sha256`. A tag alone is not accepted as sufficient model identity.

## Dry-run the artifact plan

The following forms contain placeholders and are documentation examples, not the final experiment command.

With a completed verified graph:

```bash
python -B run_phase_e_rag.py run \
  --run-id <run-id> \
  --inference <completed-inference.jsonl> \
  --verified-graph <completed-verified-graph.json> \
  --verified-graph-sha256 <sha256> \
  --dry-run
```

With the verified condition explicitly blocked:

```bash
python -B run_phase_e_rag.py run \
  --run-id <run-id> \
  --inference <completed-inference.jsonl> \
  --block-verified \
  --dry-run
```

Add `--confidence-graph <retained-confidence-graph.json>` to reuse the retained graph rather than rebuilding it. A dry run validates source identities, artifact hashes, schemas, graph shapes, and the ten-question contract without contacting Ollama or creating `output/`.

## Run the downstream workflow

Remove `--dry-run` and add both the pinned model identity and endpoint:

```bash
python -B run_phase_e_rag.py run \
  --run-id <run-id> \
  --inference <completed-inference.jsonl> \
  --verified-graph <completed-verified-graph.json> \
  --verified-graph-sha256 <sha256> \
  --ollama-url http://localhost:11434 \
  --ollama-show-sha256 <canonical-show-sha256>
```

The parent research workflow will supply one fully resolved command only after the external artifact paths, verified-graph disposition/hash, and model identity are confirmed. Do not substitute artifacts or fill unknown values by guesswork.

## Runtime output

A run creates only `output/<run-id>/`:

```text
output/<run-id>/
├── inputs/
│   ├── inference.jsonl
│   ├── ollama-show.json
│   └── verified-graph.json          # only when supplied
├── graphs/
│   └── confidence-0.7.json
├── rag-results/
│   ├── confidence.json
│   ├── verified.json                # only when supplied
│   └── gold.json
├── logs/
├── artifact-hashes.json
├── environment.json
├── method-manifest.json
├── run-status.json
└── table2-results.json
```

Existing stages are reused only when the run identity and stored hashes still match. Failed/interrupted attempt logs are retained. A model-call error, empty prediction, corrupt output, changed source blob, unexpected artifact hash/schema/count, output collision, or run-identity change fails closed.

New results are compared with legacy values only after the complete run tree is returned to the parent repository. Differences are recorded; the prompt and method are never tuned to improve agreement.
