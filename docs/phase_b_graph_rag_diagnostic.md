# Phase B output Graph RAG diagnostic

## Authority and claims boundary

C-04 makes one Graph RAG use canonical: diagnosing the graph-producing outputs
of a completed Phase B Path A run. The publication-facing command remains:

```bash
bash phase_b.sh --stage full
```

The launcher first completes the frozen `B04-PATH-A-1.3` workflow through
`CODE-STRICT-1` scoring. Only after the score completion check succeeds does it
run protocol `C-04-PHASE-B-REGIME-D-1.0`. This addition does not change Phase B
data, split, seeds, model/checkpoint, training, inference, threshold, verifier,
scoring, resume, or result semantics.

The evaluation regime is always `diagnostic_fact_probe`. Its permissible
interpretation is limited to intrinsic graph fidelity and retrieval, source
support, and fact preservation. It is not QA, compliance reasoning, evidence
that a verifier improved Phase B, or a new Phase B headline metric.

## Recovery without Phase B execution

For a run that already has a valid score manifest, including
`path-a-full-20260719T153901Z`, use:

```bash
bash phase_b.sh --run-id path-a-full-20260719T153901Z --stage graph-rag
```

The `graph-rag` stage requires only the completed score contract. It does not
call bootstrap, acquisition, preparation, training, inference, threshold
selection, a verifier, or Phase B scoring. The historical run must already be
present as `output/path-a-full-20260719T153901Z/` on the external machine. The
repository's parent `project-state/results/output-archive/` remains separate
research-management evidence, is not available to a standalone source clone,
and must not be used as a runtime output target.

For an installed, already-synchronized standalone checkout, the equivalent
no-network recovery route is:

```bash
uv run --frozen --no-sync python -B graph_rag.py phase-b-diagnostic \
  --config configs/phase_b_graph_rag_diagnostic.json \
  --run-id path-a-full-20260719T153901Z
```

This direct route performs no network or model call. The Bash route additionally
performs its normal source/environment synchronization checks before dispatch.

## Read-only parent contract

The diagnostic refuses to create its child until it has hash-verified and
cross-checked all 18 required parent artifacts:

| Role | Required artifact |
| --- | --- |
| Run identity | `manifests/00-checkout-manifest.json` |
| Full-workflow identity | `manifests/debug-full-run.json` |
| Preparation identity | `manifests/03-data-preparation-manifest.json` |
| Scoring identity and input/output hashes | `manifests/score-manifest.json` |
| Simple condition identity | `manifests/verifier-simple-live.json` |
| Corrective condition identity | `manifests/verifier-corrective-live.json` |
| Prepared test corpus/support text | `data-prepared/test.jsonl` |
| Private test gold | `data-prepared/test-gold.jsonl` |
| Split identity | `data-prepared/split-manifest.json` |
| Development-only threshold gold | `data-prepared/development-gold.jsonl` |
| Development candidate family | `predictions/dev/candidate-index.json` |
| Label-safe selected threshold | `predictions/dev/threshold-selection.json` |
| Test prediction ledger | `predictions/test/candidates.jsonl` |
| Simple verdict ledger | `verifier/simple/verdicts.jsonl` |
| Corrective verdict ledger | `verifier/corrective/verdicts.jsonl` |
| Candidate scoring cross-check | `outcomes/candidate-outcomes.jsonl` |
| Sentence/condition cross-check | `outcomes/sentence-outcomes.jsonl` |
| Independent metric cross-check | `metrics/metrics.json` |

The preparation manifest binds prepared test, private gold, split, and
development-gold bytes. The score manifest binds candidates, both verdict
ledgers, the selected threshold, both outcome ledgers, and metrics. The two
verifier manifests independently bind their prepared input, shared candidate
ledger, and complete live verdict output. The adapter reconstructs condition
emissions from candidates, threshold, and verdict decisions, then requires
exact agreement with the sentence outcomes and independently recomputed Phase B
strict counts. A missing file, symlink, identity mismatch, byte/hash mismatch,
coverage mismatch, invalid correction, or changed metric count fails before a
Graph RAG child is created.

Predicted snapshots contain only Phase B emissions:

| Diagnostic condition | Phase B derivation |
| --- | --- |
| `confidence_filtered` | Candidate triple confidence at or above the frozen development-selected threshold |
| `simple_verifier` | Candidate triples with a valid simple-verifier `KEEP` |
| `corrective_verifier` | Valid corrective `KEEP` triples plus grounded, schema-valid `CORRECT` triples |

Private gold is never used to repair a predicted graph. It supplies intrinsic
scoring targets, private probe targets/support identities, and the explicitly
labelled `gold_graph_oracle` ceiling only. Predicted snapshot identities bind
prepared text/split plus candidate, threshold, and applicable verdict hashes;
changing a private-gold or score-manifest identity cannot change a predicted
snapshot identity. The oracle identity does bind private gold.

## Predeclared Regime-D design

- Unit: one configured training seed by one eligible private gold strict triple.
- Split: the frozen `CODE-SPLIT-1:test` examples only.
- Grouping and uncertainty cluster: source document; intrinsic comparisons also
  preserve the eight training-seed replicates.
- Public query: mechanically render only the gold head surface as `What fact is
  stated about "<head>"?`; public metadata contains only the template identity.
- Private scoring fields: answer/gold tail, gold relation, full strict key,
  source document, prepared evidence chunk, gold triple, cluster, training seed,
  and graph condition.
- Exclusions: an empty endpoint, identical normalized head/tail surfaces, or a
  private target/provenance/condition surface that would collide with the
  rendered public query. Every exclusion is retained with a private-target hash.
- Denominator: every eligible gold strict triple for every configured seed.
  Retrieval failures remain in the denominator and score zero; unavailable
  metric families are typed `not_applicable`, never coerced to zero.
- Retrieval family: genuine frozen-parameter BM25 text, each predicted graph,
  BM25-plus-predicted-graph reciprocal-rank fusion, and a private gold-graph
  oracle. Every condition receives the same eight-item/256-token budget.
- Outcomes: strict Triple Precision/Recall/F1 and structure per predicted graph;
  fact preservation, fact retrieval at *k*/MRR, source support at *k*/MRR,
  support-or-fact at *k*, and retrieval failure rate.
- Comparisons: confidence-filtered versus simple-verifier and
  confidence-filtered versus corrective-verifier, separately for graph and
  hybrid retrieval. Source-document-clustered paired bootstrap intervals use
  10,000 resamples, seed `20260731`, and 95% confidence. No p-value family was
  predeclared, so p-values are not reported.

The executable leakage sentinel serializes only query text and public metadata,
then rejects any tested query containing a private answer/tail, relation,
source/evidence/triple identity, training-seed value, or condition label. Gold
answer, gold relation, source ID, evidence ID, and graph-condition label never
enter retrieval.

## Child namespace and resume contract

All new files are confined to:

```text
output/<run-id>/graph-rag/
```

The existing `output/<run-id>/` parent is expected and is never claimed as a
Graph RAG namespace. The child identity binds the parent score-manifest hash,
the hash of all verified parent artifacts, diagnostic config, complete
standalone Graph RAG source surface, and every `rag-*.schema.json` file.
An occupied child without that exact identity fails closed.

Writes are create-only. A partial rerun may reuse a byte-identical file but
cannot replace a differing file. Completion is written last. A completed resume
revalidates the parent, child identity, artifact manifest, every artifact byte
and hash, and the completion record before returning
`already_complete_verified`. No code path writes anywhere in the parent except
the owned `graph-rag/` child.

The child contains canonical documents/chunks/public probes, private targets and
exclusions, per-seed predicted snapshots, the private oracle snapshot, retrieval
traces/failures, per-probe/per-seed/aggregate/intrinsic/comparison metrics, a
TSV summary, and config/parent/protocol/identity/artifact/run/completion
manifests plus a Python/platform/`bm25s` environment manifest. The runner
requires the configured `bm25s==0.3.9` before claiming or resuming a child.

## Historical run readiness

The archived copy of `path-a-full-20260719T153901Z` was inspected read-only. Its
18 prerequisites validate as one artifact set
`72380c8f6c642bcb13b94b1b39432a372583a5fa4b9c0f64eaa5f198e8629cd3`;
the score manifest is
`24cf3b80d0a9c94805350bb2446f27c3b8ff02584487811e3d593f4d7dd524af`.
It contains 173 prepared/gold records, 586 gold strict triples, 3,174 test
candidates, all seeds 42–49, both complete verdict ledgers, and a development
threshold of 0.25. Head-only probe derivation yields 585 eligible probes and
one declared `head_tail_surface_collision` exclusion.

The derived predicted triple counts for seeds 42–49 are:

- confidence-filtered: 376, 370, 377, 427, 445, 413, 389, 372;
- simple verifier: 315, 320, 310, 356, 351, 346, 330, 314; and
- corrective verifier: 255, 260, 254, 289, 285, 279, 270, 255.

These are readiness and deterministic derivation facts, not Graph RAG results.
No diagnostic output has been generated for the retained historical evidence in
this source-development session. The external recovery command above remains
required before any diagnostic metric may be reported.

## Active-surface audit

C-04 is the sole canonical result-producing Graph RAG route. The older generic
runner remains noncanonical for three evidence-backed purposes:

- Regime Q preserves the user-defined no-domain-expert hard stop and prevents a
  diagnostic probe from being relabelled as QA.
- The synthetic adapter/generator, dense-shaped fixture, graph corruptions, and
  layered evaluator remain actively exercised as a no-network software oracle
  for leakage, unavailable metrics, graph identity, budgets, and aggregation.
  Its manifests disable scientific claims.
- SciERC remains the separately approved, blocked, intrinsic-only Phase D
  readiness adapter and realistic later compatibility surface. It is not
  reachable from `phase_b.sh`, is not a C-04 condition, and supplies no current
  result.

The CODE-ACCORD generic readiness config, generators/judges, dense retrieval,
corruptions, standalone synthetic path, and SciERC adapter are therefore not
part of canonical C-04 execution. Reference, configuration, history, and test
analysis found active constraint-validation or separately approved readiness
consumers for each, so none was deleted. The historical
`provenance/eval_graph_rag.py` remains immutable methodology provenance and is
not invoked: its answer-as-query and unique-token-overlap design is prohibited
for C-04.
