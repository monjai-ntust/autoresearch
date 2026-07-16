# Phase C migration plan: integrity gate to typed strict-view preparation

## Status and authority

This is a migration design and maintenance record. It does not activate a new
publication protocol or authorize result-producing execution. The canonical
publication path remains Phase B Path A, protocol `B04-PATH-A-1.3`, workflow
`PATH-A-WORKFLOW-1.3`, invoked through:

```text
python -B -m phase_b_pipeline
```

Official CODE-ACCORD v1.0.0 preparation must continue to retain the exhaustive
alignment audit and hard-stop before materializing gold. Phase C describes how
that integrity-focused baseline could migrate to the proposed dual-view
solution after a separate protocol revision is approved. If this document and
the current Path A workflow conflict, the current approved Path A workflow is
authoritative. If either conflicts with an applicable methodological
requirement in the target-journal paper, the journal requirement takes
priority and this plan must be revised before implementation.

## Migration objective and non-goals

The proposed destination keeps two explicitly different data views:

1. an immutable raw provenance view containing all 3,329 published positive
   relation rows; and
2. a typed strict view containing the 3,320 rows whose two marker arguments
   resolve uniquely to authoritative BIO spans, representing 3,319 unique
   directed typed triples after one exact duplicate collapses for set scoring.

The nine incompatible rows remain visible in the raw view and row-level audit
but would be ineligible for typed joint training/development gold. No final-test
row is affected by the currently verified incompatibilities.

This plan does not authorize manual endpoint repair, inferred entity types,
composite-marker expansion, ambiguous marker projection, silent row deletion,
reuse of the historical type-agnostic matcher as canonical gold, or rewriting
historical measurements to preserve a preferred result.

## Assumptions that must remain unchanged

- The input is the immutable CODE-ACCORD v1.0.0 archive identified in
  `configs/phase_b_path_a.json`, including its exact size, MD5, and six
  annotation-file SHA-256 values.
- `annotated_data/entities/all.csv` remains authoritative for typed BIO spans.
  Entity train/test files define only the verified 689/173 sentence boundary.
- The approved UUID repair remains singular, explicit, and auditable.
- `CODE-STRICT-1` remains typed, directed, and example-scoped. Display text is
  not a substitute for span/type identity.
- The 173-sentence official entity-test boundary remains isolated from model
  selection, prompt design, threshold selection, and migration decisions.
- `CODE-SPLIT-1` retains its seed, target sizes, stratification labels, stable
  tie-breaking, disjointness checks, and input-order independence. It runs only
  after the eligible typed view is frozen.
- Every workflow-created prerequisite, intermediate, checkpoint, prediction,
  audit, metric, table, manifest, and log is written beneath ignored
  `output/<run-id>/`.
- A same-seed claim requires identical frozen inputs/configuration plus the
  documented deterministic checkpoint and output-hash checks. A trend that
  looks similar is not evidence of reproducibility.

## Reuse and adjustment map

| Current code or data path | Phase C disposition | Required adjustment or constraint |
|---|---|---|
| `phase_b_pipeline/acquisition.py` | Reuse | Preserve resumable download, safe selective extraction, and exact immutable checks. Any upstream-byte change creates a new dataset identity and requires plan revision. |
| `phase_b_pipeline/config.py`, `constants.py`, `io.py`, and `paths.py` | Reuse | Add versioned dual-view identities and manifests without weakening path containment or output-root rules. |
| CSV/BIO parsing and entity partition checks in `phase_b_pipeline/preparation.py` | Reuse | Keep `entities/all.csv` authoritative and preserve all source row numbers and marker text. |
| UUID repair and marker-alignment audit in `phase_b_pipeline/preparation.py` | Reuse | The exhaustive audit remains a prerequisite. Eligibility must be derived from its recorded exact/contained/unresolved status, never from a second fuzzy matcher. |
| Current preparation hard stop | Adjust only after approval | Replace the post-audit stop with an approved dual-view gate that writes raw provenance and strict eligibility records before any split. Until approval, the stop remains mandatory. |
| Duplicate preservation/collapse logic | Reuse with explicit views | Preserve both source rows in raw provenance; collapse the one identical typed key only in set-based strict gold and record the contributing rows. |
| `phase_b_pipeline/split.py` | Reuse after input change | Regenerate, never transplant, the assignment from strict-view label presence. Bind its seed, input inventory, output membership, and tree hash to the new protocol. |
| `phase_b_pipeline/scoring.py`, `metrics.py`, and `statistics.py` | Reuse | Permit scoring only after strict gold and all candidates share the same approved data/split identities. Metric definitions do not change merely to retain a historical trend. |
| `data/code_accord.py` and historical evaluation helpers | Secondary support only | Do not use their fuzzy surface lookup or type-agnostic key for canonical gold. If reused operationally, place a versioned adapter behind canonical records and prove parity on eligible fixtures. |
| Historical `train_span.py`, inference, verifier, and graph scripts | Migrate selectively | Reuse model behavior only through thin canonical adapters that enforce output paths, frozen splits, typed identities, seed/config capture, restart manifests, and no final-test tuning. |
| `results.tsv`, `results_stage2.tsv`, historical logs, and command records | Immutable provenance | Preserve original bytes/hashes when archived. Corrections or portable reruns produce separately identified artifacts; they never overwrite the historical ledger. |

## Proposed migration workflow

Each step is independently gated and must leave machine-readable evidence under
one run directory.

1. **Freeze the baseline.** Record the source commit, configuration/protocol
   identities, archive and annotation hashes, current alignment-audit hash, and
   the expected hard-stop result.
2. **Approve a protocol revision.** Freeze the raw-view and strict-view
   definitions, row eligibility rule, duplicate semantics, publication wording,
   and manifest schemas. No materialization code runs before this approval.
3. **Materialize raw provenance.** Write all 3,329 positive rows and the 1,000
   `none` rows with original row numbers, marker fields, repair lineage, and
   raw-corpus counts. Prove the raw view is a lossless representation of the
   frozen annotation inputs.
4. **Materialize strict eligibility.** Join only through the exhaustive typed
   alignment audit; emit 3,320 eligible rows, nine ineligible-row records with
   reasons, and 3,319 unique strict keys with duplicate lineage. Generate the
   result twice and require byte/tree-hash identity before promotion.
5. **Regenerate `CODE-SPLIT-1`.** Use strict-view labels and seed 42, preserve
   the official 173-sentence test boundary, and record the disposition of every
   incompatible row. Recheck exact sizes, disjointness, order independence,
   leakage guards, distributions, and raw/strict counts.
6. **Migrate claim-bearing adapters.** Route only required historical
   training/inference behavior through canonical records. Add parity fixtures
   before retiring or demoting any old command, and keep provenance-only paths
   visibly separate in the README.
7. **Rebuild checkpoints and intermediates.** Pin model revisions,
   dependencies, hardware/runtime metadata, seeds, restart state, selection
   rules, and hashes. For seed 42, require the approved independent rebuild
   equivalence test before seeds 43-49 or final-test work.
8. **Run canonical scoring and reporting.** Bind every prediction, verdict,
   metric, table, and figure to the approved strict-view and split hashes. Keep
   raw-corpus and strict-view counts separate in all publication output.
9. **Validate from a clean source-only clone.** Re-fetch prerequisites,
   recreate all intermediates, and compare the documented same-seed hashes.
   Archive all complete and failed output runs through the parent repository's
   verified-move procedure only after writers stop.

## Secondary historical material policy

The README's retained evaluation and historical reproduction section is
secondary provenance support. Its commands may be adapted for current paths,
dependencies, diagnostics, or canonical adapters, provided that:

- the original command/result record remains preserved with its hash and
  provenance;
- every adapted rerun receives a new run ID, configuration, environment
  manifest, and output hash rather than replacing an earlier result;
- a change is accepted because its semantics and validation are correct, not
  because its numbers retain the expected direction;
- a material trend reversal, metric non-comparability, or unexplained result
  change is reported and investigated rather than tuned away; and
- no main Path A claim relies solely on a historical loader, ledger, checkpoint,
  or type-agnostic metric.

Overall trend and claim direction may be used as a discrepancy signal when
reviewing a portable rerun. They are not pass/fail criteria and cannot override
the canonical data, matcher, leakage, or reproducibility contracts.

## Conditions requiring migration-plan revision

Update this document in the same source commit whenever a discovered reuse
issue changes a migration step, assumption, dependency, or acceptance gate.
At minimum, revision is required if:

- the official archive or any required annotation byte/hash changes;
- the authoritative entity interpretation, UUID repair count, unresolved-row
  inventory, duplicate count, or raw/strict counts change;
- an endpoint-repair policy other than strict ineligibility is proposed;
- the official test boundary, split seed/labels/sizes, or matcher identity
  changes;
- reused historical code cannot preserve typed span/direction semantics,
  output containment, exact seed capture, checkpoint restart state, or final-test
  isolation without a substantive rewrite;
- clean-clone reproduction exposes hidden parent-repository, OS, path, network,
  credential, model-cache, hardware, or dependency assumptions;
- same-seed checkpoint or output hashes are not reproducible under the frozen
  deterministic contract;
- a historical rerun materially reverses a supporting trend or proves a metric
  non-comparable;
- the journal reference requires a different baseline, uncertainty analysis,
  validation design, or presentation; or
- a new paper discrepancy affects a reused path or the proposed raw/strict-view
  distinction.

When triggered, record the evidence, affected component and steps, whether the
current canonical hard stop is still safe, the proposed amendment, validation
needed, and approval status. Do not silently update protocol/config identities.

## Revision log

| Date | Revision | Evidence and effect |
|---|---|---|
| 2026-07-17 | C-MIGRATION-1.0 | Established the design-only migration from the approved integrity hard stop to the proposed raw/typed-strict dual view; classified canonical and historical reuse boundaries and froze revision triggers. No Phase C materialization or result execution was authorized. |
