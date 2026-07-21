# Phase B runner-interface inventory

## Public boundary and equivalence evidence

The sole publication-facing entry point is:

```bash
bash phase_b.sh --stage full
```

`phase_b.py` remains the internal dispatcher. The retained full run
`path-a-full-20260719T153901Z` was launched through the same script with
`--stage full`; its initial source commit was `3e4e5d9`, with recovery commits
`a125cd9` and `9eb6c7c`. Its sequence was bootstrap, independent seeds 42–49
training and development inference, development assembly and thresholding,
four live pilot captures, mandatory B-07 audit, independent seeds 42–49 test
inference, test assembly, simple and corrective live verification, then score.

The launcher cleanup does not change that sequence or any delegated
`phase_b.py` data-producing argument. It renames the entry point to `phase_b.sh`
and removes the invocation token
`--approve-b07`; the full-run manifest retains
`conditional_b07_approval: true`, matching the previous run's effective value.
The launcher still requires both `pilot_status=pass` and
`material_protocol_review_required=false` before test inference. Training,
candidate, verifier, scoring, seed, model, prompt, and dataset semantics are
unchanged. The separately approved ledger archival revises only configuration
and provenance schemas. Source-bound manifests necessarily
record the new source commit and timestamps; those provenance differences are
not scientific-data changes.

## Stage classification

| Stage | Classification | Retention reason |
| --- | --- | --- |
| `full` | Public canonical | Only supported unattended publication workflow; reproduces the retained full-run sequence. |
| `available` | Diagnostic/recovery | Checks command parsers and attempts only stages whose prerequisites exist. |
| `smoke` | Diagnostic | Isolated nonpublication GPU/Ollama path check with explicitly marked pseudo-seeds. |
| `publishable` | Diagnostic/recovery | Stops after one seed's development candidates for training/restart investigation. |
| `bootstrap` | Diagnostic/recovery | Rebuilds doctor, secondary reconciliation, acquisition, and preparation. |
| `plan` | Diagnostic/recovery | Validates one seed's frozen training plan without training. |
| `train-live` | Diagnostic/recovery | Produces or resumes one seed checkpoint. |
| `generate-live` | Diagnostic/recovery | Produces one declared live candidate artifact. |
| `generate-replay` | Diagnostic/recovery | Replays a frozen prediction ledger without model execution. |
| `threshold` | Diagnostic/recovery | Rebuilds the development-only threshold artifact. |
| `verifier-dry` | Diagnostic/recovery | Materializes verifier requests without live calls. |
| `verifier-replay` | Diagnostic/recovery | Replays frozen response evidence without live calls. |
| `verifier-live` | Diagnostic/recovery | Resumes one final verifier mode with stage-scoped response recovery. |
| `pilot-live` | Diagnostic/recovery | Creates one independently identified development-pilot capture. |
| `pilot-audit` | Diagnostic/recovery | Rebuilds the mandatory four-capture scientific gate. |
| `score` | Diagnostic/recovery | Rebuilds metrics from completed frozen inputs. |
| `legacy-train` | Provenance-only compatibility | Retains the acknowledged historical CSV trainer diagnostic; never satisfies a canonical checkpoint gate. |

No stage is deleted: each non-public stage has a distinct inspection, replay,
resume, or stage-scoped recovery role. Removing one would make an archived
failure mode or offline replay path unreachable without broadening cleanup.

## Resolved legacy-ledger cleanup

The user selected scientific-result integrity rather than byte identity of
secondary provenance manifests. `results.tsv` and `results_stage2.tsv` were
copied to external archival storage, destination hashes were verified, and the
source copies were removed. `reconcile` now validates a frozen archived-evidence
register without any external path or runtime dependency. This changes only the
secondary reconciliation manifest; candidates, outcomes, Accuracy, Precision,
Recall, F1, confusion counts, and uncertainty calculations are unaffected.
