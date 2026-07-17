# Phase B root-module mapping

This is the B-05R zero-omission mapping for the active Phase B implementation.
It is anchored to the fixed review baseline
`9feafa4029e65ab48ecfb2f452f4b0fabbff0826`.  Each listed package module was
moved as a whole: every module-level class, function, constant, and private
helper in the source file has the same-named root-module host unless this table
says otherwise.  The only edits during the move are absolute root imports,
root discovery, and the public command spelling.

| Former path | Root host | Baseline host/disposition | Mapping result |
| --- | --- | --- | --- |
| `phase_b_pipeline/acquisition.py` | `acquisition.py` | No baseline host | Whole-file move; archive contract and fetch symbols retained. |
| `phase_b_pipeline/cli.py` | `phase_b.py` | No baseline host | Whole-file move; `_parser` and `main` retained; this is the root command. |
| `phase_b_pipeline/config.py` | `config.py` | No baseline host | Whole-file move; frozen configuration validation retained. |
| `phase_b_pipeline/constants.py` | `constants.py` | No baseline host | Whole-file move; protocol constants retained. |
| `phase_b_pipeline/doctor.py` | `doctor.py` | No baseline host | Whole-file move; preflight symbols retained. |
| `phase_b_pipeline/io.py` | `phase_b_io.py` | No baseline host | Whole-file move renamed to avoid collision with Python standard-library `io`; contracts, hashing, and atomic writes retained. |
| `phase_b_pipeline/metrics.py` | `metrics.py` | No baseline host | Whole-file move; metric primitives retained. |
| `phase_b_pipeline/paths.py` | `paths.py` | No baseline host | Whole-file move; source-root sentinel changes from package directory to `phase_b.py`. |
| `phase_b_pipeline/pilot.py` | `pilot.py` | No baseline host | Whole-file move; development-only audit symbols retained. |
| `phase_b_pipeline/preparation.py` | `preparation.py` | `prepare.py` is unrelated FineWeb tokenizer preparation | Whole-file move; no function body changed. The official typed-BIO hard stop remains intact. |
| `phase_b_pipeline/reconciliation.py` | `reconciliation.py` | No baseline host | Whole-file move; secondary-ledger audit symbols retained. |
| `phase_b_pipeline/records.py` | `records.py` | No baseline host | Whole-file move; typed record symbols retained. |
| `phase_b_pipeline/scoring.py` | `scoring.py` | No baseline host | Whole-file move; canonical matcher and score symbols retained. |
| `phase_b_pipeline/split.py` | `split.py` | No baseline host | Whole-file move; deterministic split symbols retained. |
| `phase_b_pipeline/statistics.py` | `phase_b_statistics.py` | No baseline host | Whole-file move renamed to avoid collision with Python standard-library `statistics`; paired statistics symbols retained. |
| `phase_b_pipeline/verifier.py` | `verifier.py` | `verify_triples_llm.py` is incompatible SciERC provenance | Whole-file move; frozen CODE verifier symbols retained. |
| `phase_b_pipeline/__init__.py` | none | No baseline host | Removed after package deletion; it contained only package metadata. |
| `phase_b_pipeline/__main__.py` | none | No baseline host | Removed after package deletion; `phase_b.py` supplies the direct executable entry point. |

The active root command is `python -B phase_b.py`. Historical root scripts
remain separate provenance paths. In particular, `prepare.py` and
`verify_triples_llm.py` were not overwritten because their data/model contracts
are incompatible with approved Path A semantics.

## Parity checks

Before package deletion, the migration requires: no relative imports; no
`phase_b_pipeline` runtime, test, configuration, or workflow-document
reference; AST parsing of all root modules; focused and full test success; and
direct `python -B phase_b.py --help` command success. `preparation.py` is also
checked against the pre-move file with imports normalized, so changes to its
function bodies are rejected.
