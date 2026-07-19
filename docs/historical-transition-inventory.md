# Historical provenance transition inventory

This inventory implements B-05U's first gate. Historical paths are changed
because their independent CLIs cannot provide the published artifact's single,
rebuildable contract: they use incompatible ontologies/matchers, undocumented
or unavailable checkpoints, mutable external data, ad-hoc output locations, or
aggregate ledgers that cannot be replayed under the approved split and manifest
rules. Their Git lineage and confirmed generated ledgers are preserved through
the parent Phase B archive; they are not silently rewritten as canonical runs.

| Historical family | Why it cannot remain executable | Unified replacement | Disposition gate |
| --- | --- | --- | --- |
| `train_span.py`, `provenance/inference_kg.py` | Historical split/checkpoint and output contracts do not bind typed strict manifests. | `phase_b.py model generate-candidates` adapter. | Keep only until adapter parity/fixture proof. |
| `provenance/verify_triples_llm.py` | SciERC ontology and free-form parser conflict with frozen CODE verifier schema. | `phase_b.py verifier dry-run|live|replay`. | Delete after canonical adapter accepts the supported transport role. |
| `provenance/build_kg.py`, `provenance/eval_graph_rag.py`, `provenance/diagnose_evidence_paths.py` | Diagnostic questions derive from gold-bearing inputs and scoring is not a canonical task metric. | `phase_b.py secondary graph-rag` with declared diagnostic/noncomparable status. | Delete after explicit source-control fixture and manifest stage exist. |
| `provenance/train_multi.py`, dataset downloaders | Token-BIO benchmark protocol differs from typed strict scoring and has separate download lifecycle. | `phase_b.py secondary benchmark` matrix rows. | Delete after adapter/config fixtures cover supported datasets. |
| augmentation generators and Stage 2 runners | Multiple one-off generators write untracked `results/` and depend on unavailable models/checkpoints. | `phase_b.py secondary generate-data` and matrix-bound augmentation adapters. | Delete only after each paper claim is regenerated or marked unsupported. |
| GAN/Gumbel/cooperative/closed-loop paths | Negative-result variants are not independently rebuildable from tracked inputs. | Canonical provenance-normalizer plus blocked matrix rows. | Archive ledgers; delete code after noncomparability record and no consumer. |
| zh translation/DAPT/silver paths | External law sources, mutable corpora, and separate split rules break the standalone contract. | `phase_b.py secondary zh-data` acquisition/normalization/generation stages. | Delete after deterministic input manifests and final-test isolation tests. |
| `results.tsv`, `results_stage2.tsv` | Generated ledgers contain physical defects and are secondary evidence, not canonical inputs. | Canonical ledger normalizer/report stage. | Verified archive move under B-10; never delete without move proof. |

Every replacement must write under `output/<run-id>/`, bind raw/strict data and
configuration hashes, and be reachable from `phase_b.py`; unsupported claims
must be reported to the discrepancy ledger rather than retained through a
second executable workflow.
