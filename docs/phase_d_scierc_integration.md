# Phase D SciERC integration

## Status and scientific boundary

Phase D selects `dataset-specific` for SciERC and `exclude-or-block` for CUAD.
SciERC is conditionally integrated only as supervised portability of the existing
model family and implementation to a second native NER+RE ontology. It is not:

- zero-shot reuse of a CODE-ACCORD checkpoint;
- evidence that the CODE-ACCORD ontology transfers to scientific text;
- a verifier experiment;
- a Graph RAG question-answering experiment; or
- reproduction of the unretained historical `0.410` result.

CUAD receives no new adapter, model run, graph conversion, or benchmark. Its native
question-conditioned clause-span task is not represented as relation extraction.

The tracked SciERC configuration is a blocked readiness contract. It does not
authorize training or a publication result:

```powershell
python graph_rag.py doctor `
  --config configs/phase_d_graph_rag_scierc.json `
  --run-id d-scierc-readiness-<unique-id>
```

The command claims a new immutable run ID and writes its identity, environment,
configuration, and blocked checkpoint manifests under
`output/<run-id>/graph-rag/`. A blocked exit is expected until the pinned dataset
files, an approved dataset-specific checkpoint, and its prediction ledger exist.

## Native mapping

`graph_rag_eval.adapters.scierc:SciERCAdapter` maps the processed SciERC format
without importing dataset code into the core:

| Native SciERC object | Canonical object |
| --- | --- |
| One `doc_key` abstract | `Document`; `cluster_id` is the `doc_key` |
| One native sentence | `Chunk`, preserving document-global token and document-local character offsets |
| One typed mention | `Entity`, preserving the six-way type and exact source span |
| One typed endpoint pair | `Triple` plus one of seven native `Relation` definitions |
| Official split membership | `SplitMembership`, grouped at abstract level |
| Coreference clusters | Count retained as source metadata; not silently converted into relations |
| Questions/answers/evidence | Unavailable; emitted as empty/not-applicable, never derived from gold relations |
| Predicted graph | Parsed only from an independently supplied prediction ledger |

`COMPARE` and `CONJUNCTION` are symmetric and normalize endpoint order during
matching. `USED-FOR`, `FEATURE-OF`, `HYPONYM-OF`, `EVALUATE-FOR`, and `PART-OF`
remain directed.

## Pinned data contract

The adapter accepts only these official processed split identities in scientific
mode:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `train.json` | 720,664 | `04970819bd215fce8c60b3a64fccca15388b49eab2010bdc5f6d322b463568c3` |
| `dev.json` | 103,316 | `61f21b224129e580a654b036ee6fffeeb1a0311f1009a38ea931c13612db040e` |
| `test.json` | 213,898 | `7424da64e6214a90e39b09e47a74b3ded57dde86b4a7f848b3625d2c8ecdfbe1` |

The required counts are 350/50/100 abstracts. Duplicate `doc_key` values across
splits, a changed byte hash, a count mismatch, an unknown label, a cross-sentence
span, or a relation endpoint without a declared entity fails closed.

The full official processed archive is:

- `https://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz`
- 695,340,151 bytes
- SHA-256
  `bce752fb7ebe4acf570937d76ffb27c239cfc907e477a69075dc7082d0e72e9b`

The three small files exposed by the existing
`sthoran/scierc_processed_data` Hugging Face mirror were independently verified
byte-for-byte against the corresponding official archive members. The old
`data/download_scierc.py` writes beneath tracked `data/` and is retained only as
historical provenance; it is not the canonical Phase D acquisition path.

After `doctor` has claimed a unique run, place the three verified files at:

```text
output/<run-id>/graph-rag/inputs/scierc/train.json
output/<run-id>/graph-rag/inputs/scierc/dev.json
output/<run-id>/graph-rag/inputs/scierc/test.json
```

Do not commit or archive those files. The official page, paper, archives, and
upstream code contain no explicit dataset redistribution license. Public access
does not establish redistribution permission. Raw text, canonical text-bearing
records, and derived text-bearing output must remain local and must not enter the
parent output archive unless the user establishes an applicable license or
permission. Non-text metrics may be retained only with a manifest that makes this
restriction explicit.

## Prediction-ledger contract

`inputs/scierc-predictions.jsonl` is independent extractor output. Every line uses
`rag-extraction-prediction-1.0`. An entity row declares a document-global inclusive
token span, native entity type, and confidence:

```json
{"schema_version":"rag-extraction-prediction-1.0","record_type":"entity","doc_key":"A00-1024","start":0,"end":2,"entity_type":"Method","confidence":0.91}
```

A relation row references two entity rows by their exact span/type keys:

```json
{"schema_version":"rag-extraction-prediction-1.0","record_type":"relation","doc_key":"A00-1024","sentence_index":0,"head":{"start":0,"end":2,"entity_type":"Method"},"tail":{"start":7,"end":8,"entity_type":"Task"},"relation_type":"USED-FOR","confidence":0.77}
```

Unknown keys, duplicate entities/relations, invalid confidences, out-of-sentence
spans, undeclared endpoints, and ontology violations fail closed. A missing ledger
is a typed blocker and produces no zero-valued extraction result.

## Immutable run and checkpoint identity

The first Graph RAG stage to claim a run writes
`manifests/run-identity.json`. Subsequent stages may resume only when all of these
match:

- run ID and optional experiment-group ID;
- complete configuration content;
- adapter import/ID/version/options;
- checkpoint manifest content;
- standalone Graph RAG source surface; and
- every `rag-*.schema.json` identity.

An occupied directory without a valid identity or any mismatch fails before
overwriting an artifact. Existing Phase B synthetic and CODE-ACCORD configurations
remain byte-identical `rag-run-config-1.0` inputs. The new fields and manifests are
additive; no historical result is migrated or rewritten.

The proposed SciBERT base and tokenizer are pinned to
`allenai/scibert_scivocab_uncased` revision
`24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1`. A future ready checkpoint manifest
must additionally bind the downloaded model/tokenizer tree, architecture and head
vocabularies, training recipe, split, selection metric, seed, checkpoint path, and
checkpoint SHA-256. Changing a blocked readiness manifest into a ready scientific
manifest requires a new run ID.

## Proposed training protocol awaiting user approval

The following proposal is not executable until the user approves its compute/cost
and the source implementation is separately completed:

1. Use only the pinned official 350-abstract training and 50-abstract development
   splits for optimization and selection. Keep the 100-abstract test split
   untouched until one final evaluation per frozen seed.
2. Use `BertKGExtractor` with the pinned SciBERT base, 13 BIO outputs
   (`O` plus B/I for six entity types), and eight relation outputs (seven native
   labels plus `NO_REL`). This is a newly trained SciERC model.
3. Train gold-only with seeds 42–49, effective batch size 16, maximum length 256,
   AdamW learning rate `3e-5`, 1,500 update steps, 250 warm-up steps, relation-loss
   weight 1.0, and development evaluation every 100 steps.
4. Reject—not silently drop—any example whose gold entity or relation endpoint is
   truncated at 256 tokens. A preflight count of zero truncated gold annotations
   is required before training; otherwise the protocol returns for revision.
5. Select each seed's checkpoint by development end-to-end typed-relation F1;
   break a tie by the earlier update step. Test labels cannot tune thresholds,
   select checkpoints, choose seeds, or alter preprocessing.
6. Report exact entity and end-to-end typed-relation TP/FP/FN, Precision, Recall,
   F1, and per-label support. Report mean/dispersion across seeds and
   abstract-clustered uncertainty; call any project-specific triple metric
   separately from official SciERC metrics.
7. Record wall time, accelerator, peak memory, dependency/model cache tree,
   checkpoint bytes/hash, and failure/missing-data behavior per seed.

The user must decide whether to authorize this eight-seed external-accelerator
protocol and its compute cost. If it is not approved, SciERC remains
`exclude-or-block` for scientific execution while the tested adapter is retained
as interface compatibility evidence only.

## Later restoration handoff

Phase D does not restore source commit
`47e9af310bb7868eb79641e48451f08728adca14`. A later separately approved phase
must run:

```text
git revert ecd604141d4b23a96127d3a95876953ac16aa5c7
```

and perform focused and broad validation before an ordinary fast-forward push.
The Phase D SciERC changes do not overlap the 12 paths changed by that temporary
revert, so no semantic conflict is presently expected. Reset, rebase removal, and
force push remain prohibited.
