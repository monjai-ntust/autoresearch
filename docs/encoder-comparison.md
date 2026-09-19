# Historical-derived encoder comparison contract

`encoder_comparison.py` is the isolated Phase G entry point for future
Table-1-only encoder comparisons. Its presence implements and validates the
comparison boundary; it is not authorization to acquire models or execute an
experiment.

## Historical implementation boundary

The active span/relation model and trainer behavior are bounded transplants
from source commit `9feafa4029e65ab48ecfb2f452f4b0fabbff0826`:

| Historical authority | Git blob | Current host |
| --- | --- | --- |
| `train_span.py` | `104ca6e4afcb5122228fff5c73227c6b3b3edb40` | `utils/encoder_comparison/trainer.py` |
| `models/bert_kg_encoder.py` | `d05678f00097cb11176dacef6fade0b1b3f557de` | `utils/encoder_comparison/network.py` |
| `data/code_accord.py` label/batch contract | `0b11fc6bf2f16512864b4dbf77e7a610d8fa8fdb` | compatibility additions to `utils/encoder/data.py` |

Inactive historical experimental branches are not restored. The retained
historical recipe preserves span NER, ordered relation pairs, A12 inter-span
context, A21 label smoothing, `NO_REL=1.0`, and the implemented A20 behavior:
linear `5x -> 2x` decay plus a one-time development gate. A middle-threshold
result applies `3.5x` for exactly the gate batch, then the switched state uses
`2x`.

## Current utility reuse boundary

The comparison controller calls `stages.preparation.run` directly, so dataset
acquisition, relation reconstruction, `CODE-SPLIT-1`, and prepared JSONL are
identical to the canonical path. It also reuses:

- `utils.encoder.data` for prepared-record loading, first-subword BIO labels,
  batching, and the resumable sampler;
- `utils.encoder.cache` for the run-local Hugging Face cache manifest;
- `utils.encoder.model.generate_candidates` for deterministic prediction-
  ledger replay into the canonical candidate contract; and
- `utils.common` for configuration, path containment, hashes, manifests, and
  record identities.

Historical-derived code owns encoder construction, loss, optimizer, schedule,
development checkpoint selection, and raw prediction-ledger production.
Current shared code owns the corrected data boundary and artifact lineage.

## Profile and run contract

`resources/configs/encoder-comparison.json` is a finite registry. Each profile
pins a model name, immutable 40-hex model-repository revision, expected hidden
size, tokenizer assets, weight byte count/SHA-256, license, and source. Recipe
selection is independent from backbone selection. Model construction explicitly
loads the pinned `pytorch_model.bin` form rather than allowing a repository's
optional safetensors preference to select different bytes.

One run binds to exactly one profile and recipe in
`manifests/encoder-comparison-selection.json`. Every model cache, checkpoint,
restart state, prediction ledger, and candidate file stays below that physical
run. A different profile, run ID, revision, or model digest fails closed.

Offline validation does not acquire a model or create a run:

```bash
uv run --frozen --no-sync python -I -B encoder_comparison.py \
  validate-profile --profile bert-base-uncased
```

The implemented lifecycle commands are `prepare`, `plan-training`, `train`,
and `generate`. They exist for the later approved matrix, but must not be used
until Phase G freezes the arms, common recipe, statistical protocol,
compute/storage limits, and stop rules. `pipeline.py` has no import or dispatch
route to this comparison path.

## Validation boundary

Focused tests resolve the historical Git blobs, validate every profile without
network access, reject unknown/mutable profiles and model digest drift, verify
hidden-size-dependent heads, preserve the one-batch A20 middle value, exercise
prepared-data/BIO compatibility, and reject profile or run substitution. The
full canonical test suite must also pass. These are static and CPU-local
contracts; tokenizer smoke, forward/backward, checkpoint round-trip, resume
equivalence, accelerator memory, and scientific outcomes remain later gates.
