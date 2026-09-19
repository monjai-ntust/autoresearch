# Historical-derived encoder comparison contract

`encoder_comparison.py` is the isolated Phase G entry point for the approved
four-row Table-1-only encoder comparison. Model acquisition and live execution
remain gated by the smoke, accelerator, storage, and final-test rules below.

## Historical implementation boundary

The active span/relation model and trainer behavior are bounded transplants
from source commit `9feafa4029e65ab48ecfb2f452f4b0fabbff0826`:

| Historical authority | Git blob | Current host |
| --- | --- | --- |
| `train_span.py` | `104ca6e4afcb5122228fff5c73227c6b3b3edb40` | `utils/encoder_comparison/trainer.py` |
| `models/bert_kg_encoder.py` | `d05678f00097cb11176dacef6fade0b1b3f557de` | `utils/encoder_comparison/network.py` |
| `data/code_accord.py` label/batch contract | `0b11fc6bf2f16512864b4dbf77e7a610d8fa8fdb` | comparison-only adapter in `utils/encoder_comparison/data.py` |

Inactive historical experimental branches are not restored. Two recipes are
retained because they correspond to populated draft rows. The common base uses
BIO weight `0.1`, negative ratio `3.0`, no label smoothing, no inter-span
context, and no comparison boost. The method recipe preserves A12 inter-span
context, A21 label smoothing, `NO_REL=1.0`, and the effective A20 behavior in
the current canonical pipeline: linear `5x -> 2x` decay plus a one-time
development gate. A middle-threshold result applies `3.5x` for exactly the gate
batch, then the switched state uses `2x`.

## Current utility reuse boundary

The comparison controller calls `stages.preparation.run` directly, so dataset
acquisition, relation reconstruction, `CODE-SPLIT-1`, and prepared JSONL are
identical to the canonical path. It also reuses:

- `utils.encoder.data` unchanged for prepared-record loading, canonical item/
  batch construction, and the resumable sampler; the comparison-only adapter
  appends historical first-subword BIO labels, words, and example IDs;
- `utils.encoder.cache` for the run-local Hugging Face cache manifest;
- `utils.encoder.model.generate_candidates` for deterministic prediction-
  ledger replay into the canonical candidate contract; and
- `utils.common` for configuration, path containment, hashes, manifests, and
  record identities.

Historical-derived code owns encoder construction, loss, optimizer, schedule,
development checkpoint selection, and raw prediction-ledger production.
Current shared code owns the corrected data boundary and artifact lineage.

## Profile and run contract

`resources/configs/encoder-comparison.json` is a finite registry and the
machine-readable experiment/statistics contract. Each profile
pins a model name, immutable 40-hex model-repository revision, expected hidden
size, tokenizer assets, weight byte count/SHA-256, license, and source. The
exact base-parameter count is derived at G-05 by summing every tensor element
in the pinned `pytorch_model.bin` state dict after digest verification. Recipe
selection is independent from backbone selection, but runtime commands accept
only one of the four approved profile/recipe arms. Model construction
explicitly loads the pinned `pytorch_model.bin` form rather than allowing a
repository's optional safetensors preference to select different bytes.

| Arm ID | Table 1 role |
| --- | --- |
| `bert-base-common` | BERT-base backbone control |
| `deberta-base-common` | DeBERTa-base backbone control |
| `deberta-large-common` | DeBERTa-large backbone and recipe reference |
| `deberta-large-a20-a21-a12` | DeBERTa-large method row |

The old draft values are stored only as provenance. They are not inputs,
acceptance thresholds, or values to tune toward. Augmentation, external-system,
local-Qwen, Cui/focal, and optional-modern-encoder placeholders are outside the
approved matrix because they do not have an eligible populated Table 1 result.

One run binds to exactly one approved arm in
`manifests/encoder-comparison-selection.json`. Every model cache, checkpoint,
restart state, prediction ledger, and candidate file stays below that physical
run. A different profile, run ID, revision, or model digest fails closed.

Offline validation does not acquire a model or create a run:

```bash
uv run --frozen --no-sync python -I -B encoder_comparison.py \
  validate-profile --profile bert-base-uncased

uv run --frozen --no-sync python -I -B encoder_comparison.py \
  validate-arm --arm bert-base-common
```

The implemented lifecycle commands are `prepare`, `plan-training`, `train`,
and `generate`; each takes `--arm`. The four arms and statistical protocol are
frozen, but live use still requires G-05 model-cache/tokenizer/accelerator smoke
admission. `pipeline.py` has no import or dispatch route to this comparison
path.

## Validation boundary

Focused tests resolve the historical Git blobs, validate every profile and arm without
network access, reject unknown/mutable profiles and model digest drift, verify
hidden-size-dependent heads, preserve the one-batch A20 middle value, exercise
prepared-data/BIO compatibility, reject profile or run substitution, and prove
every executable/resource file in the original Phase F pipeline flow remains
byte-identical to `5db3bce`. The full canonical test suite must also pass. These
are static and CPU-local
contracts; tokenizer smoke, forward/backward, checkpoint round-trip, resume
equivalence, accelerator memory, and scientific outcomes remain later gates.
