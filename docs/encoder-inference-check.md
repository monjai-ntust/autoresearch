# External checkpoint inference check

This is a user-authorized experiment on existing pipeline checkpoints; the
initial example used `research-1022`, and the completed external test selected
`research-1020`. `comparison-1020` is left untouched. After reviewing the result,
the user authorized the production tokenizer correction on 28 October 2026.
Canonical prediction now explicitly uses `add_prefix_space=True`, matching
training and comparison prediction. The standalone tool retains its additional
meta-loading, numerical-policy and independent-repeat checks.

Comparison training and both development/test prediction already share
`utils/encoder_comparison/trainer.py::_tokenizer`, which explicitly sets
`add_prefix_space=True` for both DeBERTa profiles. Regression coverage exercises
the real worker arguments for all four comparison arms and checks those settings
at the tokenizer factory. BERT keeps its existing WordPiece tokenizer settings.

Copy **only** `check_encoder_inference.py` beside `pipeline.py` in the existing
external source checkout. Keep that checkout and its run bindings; a pull is
not needed. The script imports the codebase already present there and records
its actual Git/helper identity along with the script's own digest.

From that source checkout, use its existing frozen environment:

```bash
uv run --frozen --no-sync python -I -B check_encoder_inference.py \
  --run-id research-1022 --test-id inference-test-1022
```

Or use the environment's Python directly:

```bash
.venv/bin/python -I -B check_encoder_inference.py \
  --run-id research-1022 --test-id inference-test-1022
```

`--device cuda:0` is the default. `--device cpu` is available for a separate
diagnostic environment, with no claim of CPU/GPU bitwise equality. An existing
test directory is never overwritten or resumed: retain it and choose another
`--test-id` for another experiment. An interrupted/failed trial keeps its
available evidence and never publishes a successful aggregate.

The script validates all eight selected checkpoints, the original preparation
and model-cache manifests, test/gold/split coverage and same-parent hashes.
It consumes only `research-1022` files and never searches another run for a
cache. Optimizer/restart-state files are not consumed or needed. All original
run files remain read-only; this diagnostic uses no workflow file lock.

The specific inference correction is **`add_prefix_space=True`**, matching
the tokenizer setting used by training and comparison inference. The script
calls the existing `_live_inference_records` function with scoped loading,
tokenizer and recording adapters. It reuses the canonical network's complete
forward logic, span ordering/overlap selection, prediction-ledger candidate
replay, typed strict scorer and quality summary. It adds no entity/triple cutoff,
changes no gold matching, and performs no threshold/seed/recipe search.

To avoid randomized initialization, the existing network is built on meta
tensors from the local configuration, then every checkpoint tensor is loaded
strictly. Missing/unexpected keys, changed precision or inconsistent registered
embedding aliases fail. Only the known nonpersistent DeBERTa position-index
buffer is regenerated with `arange`. There is no sampling, shuffle, optimizer,
training or manual seed search. Evaluation/inference mode disables dropout.

The numerical policy uses float32, eager attention, one CPU thread, deterministic
algorithms with errors rather than warning-only fallback, no cuDNN benchmark,
highest float32 matmul precision and the codebase's cuDNN TF32 policy. Each
worker starts with `PYTHONHASHSEED=0` and offline model loading. Torch CPU/CUDA,
Python and NumPy RNG state must remain unchanged during model construction and
inference; otherwise the trial fails. These are explicit prospective settings,
not evidence that the earlier 0.520 execution used identical numerical policy.

Each of seeds 42–49 runs **twice in fresh processes**. The script compares exact
prediction-ledger, candidate, metric and recorded runtime hashes before accepting
each seed. It rechecks source and parent inputs before/after execution. A changed
runtime, corrupted output, unsupported deterministic operation or different
repeat fails, retaining the outputs for inspection.

Results are in a new testing directory, as requested:

```text
output/inference-test-1022/
  plan.json
  seed-42/pass-1/{prediction-ledger.jsonl,candidates.jsonl,quality.json,receipt.json,manifests/}
  seed-42/pass-2/...
  ... seed-49 ...
  statistics.json
```

`statistics.json` contains per-seed TP/FP/FN and P/R/F1, mean/sample SD/best,
pooled metrics, repeat equality and the difference from the earlier comparison
recipe mean **0.5199012003522201**. That value is a descriptive reference only;
the script does not force or optimize a score. Output has an explicit
`parent_run_id: research-1022`; it is diagnostic evidence, not a new canonical
training run or a production input. Do not transplant its files into the parent
run's predictions/manifests.

Exact repeatability is checked on the actual recorded software/hardware. The
same checkpoint alone cannot guarantee identical floating-point results across
different devices, library versions or driver builds. The two-pass check proves
the observed pair agrees; it is not proof over every future environment.
PyTorch describes both these limits and the strict algorithm option in its
[reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
The meta/strict-assignment loading pattern follows its
[checkpoint loading recipe](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html).

Return the complete testing directory, including failures if any. The completed
`inference-test-1027-uv` reports mean F1 0.5199012003522201, sample SD
0.015508657649720559 and best 0.5423406279733587, matching the earlier recipe's
eight per-seed count/metric sets. Reported checkpoint hashes also match both
retained canonical runs. This external result exercised the combined diagnostic
variant; it does not isolate a prefix-only effect or prove a fresh production
CUDA run after the source correction. Local tests compare native production and
the diagnostic variant with synthetic weights, including fresh-process equality;
they do not execute pretrained models. Existing run/source bindings and completed
products remain immutable when updating code: use the original checkout for an
old-run diagnostic and a fresh run for the corrected production workflow.
