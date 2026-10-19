# Fresh-run reproducibility

A new clone needs no `output/`, parent repository or historical Git object.
The public execution commands create their own output roots and acquire inputs
inside each run. Install with `uv sync --frozen`, run the offline validation
commands in [README](../README.md), then choose fresh IDs:

```bash
uv run --frozen --no-sync python -I -B pipeline.py run --run-id research-001
uv run --frozen --no-sync python -I -B rag.py run --run-id research-001
uv run --frozen --no-sync python -I -B encoder_comparison.py run --matrix-id comparison-001
```

RAG follows the completed canonical run under its unchanged ID. The comparison
is independent. Repeat an unchanged command to continue durable progress;
preserve every run directory after failure. No archived paper outputs are
loaded or needed. Report values are computed from the new run's records.
The recorded clean source commit identifies tracked code, configurations,
prompts and schemas. Fresh-run metadata does not repeat their file hashes or
static definitions. Hashes authenticate external inputs, weights, generated
artifacts and saved progress. Actual numerical/server settings remain shared
run records because they are not determined by the source commit.

## Encoder numerical policy

[runtime.py](../utils/encoder/runtime.py) configures every training, smoke and
live inference process before CUDA initialization or device observation:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is established before CUDA starts. A
  conflicting inherited value fails rather than being silently overridden.
- Torch deterministic algorithms are required with `warn_only=False`;
  cuDNN deterministic mode is enabled and benchmarking disabled. Recognized
  nondeterministic operations without a supported implementation raise an error. Preserve the trace
  for a source repair instead of bypassing the policy.
- FP32 matmul precision is explicitly `highest`; cuDNN TF32 remains enabled,
  preserving the locked Torch defaults. The fix does not introduce AMP or
  change initialized heads, losses, optimizer, batch order or selection rules.
  Inherited TF32 override variables are rejected.
- Mid-epoch restart restores saved RNG without an additional DataLoader base
  seed draw. Normal epoch-boundary draws and the sampler's private state remain
  intact. This prevents interrupted and uninterrupted runs diverging solely
  because an iterator was recreated.

The source commit identifies the enforced deterministic/precision policy;
those fixed settings are not repeated in runtime metadata.
`manifests/encoder-runtime.json` records actual libraries, driver availability,
device and uncontrolled process settings once per run.
`manifests/encoder-model-runtime.json` records the actual
backbone class, selected Transformers attention implementation when exposed,
and parameter dtypes. These identities are checked on continuation and bound
to restart state. They describe the executing environment, without admitting
or rejecting it because of historical hardware or model digests.

## RAG order and requests

Both breadth-first frontiers are lists. Matching start nodes follow first
appearance in the authenticated graph; adjacent edges and newly discovered
endpoints keep discovery order. Sets are used only for membership/deduplication.
This policy is applied before the existing 15-fact return limit and hybrid's
first-five limit. It uses no gold answers, correctness flags or target scores.
Question order, matching/fallback/hop rules, text ties, prompts and answer
scoring are unchanged.

`generation_options` in [table2.json](../resources/contracts/table2.json) defines
the one shared planning/dispatch policy: existing temperature/output limit and
an explicit seed following the experiment's seed convention. Exact payloads,
ordered selected context and producer identities are saved before dispatch.
Python `-I` ignores `PYTHONHASHSEED`; retrieval is now independent of hash seed.
Complete historical seedless journals replay their original bytes read-only;
an unfinished incompatible plan stops before a new call.

## External Ollama process

Use one dedicated Ollama server, keep its build/model/configuration unchanged,
and avoid concurrent model clients during the experiment. For a foreground
Linux server, use a separate terminal with serial request/model loading. Use
this option only when no existing service occupies the endpoint; otherwise
configure that service instead:

```bash
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

For an existing systemd service, apply these settings to that service's launch
environment and restart it before beginning a fresh run. Exporting variables
only in the Python client terminal does not configure an existing server.
Record the server's actual launch settings and preserve its logs externally
with the run handoff. Keep context length, Flash Attention, KV-cache precision,
GPU placement and driver/backend versions constant. Their values must describe
the actual chosen server setup; this artifact does not infer them from the
client or choose them to reproduce a table. See the
[official Ollama configuration documentation](https://docs.ollama.com/faq).

The first live verifier capture saves `runtime/ollama.json` once for the run:
actual `/api/version`, stable generation-relevant model template/defaults and
verified weight identity. RAG and later verifier dispatches reject changes to
that identity. The public API does not expose the server launch environment or
numerical library/kernel policy; no placeholder observations are logged. The version
string also does not prove an executable/build digest. A restart with the same
version/model but different unexposed settings requires external launch evidence.
Runtime checks occur at stage/continuation dispatch, rather than every generated token.

## Validation boundary

Offline tests verify graph/cap ordering across fresh hash-seeded and isolated
processes, exact request serialization, strict startup, runtime drift refusal,
and interrupted tiny CPU training equality. They acquire or execute no
pretrained model. A committed depth-one checkout can run them without output
or old Git objects.

Strict execution can select different kernels from historical runs, and ordered
retrieval/explicit generation seed can change historical answers. These are
prospective fixes, not a guarantee of equal historical statistics or improved
accuracy. Exact fresh CUDA/Qwen results require validation on the external
machine with its actual locked environment. Preserve runtime manifests, server
launch/log evidence, requests/responses and all scientific products; compare
those records before interpreting any numerical difference. Timings naturally
vary and are reported as newly measured values.

## Possible changes to statistical trends

| Change | Possible effect on the new results |
|---|---|
| Ordered graph traversal | Fact/token order and which facts survive the existing caps can change. One-hop, two-hop and hybrid hit rates, and their differences between graph conditions, can increase or decrease. |
| Explicit RAG generation seed | The request changes for every mode. It can leave greedy replies unchanged, but it does not prove identical replies across server backends. |
| Strict CUDA execution | Different supported kernels/reduction orders can change optimization trajectories, selected checkpoints, thresholds and downstream statistics. Absolute scores, relative arm rankings and significance can change. |
| Exact mid-epoch continuation | Interrupted training follows the uninterrupted random stream. This changes runs that previously incurred the extra draw; uninterrupted batch/RNG behavior is retained. |
| Metadata reduction | Removing bookkeeping does not alter model inputs, RNG or scoring. Recorded execution time can change with the reduced overhead. |

No accuracy direction, relative ranking or significance outcome is guaranteed.
Apply the same fresh numerical policy to every comparison arm and interpret the
new records as a new execution of that policy. No seed, order or numerical
setting is selected against held-out outcomes or the presentation's statistics.
