# Fresh-run workflow contract

Use the three commands in [README](../README.md) from the standalone source
checkout. `pipeline.py` owns encoder/verifier evaluation, `encoder_comparison.py`
owns the isolated four-arm matrix, and `rag.py` owns downstream Graph RAG.
Private worker actions enforce the same prerequisites as their coordinator.

## One physical run

A run ID begins with a letter or digit and contains at most 64 letters, digits,
`.`, `_` or `-`. Its physical root is `output/<run-id>/`. Source must be clean
at the actual recorded full commit. Continuing requires that source and
configuration identity. A fresh checkout doctor retains configuration bytes
under `inputs/config/`; it requires no historical ancestry or reference digest.

Checkout preflight checks the actual Git commit, clean source and effective
root-output ignore rule. Actual Python/uv versions are recorded, without a
historical version gate. The entry points disable bytecode writes themselves.
The default root-level `nohup.out` console log is ignored by Git.

Inputs are acquired into that run under immutable byte contracts. Preparation
authenticates the archive, six annotation files, deterministic repair,
CODE-SPLIT-1 membership and prepared tree. Encoder cache binds weights,
tokenizer and revision. Qwen binds actual registry identity and copied weight
bytes/SHA-256, model details and quantization. Identities must agree at every
same-run consumer. Numerical software/backend settings are recorded once and
referenced by encoder products.
Restart states and completed summaries bind these identities before reuse.
Private workers require an active recorded coordinator dispatch. A canonical
command rejects a comparison arm or matrix namespace.

The canonical command trains seeds 42–49 using development-only checkpoint
selection, generates development candidates, selects the confidence threshold,
and runs independent verifier pilot captures. After that gate it generates
test candidates, performs simple/corrective verification, and scores raw,
confidence-filtered, simple and corrective conditions.

RAG consumes the completed canonical run. It authenticates every parent
prerequisite and constructs missing canonical graphs from those inputs without
training or running the upstream verifier. It derives evaluator projections and
all eligible CODE questions. Confidence, corrective and gold oracle conditions
use the frozen five modes. Fresh `part-of` rows support later exact external
comparison. Historical output is never a live method input. Read-only old-parent
reader support preserves legacy envelopes without permitting old training,
admission or repair actions.

## Continue durable progress

Repeat the exact public command with the same ID. An operating-system lock
prevents overlapping execution and releases on process exit; the lock file
stays as evidence and is not deleted to recover a run.

If a fresh checkout was blocked before any acquisition or scientific work,
fix the reported source changes and repeat the same command with the same ID.
The failed manifest and unchanged configuration bytes are retained alongside a
new passing manifest under `manifests/checkout-recovery/`, with a hash-bound
receipt written before the active manifest is replaced. The new manifest
records the actual current source commit. An interrupted promotion can finish
from that evidence; missing or changed proof fails closed. This path accepts
only fresh lifecycle-1 checkout-only failures, including retired administrative
checks. It cannot replace a passing checkout, rebind existing downstream work
or migrate an old partial workflow.

Completed producer receipts and input/output hashes are validated before reuse.
Training loads full restart state at the last saved step, preserving optimizer,
scheduler, RNG and selection state. Candidate and score finalization reconstruct
deterministic products from saved evidence; existing products must equal the
expected bytes. Changed source, inputs, models, saved state or outputs fail.
An existing prediction ledger must have its original producer binding; a copied
or unsealed ledger cannot become a new same-run input.

Verifier journals retain original attempts and elapsed/token values before
postprocessing. Warmup, each independent pilot capture and test have separate
journals. The frozen three-attempt budget survives reinvocation; a terminal
failure is a completed scientific verdict. Reused original responses retain
`cache_hit=false` rather than becoming synthetic cache hits.

RAG saves the exact ordered prompt plan and identifies each occurrence, even
when prompts are equal. Each answer is followed immediately by hash-bound model
identity proof. Successful durable answers survive later failure. Empty or
`ERROR:` answers remain unfinished: a subsequent command retries only those
occurrences. Missing contemporaneous proof stops continuation; a later query
cannot prove which model produced an earlier answer.
All saved occurrences are checked before issuing new calls. Deleting or changing
completed evidence blocks continuation. Completed condition outputs retain their
original bytes and recorded timing; resumed intervals extend only unfinished work.

A crash before a response or training step is saved can require that unsaved
work again. The guarantee starts at validated durable progress; it does not
claim exactly-once external execution across arbitrary failure.

## Paper reports and limits

Canonical reports are in `metrics/paper-report/`; comparison reports are in
each arm's `reports/comparison/` and the coordinator's `reports/matrix/`; RAG
reports are in `table2-code-all/reports/`. Each contains deterministic
`paper-statistics.json`, `.tsv`, `.md` and a hash-bound manifest. Existing
publication tables, outcomes and metrics preserve their scientific bytes.

Reports expose per-seed strict quality, tied best seeds as descriptive
observations, existing Wilcoxon/paired-t raw and Holm-adjusted results,
verdict diagnostics, corpus/eligibility counts, unique emitted triples,
actual runtime/model identities and recorded cost. Request cost counts original
uncached attempts including retries/failures, separating warmup/pilot/test.
Tokens and timings carry measured coverage. Training exposes final trainer-summary
time and, separately, original recorded process durations including failures.
These views overlap and must not be summed. Matrix smoke and full execution
remain separate; missing finish records carry unavailable duration. Conservative
execution-budget charges are bounds, never reported as measured costs.

Graph reports distinguish canonical entities/relations/triples from evaluator
nodes/edges. RAG tables carry correct/total/hit-rate denominators for full and
`part-of` panels. Gold is a nondeployable oracle; answer matching does not
establish faithful use of retrieved evidence. Condition wall time is available;
per-call RAG latency/tokens and isolated graph construction time are not.
Disk weight size is not resident or peak accelerator memory. Energy, monetary
cost, unexecuted datasets and missing comparison inferential analyses remain
unavailable. No historical estimate is substituted.

## Existing partial runs

Fresh commands do not replace pre-refactor G-05 recovery/finalization and
review. Preserve original IDs, producer manifests and source at
`fb18028cd21239c0bf913e370149a643081958fe`. Full training/final-test access for
those partial runs still requires earlier reviewed external admission. Never
use a launcher that deletes prior run directories. Completed scientific runs
are not rerun by this refactor; extension requires their unchanged namespace,
authenticated prerequisites and separately authorized work.
