"""Typed internal worker options; public CLIs supply only run identities."""
from dataclasses import dataclass


@dataclass
class ComparisonOptions:
    mode: str | None = None
    prepared_dir: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    expected_hidden_size: int | None = None
    model_cache_dir: str | None = None
    model_local_files_only: bool | None = False
    batch_size: int | None = None
    max_length: int | None = None
    max_span_width: int | None = None
    context_between_spans: bool | None = False
    seed: int | None = None
    lr: float | None = None
    max_steps: int | None = None
    warmup_steps: int | None = None
    re_weight: float | None = None
    re_no_rel_weight: float | None = None
    neg_sample_ratio: float | None = None
    focal_gamma: float | None = None
    bio_loss_weight: float | None = None
    label_smoothing: float | None = None
    eval_every: int | None = None
    comparison_boost_schedule: str | None = None
    re_comparison_boost: float | None = None
    re_boost_adaptive_steps: int | None = None
    re_boost_adaptive_threshold: float | None = None
    re_boost_adaptive_threshold2: float | None = None
    re_boost_mid: float | None = None
    re_boost_end: float | None = None
    save_best_to: str | None = None
    save_last_to: str | None = None
    progress_log: str | None = None
    run_summary_out: str | None = None
    resume_from: str | None = None
    checkpoint: str | None = None
    sentences: str | None = None
    prediction_ledger_out: str | None = None
