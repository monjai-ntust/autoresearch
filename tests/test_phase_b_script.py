"""Static contracts for the external-machine Phase B recovery launcher.

The repository's supported launcher is Bash, while the focused local validation
environment may be Windows-only.  These tests therefore protect the important
orchestration and source-layout invariants without pretending to execute GPU,
Ollama, Git-pull, or Bash process-substitution behavior locally.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from paths import discover_source_root


SOURCE_ROOT = discover_source_root(Path(__file__))
SCRIPT = (SOURCE_ROOT / "phase_b.sh").read_text(encoding="utf-8")

MOVED_PROVENANCE_MODULES = (
    "bench_gpu.py",
    "build_kg.py",
    "dapt_zh.py",
    "diagnose_evidence_paths.py",
    "eval_graph_rag.py",
    "generate_accord_llm_aug.py",
    "generate_cycle_data.py",
    "generate_entigraph.py",
    "generate_entity_masks.py",
    "generate_paraphrase_dataset.py",
    "generate_synth_dataset.py",
    "inference_kg.py",
    "rule_engine.py",
    "train_gan.py",
    "train_gumbel.py",
    "train_multi.py",
    "train_pretrain_cooperative.py",
    "train_stage2b.py",
    "train_stage2c.py",
    "train_stage2d.py",
    "train_stage2e.py",
    "verify_triples_llm.py",
    "zh_translate_project.py",
)


class PhaseBScriptContractTests(unittest.TestCase):
    def test_full_run_derives_machine_specific_values(self):
        self.assertIn('RUN_ID=""', SCRIPT)
        self.assertIn("git pull --ff-only", SCRIPT)
        self.assertIn('value["training_seeds"]', SCRIPT)
        self.assertIn('value["verifier"]["model"]', SCRIPT)
        self.assertIn('uv run --frozen --no-sync python -B - "$CONFIG" <<\'PY\'', SCRIPT)
        self.assertIn('PHASE_B_CONFIG=', SCRIPT)
        self.assertIn('case "$config_line" in', SCRIPT)
        self.assertIn('PHASE_B_CONFIG=*) CONFIG_RECORD=', SCRIPT)
        self.assertNotIn('mapfile -t CONFIG_VALUES', SCRIPT)
        self.assertIn("IFS=' ' read -r -a TRAINING_SEEDS", SCRIPT)
        self.assertIn("uv sync --frozen", SCRIPT)
        self.assertNotIn("uv run --frozen python", SCRIPT)
        self.assertIn("uv run --frozen --no-sync python", SCRIPT)
        self.assertIn('"recovery_source_commits"', SCRIPT)
        self.assertIn('existing full-run recovery_source_commits is invalid', SCRIPT)
        self.assertIn('ALLOW_FULL_RUN_RECOVERY_DOCTOR=true', SCRIPT)
        self.assertIn(
            'reusing the existing passing checkout manifest for full-run recovery', SCRIPT
        )
        self.assertIn('ollama show --modelfile "$OLLAMA_MODEL"', SCRIPT)
        self.assertNotIn('BRANCH="refactor"', SCRIPT)
        self.assertNotIn('SEED=42', SCRIPT)
        self.assertNotIn("--python 3.10.20", SCRIPT)
        self.assertNotIn(
            "3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312",
            SCRIPT,
        )

    def test_full_run_orders_publication_stages_and_keeps_b07_audit_gate(self):
        full = SCRIPT[SCRIPT.index("ensure_full() {") : SCRIPT.index("ensure_smoke() {")]
        ordered_tokens = (
            "ensure_train_live",
            "ensure_assemble_candidates development",
            "ensure_threshold",
            "ensure_prepare_pilot",
            "ensure_pilot_captures",
            "ensure_pilot_audit",
            'pilot_status \'"pass"\'',
            "ensure_assemble_candidates test",
            "ensure_verifier_live",
            "ensure_score",
            "ensure_graph_rag",
        )
        positions = [full.index(token) for token in ordered_tokens]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            'pilot_status \'"pass"\'',
            full,
        )
        self.assertIn("material_protocol_review_required false", full)
        self.assertIn("final-test execution remains blocked", full)
        self.assertLess(full.index("ensure_score"), full.index("ensure_graph_rag"))

    def test_graph_rag_recovery_stage_is_post_score_only(self):
        graph_stage = SCRIPT[
            SCRIPT.index("ensure_graph_rag() {") : SCRIPT.index(
                "maybe_generate_live() {"
            )
        ]
        self.assertIn("graph_rag_parent_score_complete", graph_stage)
        self.assertNotIn("publication-table.tsv", graph_stage)
        self.assertNotIn("publication-summary.md", graph_stage)
        self.assertIn("graph_rag.py phase-b-diagnostic", graph_stage)
        self.assertNotIn("ensure_bootstrap", graph_stage)
        self.assertNotIn("ensure_train_live", graph_stage)
        self.assertNotIn("ensure_verifier_live", graph_stage)
        self.assertNotIn("ensure_score", graph_stage)
        self.assertIn("graph-rag) ensure_graph_rag", SCRIPT)
        self.assertIn(
            "--config configs/phase_b_graph_rag_diagnostic.json", graph_stage
        )
        self.assertIn(
            'if [[ "$ACTIVE_STAGE_ID" == "graph-rag-diagnostic" ]]; then',
            SCRIPT,
        )
        self.assertLess(
            SCRIPT.index('ACTIVE_STAGE_ID="graph-rag-diagnostic"'),
            SCRIPT.index('note "updating source checkout from its configured upstream"'),
        )

    def test_b07_invocation_is_implicit_without_changing_manifest_value(self):
        self.assertNotIn("--approve-b07", SCRIPT)
        self.assertNotIn("APPROVE_B07", SCRIPT)
        self.assertIn(
            '"conditional_b07_approval": sys.argv[7] == "true"', SCRIPT
        )
        self.assertIn(
            'true "${TRAINING_SEEDS[*]}" "$PROTOCOL_ID" "$WORKFLOW_ID"',
            SCRIPT,
        )

    def test_recovery_is_run_scoped_and_records_exact_resume(self):
        self.assertIn('[[ "$run_abs" == "$SOURCE_ROOT"/output/* ]]', SCRIPT)
        self.assertIn('[[ "$target_abs" == "$run_abs"/* ]]', SCRIPT)
        self.assertIn('[[ ! -L "$target" ]]', SCRIPT)
        self.assertIn("rm -rf -- \"$target\"", SCRIPT)
        self.assertIn("debug-recovery.jsonl", SCRIPT)
        self.assertIn("print_resume_command", SCRIPT)
        self.assertIn("resume-from-response-cache", SCRIPT)
        self.assertIn("restart-state.pt", SCRIPT)
        self.assertIn('metrics/publication-table.tsv', SCRIPT)
        self.assertIn('metrics/publication-summary.md', SCRIPT)

    def test_secondary_entry_points_are_namespaced_as_provenance(self):
        for name in MOVED_PROVENANCE_MODULES:
            with self.subTest(name=name):
                self.assertFalse((SOURCE_ROOT / name).exists())
                self.assertTrue((SOURCE_ROOT / "provenance" / name).is_file())
        self.assertTrue((SOURCE_ROOT / "train_span.py").is_file())
        self.assertTrue((SOURCE_ROOT / "phase_b.py").is_file())


if __name__ == "__main__":
    unittest.main()
