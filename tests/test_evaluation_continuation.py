"""Score reuse requires all fixed same-run inputs and publication products."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from stages import evaluation
from utils.common.artifact_io import DataContractError, atomic_write_json, sha256_file
from utils.common.paths import RunLayout
from utils.test.fixtures import load_pipeline_config

ROOT = Path(__file__).resolve().parents[1]


class EvaluationContinuationTests(unittest.TestCase):
    def test_missing_or_rebound_score_lineage_fails_closed(self):
        config = load_pipeline_config(ROOT, 'resources/configs/pipeline.json')
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout = RunLayout(Path(temporary), 'fixture-score-contract')
            layout.create()
            paths = {
                'inputs': ['data-prepared/test-gold.jsonl', 'predictions/test/candidates.jsonl',
                           'verifier/simple/verdicts.jsonl', 'verifier/corrective/verdicts.jsonl',
                           'predictions/dev/threshold-selection.json'],
                'outputs': ['outcomes/candidate-outcomes.jsonl', 'outcomes/sentence-outcomes.jsonl',
                            'metrics/metrics.json', 'metrics/publication-table.tsv',
                            'metrics/publication-summary.md'],
            }
            manifest = {'schema_version': 'phase-b-score-manifest-1.0', 'run_id': layout.run_id,
                        'nonpublication_smoke': False,
                        **{key: config.value[key] for key in ('protocol_id', 'workflow_id', 'matcher_id')}}
            for mapping, names in paths.items():
                for name in names:
                    atomic_write_json(layout.resolve(name), {'fixture': name})
                manifest[mapping] = {name: sha256_file(layout.resolve(name)) for name in names}
            receipt = layout.resolve('manifests/score-manifest.json')
            atomic_write_json(receipt, manifest)
            self.assertEqual(evaluation._validate_score(layout, config), manifest)
            for mapping in paths:
                with self.subTest(mapping=mapping):
                    invalid = {**manifest, mapping: dict(manifest[mapping])}
                    del invalid[mapping][paths[mapping][0]]
                    atomic_write_json(receipt, invalid)
                    with self.assertRaisesRegex(DataContractError, 'canonical product contract'):
                        evaluation._validate_score(layout, config)
            invalid = {**manifest, 'inputs': dict(manifest['inputs'])}
            atomic_write_json(layout.resolve('predictions/foreign.jsonl'), {'fixture': 'another population'})
            invalid['inputs'].pop('predictions/test/candidates.jsonl')
            invalid['inputs']['predictions/foreign.jsonl'] = sha256_file(layout.resolve('predictions/foreign.jsonl'))
            atomic_write_json(receipt, invalid)
            with self.assertRaisesRegex(DataContractError, 'canonical product contract'):
                evaluation._validate_score(layout, config)
            atomic_write_json(receipt, manifest)
            atomic_write_json(layout.resolve('metrics/metrics.json'), {'fixture': 'changed'})
            with self.assertRaisesRegex(DataContractError, 'score artifact changed'):
                evaluation._validate_score(layout, config)

    def test_foreign_score_paths_fail_before_scoring_or_report(self):
        config = load_pipeline_config(ROOT, 'resources/configs/pipeline.json')
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as temporary:
            layout = RunLayout(Path(temporary), 'fixture-score-paths')
            layout.create()
            with mock.patch.object(evaluation, 'score_run') as score:
                with self.assertRaisesRegex(DataContractError, 'canonical paths'):
                    evaluation.run(layout, config, test_candidates=layout.resolve('predictions/foreign.jsonl'),
                                   threshold_path=layout.resolve('predictions/dev/threshold-selection.json'))
                score.assert_not_called()
