"""Fresh metadata keeps scientific decisions while binding variable evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_pipeline_core import _candidate_record, _span, _triple, _verdict_record
from utils.common.artifact_io import DataContractError, atomic_write_jsonl
from utils.common.paths import discover_source_root
from utils.common.records import Candidate
from utils.evaluation.scoring import _condition_result, _load_verdicts
from utils.test.fixtures import load_pipeline_config
from utils.verifier.requests import _sha256_value, verifier_identity, verifier_template_id

ROOT = discover_source_root(Path(__file__))


class FreshVerdictMetadataTests(unittest.TestCase):
    def fixture(self):
        config = load_pipeline_config(ROOT, 'resources/configs/pipeline.json')
        raw = _candidate_record(42, 'fixture', 'source',
            _triple(_span(0, 0, 'Object', 'wall'), 'part-of',
                    _span(2, 2, 'Object', 'building')), .8)
        candidate = Candidate.from_mapping(raw, 'fixture')
        legacy = _verdict_record(raw, 'VER-SIMPLE', 'KEEP',
                                 identity=verifier_identity(config, 'simple'))
        fresh = {key: value for key, value in legacy.items()
                 if key not in ('prompt_sha256', 'decoding_sha256')}
        fresh.update(template_id=verifier_template_id(config, 'simple'),
                     request_sha256=_sha256_value({'model': config.value['verifier']['model'], 'fixture': 'actual dynamic input'}))
        return config, candidate, legacy, fresh

    def requests(self, path, config, candidate, fresh):
        atomic_write_jsonl(path.parent/'requests.jsonl', [{
            'condition_id': 'VER-SIMPLE', 'candidate_id': candidate.candidate_id,
            'training_seed': candidate.training_seed, 'template_id': fresh['template_id'],
            'model_manifest_sha256': fresh['model_manifest_sha256'],
            'request_sha256': fresh['request_sha256'],
            'payload': {'model': config.value['verifier']['model'], 'fixture': 'actual dynamic input'}}])

    def test_commit_bound_template_preserves_decision_without_static_hashes(self):
        config, candidate, legacy, fresh = self.fixture()
        (ROOT / 'output').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'output') as temporary:
            path = Path(temporary) / 'verdicts.jsonl'
            mapping = {candidate.candidate_id: candidate}
            self.requests(path, config, candidate, fresh)
            atomic_write_jsonl(path, [legacy])
            old = _load_verdicts(path, 'VER-SIMPLE', mapping, config=config, mode='simple')
            atomic_write_jsonl(path, [fresh])
            with mock.patch('utils.evaluation.scoring.verifier_identity',
                            side_effect=AssertionError('fresh path must not hash static resources')):
                new = _load_verdicts(path, 'VER-SIMPLE', mapping, config=config, mode='simple')
            self.assertEqual(_condition_result('VER-SIMPLE', candidate, .5, old, {}),
                             _condition_result('VER-SIMPLE', candidate, .5, new, {}))

    def test_changed_template_model_or_request_binding_cannot_be_adopted(self):
        config, candidate, _legacy, fresh = self.fixture()
        (ROOT / 'output').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'output') as temporary:
            path = Path(temporary) / 'verdicts.jsonl'
            self.requests(path, config, candidate, fresh)
            for field, value in (('template_id', 'foreign:corrective'),
                                 ('model_manifest_sha256', 'b' * 64),
                                 ('request_sha256', None),
                                 ('request_sha256', 'd' * 64),
                                 ('prompt_sha256', 'c' * 64)):
                with self.subTest(field=field):
                    atomic_write_jsonl(path, [{**fresh, field: value}])
                    with self.assertRaises(DataContractError):
                        _load_verdicts(path, 'VER-SIMPLE', {candidate.candidate_id: candidate},
                                       config=config, mode='simple')
