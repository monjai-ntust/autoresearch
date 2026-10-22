"""Synthetic-only validation of the transferable inference experiment."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('inference_check', ROOT / 'check_encoder_inference.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Module()
        self.embeddings.word_embeddings = torch.nn.Embedding(6, 2)
        self.embeddings.register_buffer('position_ids', torch.arange(8).unsqueeze(0), persistent=False)
        self.config = SimpleNamespace(hidden_size=2, _attn_implementation='eager')

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def forward(self, inputs_embeds, attention_mask):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class TinyEncoding(dict):
    def word_ids(self):
        return [None, 0, 1, None]


class InferenceCheckTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, self.rng)

    def original_model(self):
        from utils.encoder import network
        factory = SimpleNamespace(from_pretrained=lambda *a, **kw: TinyBackbone())
        with mock.patch.object(network, 'AutoModel', factory):
            model = network.BertKGExtractor('fixture', num_bio_tags=5, num_relations=3,
                num_entity_types=2, max_span_width=2)
        return model

    def test_meta_loading_materializes_without_rng_and_is_strict(self):
        model = self.original_model()
        state = copy.deepcopy(model.state_dict())
        from utils.encoder import network
        before = torch.get_rng_state().clone()
        with (torch.device('meta'), mock.patch.object(network, 'AutoModel',
                SimpleNamespace(from_pretrained=lambda *a, **kw: TinyBackbone()))):
            other = network.BertKGExtractor('fixture', num_bio_tags=5, num_relations=3,
                num_entity_types=2, max_span_width=2)
        check.strict_load(other, state, torch.device('cpu'))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(any(t.is_meta for t in list(other.parameters()) + list(other.buffers())))
        for key, value in state.items():
            self.assertTrue(torch.equal(value, other.state_dict()[key]))
        self.assertEqual(other.backbone.bert.embeddings.position_ids.tolist(), [list(range(8))])
        with self.assertRaises(RuntimeError):
            check.strict_load(other, {k: v for k, v in state.items() if k != 'ner_head.bias'}, torch.device('cpu'))

    def test_alias_mismatch_and_precision_change_are_rejected(self):
        model = self.original_model()
        state = copy.deepcopy(model.state_dict())
        state['adapters.text.word_embeddings.weight'] = state['backbone.bert.embeddings.word_embeddings.weight'].clone() + 1
        with self.assertRaisesRegex(ValueError, 'alias'):
            check.strict_load(model, state, torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'float32'):
            check.strict_load(model, {k: v.to(torch.float16) for k, v in model.state_dict().items()}, torch.device('cpu'))

    def test_existing_live_loop_prefix_override_and_runtime_writes_are_scoped(self):
        import transformers
        from utils.encoder import network, runtime
        from utils.encoder.inference import _live_inference_records
        from utils.common.constants import ENTITY_TYPES, RELATION_TYPES
        from utils.verifier.requests import PreparedSentence
        (ROOT / 'output').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'output') as temporary:
            base = Path(temporary)
            checkpoint = base / 'original/checkpoints/seed-42/checkpoint.pt'
            checkpoint.parent.mkdir(parents=True)
            with mock.patch.object(network, 'AutoModel', SimpleNamespace(from_pretrained=lambda *a, **kw: TinyBackbone())):
                fixture = network.BertKGExtractor('fixture', num_bio_tags=1+2*len(ENTITY_TYPES),
                    num_relations=1+len(RELATION_TYPES), num_entity_types=len(ENTITY_TYPES), max_span_width=2)
            # Explicit fixture parameters force entity/relation decoding without random prediction.
            fixture.re_context_span = True
            fixture.re_head = torch.nn.Sequential(torch.nn.Linear(6, 2), torch.nn.GELU(),
                torch.nn.Dropout(.1), torch.nn.Linear(2, 1+len(RELATION_TYPES)))
            for p in fixture.parameters():
                torch.nn.init.zeros_(p)
            fixture.span_ner_head.bias.data[1] = 2
            fixture.re_head[-1].bias.data[1] = 2
            torch.save({'encoder': fixture.state_dict(), 'step': 1, 'metrics': {}}, checkpoint)
            frozen = checkpoint.read_bytes()
            config = SimpleNamespace(value={'training': {'max_length': 128, 'max_span_width': 2,
                'base_model_revision': 'fixture', 'context_between_spans': True}})
            sentence = PreparedSentence('x', 'x', 'test', 'a b', 'a b', ('a', 'b'))
            expected_class = network.BertKGExtractor
            kwargs_seen = []
            paths = []

            def tokenizer(*args, **kwargs):
                kwargs_seen.append(kwargs)
                return lambda *a, **kw: TinyEncoding(input_ids=[1, 2, 3, 2], attention_mask=[1]*4)

            before = torch.get_rng_state().clone()
            with (mock.patch.object(transformers.AutoConfig, 'from_pretrained', return_value=SimpleNamespace(model_type='deberta')),
                  mock.patch.object(transformers.AutoTokenizer, 'from_pretrained', side_effect=tokenizer),
                  mock.patch.object(transformers.AutoModel, 'from_config', side_effect=lambda *a, **kw: TinyBackbone()),
                  mock.patch.object(runtime, 'configure_numerics'),
                  mock.patch.object(runtime, 'record_runtime', side_effect=lambda p: paths.append(p)),
                  mock.patch.object(runtime, 'record_model_runtime', side_effect=lambda p, m: paths.append(p))):
                with check.corrected_dependencies(config, base / 'snapshot', base / 'testing'), torch.inference_mode():
                    records = _live_inference_records(config, SimpleNamespace(training_seed=42), {'x': sentence},
                        checkpoint, 'fixture', 'cpu', base / 'cache')
            self.assertTrue(kwargs_seen[0]['add_prefix_space'])
            self.assertTrue(kwargs_seen[0]['local_files_only'])
            self.assertTrue(records[0]['predicted_relations'])
            self.assertTrue(all(p == base / 'testing/data-prepared' for p in paths))
            self.assertEqual(frozen, checkpoint.read_bytes())
            self.assertEqual(network.BertKGExtractor, expected_class)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            if os.environ.get('INFERENCE_FIXTURE_RESULT'):
                Path(os.environ['INFERENCE_FIXTURE_RESULT']).write_bytes(check.encode(records))

    def test_independent_processes_agree_on_real_live_loop_with_synthetic_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ('one.json', 'two.json')]
            for path in paths:
                result = subprocess.run([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                    'InferenceCheckTests.test_existing_live_loop_prefix_override_and_runtime_writes_are_scoped'],
                    env={**os.environ, 'INFERENCE_FIXTURE_RESULT': str(path), 'PYTHONHASHSEED': '0'},
                    capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_repeat_tamper_and_runtime_difference_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = (Path(temporary) / n for n in ('a', 'b'))
            for p in (a, b):
                check.save(p / 'quality.json', {'counts': {'tp': 1, 'fp': 0, 'fn': 0}})
                check.save(p / 'runtime.json', {'gpu': 'fixture'})
                check.save(p / 'receipt.json', {'files': {n: check.digest(p / n) for n in ('quality.json', 'runtime.json')}})
            self.assertEqual(check.compare_passes(a, b)['counts']['tp'], 1)
            (b / 'quality.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'modified'):
                check.compare_passes(a, b)
            (b / 'quality.json').write_bytes((a / 'quality.json').read_bytes())
            check.save(b / 'different.json', {'gpu': 'other'})
            receipt = check.load(b / 'receipt.json')
            receipt['files']['different.json'] = check.digest(b / 'different.json')
            (b / 'receipt.json').write_bytes(check.encode(receipt))
            with self.assertRaisesRegex(ValueError, 'passes differ'):
                check.compare_passes(a, b)

    def test_outputs_never_overwrite_and_run_ids_cannot_escape(self):
        from utils.common.paths import RunLayout, PathContractError
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'q.json'
            check.save(p, {'x': 1})
            with self.assertRaises(FileExistsError):
                check.save(p, {'x': 2})
            with self.assertRaises(PathContractError):
                RunLayout(Path(temporary), '../original')
            self.assertEqual(check.load(p), {'x': 1})

    def test_preflight_binds_all_checkpoints_and_rejects_parent_tampering(self):
        from utils.common.config import load_pipeline_config
        from utils.common.paths import RunLayout
        from utils.encoder.cache import write_cache_manifest
        from utils.common.artifact_io import atomic_write_json, atomic_write_jsonl
        from test_model import _sentence, EXAMPLE_ID
        original_config = load_pipeline_config(ROOT, 'resources/configs/pipeline.json')
        config = copy.deepcopy(original_config)
        config.value['split']['test_sentences'] = 1
        with tempfile.TemporaryDirectory(dir=ROOT / 'output') as temporary:
            parent = Path(temporary)
            layout = RunLayout(ROOT, parent.name)
            def write(relative, value):
                atomic_write_json(layout.resolve(relative), value)
            def h(relative):
                return check.digest(layout.resolve(relative))
            commit = check.git(ROOT, 'rev-parse', 'HEAD')
            write('manifests/00-checkout-manifest.json', {'run_id': parent.name, 'status': 'pass', 'source': {'commit': commit}})
            write('manifests/02-input-acquisition-manifest.json', {'archive': {'sha256': 'c'*64}})
            for split in ('train', 'development', 'test'):
                atomic_write_jsonl(layout.resolve('data-prepared/' + split + '.jsonl'), [_sentence(split)])
            write('data-prepared/split-manifest.json', {'test_ids': [EXAMPLE_ID]})
            atomic_write_jsonl(layout.resolve('data-prepared/test-gold.jsonl'), [{
                'protocol_id': 'B04-PATH-A-1.3', 'split_id': 'CODE-SPLIT-1:test',
                'example_id': EXAMPLE_ID, 'source_document_id': 'doc-a', 'gold_triples': [], 'input_hashes': {'fixture': 'a'*64}}])
            names = ('train.jsonl', 'development.jsonl', 'test.jsonl', 'test-gold.jsonl', 'split-manifest.json')
            prep = {'acquisition_manifest_sha256': h('manifests/02-input-acquisition-manifest.json'),
                'archive_sha256': 'c'*64, 'annotation_bundle_sha256': 'd'*64, 'dataset_tree_sha256': 'e'*64,
                'artifacts': [{'path': name, 'sha256': h('data-prepared/' + name)} for name in names]}
            write('manifests/03-data-preparation-manifest.json', prep)
            training = config.value['training']
            snapshot = 'inputs/huggingface/models--' + training['base_model'].replace('/', '--') + '/snapshots/' + training['base_model_revision']
            for name in ('config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
                path = layout.resolve(snapshot + '/' + name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture\n')
            cache = write_cache_manifest(layout, model=training['base_model'], revision=training['base_model_revision'])
            for seed in range(42, 50):
                checkpoint = layout.resolve(f'checkpoints/seed-{seed}/checkpoint.pt')
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b'synthetic checkpoint')
                write(f'checkpoints/seed-{seed}/checkpoint-manifest.json', {
                    'schema_version': 'encoder-checkpoint-1', 'protocol_id': 'B04-PATH-A-1.3',
                    'split_id': 'CODE-SPLIT-1', 'training_seed': seed, 'checkpoint_sha256': check.digest(checkpoint),
                    'checkpoint_step': 1500, 'split_manifest_sha256': h('data-prepared/split-manifest.json'),
                    'archive_sha256': 'c'*64, 'annotation_bundle_sha256': 'd'*64, 'prepared_dataset_tree_sha256': 'e'*64,
                    'train_jsonl_sha256': h('data-prepared/train.jsonl'), 'development_jsonl_sha256': h('data-prepared/development.jsonl'),
                    'acquisition_manifest_sha256': h('manifests/02-input-acquisition-manifest.json'),
                    'config': config.source_path, 'model_cache_manifest': 'manifests/huggingface-model-cache.json',
                    'model_cache_manifest_sha256': h('manifests/huggingface-model-cache.json'),
                    'model_cache_tree_sha256': cache['tree_sha256'], 'source_commit': commit,
                    'selected_metric_value': .5, 'restart_state_sha256': 'f'*64})
                write(f'checkpoints/seed-{seed}/training-summary.json', {'status': 'completed', 'seed': seed,
                    'completed_steps': training['max_steps'], 'selected_step': 1500,
                    'selected_metrics': {'triple_f1': .5}})
            before = {p.relative_to(parent).as_posix(): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
            test_id = parent.name + '-testing'
            with mock.patch('utils.common.config.load_pipeline_config', return_value=config):
                _, _, testing, plan = check.prepare(ROOT, parent.name, test_id, 'cpu')
                self.assertFalse(testing.exists())
                self.assertEqual(before, {p.relative_to(parent).as_posix(): p.read_bytes() for p in parent.rglob('*') if p.is_file()})
                self.assertEqual(plan['parent_run_id'], parent.name)
                check.require_inputs(ROOT, plan)
                with self.assertRaisesRegex(ValueError, 'differ'):
                    check.prepare(ROOT, parent.name, parent.name, 'cpu')
                checkpoint.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    check.require_inputs(ROOT, plan)
                with self.assertRaisesRegex(ValueError, 'hash differs'):
                    check.prepare(ROOT, parent.name, test_id, 'cpu')


if __name__ == '__main__':
    unittest.main()
