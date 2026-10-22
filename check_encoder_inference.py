"""External-only prefix-space inference check; does not replace production.

Copy this file beside pipeline.py in the existing clean source checkout:
  python -I -B check_encoder_inference.py --run-id research-1022 --test-id inference-test-1022
All eight checkpoints run twice in fresh processes; original run files are read-only.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
POLICY = {"add_prefix_space": True, "dtype": "torch.float32", "attention": "eager",
          "deterministic_algorithms": True, "cpu_threads": 1, "passes": 2,
          "entity_threshold": None, "triple_threshold": None}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('utf-8')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(encode(value))


def load(path):
    return json.loads(path.read_bytes())


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode().strip()


def setup(root):
    root = root.resolve(strict=True)
    if not (root / 'pipeline.py').is_file() or not (root / 'utils/encoder/inference.py').is_file():
        raise ValueError('--source-root must contain pipeline.py and utils/encoder/inference.py')
    sys.path.insert(0, str(root))
    return root


def prepare(root, run_id, test_id, device):
    from utils.common.paths import RunLayout, validate_run_id
    from utils.common.config import load_pipeline_config
    from utils.encoder.cache import verify_cache_manifest
    from utils.encoder.model import _load_checkpoint_manifest
    from utils.verifier.requests import _load_sentences
    from utils.evaluation.scoring import _load_gold
    from utils.encoder import inference, network, candidates, data, runtime
    from utils.evaluation import metrics, reporting

    validate_run_id(test_id)
    layout = RunLayout(root, run_id)
    layout.require_existing()
    testing = RunLayout(root, test_id).run_root
    if testing == layout.run_root:
        raise ValueError('test-id must differ from the original run-id')
    if git(root, 'diff', 'HEAD', '--name-only'):
        raise ValueError('tracked source is dirty; keep the original clean checkout')
    config = load_pipeline_config(root, 'resources/configs/pipeline.json')
    checkout = load(layout.resolve('manifests/00-checkout-manifest.json', must_exist=True))
    if checkout.get('run_id') != run_id or checkout.get('status') != 'pass':
        raise ValueError('parent checkout manifest run_id differs')
    cache = verify_cache_manifest(layout, model=config.value['training']['base_model'],
                                  revision=config.value['training']['base_model_revision'])
    # Read preparation's original hash bindings, never manufacture a new parent identity.
    preparation = load(layout.resolve('manifests/03-data-preparation-manifest.json', must_exist=True))
    if preparation.get('acquisition_manifest_sha256') != digest(layout.resolve('manifests/02-input-acquisition-manifest.json')):
        raise ValueError('parent acquisition/preparation lineage differs')
    prepared = layout.resolve('data-prepared', must_exist=True)
    required = ('train.jsonl', 'development.jsonl', 'test.jsonl', 'test-gold.jsonl', 'split-manifest.json')
    artifacts = {row['path']: row['sha256'] for row in preparation['artifacts']}
    inputs = {}

    def pin(relative, expected=None):
        path = layout.resolve(relative, must_exist=True)
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'parent input hash differs: {relative}')
        inputs[relative] = actual
        return path

    for name in required:
        if name not in artifacts:
            raise ValueError(f'parent preparation lacks {name}')
        pin('data-prepared/' + name, artifacts[name])
    for relative in ('manifests/00-checkout-manifest.json', 'manifests/03-data-preparation-manifest.json',
                     'manifests/02-input-acquisition-manifest.json', 'manifests/huggingface-model-cache.json'):
        pin(relative)
    snapshot_relative = ('inputs/huggingface/models--' + config.value['training']['base_model'].replace('/', '--')
                         + '/snapshots/' + config.value['training']['base_model_revision'])
    for name in ('config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
        pin(snapshot_relative + '/' + name)
    sentences = _load_sentences(prepared / 'test.jsonl')
    gold = _load_gold(prepared / 'test-gold.jsonl', 'CODE-SPLIT-1:test')
    split = load(prepared / 'split-manifest.json')
    if set(sentences) != set(gold) or set(sentences) != set(split['test_ids']):
        raise ValueError('test sentence/gold/split coverage differs')
    if len(sentences) != config.value['split']['test_sentences']:
        raise ValueError('fixed test sentence count differs')
    seeds = config.value['training_seeds']
    for seed in seeds:
        relative = f'checkpoints/seed-{seed}/checkpoint-manifest.json'
        manifest = load(pin(relative))
        identity = _load_checkpoint_manifest(layout.resolve(relative), config)
        bindings = {'split_manifest_sha256': inputs['data-prepared/split-manifest.json'],
            'train_jsonl_sha256': inputs['data-prepared/train.jsonl'],
            'development_jsonl_sha256': inputs['data-prepared/development.jsonl'],
            'acquisition_manifest_sha256': inputs['manifests/02-input-acquisition-manifest.json'],
            'model_cache_manifest_sha256': inputs['manifests/huggingface-model-cache.json'],
            'source_commit': checkout['source']['commit'], 'model_cache_tree_sha256': cache['tree_sha256'],
            'archive_sha256': preparation['archive_sha256'],
            'annotation_bundle_sha256': preparation['annotation_bundle_sha256'],
            'prepared_dataset_tree_sha256': preparation['dataset_tree_sha256']}
        if identity.training_seed != seed or any(manifest[k] != v for k, v in bindings.items()):
            raise ValueError(f'seed {seed} checkpoint is not bound to this parent run')
        summary = load(pin(f'checkpoints/seed-{seed}/training-summary.json'))
        if (summary.get('status') != 'completed' or summary.get('seed') != seed
                or summary.get('completed_steps') != config.value['training']['max_steps']
                or summary.get('selected_step') != identity.checkpoint_step
                or summary.get('selected_metrics', {}).get('triple_f1') != manifest['selected_metric_value']):
            raise ValueError(f'seed {seed} lacks completed training bound to this selected checkpoint')
        pin(f'checkpoints/seed-{seed}/checkpoint.pt', identity.checkpoint_sha256)
    helpers = {Path(inspect.getfile(module)).relative_to(root).as_posix(): digest(inspect.getfile(module))
               for module in (inference, network, candidates, data, runtime, metrics, reporting)}
    identity = {'parent_run_id': run_id, 'test_id': test_id, 'device': device, 'seeds': seeds,
        'test_sentences': len(sentences), 'gold_triples': sum(len(row.triples) for row in gold.values()),
        'parent_source_commit': checkout['source']['commit'], 'actual_source_commit': git(root, 'rev-parse', 'HEAD'),
        'script_sha256': digest(__file__), 'helpers': helpers, 'inputs': inputs, 'policy': POLICY,
        'purpose': 'user-authorized diagnostic; original parent remains immutable; no training or production replacement'}
    return layout, config, testing, identity


def require_inputs(root, plan, seed=None):
    from utils.common.paths import RunLayout
    layout = RunLayout(root, plan['parent_run_id'])
    for relative, expected in plan['inputs'].items():
        if seed is not None and relative.startswith('checkpoints/') and not relative.startswith(f'checkpoints/seed-{seed}/'):
            continue
        if digest(layout.resolve(relative, must_exist=True)) != expected:
            raise ValueError('parent input changed: ' + relative)
    for relative, expected in plan['helpers'].items():
        if digest(root / relative) != expected:
            raise ValueError('source helper changed: ' + relative)
    if (digest(__file__) != plan['script_sha256'] or git(root, 'rev-parse', 'HEAD') != plan['actual_source_commit']
            or git(root, 'diff', 'HEAD', '--name-only')):
        raise ValueError('script/source identity changed')
    return layout


def strict_load(model, state, device):
    """Materialize all weights; regenerate only DeBERTa's nonpersistent position IDs."""
    import torch
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in state.values()):
        raise ValueError('diagnostic requires original float32 checkpoint tensors')
    # Check duplicated embedding registrations before strict assign (no random fallback).
    aliases = ('backbone.bert.embeddings.word_embeddings.weight', 'adapters.text.word_embeddings.weight')
    if all(key in state for key in aliases) and not torch.equal(state[aliases[0]], state[aliases[1]]):
        raise ValueError('checkpoint embedding alias values differ')
    result = torch.nn.Module.load_state_dict(model, state, strict=True, assign=True)
    for name, value in list(model.named_buffers()):
        if value.is_meta:
            if not name.endswith('.position_ids') or value.ndim != 2 or value.shape[0] != 1:
                raise ValueError('unmaterialized nonpersistent buffer: ' + name)
            owner, attribute = name.rsplit('.', 1)
            setattr(model.get_submodule(owner), attribute,
                    torch.arange(value.shape[1], dtype=value.dtype, device=device).unsqueeze(0))
    torch.nn.Module.to(model, device)
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError('checkpoint did not materialize the complete model')
    return result


@contextlib.contextmanager
def corrected_dependencies(config, snapshot, output):
    """Reuse the unchanged live loop with scoped tokenizer/loading/record destinations."""
    import torch
    import transformers
    from utils.encoder import network, runtime
    original_class = network.BertKGExtractor
    original_tokenizer = transformers.AutoTokenizer.from_pretrained
    original_runtime = runtime.record_runtime
    original_model_runtime = runtime.record_model_runtime
    model_config = transformers.AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    if model_config.model_type != 'deberta':
        raise ValueError('this prefix-space experiment supports DeBERTa v1 only')

    class CheckpointOnlyModel(original_class):
        def to(self, device):
            # Original live loop calls .to before load_state_dict. Defer meta materialization.
            self._test_device = device
            return self

        def load_state_dict(self, state, strict=True):
            if strict is not True:
                raise ValueError('strict checkpoint loading is required')
            return strict_load(self, state, self._test_device)

    def meta_backbone(*args, **kwargs):
        return transformers.AutoModel.from_config(model_config, attn_implementation='eager', dtype=torch.float32)

    def tokenizer(*args, **kwargs):
        return original_tokenizer(str(snapshot), local_files_only=True, add_prefix_space=True)

    with (patch.object(network, 'BertKGExtractor', CheckpointOnlyModel),
          patch.object(network, 'AutoModel', SimpleNamespace(from_pretrained=meta_backbone)),
          patch.object(transformers.AutoTokenizer, 'from_pretrained', side_effect=tokenizer),
          patch.object(runtime, 'record_runtime', side_effect=lambda _: original_runtime(output / 'data-prepared')),
          patch.object(runtime, 'record_model_runtime', side_effect=lambda _, m: original_model_runtime(output / 'data-prepared', m)),
          torch.device('meta')):
        yield


def run_worker(root, plan, seed, pass_number):
    # Set before importing torch or initializing CUDA. No seeds/sampling are needed.
    for name, value in {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'DISABLE_SAFETENSORS_CONVERSION': '1'}.items():
        os.environ[name] = value
    import random
    import numpy as np
    import torch
    from utils.encoder.runtime import configure_numerics
    configure_numerics()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(plan['device'])
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('requested CPU/CUDA device is unavailable; no device fallback')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.init()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    layout = require_inputs(root, plan, seed)
    from utils.common.config import load_pipeline_config
    from utils.encoder.model import _load_checkpoint_manifest
    from utils.encoder.inference import _live_inference_records
    from utils.encoder.candidates import _load_prediction_ledger
    from utils.verifier.requests import _load_sentences
    from utils.evaluation.scoring import _load_gold
    from utils.evaluation.metrics import triple_metrics
    from utils.common.records import Candidate
    config = load_pipeline_config(root, 'resources/configs/pipeline.json')
    manifest = layout.resolve(f'checkpoints/seed-{seed}/checkpoint-manifest.json', must_exist=True)
    checkpoint = _load_checkpoint_manifest(manifest, config)
    sentences = _load_sentences(layout.resolve('data-prepared/test.jsonl'))
    output = root / 'output' / plan['test_id'] / f'seed-{seed}' / f'pass-{pass_number}'
    output.mkdir(parents=True, exist_ok=False)
    snapshot = layout.resolve('inputs/huggingface') / ('models--' + config.value['training']['base_model'].replace('/', '--')) / 'snapshots' / config.value['training']['base_model_revision']
    with corrected_dependencies(config, snapshot, output), torch.inference_mode():
        records = _live_inference_records(config, checkpoint, sentences,
            layout.resolve(f'checkpoints/seed-{seed}/checkpoint.pt'), config.value['training']['base_model'],
            str(device), layout.resolve('inputs/huggingface'))
    numpy_after = np.random.get_state()
    if (not torch.equal(cpu_rng, torch.get_rng_state())
            or (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
            or python_rng != random.getstate() or numpy_rng[0] != numpy_after[0]
            or not np.array_equal(numpy_rng[1], numpy_after[1]) or numpy_rng[2:] != numpy_after[2:]):
        raise ValueError('RNG state changed during construction/inference; repeatability is not accepted')
    ledger = output / 'prediction-ledger.jsonl'
    with ledger.open('xb') as handle:
        for row in records:
            handle.write(encode(row))
    candidates, n, selected, duplicates = _load_prediction_ledger(ledger, sentences, checkpoint,
        {'checkpoint': checkpoint.checkpoint_sha256, 'sentences': plan['inputs']['data-prepared/test.jsonl'], 'script': plan['script_sha256']})
    candidate_path = output / 'candidates.jsonl'
    with candidate_path.open('xb') as handle:
        for row in candidates:
            handle.write(encode(row))
    gold = _load_gold(layout.resolve('data-prepared/test-gold.jsonl'), 'CODE-SPLIT-1:test')
    truth = {t for row in gold.values() for t in row.triples}
    predicted = {Candidate.from_mapping(r, 'inference test').triple for r in candidates}
    quality = triple_metrics(tp=len(truth & predicted), fp=len(predicted - truth), fn=len(truth - predicted))
    save(output / 'quality.json', quality)
    require_inputs(root, plan, seed)
    files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    save(output / 'receipt.json', {'seed': seed, 'checkpoint_sha256': checkpoint.checkpoint_sha256,
        'sentence_count': n, 'selected_entities': selected, 'duplicates': duplicates,
        'rng_unchanged': True, 'files': files})


def compare_passes(first, second):
    a, b = load(first / 'receipt.json'), load(second / 'receipt.json')
    for directory, receipt in ((first, a), (second, b)):
        for name, expected in receipt['files'].items():
            if digest(directory / name) != expected:
                raise ValueError('completed test output was modified: ' + name)
    if a != b:
        raise ValueError('independent passes differ; inspect preserved outputs, do not accept these statistics')
    return load(first / 'quality.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--test-id', required=True)
    parser.add_argument('--device', default='cuda:0', choices=('cpu', 'cuda:0'))
    parser.add_argument('--worker', type=int, choices=range(42, 50), help=argparse.SUPPRESS)
    parser.add_argument('--pass-number', type=int, choices=(1, 2), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = setup(args.source_root)
    from utils.common.paths import RunLayout
    testing = RunLayout(root, args.test_id).run_root
    if args.worker is not None:
        if args.pass_number is None or os.environ.get('PYTHONHASHSEED') != '0':
            raise ValueError('workers require parent dispatch with fixed startup hash seed')
        plan = load(testing / 'plan.json')
        if plan['parent_run_id'] != args.run_id or plan['device'] != args.device or plan['test_id'] != args.test_id:
            raise ValueError('worker dispatch differs from saved plan')
        run_worker(root, plan, args.worker, args.pass_number)
        return 0
    _, _, testing, plan = prepare(root, args.run_id, args.test_id, args.device)
    testing.mkdir(parents=False, exist_ok=False)
    save(testing / 'plan.json', plan)
    from utils.evaluation.reporting import quality_summary
    per_seed = {}
    for seed in plan['seeds']:
        for repeat in (1, 2):
            print(f'Seed {seed}, independent pass {repeat}/2', flush=True)
            command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--source-root', str(root),
                '--run-id', args.run_id, '--test-id', args.test_id, '--device', args.device,
                '--worker', str(seed), '--pass-number', str(repeat)]
            subprocess.run(command, env={**os.environ, 'PYTHONHASHSEED': '0'}, check=True)
        per_seed[str(seed)] = compare_passes(testing / f'seed-{seed}/pass-1', testing / f'seed-{seed}/pass-2')
    require_inputs(root, plan)
    quality = quality_summary(per_seed)
    save(testing / 'statistics.json', {'parent_run_id': args.run_id, 'test_id': args.test_id,
        'independent_repeats_identical': True, 'quality': quality,
        'reference_recipe_mean_f1': 0.5199012003522201,
        'mean_difference_from_reference': quality['mean_f1'] - 0.5199012003522201,
        'reference_scope': 'descriptive earlier comparison result; no score-driven tuning or guaranteed parity'})
    print(f"F1 {quality['mean_f1']:.9f} +/- {quality['sample_sd_f1']:.9f} (best {quality['best_seed_f1']:.9f})")
    print(f'PASS: independent prediction/statistic bytes match. Results: {testing / "statistics.json"}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Inference check failed: {exc}', file=sys.stderr)
        raise SystemExit(2)
