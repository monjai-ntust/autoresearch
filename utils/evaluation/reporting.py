"""Deterministic descriptive paper reports from authenticated same-run records.

These readers aggregate retained measurements. They run no experiment, model,
resampling procedure or significance test.
"""
from __future__ import annotations

import csv
import io
import math
import statistics

from utils.common.artifact_io import (
    DataContractError, atomic_write_bytes, canonical_json_bytes, iter_jsonl,
    load_json, sha256_file,
)
from utils.common.execution import bind_json


def response_statistics(rows: list[dict]) -> dict:
    """Original uncached attempt timings, including failed/retried requests."""
    attempts = [attempt for row in rows if row.get('cache_hit') is False
                for attempt in row.get('attempts', [])]
    elapsed = [a['elapsed_seconds'] for a in attempts
               if isinstance(a.get('elapsed_seconds'), (float, int))
               and not isinstance(a.get('elapsed_seconds'), bool)
               and math.isfinite(a['elapsed_seconds']) and a['elapsed_seconds'] >= 0]
    tokens = {}
    for key in ('prompt_eval_count', 'eval_count'):
        values = [(a.get('raw_response') or {}).get(key) for a in attempts
                  if isinstance(a.get('raw_response'), dict)]
        values = [v for v in values if isinstance(v, int) and not isinstance(v, bool) and v >= 0]
        tokens[key] = {'sum': sum(values) if values else None,
                       'attempts_with_measurement': len(values), 'attempts': len(attempts)}
    return {'requests': len(rows), 'uncached_requests': sum(r.get('cache_hit') is False for r in rows),
        'cached_requests': sum(r.get('cache_hit') is True for r in rows),
        'unique_candidates': len({r['candidate_id'] for r in rows if 'candidate_id' in r}),
        'uncached_attempts': len(attempts),
        'retry_attempts': sum(max(0, len(r.get('attempts', []))-1) for r in rows if r.get('cache_hit') is False),
        'transport_failures': sum(a.get('transport_status') != 'response' or a.get('http_status') != 200 for a in attempts),
        'recorded_request_seconds': sum(elapsed) if elapsed else None,
        'mean_seconds_per_measured_uncached_attempt': sum(elapsed)/len(elapsed) if elapsed else None,
        'timed_attempts': len(elapsed), 'tokens': tokens}


def training_statistics(summaries: list[dict], expected_seeds: list[int]) -> dict:
    measured = [s for s in summaries if isinstance(s.get('elapsed_seconds'), (float, int))
                and math.isfinite(s['elapsed_seconds']) and s['elapsed_seconds'] >= 0]
    return {'per_seed': [{'seed': s.get('seed'), 'recorded_seconds': s.get('elapsed_seconds'),
                          'resumed': s.get('resumed_from') is not None} for s in summaries],
        'recorded_wall_hours': sum(s['elapsed_seconds'] for s in measured)/3600 if measured else None,
        'seeds_with_measurement': len(measured), 'expected_seeds': expected_seeds,
        'coverage': 'final recorded trainer invocation per seed; earlier failed invocations may be absent',
        'measured_gpu_hours': None, 'peak_resident_memory_bytes': None}


def quality_summary(per_seed):
    """The approved Table-1 descriptive aggregation and strict pooled counts."""
    from utils.evaluation.metrics import triple_metrics
    values = [v['f1']['value'] for v in per_seed.values() if v['f1']['value'] is not None]
    counts = {key: sum(v['counts'][key] for v in per_seed.values()) for key in ('tp', 'fp', 'fn')}
    return {'per_seed': per_seed, 'mean_f1': statistics.mean(values) if values else None,
        'sample_sd_f1': statistics.stdev(values) if len(values) > 1 else None,
        'best_seed_f1': max(values) if values else None,
        'tied_best_seeds': sorted(int(seed) for seed, value in per_seed.items()
                                 if values and value['f1']['value'] == max(values)),
        'pooled': triple_metrics(**counts)}


class Evidence:
    def __init__(self, layout):
        self.layout = layout
        self.hashes = {}

    def file(self, relative, expected=None):
        path = self.layout.resolve(relative, must_exist=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise DataContractError(f'paper report evidence changed: {relative}')
        if relative in self.hashes and self.hashes[relative] != digest:
            raise DataContractError('paper report inputs changed during aggregation')
        self.hashes[relative] = digest
        return path

    def manifest(self, relative):
        value = load_json(self.file(relative))
        if value.get('run_id', self.layout.run_id) != self.layout.run_id:
            raise DataContractError('paper report manifest belongs to another run')
        for key in ('inputs', 'outputs', 'output_hashes'):
            for path, identity in value.get(key, {}).items():
                if isinstance(identity, str):
                    if (('/' in path or '.' in path) and len(identity) == 64
                            and all(c in '0123456789abcdef' for c in identity)):
                        self.file(path, identity)
                elif isinstance(identity, dict) and isinstance(identity.get('path'), str):
                    self.file(identity['path'], identity.get('sha256'))
        return value


def _write(evidence, directory, observations):
    layout = evidence.layout
    document = {'schema_version': 1, 'run_id': layout.run_id,
        'inputs': dict(sorted(evidence.hashes.items())), 'observations': observations,
        'definitions': {
            'quality': 'typed strict triples; per-seed counts and F1; sample SD uses n-1',
            'best_seed': 'all tied final-test maxima; descriptive, never a selection input',
            'request_cost': 'original uncached attempts, including retries and failures; workload paths remain separate',
            'training_cost': 'recorded final trainer invocation per seed; missing earlier invocations remain outside coverage',
            'precision': 'full recorded numerical precision; no rounding in JSON/TSV/Markdown',
            'missingness': 'null means unavailable; token/timing totals carry measured coverage',
        },
        'limits': ['Unavailable values are null; no historical estimates are imported.',
                   'Recorded wall time is not GPU utilization or complete reproduction cost.',
                   'Disk model bytes are not resident or peak accelerator memory.']}
    products = {'paper-statistics.json': canonical_json_bytes(document)}
    flattened = []
    def flatten(value, path):
        if isinstance(value, dict):
            for key, item in value.items(): flatten(item, f'{path}.{key}' if path else key)
        elif isinstance(value, list):
            for i, item in enumerate(value): flatten(item, f'{path}[{i}]')
        else:
            flattened.append((path, 'unavailable' if value is None else str(value)))
    flatten(observations, '')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(['measurement', 'value'])
    writer.writerows(flattened)
    products['paper-statistics.tsv'] = stream.getvalue().encode('utf-8')
    lines = [f'# Recorded paper statistics: {layout.run_id}', '',
             '| Measurement | Value |', '|---|---|']
    def escape(value): return value.replace('|', '\\|').replace('\n', '<br>')
    lines.extend(f'| {escape(k)} | {escape(v)} |' for k, v in flattened)
    lines += ['', *document['limits'], '']
    products['paper-statistics.md'] = '\n'.join(lines).encode('utf-8')
    # Check all existing products before writing any; receipt finalization is pure.
    for name, raw in products.items():
        path = layout.resolve(f'{directory}/{name}')
        if path.exists() and path.read_bytes() != raw:
            raise DataContractError(f'existing paper report differs: {name}')
    for name, raw in products.items():
        path = layout.resolve(f'{directory}/{name}')
        if not path.exists(): atomic_write_bytes(path, raw)
    bind_json(layout.resolve(f'{directory}/manifest.json'), {
        'run_id': layout.run_id, 'inputs': document['inputs'],
        'outputs': {f'{directory}/{name}': sha256_file(layout.resolve(f'{directory}/{name}')) for name in products}})
    return document


def _cost(evidence, seeds, comparison=False):
    layout = evidence.layout
    summaries = []
    environments = {}
    for seed in seeds:
        relative = f'checkpoints/seed-{seed}/training-summary.json'
        checkpoint = load_json(evidence.file(f'checkpoints/seed-{seed}/checkpoint-manifest.json'))
        producer = f'manifests/encoder-comparison-train-live-seed-{seed}.json' if comparison else f'manifests/model-train-live-seed-{seed}.json'
        manifest = evidence.manifest(producer)
        aliases = {'checkout_manifest_sha256': 'manifests/00-checkout-manifest.json',
            'acquisition_manifest_sha256': 'manifests/02-input-acquisition-manifest.json',
            'preparation_manifest_sha256': 'manifests/03-data-preparation-manifest.json',
            'split_manifest_sha256': 'data-prepared/split-manifest.json',
            'train_jsonl_sha256': 'data-prepared/train.jsonl',
            'development_jsonl_sha256': 'data-prepared/development.jsonl',
            'model_cache_manifest_sha256': 'manifests/model-cache-huggingface.json'}
        for name, path in aliases.items():
            if name in manifest.get('inputs', {}):
                if name == 'model_cache_manifest_sha256': path = checkpoint['model_cache_manifest']
                evidence.file(path, manifest['inputs'][name])
        expected = manifest.get('output_hashes', manifest.get('outputs', {})).get(relative)
        if not expected:
            raise DataContractError('training report summary is not authenticated by its producer')
        summary = load_json(evidence.file(relative, expected))
        if summary.get('seed') != seed or summary.get('status') != 'completed':
            raise DataContractError('training report summary is incomplete')
        summaries.append(summary)
        environment = summary.get('environment')
        if isinstance(summary.get('runtime_environment'), dict):
            ref = summary['runtime_environment']
            environment = load_json(evidence.file(ref['path'], ref['sha256']))
        if environment is not None:
            environments[str(seed)] = environment
        evidence.file(checkpoint['model_cache_manifest'], checkpoint['model_cache_manifest_sha256'])
    result = training_statistics(summaries, list(seeds))
    result['environments'] = environments
    return result


def _corpus(evidence):
    """Counts from the authenticated prepared tree, including eligibility limits."""
    from utils.preparation.materialization import _tree_inventory
    preparation = evidence.manifest('manifests/03-data-preparation-manifest.json')
    rows, digest = _tree_inventory(evidence.layout.resolve('data-prepared'))
    if digest != preparation.get('dataset_tree_sha256'):
        raise DataContractError('paper report prepared tree differs from its producer')
    for row in rows:
        evidence.file('data-prepared/' + row['path'], row['sha256'])
    data = load_json(evidence.file('data-prepared/data-manifest.json'))
    return {'split_distributions': load_json(evidence.file('data-prepared/distributions.json')),
            'counts': data.get('counts'), 'eligibility': data.get('alignment'),
            'split': data.get('split')}


def report_canonical(layout, config):
    evidence = Evidence(layout)
    checkout = evidence.manifest('manifests/00-checkout-manifest.json')
    score = evidence.manifest('manifests/score-manifest.json')
    metrics_relative = next(p for p in score['outputs'] if p.endswith('/metrics.json') or p == 'metrics.json')
    metrics = load_json(evidence.file(metrics_relative))
    observations = {'source': checkout.get('source'),
                    'training': _cost(evidence, config.value['training_seeds']),
                    'quality': metrics.get('conditions'),
                    'paired_statistics': metrics.get('statistics'),
                    'corpus': _corpus(evidence),
                    'candidate_tuples': metrics.get('n_raw_candidates'),
                    'verdict_diagnostics': {key: value.get('diagnostics')
                                            for key, value in metrics.get('conditions', {}).items()},
                    'verifier': {}}
    for path in sorted(layout.resolve('manifests').glob('verifier-*-live.json')):
        relative = layout.relative_identity(path)
        manifest = evidence.manifest(relative)
        responses = [p for p in manifest.get('outputs', {}) if p.endswith('/responses.jsonl') or p.endswith('/warmup-response.jsonl')]
        for name in responses:
            observations['verifier'][name] = response_statistics([row for _, row in iter_jsonl(evidence.file(name))])
        for name in manifest.get('outputs', {}):
            if name.endswith('/environment-manifest.json'):
                observations.setdefault('verifier_environments', {})[name] = load_json(evidence.file(name))
    best = {}
    for condition, value in (metrics.get('conditions') or {}).items():
        per_seed = value.get('per_seed', {})
        ranked = [(v['end_to_end_strict_triple']['f1']['value'], int(seed)) for seed, v in per_seed.items()
                  if isinstance(v.get('end_to_end_strict_triple', {}).get('f1', {}).get('value'), (int, float))]
        if ranked:
            chosen = sorted(ranked, key=lambda x: (-x[0], x[1]))[0]
            best[condition] = {'seed': chosen[1], 'strict_f1': chosen[0],
                'tied_seeds': sorted(seed for value, seed in ranked if value == chosen[0]),
                'tie_rule': 'highest recorded fixed-test F1; show all ties, lowest seed labels the row; descriptive only'}
    observations['quality_supplement'] = {}
    observations['emitted_unique_triples'] = {}
    for condition, value in (metrics.get('conditions') or {}).items():
        per_seed = {seed: row['end_to_end_strict_triple'] for seed, row in value.get('per_seed', {}).items()}
        if all('counts' in row for row in per_seed.values()) and per_seed:
            observations['quality_supplement'][condition] = quality_summary(per_seed)
            observations['emitted_unique_triples'][condition] = {
                seed: row['counts']['tp'] + row['counts']['fp'] for seed, row in per_seed.items()}
    observations['descriptive_best_seed'] = best
    blobs = []
    for path in sorted(layout.resolve('inputs').rglob('*')):
        if path.is_file() and ('blobs' in path.parts or path.suffix in {'.bin', '.safetensors', '.gguf'}):
            # Actual run-local weight identity and disk size, never another run's cache.
            relative = layout.relative_identity(path)
            evidence.file(relative)
            blobs.append({'path': relative, 'disk_bytes': path.stat().st_size, 'sha256': evidence.hashes[relative]})
    observations['model_files'] = blobs
    observations['model_memory'] = {'resident_bytes': None, 'peak_bytes': None}
    observations['unavailable_measurements'] = {
        'energy_and_monetary_cost': 'not measured',
        'thinking_mode_verifier': 'outside the frozen non-thinking protocol',
        'optional_datasets': 'CUAD, SciERC and zh are not executed by this workflow',
        'canonical_graph_scale': 'graphs are produced and reported by the independent RAG workflow',
    }
    return _write(evidence, 'metrics/paper-report', observations)


def report_comparison(layout, selection):
    evidence = Evidence(layout)
    evidence.manifest('manifests/00-checkout-manifest.json')
    evidence.manifest('manifests/encoder-comparison-selection.json')
    from utils.common.records import Candidate
    from utils.evaluation.scoring import _load_gold
    from utils.evaluation.metrics import triple_metrics
    gold = {triple for row in _load_gold(evidence.file('data-prepared/test-gold.jsonl'), 'CODE-SPLIT-1:test').values()
            for triple in row.triples}
    quality = {}
    for seed in selection.config.value['training_seeds']:
        producer = evidence.manifest(f'manifests/model-generate-candidates-replay-seed-{seed}-test.json')
        relative = producer['candidates_output']
        predictions = {Candidate.from_mapping(row, label='comparison report').triple
                       for _, row in iter_jsonl(evidence.file(relative, producer['output_hashes'][relative]))}
        quality[str(seed)] = triple_metrics(tp=len(gold & predictions), fp=len(predictions-gold), fn=len(gold-predictions))
    return _write(evidence, 'reports/comparison', {
        'arm': selection.manifest(layout.run_id), 'quality': quality_summary(quality),
        'training': _cost(evidence, selection.config.value['training_seeds'], comparison=True),
        'scope': 'Table 1 only; no method-superiority claim from backbone sensitivity'})


def report_matrix(layout, config, identity):
    from utils.common.paths import RunLayout
    evidence = Evidence(layout)
    evidence.file('manifests/comparison-matrix.json')
    members = {}
    for arm, run_id in identity['members'].items():
        member = RunLayout(layout.source_root, run_id)
        report = member.resolve('reports/comparison/paper-statistics.json', must_exist=True)
        receipt = load_json(member.resolve('reports/comparison/manifest.json', must_exist=True))
        reader = Evidence(member)
        reader.manifest('reports/comparison/manifest.json')
        document = load_json(report)
        if (document.get('run_id') != run_id or document.get('inputs') != receipt.get('inputs')
                or receipt.get('outputs', {}).get('reports/comparison/paper-statistics.json') != sha256_file(report)):
            raise DataContractError('matrix report member identity differs')
        # This coordinator references isolated authenticated reports; no model or cache is shared.
        members[arm] = {'run_id': run_id, 'report_sha256': sha256_file(report),
                        'quality': document['observations']['quality']}
    return _write(evidence, 'reports/matrix', {'members': members,
        'predeclared_statistics': config.value['statistical_protocol'],
        'unrecorded_paired_or_bootstrap_measurements': None,
        'scope': 'descriptive same-protocol Table-1 comparison; no new statistical procedure in Phase I'})


def report_rag(layout, summary, runtime, child_dir):
    evidence = Evidence(layout)
    relative = layout.relative_identity(child_dir/'artifact-hashes.json')
    hashes = load_json(evidence.file(relative))
    entries = hashes.get('files', hashes.get('artifacts', []))
    if isinstance(entries, dict): entries = [{'path': k, 'sha256': v} for k, v in entries.items()]
    for item in entries:
        path = item['path']
        path = layout.relative_identity(child_dir/path)
        evidence.file(path, item['sha256'])
    observations = {'coverage': summary['questions'], 'conditions': {}, 'graph_scale': {},
        'canonical_graph_scale': {condition: {
            'seed': graph.get('training_seed'), 'condition': graph.get('condition'),
            'entities': len(graph['entities']), 'relations': len(graph['relations']),
            'triples': len(graph['triples'])} for condition, graph in runtime.get('graphs', {}).items()},
        'graph_construction_time_seconds': None,
        'timing_scope': 'retained evaluator condition wall time; no per-call token or latency measurement'}
    for condition in ('confidence', 'corrective', 'gold'):
        result = load_json(evidence.file(layout.relative_identity(child_dir/'rag-results'/f'{condition}.json')))
        subset = load_json(evidence.file(layout.relative_identity(child_dir/'part-of-results'/f'{condition}.json')))
        observations['conditions'][condition] = {'questions': result['metadata']['n_questions'],
            'hits': result['correct_counts'], 'accuracy': result['accuracy'],
            'recorded_condition_seconds': result['metadata']['time_seconds'],
            'part_of': {mode: {'questions': len(rows), 'hits': sum(row['correct'] for row in rows),
                        'answer_hit_rate': sum(row['correct'] for row in rows)/len(rows) if rows else None}
                        for mode, rows in subset['results'].items()},
            'error_rows': sum(str(row['pred']).startswith('ERROR:') or not str(row['pred']).strip()
                              for rows in result['results'].values() for row in rows)}
        projection = runtime['projections'][condition] if 'projections' in runtime else None
        if projection is not None:
            if isinstance(projection, bytes):
                import json
                projection = json.loads(projection)
            observations['graph_scale'][condition] = {'entities': len(projection.get('nodes', [])),
                'edges': len(projection.get('edges', [])),
                'unique_edges': len({canonical_json_bytes(e) for e in projection.get('edges', [])})}
    return _write(evidence, f'{layout.relative_identity(child_dir)}/reports', observations)
