"""Deterministic descriptive paper reports from authenticated same-run records.

These readers aggregate retained measurements and the approved frozen Table-1
paired statistics. They run no experiment or model.
"""
from __future__ import annotations

import csv
import hashlib
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
                and not isinstance(s['elapsed_seconds'], bool)
                and math.isfinite(s['elapsed_seconds']) and s['elapsed_seconds'] >= 0]
    return {'per_seed': [{'seed': s.get('seed'), 'recorded_seconds': s.get('elapsed_seconds') if s in measured else None,
                          'resumed': s.get('resumed_from') is not None} for s in summaries],
        'recorded_wall_hours': sum(s['elapsed_seconds'] for s in measured)/3600 if measured else None,
        'seeds_with_measurement': len(measured), 'expected_seeds': expected_seeds,
        'coverage': 'final recorded trainer invocation per seed; earlier failed invocations may be absent',
        'measured_gpu_hours': None, 'peak_resident_memory_bytes': None}


def _invocation_cost(evidence, label):
    """Read original process timings; unfinished starts have no measured cost."""
    directory = evidence.layout.resolve(f'logs/invocations/{label}')
    starts = sorted(directory.glob('*-start.json'))
    finishes = set(directory.glob('*-finish.json'))
    rows = []
    for index, path in enumerate(starts, 1):
        if path.name != f'{index:04d}-start.json':
            raise DataContractError('report invocation accounting is not contiguous')
        relative = evidence.layout.relative_identity(path)
        start = load_json(evidence.file(relative))
        if start.get('run_id') != evidence.layout.run_id or start.get('label') != label:
            raise DataContractError('report invocation accounting belongs to another run')
        finish_path = path.with_name(path.name.replace('-start', '-finish'))
        row = {'ordinal': index, 'status': 'missing_finish', 'recorded_seconds': None}
        if finish_path in finishes:
            finishes.remove(finish_path)
            finish = load_json(evidence.file(evidence.layout.relative_identity(finish_path)))
            if finish.get('start_sha256') != evidence.hashes[relative]:
                raise DataContractError('report invocation accounting hash differs')
            elapsed = finish.get('elapsed_seconds')
            if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
                    or not math.isfinite(elapsed) or elapsed < 0
                    or finish.get('status') not in {'completed', 'failed'}):
                raise DataContractError('report invocation timing is malformed')
            row.update(status=finish['status'], recorded_seconds=elapsed)
        rows.append(row)
    if finishes:
        raise DataContractError('report invocation finish has no start')
    measured = [row['recorded_seconds'] for row in rows if row['recorded_seconds'] is not None]
    seconds = sum(measured) if measured else None
    return {'per_invocation': rows, 'recorded_seconds': seconds,
        'recorded_wall_hours': seconds/3600 if seconds is not None else None,
        'retained_starts': len(rows), 'invocations_with_measurement': len(measured),
        'completed_invocations': sum(row['status'] == 'completed' for row in rows),
        'failed_invocations': sum(row['status'] == 'failed' for row in rows),
        'invocations_without_finish': len(rows)-len(measured),
        'coverage': 'original finished process timings only, including failures; missing finishes and unrecorded invocations are unavailable; no current-time or conservative budget charge is a measurement'}


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


def canonical_record_statistics(candidates, candidate_outcomes, sentence_outcomes,
                                selected_threshold, expected_seeds):
    """Descriptive intersections of original saved emissions and confidence.

    These views retain every gold denominator and deduplicate strict emitted
    keys after filtering original candidates. They do not change the scorer's
    four conditions, statistical family, or development-selected operating point.
    """
    from utils.evaluation.metrics import triple_metrics
    seeds = list(expected_seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or isinstance(selected_threshold, bool) or not isinstance(selected_threshold, (float, int))
            or not math.isfinite(selected_threshold) or not 0 <= selected_threshold <= 1):
        raise DataContractError('paper report selected threshold or seeds are malformed')
    confidence = {}
    for row in candidates:
        identity = (row['training_seed'], row['candidate_id'])
        value = row['triple_confidence']
        if (identity in confidence or identity[0] not in seeds
                or isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise DataContractError('paper report candidate identity or confidence is malformed')
        confidence[identity] = value
    gold = {}
    for row in sentence_outcomes:
        if row['condition_id'] == 'VER-RAW':
            identity = (row['training_seed'], row['example_id'])
            if identity in gold or identity[0] not in seeds:
                raise DataContractError('paper report gold sentence identity differs')
            gold[identity] = {tuple(key) for key in row['gold_strict_keys']}
    examples = {seed: {example for s, example in gold if s == seed} for seed in seeds}
    if (not gold or any(not values for values in examples.values())
            or any(values != examples[seeds[0]] for values in examples.values())):
        raise DataContractError('paper report gold coverage differs across training seeds')
    emissions = {condition: {} for condition in ('VER-RAW', 'VER-CONFIDENCE', 'VER-SIMPLE', 'VER-CORRECTIVE')}
    for row in candidate_outcomes:
        identity = (row['training_seed'], row['candidate_id'])
        condition = row['condition_id']
        if (identity not in confidence or condition not in emissions
                or identity in emissions[condition]
                or (identity[0], row['example_id']) not in gold):
            raise DataContractError('paper report outcome identity differs from original candidates')
        emitted = row['final_emitted_key']
        if emitted is not None and (not isinstance(emitted, list) or not emitted
                                    or emitted[0] != row['example_id']):
            raise DataContractError('paper report emitted strict key belongs to another sentence')
        emissions[condition][identity] = ((identity[0], row['example_id']),
                                          tuple(emitted) if emitted is not None else None)
    if any(set(rows) != set(confidence) for rows in emissions.values()):
        raise DataContractError('paper report outcome coverage differs from original candidates')

    def subset(condition, threshold):
        predicted = {identity: set() for identity in gold}
        for identity, (sentence, emitted) in emissions[condition].items():
            if confidence[identity] >= threshold and emitted is not None:
                predicted[sentence].add(emitted)
        counts = {str(seed): {'tp': 0, 'fp': 0, 'fn': 0} for seed in seeds}
        for (seed, example), reference in gold.items():
            actual = predicted[(seed, example)]
            for key, amount in (('tp', len(actual & reference)), ('fp', len(actual-reference)),
                                ('fn', len(reference-actual))):
                counts[str(seed)][key] += amount
        summary = quality_summary({seed: triple_metrics(**row) for seed, row in counts.items()})
        summary.update(original_candidate_confidence_threshold=threshold,
                       saved_emission_condition=condition)
        summary['emitted_unique_triples'] = summary['pooled']['counts']['tp'] + summary['pooled']['counts']['fp']
        return summary

    base = {condition: subset(condition, 0.) for condition in emissions}
    joint = subset('VER-SIMPLE', selected_threshold)
    supplements = {**base, 'VER-CONFIDENCE-SIMPLE': joint}
    raw = base['VER-RAW']
    effects = {}
    seed_effects = {}
    for condition, summary in supplements.items():
        if condition == 'VER-RAW':
            continue
        effects[condition] = {metric: (
            100*(summary['pooled'][metric]['value']-raw['pooled'][metric]['value'])
            if summary['pooled'][metric]['value'] is not None and raw['pooled'][metric]['value'] is not None else None)
            for metric in ('precision', 'recall', 'f1')}
        seed_effects[condition] = {}
        for metric in ('precision', 'recall', 'f1'):
            pairs = [(summary['per_seed'][str(seed)][metric]['value'], raw['per_seed'][str(seed)][metric]['value'])
                     for seed in seeds]
            seed_effects[condition][metric] = {
                'improved': sum(a > b for a, b in pairs if a is not None and b is not None),
                'worsened': sum(a < b for a, b in pairs if a is not None and b is not None),
                'tied': sum(a == b for a, b in pairs if a is not None and b is not None),
                'unavailable': sum(a is None or b is None for a, b in pairs)}
    return {'selected_confidence_threshold': selected_threshold,
        'joint_selected_threshold': joint,
        'threshold_sensitivity': {condition: subset(condition, .5)
            for condition in ('VER-RAW', 'VER-SIMPLE', 'VER-CORRECTIVE')},
        'pooled_effects_vs_raw_percentage_points': effects,
        'per_seed_effect_counts_vs_raw': seed_effects,
        'scope': 'descriptive saved-record intersections only; 0.5 is sensitivity, not a selected operating point; no additional significance family'}


def rag_record_statistics(result):
    """Recount original hit flags and pair mode transitions by question/target."""
    from utils.rag.evaluator import MODES, check_answer
    if any(set(result.get(key, {})) != set(MODES) for key in ('results', 'correct_counts', 'accuracy')):
        raise DataContractError('paper report RAG mode coverage differs from the frozen evaluator')
    identities = lambda rows: [(row.get('q'), row.get('gold')) for row in rows]
    reference = identities(result['results'][MODES[0]])
    full = {}
    for mode, rows in result['results'].items():
        if (not rows or any(not isinstance(row.get('correct'), bool) for row in rows)
                or any(row['correct'] != check_answer(row['pred'], row['gold']) for row in rows)):
            raise DataContractError('paper report RAG hit flags are malformed')
        if identities(rows) != reference:
            raise DataContractError('paper report RAG mode question/target identities differ')
        hits = sum(row['correct'] for row in rows)
        rate = hits/len(rows)
        if (len(rows) != result['metadata']['n_questions']
                or result['correct_counts'].get(mode) != hits or result['accuracy'].get(mode) != rate):
            raise DataContractError('paper report RAG aggregate differs from saved answer rows')
        full[mode] = {'questions': len(rows), 'hits': hits, 'answer_hit_rate': rate}
    transitions = None
    if {'kg_1hop', 'hybrid'} <= set(result['results']):
        before, after = result['results']['kg_1hop'], result['results']['hybrid']
        if identities(before) != identities(after):
            raise DataContractError('paper report RAG mode question/target identities differ')
        losses = sum(a['correct'] and not b['correct'] for a, b in zip(before, after))
        gains = sum(not a['correct'] and b['correct'] for a, b in zip(before, after))
        transitions = {'lost_hits': losses, 'gained_hits': gains,
            'net_hit_change': gains-losses,
            'both_hit': sum(a['correct'] and b['correct'] for a, b in zip(before, after)),
            'neither_hit': sum(not a['correct'] and not b['correct'] for a, b in zip(before, after)),
            'scope': 'same question/target rows; evidence modes also change fact budgets, so this is not an isolated text effect'}
    hybrid = result['results']['hybrid']
    losses_one = {index for index, (before, after) in enumerate(zip(result['results']['kg_1hop'], hybrid))
                  if before['correct'] and not after['correct']}
    losses_two = {index for index, (before, after) in enumerate(zip(result['results']['kg_2hop'], hybrid))
                  if before['correct'] and not after['correct']}
    return {'full': full, 'transitions_1hop_to_hybrid': transitions,
        'graph_hit_to_hybrid_loss_population': {
            'one_hop_losses': len(losses_one), 'two_hop_losses': len(losses_two),
            'both_graph_modes_hit_hybrid_missed': len(losses_one & losses_two),
            'one_hop_only': len(losses_one-losses_two), 'two_hop_only': len(losses_two-losses_one),
            'unique_losses': len(losses_one | losses_two)}}


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
                    if '/' in path or '.' in path:
                        self.file(path, identity)
                elif isinstance(identity, dict) and isinstance(identity.get('path'), str):
                    if not isinstance(identity.get('sha256'), str):
                        raise DataContractError('paper report evidence reference lacks its hash')
                    self.file(identity['path'], identity['sha256'])
        return value


def _write(evidence, directory, observations):
    layout = evidence.layout
    document = {'schema_version': 1, 'run_id': layout.run_id,
        'inputs': dict(sorted(evidence.hashes.items())), 'observations': observations,
        'definitions': {
            'quality': 'typed strict triples; per-seed counts and F1; sample SD uses n-1',
            'best_seed': 'all tied final-test maxima; descriptive, never a selection input',
            'paired_statistics': 'Wilcoxon is primary; paired t is sensitivity; p_value is raw and holm_adjusted_p_value is adjusted within its existing family',
            'request_cost': 'original uncached attempts, including retries and failures; workload paths remain separate',
            'training_cost': 'final trainer-summary invocation per seed; separate invocation_cost includes retained process timings and earlier failures where available; these views overlap',
            'invocation_cost': 'original elapsed time of hash-bound process finishes, including failures; overlaps trainer summaries and must not be added to them; unfinished starts have unavailable duration',
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
    receipt = {'run_id': layout.run_id, 'inputs': document['inputs'],
        'outputs': {f'{directory}/{name}': hashlib.sha256(raw).hexdigest()
                    for name, raw in products.items()}}
    receipt_path = layout.resolve(f'{directory}/manifest.json')
    if receipt_path.exists() and receipt_path.read_bytes() != canonical_json_bytes(receipt):
        raise DataContractError('existing paper report receipt differs')
    for relative, digest in document['inputs'].items():
        evidence.file(relative, digest)
    for name, raw in products.items():
        path = layout.resolve(f'{directory}/{name}')
        if path.exists() and path.read_bytes() != raw:
            raise DataContractError(f'existing paper report differs: {name}')
    for name, raw in products.items():
        path = layout.resolve(f'{directory}/{name}')
        if not path.exists(): atomic_write_bytes(path, raw)
    bind_json(receipt_path, receipt)
    return document


def _cost(evidence, seeds, comparison=False):
    layout = evidence.layout
    summaries = []
    environments = {}
    runtime_reference = None
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
            evidence.file(ref['path'], ref['sha256'])
            if runtime_reference is not None and ref != runtime_reference:
                raise DataContractError('training summaries use different numerical runtimes')
            runtime_reference = ref
        elif environment is not None:
            environments[str(seed)] = environment
        evidence.file(checkpoint['model_cache_manifest'], checkpoint['model_cache_manifest_sha256'])
    result = training_statistics(summaries, list(seeds))
    if runtime_reference is not None:
        result['runtime_environment'] = runtime_reference
    elif environments:
        result['environments'] = environments  # Original archived summaries only.
    if not comparison:
        per_seed = {str(seed): _invocation_cost(evidence, f'canonical-train-seed-{seed}')
                    for seed in seeds}
        measured = [value['recorded_seconds'] for value in per_seed.values()
                    if value['recorded_seconds'] is not None]
        result['invocations'] = {'per_seed': per_seed,
            'recorded_wall_hours': sum(measured)/3600 if measured else None,
            'seeds_with_measurement': len(measured), 'expected_seeds': list(seeds),
            'scope': 'canonical trainer processes, including startup and failed attempts; overlaps final trainer summaries above'}
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
    required_inputs = ('predictions/test/candidates.jsonl', 'predictions/dev/threshold-selection.json')
    required_outputs = ('outcomes/candidate-outcomes.jsonl', 'outcomes/sentence-outcomes.jsonl')
    if (any(name not in score.get('inputs', {}) for name in required_inputs)
            or any(name not in score.get('outputs', {}) for name in required_outputs)):
        raise DataContractError('paper report lacks authenticated original scoring records')
    threshold_trace = load_json(evidence.file(required_inputs[1], score['inputs'][required_inputs[1]]))
    if threshold_trace.get('selected_threshold') != metrics.get('selected_confidence_threshold'):
        raise DataContractError('paper report selected threshold differs from scored operating point')
    read_rows = lambda name: [row for _, row in iter_jsonl(evidence.file(name))]
    supplements = canonical_record_statistics(read_rows(required_inputs[0]),
        read_rows(required_outputs[0]), read_rows(required_outputs[1]),
        threshold_trace['selected_threshold'], metrics.get('observed_training_seeds', config.value['training_seeds']))
    observations = {'source': checkout.get('source'),
                    'training': _cost(evidence, config.value['training_seeds']),
                    'coverage': {key: metrics.get(key) for key in (
                        'protocol_id', 'matcher_id', 'evaluation_split_id', 'n_gold_examples',
                        'observed_training_seeds', 'required_training_seeds',
                        'publication_seed_coverage_complete', 'publication_data_coverage_complete',
                        'nonpublication_smoke', 'uncertainty_status')},
                    'quality': metrics.get('conditions'),
                    'paired_statistics': metrics.get('statistics'),
                    'corpus': _corpus(evidence),
                    'candidate_tuples': metrics.get('n_raw_candidates'),
                    'threshold_selection': threshold_trace,
                    'saved_record_supplement': supplements,
                    'verdict_diagnostics': {key: value.get('diagnostics')
                                            for key, value in metrics.get('conditions', {}).items()},
                    'verifier': {}}
    for mode in ('simple', 'corrective'):
        manifest = evidence.manifest(f'manifests/verifier-{mode}-live.json')
        required = {f'verifier/{mode}/{name}' for name in (
            'responses.jsonl', 'warmup-response.jsonl', 'environment-manifest.json')}
        if not required <= set(manifest.get('outputs', {})):
            raise DataContractError(f'paper report lacks authenticated {mode} verifier measurements')
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
    selected = evidence.manifest('manifests/encoder-comparison-selection.json')
    if selected != selection.manifest(layout.run_id):
        raise DataContractError('comparison report selected arm identity differs')
    corpus = _corpus(evidence)
    split_sha256 = sha256_file(evidence.file('data-prepared/split-manifest.json'))
    from utils.common.records import Candidate
    from utils.evaluation.scoring import _load_gold
    from utils.evaluation.metrics import triple_metrics
    gold_rows = _load_gold(evidence.file('data-prepared/test-gold.jsonl'), 'CODE-SPLIT-1:test')
    gold = {triple for row in gold_rows.values()
            for triple in row.triples}
    quality = {}
    for seed in selection.config.value['training_seeds']:
        producer = evidence.manifest(f'manifests/model-generate-candidates-replay-seed-{seed}-test.json')
        if (producer.get('training_seed') != seed or producer.get('status') != 'completed'
                or producer.get('checkpoint', {}).get('split_manifest_sha256') != split_sha256):
            raise DataContractError('comparison report candidate producer seed or completion differs')
        relative = producer['candidates_output']
        candidates = [Candidate.from_mapping(row, label='comparison report')
                      for _, row in iter_jsonl(evidence.file(relative, producer['output_hashes'][relative]))]
        if any(candidate.training_seed != seed or candidate.split_id != 'CODE-SPLIT-1:test'
                or candidate.triple.example_id not in gold_rows
                or candidate.source_document_id != gold_rows[candidate.triple.example_id].source_document_id
                for candidate in candidates):
            raise DataContractError('comparison report candidate seed, split or source identity differs')
        predictions = {candidate.triple for candidate in candidates}
        quality[str(seed)] = triple_metrics(tp=len(gold & predictions), fp=len(predictions-gold), fn=len(gold-predictions))
    return _write(evidence, 'reports/comparison', {
        'arm': selection.manifest(layout.run_id), 'quality': quality_summary(quality),
        'corpus': corpus,
        'pairing': {name: sha256_file(evidence.file(name)) for name in (
            'data-prepared/split-manifest.json', 'data-prepared/test-gold.jsonl')},
        'training': _cost(evidence, selection.config.value['training_seeds'], comparison=True),
        'scope': 'Table 1 only; no method-superiority claim from backbone sensitivity'})


def report_matrix(layout, config, identity):
    from utils.common.paths import RunLayout
    evidence = Evidence(layout)
    recorded_identity = load_json(evidence.file('manifests/comparison-matrix.json'))
    if recorded_identity != identity or identity.get('matrix_id') != layout.run_id:
        raise DataContractError('matrix report identity differs')
    if (identity.get('config') != config.source_path
            or identity.get('pipeline_config') != config.pipeline.source_path
            or set(identity.get('members', {})) != set(config.arms)):
        raise DataContractError('matrix report configuration or declared member family differs')
    members = {}
    readers = []
    shared_pairing = None
    for arm, run_id in identity['members'].items():
        member = RunLayout(layout.source_root, run_id)
        report = member.resolve('reports/comparison/paper-statistics.json', must_exist=True)
        receipt = load_json(member.resolve('reports/comparison/manifest.json', must_exist=True))
        reader = Evidence(member)
        reader.manifest('reports/comparison/manifest.json')
        document = load_json(reader.file('reports/comparison/paper-statistics.json'))
        if (document.get('run_id') != run_id or document.get('inputs') != receipt.get('inputs')
                or receipt.get('outputs', {}).get('reports/comparison/paper-statistics.json') != sha256_file(report)):
            raise DataContractError('matrix report member identity differs')
        observations = document['observations']
        if observations.get('arm') != config.select_arm(arm).manifest(run_id):
            raise DataContractError('matrix report arm identity differs from its declared member')
        pairing = observations.get('pairing')
        required = {'data-prepared/split-manifest.json', 'data-prepared/test-gold.jsonl'}
        if not isinstance(pairing, dict) or set(pairing) != required:
            raise DataContractError('matrix report member lacks split/gold pairing evidence')
        if any(document['inputs'].get(path) != digest for path, digest in pairing.items()):
            raise DataContractError('matrix report pairing is not authenticated by member evidence')
        if shared_pairing is not None and pairing != shared_pairing:
            raise DataContractError('matrix report members have different split or gold identities')
        shared_pairing = pairing
        # This coordinator references isolated authenticated reports; no model or cache is shared.
        members[arm] = {'run_id': run_id, 'report_sha256': sha256_file(report),
                        'receipt_sha256': sha256_file(member.resolve('reports/comparison/manifest.json')),
                        'pairing': pairing,
                        'quality': document['observations']['quality'],
                        'training': document['observations']['training']}
        readers.append(reader)
    from utils.evaluation.comparison_statistics import comparison_statistics
    paired = comparison_statistics(members, config.value['statistical_protocol'], config.value['training_seeds'])
    measured = [row['recorded_seconds'] for member in members.values()
                for row in member['training']['per_seed'] if row['recorded_seconds'] is not None]
    # Sum the original arm totals in declared arm order, matching the archived
    # 32-invocation analysis rather than adding already-converted hours.
    arm_seconds = [sum(row['recorded_seconds'] for row in member['training']['per_seed']
                       if row['recorded_seconds'] is not None) for arm in
                   [row['arm_id'] for row in config.value['table1_arms']]
                   if (member := members.get(arm)) is not None]
    training_seconds = sum(arm_seconds) if measured else None
    for reader in readers:
        for relative, digest in list(reader.hashes.items()):
            reader.file(relative, digest)
    return _write(evidence, 'reports/matrix', {'members': members,
        'paired_statistics': paired,
        'training_cost': {'seconds_sum': training_seconds,
            'hours_sum': training_seconds/3600 if training_seconds is not None else None,
            'invocations_with_measurement': len(measured),
            'expected_invocations': len(identity['members'])*len(config.value['training_seeds']),
            'scope': 'recorded final trainer invocation per arm/seed; cumulative workload, not end-to-end wall time'},
        'recorded_execution': {
            'smoke': _invocation_cost(evidence, 'comparison-smoke'),
            'full': _invocation_cost(evidence, 'comparison-full'),
            'scope': 'smoke is separate from full training and candidate-generation processes; full execution includes failed/resumed attempts and overlaps member training summaries'},
        'scope': 'same-protocol Table-1 comparison with the approved frozen four-contrast analysis; underlying-document uncertainty remains unproved'})


def report_rag(layout, summary, runtime, child_dir):
    evidence = Evidence(layout)
    relative = layout.relative_identity(child_dir/'artifact-hashes.json')
    hashes = load_json(evidence.file(relative))
    if hashes.get('run_id') != layout.run_id:
        raise DataContractError('RAG paper report evidence belongs to another run')
    entries = hashes.get('files', hashes.get('artifacts', []))
    if isinstance(entries, dict): entries = [{'path': k, 'sha256': v} for k, v in entries.items()]
    for item in entries:
        path = item['path']
        path = layout.relative_identity(child_dir/path)
        evidence.file(path, item['sha256'])
    required = ['environment.json', *(f'{directory}/{condition}.json'
        for directory in ('rag-results', 'part-of-results')
        for condition in ('confidence', 'corrective', 'gold'))]
    if any(layout.relative_identity(child_dir/path) not in evidence.hashes for path in required):
        raise DataContractError('RAG paper report lacks authenticated result or environment evidence')
    environment = load_json(evidence.file(layout.relative_identity(child_dir/'environment.json')))
    observations = {'coverage': summary['questions'], 'conditions': {}, 'graph_scale': {},
        'environment': environment,
        'claim_limits': {
            'result_status': 'expanded CODE diagnostic, not the original paper Table 2',
            'gold_condition': 'nondeployable gold-graph oracle',
            'metric': 'frozen answer-hit check, not a factuality or deployment evaluation',
            'historical_parity': 'not established by these reports; external comparison is separate',
        },
        'canonical_graph_scale': {condition: {
            'seed': graph.get('training_seed'), 'condition': graph.get('condition'),
            'entities': len(graph['entities']), 'relations': len(graph['relations']),
            'triples': len(graph['triples'])} for condition, graph in runtime.get('graphs', {}).items()},
        'graph_construction_time_seconds': None,
        'timing_scope': 'retained evaluator condition wall time; no per-call token or latency measurement'}
    for condition in ('confidence', 'corrective', 'gold'):
        result = load_json(evidence.file(layout.relative_identity(child_dir/'rag-results'/f'{condition}.json')))
        subset = load_json(evidence.file(layout.relative_identity(child_dir/'part-of-results'/f'{condition}.json')))
        recounted = rag_record_statistics(result)
        observations['conditions'][condition] = {'questions': result['metadata']['n_questions'],
            'hits': result['correct_counts'], 'accuracy': result['accuracy'],
            **recounted,
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
