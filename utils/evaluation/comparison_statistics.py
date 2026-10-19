"""The frozen four-contrast Table-1 analysis, using paired seed records only."""
from __future__ import annotations

import math
import statistics

from utils.common.artifact_io import DataContractError
from utils.evaluation.statistics import exact_wilcoxon_signed_rank, holm_adjust


def comparison_statistics(members: dict, protocol: dict, seeds: list[int]) -> dict:
    """Reproduce the approved exact tests and shared PCG64 seed bootstrap.

    This is an offline calculation on saved outcomes. It neither selects a model
    nor establishes uncertainty across the underlying regulatory documents.
    """
    if (protocol.get('paired_test') != 'two_sided_exact_wilcoxon_signed_rank'
            or protocol.get('multiplicity') != 'holm'
            or protocol.get('confidence_level') != .95):
        raise DataContractError('comparison report statistical protocol differs')
    contrasts = protocol.get('paired_contrasts')
    replicates = protocol.get('paired_seed_bootstrap_replicates')
    if (not isinstance(contrasts, list) or not contrasts
            or isinstance(replicates, bool) or not isinstance(replicates, int)
            or replicates <= 0 or len(set(seeds)) != len(seeds) or not seeds):
        raise DataContractError('comparison report contrast or bootstrap plan is malformed')
    required = {arm for pair in contrasts for arm in pair}
    absent = sorted(required - set(members))
    missing = {}
    values = {}
    for arm in sorted(required & set(members)):
        records = members[arm]['quality'].get('per_seed', {})
        if set(records) != {str(seed) for seed in seeds}:
            missing[arm] = 'per-seed identities do not cover the declared seeds exactly'
            continue
        sample = [records[str(seed)].get('f1', {}).get('value') for seed in seeds]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or not 0 <= value <= 1 for value in sample):
            missing[arm] = 'one or more seed F1 values are unavailable or malformed'
            continue
        values[arm] = sample
    limits = {
        'seed_interval': 'variation across the declared training seeds on the fixed test set; unadjusted intervals',
        'document_cluster_uncertainty': None,
        'document_cluster_status': 'underlying regulation-document clusters are not established by recorded source IDs',
    }
    if absent or missing:
        return {'status': 'withheld_incomplete_declared_family', 'missing_arms': absent,
                'incomplete_arms': missing, 'contrasts': [], 'per_arm': {}, 'limits': limits}

    import numpy as np
    generator = np.random.Generator(np.random.PCG64(protocol['bootstrap_seed']))
    draws = generator.integers(0, len(seeds), size=(replicates, len(seeds)))
    # The archived method subtracts the separately averaged arm replicates.
    # Averaging precomputed differences can produce different floating bytes.
    bootstraps = {arm: np.array(sample)[draws].mean(axis=1) for arm, sample in values.items()}
    interval = lambda sample: np.quantile(sample, [.025, .975], method='linear').tolist()
    results = []
    for arm, reference in contrasts:
        deltas = [a-b for a, b in zip(values[arm], values[reference])]
        test = exact_wilcoxon_signed_rank(deltas)
        results.append({'arm': arm, 'reference': reference,
            'per_seed_differences': deltas, 'mean_difference': statistics.mean(deltas),
            'wilcoxon': test, 'raw_p': test['p_value'],
            'seed_bootstrap_95ci': interval(bootstraps[arm]-bootstraps[reference]),
            'seeds_improved': sum(delta > 0 for delta in deltas),
            'seeds_worsened': sum(delta < 0 for delta in deltas),
            'seeds_tied': sum(delta == 0 for delta in deltas)})
    adjusted = holm_adjust({str(index): row['raw_p'] for index, row in enumerate(results)})
    for index, row in enumerate(results):
        row['holm_p'] = adjusted[str(index)]
        row['significant_at_0_05'] = row['holm_p'] < protocol['alpha']
        row['mean_reaches_0_01_effect'] = abs(row['mean_difference']) >= protocol['minimum_meaningful_absolute_f1']
    return {'status': 'defined', 'seeds': list(seeds), 'contrasts': results,
        'per_arm': {arm: {'seed_bootstrap_mean_f1_95ci': interval(sample)}
                    for arm, sample in bootstraps.items()},
        'bootstrap_implementation': {'numpy': np.__version__, 'generator': 'PCG64',
            'replicates': replicates, 'seed': protocol['bootstrap_seed'],
            'interval': 'linear percentile 2.5/97.5',
            'pairing': 'the same seed draws across arms; subtract separately averaged arm replicates'},
        'limits': limits}
