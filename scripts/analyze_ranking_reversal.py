"""Paired test of whether canonicalization reverses generator rankings.

Reads the frozen per-draw metrics of the HumanML3D release audits (primary and
secondary official T2M checkpoints) and, for every pair of releases, computes
per-draw score differences under raw and canonical inputs.  All releases share
the same 20 native caption/crop draws and the same saved real-reference draws,
so differences are paired across draws.  No model is run; inputs are read only.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / 'reports/cvpr_motion_prevalence_20261012'
ADD = ROOT / 'reports/cvpr_additional_motion_audit_20261013/humanml'
MD = ROOT / 'reports/cvpr_motiondiffuse_compiled_20261013/humanml'
OUT = ROOT / 'reports/cvpr_ranking_reversal_20261014'
REPS = 20
ORIGINAL = ['mdm50', 'mld', 'motionlcm_v1', 't2mgpt', 't2m']
ADDITIONAL = ['momask', 'dip', 'motiondiffuse']
OUTPUTS = {'mdm50': 1024, 'motiondiffuse': 1024, 'dip': 1024}


def load_primary(method):
    vals = {'raw': {}, 'canonical': {}}
    for r in range(REPS):
        if method in ORIGINAL:
            m = json.loads((PREV / method / f'rep_{r:02d}' / 'metrics.json').read_text())
            res = m['results']
            assert m['repetition'] == r
        else:
            base = MD if method == 'motiondiffuse' else ADD
            m = json.loads((base / method / f'rep_{r:02d}' / 'metrics.json').read_text())
            res = m['scores']
            assert m['rep'] == r
        for enc in ('raw', 'canonical'):
            vals[enc].setdefault('FID', []).append(res[enc]['FID'])
            vals[enc].setdefault('R1', []).append(res[enc]['R_precision'][0])
    return vals


def load_secondary(method):
    vals = {'raw': {}, 'canonical': {}}
    for r in range(REPS):
        m = json.loads((PREV / 'secondary_gpt_reader' / method / f'rep_{r:02d}' / 'metrics.json').read_text())
        res = m.get('results', m.get('scores'))
        for enc in ('raw', 'canonical'):
            vals[enc].setdefault('FID', []).append(res[enc]['FID'])
            vals[enc].setdefault('R1', []).append(res[enc]['R_precision'][0])
    return vals


def interval(x, family):
    x = np.asarray(x, float)
    n = len(x)
    se = x.std(ddof=1) / math.sqrt(n)
    q = stats.t.ppf(1 - 0.05 / (2 * family), n - 1)
    return float(x.mean()), [float(x.mean() - q * se), float(x.mean() + q * se)]


def sign(ci):
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def analyze(name, data, metric, lower_better):
    methods = list(data)
    pairs = list(itertools.combinations(methods, 2))
    family = 2 * len(pairs)
    rows = []
    for a, b in pairs:
        row = {'a': a, 'b': b, 'equal_output_counts': OUTPUTS.get(a, 4640) == OUTPUTS.get(b, 4640)}
        for enc in ('raw', 'canonical'):
            d = np.array(data[a][enc][metric]) - np.array(data[b][enc][metric])
            mean, ci = interval(d, family)
            row[enc] = {'mean_a_minus_b': mean, 'family_CI95': ci, 'supported_sign': sign(ci)}
        dd = (np.array(data[a]['raw'][metric]) - np.array(data[b]['raw'][metric])) - \
             (np.array(data[a]['canonical'][metric]) - np.array(data[b]['canonical'][metric]))
        mean, ci = interval(dd, len(pairs))
        row['change_in_difference'] = {'mean': mean, 'family_CI95': ci}
        row['mean_order_flips'] = np.sign(row['raw']['mean_a_minus_b']) != np.sign(row['canonical']['mean_a_minus_b'])
        rs, cs = row['raw']['supported_sign'], row['canonical']['supported_sign']
        row['reliable_reversal'] = rs != 0 and cs != 0 and rs != cs
        row['significance_gained_or_lost'] = (rs == 0) != (cs == 0)
        rows.append(row)
    rank = {}
    for enc in ('raw', 'canonical'):
        means = {m: float(np.mean(data[m][enc][metric])) for m in methods}
        rank[enc] = sorted(methods, key=lambda m: means[m], reverse=not lower_better)
    taus = stats.kendalltau([rank['raw'].index(m) for m in methods],
                            [rank['canonical'].index(m) for m in methods])
    return {'reader': name, 'metric': metric, 'methods': methods, 'family_size': family,
            'ranking_raw': rank['raw'], 'ranking_canonical': rank['canonical'],
            'kendall_tau_raw_vs_canonical': float(taus.statistic),
            'mean_order_flips': [f"{r['a']}/{r['b']}" for r in rows if r['mean_order_flips']],
            'reliable_reversals': [f"{r['a']}/{r['b']}" for r in rows if r['reliable_reversal']],
            'significance_changes': [f"{r['a']}/{r['b']}" for r in rows if r['significance_gained_or_lost']],
            'pairs': rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    primary = {m: load_primary(m) for m in ORIGINAL + ADDITIONAL}
    secondary = {m: load_secondary(m) for m in ORIGINAL}
    results = []
    for metric, lower in (('FID', True), ('R1', False)):
        results.append(analyze('primary_T2M_MDM_linked', primary, metric, lower))
        results.append(analyze('secondary_T2M_GPT_checkpoint', secondary, metric, lower))
    summary = {'protocol': 'paired t_19 over 20 shared native draws; Bonferroni over 2 x pairs per reader/metric',
               'inputs_read_only': [str(PREV), str(ADD), str(MD)],
               'note': 'MDM, MotionDiffuse and DiP emit 1,024 outputs, others 4,640; pairs with unequal counts are flagged',
               'results': results}
    (OUT / 'analysis.json').write_text(json.dumps(summary, indent=1, default=bool))
    for r in results:
        print(f"\n{r['reader']} {r['metric']}: tau={r['kendall_tau_raw_vs_canonical']:.3f}")
        print('  raw  :', r['ranking_raw'])
        print('  canon:', r['ranking_canonical'])
        print('  mean flips:', r['mean_order_flips'])
        print('  reliable reversals:', r['reliable_reversals'])
        print('  significance changes:', r['significance_changes'])


if __name__ == '__main__':
    main()
