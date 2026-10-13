"""Paired within-generator effects and existing raw-reader training evidence."""
import json
from pathlib import Path

import numpy as np
from analyze_signspark_geometry_dev import statistics, b4
from verify_signspark_studies import sha

ROOT = Path('reports/cvpr_reviewer_followup_20261011')
OLD = Path('reports/cvpr_evaluation_confirmation_20261011')
REPLICATES = 20000


def read(path):
    return json.loads(path.read_text())


def bootstrap(records):
    first = next(iter(records.values()))
    refs = first['references']; ids = first['clip_ids']; n = len(refs)
    weights = np.random.default_rng(42).multinomial(n, np.ones(n)/n, size=REPLICATES)
    points, samples = {}, {}
    for name, record in records.items():
        assert record['references'] == refs and record['clip_ids'] == ids
        stats = statistics(refs, record['hypotheses'])
        points[name] = b4(stats.sum(0))
        samples[name] = np.asarray([b4(row) for row in weights @ stats])
        assert abs(points[name] - record['scores']['bleu']['bleu4']) < 1e-10
    return points, samples


def interval(point, samples, comparisons=1):
    tail = 2.5/comparisons
    return {'delta_bleu4': float(point), 'ci95': np.percentile(samples, [2.5, 97.5]).tolist(),
            'bonferroni_family_interval': np.percentile(samples, [tail, 100-tail]).tolist(),
            'family_comparisons': comparisons}


def main():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'manifest_sha256.json').exists()
    output = {'analysis_status': 'post-review exploratory analysis of previously observed outputs',
              'replicates': REPLICATES, 'seed': 42, 'bootstrap_unit': 'paired clip',
              'training_variability_estimated': False, 'input_hashes': {},
              'within_generator': {}, 'existing_raw_reader_training': {}}
    arms = ['geometry', 'velocity', 'physical_ce', 'physical_generated_near']
    pairs = [('canonical', 'raw'), ('gauge_scale090', 'raw'), ('gauge_scale110', 'raw'),
             ('gauge_shear010', 'raw'), ('gauge_scale110', 'gauge_scale090')]
    for arm in arms:
        records = {}
        for kind in {x for pair in pairs for x in pair}:
            path = OLD/'rankings'/arm/(kind+'_scores.json')
            records[kind] = read(path); output['input_hashes'][str(path)] = sha(path)
        points, samples = bootstrap(records)
        output['within_generator'][arm] = {'clips': len(records['raw']['references']), 'scores': points,
            'comparisons': {a+' minus '+b: interval(points[a]-points[b], samples[a]-samples[b], 20)
                            for a, b in pairs}}
    # Both matched raw/canonical objectives already ran before this review.
    for label, folder, arms in [
        ('original_semantic', Path('reports/cvpr_physical_semantic_ddpm_20261010'),
         ['geometry', 'semantic_raw', 'semantic_physical']),
        ('fixed_prefix_contrastive', Path('reports/cvpr_fixed_prefix_diffusion_20261010'),
         ['raw_generated_near', 'physical_generated_near']),
    ]:
        records = {}
        for arm in arms:
            path = folder/(arm+'_scores.json'); r = read(path)
            output['input_hashes'][str(path)] = sha(path)
            for kind in ['raw', 'canonical']:
                records[arm+':'+kind] = {'clip_ids': r['clip_ids'], 'references': r['references'],
                    'hypotheses': r[kind+'_hypotheses'], 'scores': r[kind+'_scores']}
        points, samples = bootstrap(records)
        left = 'semantic_raw' if label == 'original_semantic' else 'raw_generated_near'
        right = 'geometry' if label == 'original_semantic' else 'physical_generated_near'
        comparisons = {}
        for kind in ['raw', 'canonical']:
            a = left+':'+kind; b = right+':'+kind
            comparisons[kind+'_objective_gain'] = interval(points[a]-points[b], samples[a]-samples[b], 6)
        a,b,c,d = left+':raw',right+':raw',left+':canonical',right+':canonical'
        comparisons['excess_raw_gain_difference_in_differences'] = interval(
            (points[a]-points[b])-(points[c]-points[d]),
            (samples[a]-samples[b])-(samples[c]-samples[d]), 6)
        output['existing_raw_reader_training'][label] = {
            'clips': len(next(iter(records.values()))['references']), 'scores': points,
            'comparisons': comparisons, 'one_training_seed': True,
            'warning': 'original_semantic uses different wrong-caption decoder prefixes; fixed_prefix comparison removes this confound'}
    (ROOT/'paired_effects.json').write_text(json.dumps(output, indent=2)+'\n')
    lines = ['# Análise pareada após o parecer', '',
        'Análise exploratória de outputs existentes; 20.000 reamostragens pareadas, seed 42. '
        'Intervalos condicionados aos modelos treinados, sem medir variação entre seeds. '
        'Famílias declaradas: 20 contrastes nas quatro variantes; seis contrastes nos dois pilotos de treinamento.', '',
        '| Variante | Transformação − raw | Δ BLEU-4 | IC 95% | Intervalo Bonferroni (20) |',
        '|---|---|---:|---|---|']
    for arm, result in output['within_generator'].items():
        for pair, effect in result['comparisons'].items():
            if pair == 'canonical minus raw': continue
            fmt = lambda v: '['+', '.join(f'{x:.4f}' for x in v)+']'
            lines.append(f"| {arm} | {pair} | {effect['delta_bleu4']:.4f} | {fmt(effect['ci95'])} | {fmt(effect['bonferroni_family_interval'])} |")
    lines += ['', '## Treinamento com leitor bruto já executado', '',
        'O parecer identifica uma lacuna no manuscrito, mas o repositório já contém dois pilotos com loss no leitor bruto. '
        'Não são evidência nova nem confirmação independente. A comparação com prefixo fixo é a mais adequada. '
        'A primeira comparação tinha um confundimento por prefixos distintos nos negativos.', '']
    for label, result in output['existing_raw_reader_training'].items():
        lines.append(f"- {label}: "+json.dumps(result['comparisons'], ensure_ascii=False))
    lines += ['', 'Esses contrastes não provam inflação de resultados publicados nem um ranking robustamente invertido. '
        'Efeito pareado dentro de uma variante, ganho por treinamento e ranking entre sistemas são hipóteses diferentes.', '']
    (ROOT/'EFEITOS_PAREADOS.md').write_text('\n'.join(lines))
    print('PAIRED_EFFECTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
