"""Verify new review experiments, write reports/figures and freeze their bytes."""
import ast
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_reviewed_manuscript_evidence import metric
from analyze_signspark_geometry_dev import statistics, b4
from verify_signspark_studies import sha, matrices

ROOT = Path('reports/cvpr_reviewer_followup_20261011')


def read(p): return json.loads(Path(p).read_text())
def write(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def verify():
    metrics = poses = 0; max_angle = max_reconstruction = 0.
    for r in read(ROOT/'training_protocol.json')['inputs'].items(): assert sha(Path(r[0])) == r[1]
    assert sha(Path('scripts/run_reviewer_training.py')) == read(ROOT/'training_protocol.json')['code_sha256']
    for seed in [42,123,777]:
        stochastic = []
        for arm in ['geometry','raw_ce','canonical_ce']:
            folder = ROOT/f'training/seed{seed}'
            stochastic.append(read(folder/(arm+'_stochastic.json')))
            raw,can = [read(folder/(arm+'_'+kind+'_scores.json')) for kind in ['raw','canonical']]
            assert raw['clip_ids'] == can['clip_ids'] and len(raw['clip_ids']) == 519
            for r in [raw,can]:
                assert abs(metric(r['references'],r['hypotheses'])-r['scores']['bleu']['bleu4']) < 1e-10
                metrics += 519
            with np.load(folder/(arm+'_poses.npz')) as z:
                assert len(z.files) == 519
                for p in z.files: assert z[p].ndim == 2 and z[p].shape[1] == 240 and np.isfinite(z[p]).all()
            if arm == 'canonical_ce': assert read(folder/(arm+'_gradient.json'))['maximum_vertical_dot'] < 1e-5
        assert stochastic[0] == stochastic[1] == stochastic[2]
    budget = read(ROOT/'attack_ablation/summary.json'); assert len(budget['conditions']) == 60
    for path in sorted((ROOT/'attack_ablation').glob('*_poses.npz')):
        record = read(path.with_name(path.name.removesuffix('_poses.npz')+'.json'))
        refs,targets = budget['references'],budget['targets']
        assert abs(metric(targets,record['raw_hypotheses'])-record['target_bleu4']) < 1e-10
        assert abs(metric(refs,record['raw_hypotheses'])-record['actual_bleu4']) < 1e-10
        assert record['canonical_hypotheses'] == budget['baseline_canonical_hypotheses']
        metrics += 128
        with np.load(path) as z:
            assert len(z.files) == 192
            for i in range(64):
                x,y,c = [z[f'{prefix}{i:04d}'] for prefix in ['source','attacked','coefficient']]
                assert x.shape == y.shape and c.shape == (*x.reshape(-1,40,6).shape[:-1],3)
                assert all(np.isfinite(v).all() for v in [x,y,c])
                assert abs(c).max() <= record['parameters']['budget']+1e-6
                assert np.array_equal(x[:,60:150],y[:,60:150]) and (c[:,10:25] == 0).all()
                d = x.reshape(-1,40,6); a = d[...,:3]
                reconstructed = np.concatenate([np.exp(c[...,0:1])*a,
                    np.exp(c[...,1:2])*d[...,3:]+c[...,2:3]*a],-1).reshape(y.shape)
                max_reconstruction = max(max_reconstruction,float(abs(y-reconstructed).max()))
                angle = float(np.degrees(Rotation.from_matrix((matrices(x).transpose(0,1,3,2)@matrices(y)).reshape(-1,3,3)).magnitude()).max())
                max_angle = max(max_angle,angle); poses += 1
        assert max_reconstruction < 2e-6 and max_angle < .01
    readers = {}
    for arm in ['raw','canonical']:
        real = read(ROOT/f'readers/{arm}/dev_scores.json')
        random = read(ROOT/f'readers/{arm}/random_perturbation_scores.json')
        targeted = read(ROOT/f'readers/{arm}/white_box_wrong_target.json')
        for r in [real,random,targeted['baseline'],targeted['attacked']]:
            assert abs(metric(r['references'],r['hypotheses'])-r['scores']['bleu']['bleu4']) < 1e-10
            metrics += len(r['references'])
        assert abs(metric(targeted['targets'],targeted['attacked']['hypotheses'])-targeted['attacked_target_bleu4']) < 1e-10
        readers[arm] = {'dev_bleu4':real['scores']['bleu']['bleu4'],
            'random_bleu4':random['scores']['bleu']['bleu4'],'random_changed_hypotheses':random['changed_hypotheses'],
            'whitebox_target_bleu4':targeted['attacked_target_bleu4'],'whitebox_changed_hypotheses':targeted['changed_hypotheses'],
            'whitebox_max_rotation_error_deg':targeted['max_rotation_error_deg']}
        if arm == 'canonical':
            assert random['hypotheses'] == real['hypotheses']
            assert targeted['baseline']['hypotheses'] == targeted['attacked']['hypotheses']
    original = read(Path('reports/cvpr_signspark_baseline_20261010/PHOENIX-2014T/original.json'))
    fine = read(ROOT/'readers/canonical/dev_scores.json')
    assert fine['references'] == [x['reference'] for x in read(Path('reports/cvpr_signspark_baseline_20261010/PHOENIX-2014T/references.json'))]
    old_stats = statistics(fine['references'],original['hypotheses']); new_stats = statistics(fine['references'],fine['hypotheses'])
    n = len(fine['references']); weights = np.random.default_rng(42).multinomial(n,np.ones(n)/n,size=20000)
    delta = np.asarray([b4(a)-b4(b) for a,b in zip(weights@new_stats,weights@old_stats)])
    readers['fine_tuning_minus_original'] = {'delta_bleu4':b4(new_stats.sum(0))-b4(old_stats.sum(0)),
        'ci95':np.percentile(delta,[2.5,97.5]).tolist(),
        'warning':'CI containing zero is not a formal equivalence/noninferiority demonstration'}
    write('reader_analysis.json',readers)
    write('verification.json',{'passed':True,'independent_unsmoothed_BLEU_sentence_evaluations':metrics,
        'budget_pairs_reconstructed':poses,'max_rotation_error_deg':max_angle,
        'maximum_reconstruction_error':max_reconstruction,'three_seeds_training_inputs_matched':True,
        'canonical_reader_wrong_target_hypotheses_unchanged':True,
        'all_registered_input_hashes_match':True,'independent_verification_is_saved_output_analysis':True})
    print('NEW_EXPERIMENTS_VERIFIED',metrics,poses,max_angle,flush=True)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
    folder = ROOT/'figures'; folder.mkdir(exist_ok=True)
    budget = read(ROOT/'attack_ablation/summary.json')['conditions']
    fig,axes = plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    bounds = [.01,.05,.1,.35]
    for scope,label in [('global_per_clip','3 parameters / clip'),('per_frame','3 parameters / frame'),('per_frame_joint','3 parameters / frame / joint')]:
        axes[0].plot(bounds,[budget[f'{scope}_bound{x}_steps80']['target_bleu4'] for x in bounds],marker='o',label=label)
    axes[0].set(xlabel='Coefficient bound',ylabel='BLEU-4 against wrong target',title='80 optimization steps')
    steps = [1,5,20,80]
    for bound in bounds:
        axes[1].plot(steps,[budget[f'per_frame_joint_bound{bound}_steps{s}']['target_bleu4'] for s in steps],marker='o',label=f'Bound {bound}')
    axes[1].set(xlabel='Optimization steps',ylabel='BLEU-4 against wrong target',title='Frame and joint parameters')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('PHOENIX DEV, 64 clips; exploratory white-box capability test',fontsize=10)
    fig.savefig(folder/'attack_budget.pdf');fig.savefig(folder/'attack_budget.png',dpi=180);plt.close(fig)
    pairs = read(ROOT/'paired_effects.json')['within_generator']; fig,ax = plt.subplots(figsize=(8,3.7),layout='constrained')
    names = list(pairs); points = [pairs[n]['comparisons']['gauge_scale090 minus raw']['delta_bleu4'] for n in names]
    limits = [pairs[n]['comparisons']['gauge_scale090 minus raw']['bonferroni_family_interval'] for n in names]
    error = np.asarray([[p-l[0],l[1]-p] for p,l in zip(points,limits)]).T
    ax.errorbar(np.arange(len(names)),points,yerr=error,fmt='o',capsize=5)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(np.arange(len(names)),['Geometry','Velocity','Canonical CE','Generated near'])
    ax.set(ylabel='Change in BLEU-4: scale 0.9 − raw',title='Identical rotations, within-generator effects')
    ax.text(.02,.97,'642 PHOENIX TEST clips; paired bootstrap\nFamily intervals: 20 contrasts; one training seed',transform=ax.transAxes,va='top',fontsize=8)
    ax.grid(axis='y',alpha=.2);fig.savefig(folder/'within_generator.pdf');fig.savefig(folder/'within_generator.png',dpi=180);plt.close(fig)


def report():
    analysis = read(ROOT/'training_analysis.json'); readers = read(ROOT/'reader_analysis.json')
    budget = read(ROOT/'attack_ablation/summary.json')['conditions']; profiles = read(ROOT/'feature_profiles.json')['profiles']
    effect = read(ROOT/'paired_effects.json')['within_generator']['physical_generated_near']['comparisons']
    ci = lambda values: '['+', '.join(f'{v:.4f}' for v in values)+']'
    lines = ['# Resposta experimental ao parecer', '',
        'A crítica central é pertinente: um ataque white-box alto não demonstra que geradores publicados exploram '
        'o canal no treinamento. A linha tem uma falha de avaliação demonstrada, mas os novos testes ainda não '
        'demonstram essa exploração espontânea. Este relatório acrescenta evidência e delimita o que falta; '
        'não substitui o manuscrito congelado nem promete aceitação no CVPR.', '',
        '## 1. Efeitos pareados dentro de cada gerador', '',
        'A análise sugerida foi executada: 20.000 reamostragens pareadas, seed42, 642 clips TEST já pontuados; '
        'sem nova inferência. A família contém 20 contrastes. Para **generated near**, escala0,9−raw é '
        f"{effect['gauge_scale090 minus raw']['delta_bleu4']:.4f} BLEU, IC95% {ci(effect['gauge_scale090 minus raw']['ci95'])}, "
        f"intervalo Bonferroni {ci(effect['gauge_scale090 minus raw']['bonferroni_family_interval'])}. "
        'Escala1,1−escala0,9 é '
        f"{effect['gauge_scale110 minus gauge_scale090']['delta_bleu4']:.4f}, intervalo Bonferroni "
        f"{ci(effect['gauge_scale110 minus gauge_scale090']['bonferroni_family_interval'])}. "
        'Há sensibilidade real ao representante nesse sistema; isso não equivale a rankings robustamente invertidos '
        'entre os quatro geradores. Demais valores em `EFEITOS_PAREADOS.md`.', '',
        '## 2. Treinamento com loss do leitor bruto', '',
        'Já existiam dois pilotos com leitor bruto. O piloto com negativos de prefixo fixo tinha ganho bruto '
        '+0,359 e canônico−0,004, mas diferença de diferenças com IC95% incluindo zero. Ele foi preservado na limpeza.', '',
        'Novo experimento: **9 fine-tunes**, três seeds42/123/777, objetivos geometria/CE bruto/CE canônico. '
        'Mesma inicialização DDPM congelada, cinco épocas, lr1e−4, peso CE0,02, TRAIN7096; '
        'mesmos batches, timesteps, dropout e ruído por seed. DEV519 completo, 50 passos DDIM, CFG3, '
        'sem leitura de poses reais pelo sampler, com durações de referência. Não houve seleção por BLEU '
        'nem nova avaliação TEST. São três seeds de fine-tuning, com uma inicialização compartilhada.', '',
        '| Objetivo | BLEU bruto médio | BLEU canônico médio |', '|---|---:|---:|']
    for name,label in [('geometry','Geometria'),('raw_ce','CE no leitor bruto'),('canonical_ce','CE no leitor canônico')]:
        r = analysis['mean_scores'][name]; lines.append(f"| {label} | {r['raw']:.4f} | {r['canonical']:.4f} |")
    comparison = analysis['mean_comparisons']['geometry']; excess = comparison['excess_raw_gain']
    lines += ['', f"CE bruto−geometria: ganho bruto {comparison['raw']['mean_delta_bleu4']:+.4f}; "
        f"ganho canônico {comparison['canonical']['mean_delta_bleu4']:+.4f}. "
        f"Excesso de ganho bruto {excess['mean_delta_bleu4']:+.4f}, IC exploratório seed+clip "
        f"{ci(excess['exploratory_seed_and_clip_bootstrap_ci95'])}. "
        '**O critério “bruto sobe sem melhora canônica” não foi confirmado nesse protocolo.** '
        'Nem isso prova que o canal nunca seja explorado: só testa este DDPM, esta loss/peso e este fine-tune curto. '
        'Não substituir esse resultado por uma busca em TEST até obter um positivo. '
        'Valores por seed e limites do bootstrap de apenas três seeds estão em `training_analysis.json`.', '',
        '## 3. Leitor com entradas canônicas', '',
        'Dois fine-tunes do leitor público PHOENIX, cinco épocas, lr1e−5, seed42, '
        'TRAIN7096, mesma inicialização e último checkpoint: bruto e canônico. '
        f"Ambos alcançam **{readers['canonical']['dev_bleu4']:.4f} BLEU** em DEV519. "
        'É fine-tuning supervisionado, não treinamento do zero. O leitor original pontuava15,2637: '
        f"mudança {readers['fine_tuning_minus_original']['delta_bleu4']:+.4f}, "
        f"IC95% {ci(readers['fine_tuning_minus_original']['ci95'])}. "
        'Um intervalo contendo zero não constitui teste formal de equivalência ou não inferioridade.', '',
        f"Perturbação aleatória sem alvo: bruto {readers['raw']['random_bleu4']:.4f}, "
        f"{readers['raw']['random_changed_hypotheses']} hipóteses alteradas; canônico "
        f"{readers['canonical']['random_bleu4']:.4f}, nenhuma alteração. "
        'Ataque novo white-box contra cada leitor, alvo errado, DEV64, 80passos e bound0,35: '
        f"bruto **{readers['raw']['whitebox_target_bleu4']:.4f}** BLEU contra o alvo errado; "
        f"canônico **{readers['canonical']['whitebox_target_bleu4']:.4f}**, "
        f"{readers['canonical']['whitebox_changed_hypotheses']} hipóteses alteradas. "
        'A canonicalização faz parte do caminho de loss/inferência da defesa. '
        'Isso fecha o canal nesse caso; PHOENIX já era aproximadamente canônico, portanto '
        '**não resolve o custo da defesa no How2Sign/CSL**. Faltam TRAIN e retreino apropriado nesses corpora.', '',
        '## 4. Orçamento, passos e aleatoriedade', '',
        '**60 condições** no mesmo DEV64: três escopos de parâmetros, quatro bounds '
        '0,01/0,05/0,1/0,35, passos1/5/20/80 e um controle aleatório por combinação. '
        'As curvas são exploratórias no conjunto DEV já usado; não são confirmação independente. '
        'Nenhum dos controles aleatórios obteve quadgramas contra os alvos errados — BLEU4zero não significa ausência de alterações nas hipóteses.', '',
        '| Escopo | Bound | Passos | BLEU alvo errado | BLEU referência real |', '|---|---:|---:|---:|---:|']
    for scope, bounds in [('global_per_clip',[.35]),('per_frame',[.35]),('per_frame_joint',[.01,.05,.1,.35])]:
        for bound in bounds:
            r = budget[f'{scope}_bound{bound}_steps80']
            lines.append(f"| {scope} | {bound} | 80 | {r['target_bleu4']:.4f} | {r['actual_bleu4']:.4f} |")
    lines += ['', 'O sucesso alto depende da flexibilidade por frame/junta e do orçamento: '
        'o efeito global é muito menor. Isso restringe a extrapolação para problemas de avaliação real. '
        'Todas as3840comparações de poses das ablações foram reconstruídas por coeficientes salvos; '
        'o verificador independente reconfirma rotações, bounds e métricas. '
        'Curvas em `figures/attack_budget.pdf`.', '',
        '## 5. Por que PHOENIX muda pouco e os outros corpora mudam mais', '',
        'Diagnóstico de128movimentos reais de cada corpus já salvos nos ataques. '
        'RMS médio entre features brutas e canônicas:', '',
        '| Corpus | RMS médio da canonicalização |', '|---|---:|']
    for corpus,key in [('PHOENIX','real_PHX_TEST_subset'),('How2Sign','real_How2Sign_TEST_subset'),('CSL-Daily','real_CSL_TEST_subset')]:
        lines.append(f"| {corpus} | {profiles[key]['clip_mean']['rms_canonicalization_change']:.8g} |")
    lines += ['', 'PHOENIX está próximo da precisão float32; How2Sign/CSL apresentam desvios materiais. '
        'Os quatro DDPMs locais têm RMS≈0,10, maior que os dados reais. '
        'Isso sustenta uma diferença de domínio de entrada, mas não identifica sua origem nem '
        'prova que as coordenadas redundantes contenham informação linguística. '
        'Quantis de norma e cisalhamento por stream ficam em `feature_profiles.json`; '
        'são amostras de até20frames por clip, explicitamente separadas das médias completas.', '',
        '## Ajustes ao parecer e ao texto', '',
        '- A equivalência é exata no modelo matemático; implementações apresentam erros de≈1e−5graus. '
        'Não chamar as matrizes de “bit a bit idênticas”.',
        '- SiLVERScore e BackTranslation2.0 já estão na bibliografia atual. '
        'A publicação SiLVERScore em RANLP2025 permite atualizar a referência de preprint. '
        '[Registro primário](https://aclanthology.org/2025.ranlp-1.54/) e '
        '[BackTranslation2.0](https://arxiv.org/abs/2606.28673).',
        '- Levinson et al., *An Analysis of SVD for Deep Rotation Estimation*, é '
        '[NeurIPS2020](https://proceedings.nips.cc/paper_files/paper/2020/hash/fec3392b0dc073244d38eba1feb8e6b7-Abstract.html), '
        'não CVPR2020. É antecedente a acrescentar sem apresentar Gram–Schmidt como contribuição nova.',
        '- Houve uma avaliação da geração pública SignSparK em DEV nos relatórios preservados, '
        'mas as quatro variantes principais são locais e não constituem comparação de quatro geradores publicados. '
        'A crítica à evidência estreita permanece.',
        '- Reescrever abstract/introdução para abrir com alvo errado e equivalência física; '
        'trocar a reivindicação de ranking pelo efeito pareado demonstrado; inserir barras de erro e curvas de budget; '
        'mover depuração/controleXYZ para o suplemento e retirar jargão de auditoria do corpo. '
        'Não inserir alegação de exploração por treinamento enquanto ela não estiver demonstrada.', '',
        '## O que ainda sustenta risco alto no CVPR principal', '',
        'Há contribuição empírica verificável sobre sensibilidade de uma família pública de leitores6D. '
        'A defesa e as ablações ficaram mais bem sustentadas. Entretanto, falta demonstrar '
        'exploração aprendida em geração plausível, validar a defesa nos corpora com features não canônicas, '
        'avaliar leitores/geradores independentes e aferir relevância linguística. '
        'Esses resultados não autorizam afirmar inflação de números publicados. '
        'A avaliação editorial do parecer é opinião; executar três experimentos não garante aceitação.', '',
        '## Reprodução e conservação', '',
        'Os scripts e logs desta rodada estão arquivados aqui. `training_protocol.json` registra '
        'inputs e decisões antes do treinamento. Todos os produtores escrevem neste report novo e '
        'recusam um diretório com manifesto. Para replay, usar diretório novo e redirecionar ROOT '
        'em todos os módulos que a compartilham, incluindo `run_reviewer_training` e o finalizador. '
        'Executar: análise pareada → perfil de features → generators → readers → ablações → '
        'ataque aos leitores ajustados → análise do treinamento → finalizador. '
        'O uso de TEST aqui se limita a reanálise de outputs/poses previamente salvos; '
        'treinamento e novas inferências ocorreram exclusivamente em TRAIN/DEV. '
        'A limpeza anterior mantém os13reports científicos intactos e registra os artefatos antigos excluídos.', '']
    (ROOT/'RESPOSTA_AO_PARECER.md').write_text('\n'.join(lines))


def archive():
    producers = ['analyze_reviewer_followup','profile_rotation_representations','run_reviewer_training',
                 'run_reviewer_attack_ablation','run_reviewer_reader_attack','analyze_reviewer_training','finalize_reviewer_followup']
    local = {p.stem:p for p in Path('scripts').glob('*.py')}; keep = set(producers); pending = list(keep)
    while pending:
        for node in ast.walk(ast.parse(local[pending.pop()].read_text())):
            modules = [v.name for v in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                name = module.split('.')[0]
                if name in local and name not in keep: keep.add(name); pending.append(name)
    folder = ROOT/'scripts'; folder.mkdir(exist_ok=True)
    for name in sorted(keep): shutil.copy2(local[name],folder/local[name].name)
    logs = ROOT/'logs'; logs.mkdir(exist_ok=True)
    for name in ['paired','profiles','training','readers','training_analysis','ablation','reader_attacks']:
        path = Path('/tmp')/f'cvpr_reviewer_{name}_20261011.log'
        if path.exists(): shutil.copy2(path,logs/path.name)
    write('source_provenance.json',{'producers':producers,'import_closure':sorted(keep),
        'archived_scripts_match_current':all(sha(local[name]) == sha(folder/local[name].name) for name in keep)})


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    verify();figures();report();archive()
    manifest = {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest_sha256.json'}
    write('manifest_sha256.json',manifest)
    print('REVIEWER_FOLLOWUP_FROZEN',len(manifest),sha(ROOT/'manifest_sha256.json'),flush=True)


if __name__ == '__main__':main()
