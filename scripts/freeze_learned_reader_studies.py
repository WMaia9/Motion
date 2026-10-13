"""Independently verify and freeze the two TRAIN/DEV intervention studies."""
import ast
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from verify_signspark_studies import sha,matrices
from audit_reviewed_manuscript_evidence import metric

LEARNED=Path('reports/cvpr_learned_equivalence_20261011')
READERS=Path('reports/cvpr_canonical_readers_20261011')


def read(p):return json.loads(Path(p).read_text())
def store(root,name,value):(root/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def pair(x,y,c):
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(c).all()
    assert x.shape==y.shape and x.ndim==2 and x.shape[1]==240
    assert c.shape==(len(x),40,3) and abs(c).max()<=.350001
    assert np.array_equal(x[:,60:150],y[:,60:150]) and (c[:,10:25]==0).all()
    d=x.reshape(-1,40,6);a=d[...,:3]
    rebuilt=np.concatenate([np.exp(c[...,0:1])*a,np.exp(c[...,1:2])*d[...,3:]+c[...,2:3]*a],-1).reshape(x.shape)
    difference=float(abs(rebuilt-y).max());assert difference<2e-6
    angle=float(np.degrees(Rotation.from_matrix((matrices(x).transpose(0,1,3,2)@matrices(y)).reshape(-1,3,3)).magnitude()).max())
    assert angle<.01
    return difference,angle


def archive(root,producers,logs):
    mapping={p.stem:p for p in Path('scripts').glob('*.py')};pending=list(producers);keep=set(producers)
    while pending:
        for node in ast.walk(ast.parse(mapping[pending.pop()].read_text())):
            modules=[v.name for v in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                name=module.split('.')[0]
                if name in mapping and name not in keep:keep.add(name);pending.append(name)
    folder=root/'scripts';folder.mkdir(exist_ok=True)
    for name in keep:shutil.copy2(mapping[name],folder/mapping[name].name)
    (root/'logs').mkdir(exist_ok=True)
    for name in logs:
        p=Path('/tmp')/name
        if p.exists():shutil.copy2(p,root/'logs'/name)
    store(root,'source_provenance.json',{'producers':producers,'import_closure':sorted(keep),
        'all_archived_scripts_match_live':all(sha(mapping[n])==sha(folder/mapping[n].name) for n in keep)})


def learned():
    root=LEARNED;assert not (root/'manifest_sha256.json').exists()
    protocol=read(root/'protocol.json')
    assert sha(Path('scripts/run_learned_equivalence.py'))==protocol['code_sha256']
    for p,h in protocol['inputs'].items():assert sha(Path(p))==h
    assert read(root/'training/complete.json')['all_first_epoch_batches_matched']
    assert read(root/'evaluation/complete.json')['canonical_hypotheses_unchanged']
    baseline=read(root/'evaluation/baseline.json');references=baseline['references']
    source={}
    for path in (root/'generated/dev').glob('*.npz'):
        with np.load(path) as z:source.update({k:z[k].copy() for k in z.files})
    assert len(source)==519
    count=metrics=0;max_error=max_angle=0.
    for path in sorted((root/'evaluation').glob('seed*_poses.npz')):
        result=read(path.with_name(path.name.removesuffix('_poses.npz')+'.json'))
        assert result['canonical']['hypotheses']==baseline['hypotheses']
        for kind in ['raw','canonical']:
            r=result[kind];assert abs(metric(references,r['hypotheses'])-r['scores']['bleu']['bleu4'])<1e-10;metrics+=len(references)
        with np.load(path) as z:
            assert len(z.files)==1038
            for i in range(519):
                error,angle=pair(source[f'clip{i:05d}'],z[f'attacked{i:04d}'],z[f'coefficient{i:04d}'])
                max_error=max(max_error,error);max_angle=max(max_angle,angle);count+=1
    public=read(root/'public_transfer/baseline.json')
    for path in sorted((root/'public_transfer').glob('seed*_poses.npz')):
        r=read(path.with_name(path.name.removesuffix('_poses.npz')+'.json'));assert r['canonical']['hypotheses']==public['hypotheses']
        for kind in ['raw','canonical']:
            score=r[kind];assert abs(metric(public['references'],score['hypotheses'])-score['scores']['bleu']['bleu4'])<1e-10;metrics+=128
        with np.load(path) as z:
            for i in range(128):
                error,angle=pair(*[z[f'{tag}{i:04d}'] for tag in ['source','attacked','coefficient']])
                max_error=max(max_error,error);max_angle=max(max_angle,angle);count+=1
    store(root,'verification.json',{'passed':True,'rotation_preserving_pairs_reconstructed':count,
        'maximum_coefficient_reconstruction_error':max_error,'max_rotation_error_deg_independent':max_angle,
        'standalone_BLEU_sentence_evaluations':metrics,'frozen_generator_and_teacher_hashes_match':True,
        'all_canonical_hypotheses_unchanged':True,'new_TEST_use':False})
    report='''# Camada aprendida com rotações preservadas

Treinamos uma camada limitada de escala/cisalhamento sobre movimentos de um
DDPM congelado. O leitor e o gerador permanecem congelados; apenas a camada
aprende com captions de TRAIN. Ela recebe movimento e texto de origem, sem ID
do clip e sem poses de referência. Inferência em DEV é uma passagem, sem
otimização por vídeo. Isso isola uma exploração aprendida do canal, construída
explicitamente para estudá-lo; não mede sua frequência em treino convencional.

7096 movimentos TRAIN foram gerados de texto e duração. Três seeds42/123/777,
20 épocas, lr0,001, bound0,35, mesmo ordenamento de batches por seed, último
checkpoint. Controles: loss no leitor canônico e camada sem texto. DEV519.

Texto+movimento/loss bruta: BLEU6,0871 → média14,2404, ganho8,1533; IC pareado
condicional[7,0992;9,2695], sensibilidade seed+clip[7,0214;9,3580]. Todas as
hipóteses canônicas permanecem iguais. O controle de loss canônica tem ganho
bruto médio0,0058. Movimento sem texto tem ganho1,0416. Em494captions DEV sem
correspondência exata em TRAIN, o ganho da camada completa permanece7,9187.

Trocar só o texto que condiciona a camada, preservando as rotações de origem,
leva a9,0792BLEU contra a caption errada (baseline1,3331), enquanto o BLEU contra
a caption original cai de6,0871 para3,5579. Todas as519captions trocadas diferem.
Esse resultado identifica influência do texto pelas coordenadas redundantes.

Transferência sem ajuste para128movimentos do SignSparK público, de um replay
nativo texto+duração: BLEU0,7381 → média12,6456; ganho11,9075 com IC pareado
[8,7772;15,6062]. A avaliação canônica permanece0,7381. O replay público é fraco
e não é apresentado como reprodução de scores SOTA publicados.

Os números completos, a incerteza entre apenas três seeds de camada e as
limitações da inicialização única estão em analysis.json. O piloto anterior
de fine-tuning convencional não confirmou inflação seletiva; sua evidência
negativa está preservada em cvpr_reviewer_followup_20261011.

Protocolo, fontes, pesos e logs estão registrados. Uma falha inicial de device
ocorreu antes do primeiro passo de otimização; a correção preserva hiperparâmetros
e inputs e está em sources/device_amendment.json. Nenhum resultado foi apagado
para escolher uma configuração positiva. Canonicalização fecha o canal em
aritmética exata; pequenos erros numéricos de matriz são medidos separadamente.
'''
    (root/'RESULTADOS.md').write_text(report)
    archive(root,['run_learned_equivalence','evaluate_learned_public_transfer','analyze_learned_equivalence','freeze_learned_reader_studies'],
        ['cvpr_learned_prepare_20261011.log','cvpr_learned_train_20261011.log','cvpr_learned_train_fixed_20261011.log',
         'cvpr_learned_evaluate_20261011.log','cvpr_learned_public_transfer_20261011.log',
         'cvpr_learned_public_transfer_fixed_20261011.log','cvpr_learned_analysis_20261011.log'])
    print('LEARNED_STUDY_VERIFIED',count,metrics,max_angle,flush=True)


def readers():
    root=READERS;assert not (root/'manifest_sha256.json').exists()
    protocol=read(root/'protocol.json');assert sha(Path('scripts/run_canonical_readers.py'))==protocol['code_sha256']
    for p,h in protocol['checkpoints'].items():assert sha(Path(p))==h
    for r in read(root/'sources/downloads.json'):assert sha(Path(r['path']))==r['sha256']
    assert read(root/'training_complete.json')['paired_minibatches_matched']
    metrics=count=0;max_error=max_angle=0.
    for corpus in ['How2Sign','CSL-Daily']:
        char=corpus=='CSL-Daily';audit=read(root/corpus/'data_audit.json')
        assert not set(audit['train_ids'])&set(audit['dev_ids'])
        for seed in [42,123,777]:
            for arm in ['raw','canonical']:
                folder=root/corpus/f'seed{seed}'/arm
                for name in ['native_scores','canonical_scores']:
                    r=read(folder/(name+'.json'));assert abs(metric(r['references'],r['hypotheses'],char)-r['scores']['bleu']['bleu4'])<1e-10;metrics+=len(r['references'])
                r=read(folder/'wrong_target_attack.json')
                for name in ['baseline','after']:
                    score=r[name];assert abs(metric(score['references'],score['hypotheses'],char)-score['scores']['bleu']['bleu4'])<1e-10;metrics+=64
                assert abs(metric(r['targets'],r['after']['hypotheses'],char)-r['after_target_bleu4'])<1e-10
                if arm=='canonical':assert r['baseline']['hypotheses']==r['after']['hypotheses']
                with np.load(folder/'attack_poses.npz') as z:
                    for i in range(64):
                        error,angle=pair(*[z[f'{tag}{i:04d}'] for tag in ['source','attacked','coefficient']])
                        max_error=max(max_error,error);max_angle=max(max_angle,angle);count+=1
    store(root,'verification.json',{'passed':True,'fresh_attack_pairs_reconstructed':count,
        'max_rotation_error_deg_independent':max_angle,'maximum_reconstruction_error':max_error,
        'standalone_BLEU_sentence_evaluations':metrics,'all_canonical_attacks_leave_hypotheses_unchanged':True,
        'TRAIN_DEV_clip_overlap':0,'source_download_and_checkpoint_hashes_match':True,'no_TEST_use':True})
    summary='''# Leitores canônicos em How2Sign e CSL

Doze fine-tunes pareados: três seeds, dois caminhos de input e dois corpora.
Todos partem do mesmo checkpoint público por corpus, usam cinco épocas,
lr1e−5, TRAIN completo válido e último checkpoint, sem seleção por BLEU.
How2Sign:29630TRAIN/1650DEV. CSL:18201TRAIN/1064DEV. As contagens são as da
release reotimizada com os filtros nativos, não as do dataset original.

How2Sign: médias4,0770(bruto ajustado) e4,0150(canônico ajustado); diferença
−0,0620BLEU, intervalo exploratório seed+VIDEO_ID[−0,2646;0,1424]. A margem
exploratória0,2não estabelece não inferioridade. Contra o leitor original bruto,
a diferença é−0,2565. Não declarar defesa sem perda geral de acurácia.

CSL: médias7,0023(bruto ajustado) e7,1600(canônico ajustado); diferença+0,1577,
intervalo seed+clip[−0,1529;0,5316]. Contra o leitor público bruto6,1956, a
mudança é+0,9644com intervalo[0,4444;1,5541]. Margens são exploratórias e não
validam correlação com compreensão linguística humana.

Ataques novos contra cada leitor: DEV64porcorpus, alvo trocado, 80passos,
bound0,35. Todos os384pares atacados dos leitores canônicos mantêm hipóteses
idênticas às originais; BLEU contra o alvo errado pode ser não zero por n-grams
comuns na baseline. Os leitores brutos continuam redirecionáveis. Dados,
seleções, hipóteses e matrizes foram verificados independentemente.

Protocolo e comparações completas em protocol.json e analysis.json. Bootstrap
How2Sign agrupa115VIDEO_ID; esses IDs não são identidades de signers. Três
seeds de fine-tuning não estimam a variabilidade de todo o pré-treinamento.
Nenhuma nova inferência TEST ocorreu nesta fase de desenvolvimento.
'''
    (root/'RESULTADOS.md').write_text(summary)
    archive(root,['download_canonical_reader_training','run_canonical_readers','analyze_canonical_readers','freeze_learned_reader_studies'],
        ['cvpr_canonical_reader_downloads_20261011.log','cvpr_canonical_reader_train_20261011.log',
         'cvpr_canonical_reader_attacks_20261011.log','cvpr_canonical_reader_analysis_20261011.log'])
    print('CANONICAL_READER_STUDY_VERIFIED',count,metrics,max_angle,flush=True)


def main():
    for root,verify in [(LEARNED,learned),(READERS,readers)]:
        verify()
        manifest={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
                  if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest_sha256.json'}
        store(root,'manifest_sha256.json',manifest)
        print('FROZEN_NEW_STUDY',str(root),len(manifest),sha(root/'manifest_sha256.json'),flush=True)


if __name__=='__main__':main()
