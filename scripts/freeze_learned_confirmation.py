"""Independent verification and immutable snapshot of fixed TEST confirmation."""
import json
from pathlib import Path
import numpy as np
from freeze_learned_reader_studies import pair,archive
from audit_reviewed_manuscript_evidence import metric
from run_signspark_geometry_dev import project
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_learned_confirmation_20261012')
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (ROOT/'manifest_sha256.json').exists()
 proto=read(ROOT/'protocol.json')
 assert sha('scripts/run_learned_confirmation.py')==proto['code_sha256']
 for p,h in proto['checkpoint_hashes'].items():assert sha(p)==h,p
 for p,h in proto['discovery_manifest_hashes'].items():assert sha(Path(p)/'manifest_sha256.json')==h,p
 original=Path('reports/cvpr_evaluation_confirmation_20261011/rankings/geometry/poses.npz')
 assert sha(original)==proto['frozen_TEST_motion_sha256']
 with np.load(original) as z:xs=[project(z[f'clip{i:04d}']) for i in range(642)]
 baseline=read(ROOT/'learned/baseline.json');count=metrics=0;err=angle=0.
 for path in sorted((ROOT/'learned').glob('seed*_poses.npz')):
  r=read(path.with_name(path.name.removesuffix('_poses.npz')+'.json'))
  assert r['canonical']['hypotheses']==baseline['hypotheses'] and r['evaluation_optimization_steps']==0
  for k in ['raw','canonical']:
   d=r[k];assert abs(metric(d['references'],d['hypotheses'])-d['scores']['bleu']['bleu4'])<1e-10;metrics+=len(d['references'])
  assert abs(metric(r['adapter_target_references'],r['raw']['hypotheses'])-r['raw_adapter_target_bleu4'])<1e-10
  with np.load(path) as z:
   for i,x in enumerate(xs):
    e,a=pair(x,z[f'attacked{i:04d}'],z[f'coefficient{i:04d}']);err=max(err,e);angle=max(angle,a);count+=1
 assert count==7704
 invariant=0
 for corpus in ['How2Sign','CSL-Daily']:
  for path in (ROOT/'readers'/corpus).glob('*.json'):
   d=read(path)
   if 'hypotheses' in d:
    assert abs(metric(d['references'],d['hypotheses'],corpus=='CSL-Daily')-d['scores']['bleu']['bleu4'])<1e-10;metrics+=len(d['references'])
  for seed in [42,123,777]:
   a=read(ROOT/'readers'/corpus/f'seed{seed}_canonical_native.json');b=read(ROOT/'readers'/corpus/f'seed{seed}_canonical_fixed_gauge.json')
   assert a['hypotheses']==b['hypotheses'];invariant+=len(a['hypotheses'])
 assert invariant==10035
 checks={'passed':True,'independent_saved_rotation_pairs':count,'max_rotation_error_deg':angle,'max_coefficient_reconstruction_error':err,
 'standalone_BLEU_sentence_evaluations':metrics,'canonical_reader_fixed_gauge_hypothesis_pairs':invariant,'unchanged_checkpoints':len(proto['checkpoint_hashes']),
 'no_evaluation_optimization':True,'all_canonical_hypotheses_unchanged':True,'prior_project_TEST_use_disclosed':True}
 (ROOT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
 (ROOT/'RESULTADOS.md').write_text('''# Confirmação fixa nos splits TEST\n\nPesos, configurações e resultados de desenvolvimento foram congelados antes\nda nova inferência. Não houve treinamento, otimização por clip ou seleção de\ncheckpoint nesta etapa. Os splits públicos já haviam sido usados em fases\nanteriores do projeto; não são holdouts intocados durante todo o projeto.\n\nPHOENIX: 642 clips. A camada texto+movimento/loss bruta eleva BLEU de 6,3905\npara média 15,2053 (seeds 15,0190/15,1964/15,4005). Ganho 8,8148; IC pareado\ncondicional [7,8196;9,8706]. As hipóteses canônicas são idênticas. Controle de\nloss canônica: ganho −0,1347, intervalo contendo zero. Controle sem texto:\nganho 1,2091. Excluindo 32 captions já vistas exatamente em TRAIN, restam\n610 clips com ganho 8,3745. Trocar somente o texto da camada leva a 9,5961\ncontra a caption errada e 4,2577 contra a original, sem mudar as rotações.\n\nLeitores adaptados: How2Sign TEST 2183 clips, diferença canônico−bruto\n+0,0991; CSL TEST 1162, −0,0217. Os intervalos pareados contêm zero. As margens\nexploratórias passam na comparação pareada TEST, mas How2Sign não demonstra\nnão inferioridade contra o leitor público original no intervalo seed+grupo,\ne o resultado DEV também exige cautela. Não afirmar defesa universal sem custo.\n10.035 hipóteses dos leitores canônicos permanecem iguais sob encoding\nequivalente fixo. Matrizes e aritmética BLEU foram verificadas separadamente.\n\nanalysis.json contém resultados completos e dois níveis de incerteza. Três\nseeds de camada/adaptação não estimam variabilidade do pré-treinamento inteiro.\nA camada foi projetada explicitamente para explorar redundância; o piloto\nconvencional anterior não confirmou exploração espontânea.\n''')
 archive(ROOT,['run_learned_confirmation','analyze_learned_confirmation','freeze_learned_confirmation'],['cvpr_learned_TEST_confirmation_20261012.log','cvpr_readers_TEST_confirmation_20261012.log','cvpr_TEST_confirmation_analysis_20261012.log'])
 manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest_sha256.json'}
 (ROOT/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('CONFIRMATION_FROZEN',checks,len(manifest),sha(ROOT/'manifest_sha256.json'),flush=True)
if __name__=='__main__':main()
