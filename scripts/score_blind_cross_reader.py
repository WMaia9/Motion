"""Transfer sealed blind attacks to an untargeted official evaluator checkpoint."""
from pathlib import Path
import numpy as np
import torch
from humanml_audit_common import configure,load_evaluator,encode_raw,text_embeddings,metrics,read,write,sha
R=Path('reports/cvpr_blind_transfer_20261013')
COHORTS={
 'mdm42':Path('reports/cvpr_humanml3d_blind_fid_20261012'),
 'mdm4242':Path('reports/cvpr_humanml_blind_repetitions_20261012/cohort_1'),
 'mdm7777':Path('reports/cvpr_humanml_blind_repetitions_20261012/cohort_2'),
 'mld':R/'mld',
}
CK=Path('models/motion_release_audit/gpt_evaluator/t2m/text_mot_match/model/finest.tar')

def main():
 R.mkdir(exist_ok=True);assert not (R/'manifest_sha256.json').exists();files=[Path(__file__),Path('scripts/humanml_audit_common.py'),CK]
 for source in COHORTS.values():
  seal=read(source/'attack_outputs_before_TEST_scoring.json');assert seal['all_conditions_completed']
  for p,h in seal['files_sha256'].items():assert sha(p)==h
  files+=[source/'attack_outputs_before_TEST_scoring.json',source/'protocol.json',source/'test_selection.json',source/'test_real.npz']
 protocol={'scope':'no optimization against destination reader; all sealed fixed blind conditions, both budgets, all four cohorts','reader':'distinct released T2M-GPT checkpoint of same Guo architecture','attacks_already_sealed_before_destination_scoring':True,'primary_attack_access':'TRAIN motion statistics and source-reader gradients, no caption/reference consumed before seal','retrieval_orders':[11,97,909,1337,2027],'repetitions_are_overlapping_TEST_sources_not_independent_samples':True,'source_sha256':{str(p):sha(p) for p in files}}
 if (R/'cross_reader_protocol.json').exists():assert read(R/'cross_reader_protocol.json')==protocol
 else:write(R/'cross_reader_protocol.json',protocol)
 configure();movement,motion,text,mean,std=load_evaluator();ck=torch.load(CK,map_location='cpu',weights_only=False)
 for model,key in [(movement,'movement_encoder'),(motion,'motion_encoder'),(text,'text_encoder')]:model.load_state_dict(ck[key],strict=True)
 mean=torch.tensor(mean,device='cuda');std=torch.tensor(std,device='cuda');result={}
 @torch.no_grad()
 def embed(X,L):return np.concatenate([encode_raw(torch.tensor(X[i:i+32],device='cuda'),torch.tensor(L[i:i+32],device='cuda'),movement,motion,mean,std).cpu().numpy() for i in range(0,len(X),32)])
 for name,source in COHORTS.items():
  out=R/'secondary_reader'/name;out.mkdir(parents=True,exist_ok=True)
  if (out/'metrics.json').exists():result[name]=read(out/'metrics.json');continue
  with np.load(source/'test_real.npz') as z:reference=embed(z['raw'],z['lengths'])
  with np.load(source/'test_mdm.npz') as z:L=z['lengths'].copy()
  T=text_embeddings(read(source/'test_selection.json'),text);orders=np.array([np.random.default_rng(s).permutation(len(L)) for s in [11,97,909,1337,2027]])
  np.savez_compressed(out/'reference.npz',motion=reference,text=T,orders=orders);scores={}
  from humanml_audit_common import canonical
  for condition in read(source/'protocol.json')['conditions']:
   with np.load(source/'attacks'/(condition+'.npz')) as z:X=z['raw']
   E=embed(X,L);C=np.concatenate([canonical(torch.tensor(X[i:i+32])).numpy() for i in range(0,len(X),32)]);CE=embed(C,L)
   scores[condition]={'raw':metrics(E,T,reference,orders),'canonical':metrics(CE,T,reference,orders)};np.savez_compressed(out/(condition+'.npz'),embedding=E,canonical_embedding=CE)
  baseline=scores['identity']['raw']['FID'];basecan=np.load(out/'identity.npz')['canonical_embedding'];cmax=0.
  for condition,row in scores.items():
   row['FID_change']=row['raw']['FID']-baseline;row['relative_FID_reduction']=1-row['raw']['FID']/baseline
   change=float(abs(np.load(out/(condition+'.npz'))['canonical_embedding']-basecan).max());assert change<2e-4;cmax=max(cmax,change)
  primary=read(source/'analysis.json');entry={'conditions':scores,'canonical_embedding_max_change':cmax,'source_reader_metrics':primary['conditions'],'untargeted_destination_reader':True,'no_destination_gradient':True};write(out/'metrics.json',entry);result[name]=entry;print('BLIND_TRANSFER_SCORED',name,{k:v['relative_FID_reduction'] for k,v in scores.items()},flush=True)
 write(R/'cross_reader_analysis.json',result);write(R/'cross_reader_complete.json',{'all_four_cohorts_and_all_conditions':True,'no_reoptimization':True});print('BLIND_CROSS_READER_COMPLETE',flush=True)
if __name__=='__main__':main()
