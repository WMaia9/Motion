"""Real-data accuracy cost, separate from same-motion attack closure."""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from motion_release_audit_common import read,write,sha,seed,evaluator,score_embeddings,canonical
R=Path('reports/cvpr_motion_defense_cost_20261013');OLD=Path('reports/cvpr_motion_prevalence_20261012');TMR=Path('reports/cvpr_tmr_audit_20261013');KIT=Path('reports/cvpr_additional_motion_audit_20261013/kit/real_controls');CK=Path('models/motion_release_audit/gpt_evaluator/t2m/text_mot_match/model/finest.tar')

def retrieval(T,E):
 from data_loaders.humanml.utils.metrics import calculate_R_precision
 return (sum((calculate_R_precision(T[i:i+32],E[i:i+32],3,True) for i in range(0,len(E),32)),np.zeros(3))/len(E)).tolist()

def main():
 R.mkdir(exist_ok=True);files=[Path(__file__),Path('scripts/motion_release_audit_common.py'),CK,OLD/'manifest_sha256.json',TMR/'complete.json',KIT/'complete.json'];p={'scope':'real data raw/C R1 cost, all20saved native cohorts; no attacks or reader fitting','readers':['Guo_primary','Guo_secondary','TMR'],'KIT_reader':'own251-feature Guo, no incompatible263TMRKIT','candidate_pool':32,'primary':'canonical-minus-raw retrieval difference','source_sha256':{str(f):sha(f) for f in files}}
 if (R/'protocol.json').exists():assert read(R/'protocol.json')==p
 else:write(R/'protocol.json',p)
 seed(414);wrapper,mean,std=evaluator();state=torch.load(CK,map_location='cpu',weights_only=False)
 for module,key in [(wrapper.movement_encoder,'movement_encoder'),(wrapper.motion_encoder,'motion_encoder'),(wrapper.text_encoder,'text_encoder')]:module.load_state_dict(state[key],strict=True)
 rows={key:[] for key in ['HumanML_primary','HumanML_secondary','HumanML_TMR','KIT_primary']};emax=0.
 for rep in range(20):
  old=OLD/f'real_controls/rep_{rep:02d}';folder=R/f'rep_{rep:02d}';folder.mkdir(exist_ok=True);primary=read(old/'metrics.json')['results'];tmr=read(TMR/f'real_controls/rep_{rep:02d}/metrics.json')['scores'];kit=read(KIT/f'rep_{rep:02d}/metrics.json')['scores']
  for key,row,names in [('HumanML_primary',primary,('identity','canonical')),('HumanML_TMR',tmr,('raw','canonical')),('KIT_primary',kit,('raw','canonical'))]:rows[key].append({'raw':row[names[0]]['R_precision'],'canonical':row[names[1]]['R_precision']})
  if (folder/'secondary.json').exists():result=read(folder/'secondary.json')
  else:
   with np.load(old/'conditioning.npz') as z:cohort={k:z[k].copy() for k in z.files}
   X=np.load(old/'source_raw.npy');T,E=score_embeddings(wrapper,X,cohort,mean,std);TC,EC=score_embeddings(wrapper,canonical(X),cohort,mean,std);assert np.array_equal(T,TC);result={'raw':retrieval(T,E),'canonical':retrieval(T,EC),'embedding_max_change':float(abs(E-EC).max())};np.savez_compressed(folder/'secondary_embeddings.npz',text=T,raw=E,canonical=EC);write(folder/'secondary.json',result)
  rows['HumanML_secondary'].append(result);emax=max(emax,result['embedding_max_change']);print('REAL_DEFENSE_COST',rep,result,flush=True)
 out={}
 for key,entries in rows.items():
  a=np.array([r['raw'][0] for r in entries]);b=np.array([r['canonical'][0] for r in entries]);d=b-a;se=d.std(ddof=1)/np.sqrt(20);out[key]={'raw_R1':float(a.mean()),'canonical_R1':float(b.mean()),'canonical_minus_raw':float(d.mean()),'family4_t19_CI95':(d.mean()+np.array([-1,1])*t.ppf(1-.05/8,19)*se).tolist(),'repetitions':20}
 write(R/'analysis.json',{'real_data_R1':out,'secondary_embedding_max_change':emax,'different_architecture_TMR_native_cosine':True,'no_free_accuracy_theorem':True});write(R/'complete.json',{'all_four_real_reader_dataset_cells':True});print('REAL_DEFENSE_COST_COMPLETE',out,flush=True)
if __name__=='__main__':main()
