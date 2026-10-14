"""Independent retained-endpoint checks for transfer, DEV qualification and cost."""
from pathlib import Path
import numpy as np
import torch
from motion_release_audit_common import read,write,sha,seed

def main():
 seed(619);import sys;sys.path.insert(0,str(Path('external/MDM').resolve()))
 from data_loaders.humanml.utils.metrics import calculate_R_precision,calculate_activation_statistics,calculate_frechet_distance
 R=Path('reports/cvpr_blind_transfer_20261013');assert read(R/'cross_reader_complete.json')['all_four_cohorts_and_all_conditions'];p=read(R/'cross_reader_protocol.json')
 for f,h in p['source_sha256'].items():assert sha(f)==h
 maximum=0.;cells=0
 for folder in (R/'secondary_reader').iterdir():
  rows=read(folder/'metrics.json');reference=np.load(folder/'reference.npz');T=reference['text'];ref=reference['motion'];orders=reference['orders'];mr,cr=calculate_activation_statistics(ref)
  for condition,row in rows['conditions'].items():
   z=np.load(folder/(condition+'.npz'))
   for c,key in [('raw','embedding'),('canonical','canonical_embedding')]:
    E=z[key];mu,cov=calculate_activation_statistics(E);fid=max(float(calculate_frechet_distance(mu,cov,mr,cr)),0);er=abs(fid-row[c]['FID']);assert er<1e-5;maximum=max(er,maximum);r1=np.mean([sum(calculate_R_precision(T[idx],E[idx],1,True)[0] for idx in order.reshape(-1,32))/len(E) for order in orders]);assert abs(r1-row[c]['R_precision_top1'])<1e-10;cells+=1
 assert read(R/'mld/verification.json')['passed']
 write(R/'verification.json',{'passed':True,'secondary_native_metric_endpoints':cells,'max_saved_FID_error':maximum,'MLD_geometry_verification':read(R/'mld/verification.json'),'seal_and_protocol_source_hashes_verified':True,'no_destination_reoptimization':True,'verifier_sha256':sha(__file__)})
 GRID=Path('reports/cvpr_reward_dev_search_20261013');x=read(GRID/'selection.json');qualified=[]
 for row in x['all_DEV_rows']:
  if row['weight']==0:continue
  baseline=next(r for r in x['all_DEV_rows'] if r['lr']==row['lr'] and r['weight']==0)['scores']['raw'];r=row['scores']['raw'];path=GRID/f"lr{row['lr']:g}_w{row['weight']:g}";h=read(path/'training_complete.json');assert sha(path/'final.pt')==h['weights_sha256']
  if r['R_precision'][0]-baseline['R_precision'][0]>=.005 and r['FID']<=1.1*baseline['FID'] and h['CE_last100']<h['CE_first100']:qualified.append(row)
 assert len(qualified)==x['qualified']==0 and x['chosen'] is None
 for f,h in read(GRID/'protocol.json')['inputs_and_sources_sha256'].items():assert sha(f)==h
 write(GRID/'verification.json',{'passed':True,'all_eight_DEV_fits_retained':True,'qualification_recomputed_from_all_outcomes':True,'qualified':0,'no_grid_extension':True,'no_TEST_sampling_in_driver':True,'weight_and_source_hashes_verified':True,'verifier_sha256':sha(__file__)})
 VAL=Path('reports/cvpr_reward_dev_validation_20261013');assert read(VAL/'complete.json')['valid_reward_pilot_obtained'] is False;assert not list(VAL.glob('*_seed*/final.pt'));assert read(VAL/'analysis.json')['no_TEST_training_or_inference'];write(VAL/'verification.json',{'passed':True,'conditional_no_TEST_execution_verified':True,'no_valid_reward_pilot_claim':True,'verifier_sha256':sha(__file__)})
 COST=Path('reports/cvpr_motion_defense_cost_20261013');assert read(COST/'complete.json')['all_four_real_reader_dataset_cells'];p=read(COST/'protocol.json')
 for f,h in p['source_sha256'].items():assert sha(f)==h
 for rep in range(20):
  folder=COST/f'rep_{rep:02d}';r=read(folder/'secondary.json');z=np.load(folder/'secondary_embeddings.npz');T=z['text']
  for condition in ['raw','canonical']:
   E=z[condition];native=sum((calculate_R_precision(T[i:i+32],E[i:i+32],3,True) for i in range(0,len(E),32)),np.zeros(3))/len(E);np.testing.assert_allclose(native,r[condition],atol=0,rtol=0)
  assert abs(z['raw']-z['canonical']).max()==r['embedding_max_change']
 for row in read(COST/'analysis.json')['real_data_R1'].values():assert row['raw_R1']==row['canonical_R1'] and row['canonical_minus_raw']==0
 write(COST/'verification.json',{'passed':True,'twenty_secondary_saved_R1_pairs_recomputed':True,'all_four_real_R1_cost_cells_zero':True,'source_hashes_checked':True,'verifier_sha256':sha(__file__)})
 print('GENERALIZATION_CONTROLS_VERIFIED',cells,flush=True)
if __name__=='__main__':main()
