"""All registered release endpoints, paired20-repetition intervals, no selection."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.stats import t
from motion_release_audit_common import R,read,write,sha
METHODS=['mdm50','mld','motionlcm_v1','t2mgpt','t2m']

def statistics(rows,field='results'):
    out={}
    for metric,index in [('FID',None),('R_precision',0),('MM_distance',None)]:
        a=np.array([x[field]['raw'][metric] if index is None else x[field]['raw'][metric][index] for x in rows]);b=np.array([x[field]['canonical'][metric] if index is None else x[field]['canonical'][metric][index] for x in rows]);d=a-b;se=d.std(ddof=1)/np.sqrt(20);family=10 if metric!='MM_distance' else 5
        out['R1' if index==0 else metric]={'raw_mean':float(a.mean()),'canonical_mean':float(b.mean()),'raw_native95_halfwidth':float(1.96*a.std(ddof=0)/np.sqrt(20)),'canonical_native95_halfwidth':float(1.96*b.std(ddof=0)/np.sqrt(20)),'paired_raw_minus_canonical':float(d.mean()),'paired_CI95':(d.mean()+np.array([-1,1])*t.ppf(.975,19)*se).tolist(),'paired_family_CI95':(d.mean()+np.array([-1,1])*t.ppf(1-.05/(2*family),19)*se).tolist(),'family_size':family,'relative_raw_minus_canonical_over_raw':float(d.mean()/a.mean())}
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--secondary',action='store_true');args=parser.parse_args();results={};inputs=[]
    for method in METHODS:
        paths=sorted((R/method).glob('rep_*/metrics.json'));assert len(paths)==20,(method,len(paths));rows=[read(p) for p in paths];assert all(x['results']['raw']['clips']==(1024 if method=='mdm50' else 4640) for x in rows);results[method]=statistics(rows);inputs.extend(paths)
    real=read(R/'real20_summary.json');order={mode:sorted(METHODS,key=lambda m:results[m]['FID'][mode+'_mean']) for mode in ['raw','canonical']}
    report={'replications':20,'methods':results,'real_fixed_controls':real,'FID_ranking':order,'FID_rank_order_changed':order['raw']!=order['canonical'],'same_output_comparisons_not_intentional_reward_exploitation':True,'evaluation_candidate_batch':32,'MDM_generated_cap':1024,'other_generated_and_all_reference_clips':4640,'no_exact_published_score_reproduction_claim':True,'source_sha256':{str(p):sha(p) for p in inputs},'analysis_code_sha256':sha(__file__)}
    if args.secondary:
        second={}
        for method in METHODS:
            paths=sorted((R/'secondary_gpt_reader'/method).glob('rep_*/metrics.json'));assert len(paths)==20,(method,'secondary',len(paths));second[method]=statistics([read(p) for p in paths],field='scores')
        report['secondary_official_GPT_checkpoint_same_architecture']=second
    write(R/'analysis.json',report);print(json.dumps({m:{k:results[m][k] for k in ['FID','R1']} for m in METHODS},indent=2),flush=True)
if __name__=='__main__':main()
