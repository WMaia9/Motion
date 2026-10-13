"""Fresh feature-only inference and standalone Counter/math corpus BLEU audit."""
import hashlib,json,math,shutil,inspect
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from run_signspark_geometry_dev import make_back_translation_model,back_translate,project

PRIOR=Path('reports/cvpr_representation_paper_20261011')
ROOT=Path('reports/csl_bleu_recheck_20261011')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def metric(refs,hyps):
    """Single-reference BLEU-4, character tokens; no imported metric code."""
    assert len(refs)==len(hyps)
    matched=[0]*4;total=[0]*4;rh=hh=0
    for ref,hyp in zip(refs,hyps):
        r=list(ref.replace(' ',''));h=list(hyp.replace(' ',''));rh+=len(r);hh+=len(h)
        for n in range(1,5):
            rc=Counter(tuple(r[i:i+n]) for i in range(max(0,len(r)-n+1)))
            hc=Counter(tuple(h[i:i+n]) for i in range(max(0,len(h)-n+1)))
            matched[n-1]+=sum(min(v,rc[k]) for k,v in hc.items());total[n-1]+=sum(hc.values())
    precision=[m/t if t else 0 for m,t in zip(matched,total)]
    bp=min(1.,math.exp(1-rh/hh)) if hh else 0.
    score=100*bp*math.exp(sum(math.log(p) for p in precision)/4) if min(precision)>0 else 0.
    return {'bleu4':score,'matched_ngrams':matched,'total_ngrams':total,'precisions':precision,'reference_characters':rh,'hypothesis_characters':hh,'brevity_penalty':bp,'exact_sentences_no_spaces':sum(r.replace(' ','')==h.replace(' ','') for r,h in zip(refs,hyps))}

def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    ROOT.mkdir(exist_ok=True);(ROOT/'scripts').mkdir(exist_ok=True)
    torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    registered=json.loads((PRIOR/'sources/registered_inputs.json').read_text())
    prior_hash=sha(PRIOR/'manifest_sha256.json')
    assert prior_hash=='e2f1ee0d7cad334e2aa58c21d15a0dc3828d3db95a2de367e95a720c7359b3f5'
    ck=PRIOR/'checkpoints/CSL-Daily/best.ckpt';ckhash=sha(ck)
    assert ckhash=='5ea787537b11d7b3e14e6a5c3ae711931b8abf939f6a7f25e5a9b8dd3e91a2c0'
    # No caption/reference arrays are loaded until all decoding has completed.
    data={}
    for condition in ['correct_caption','rolled_wrong_caption']:
        with np.load(PRIOR/'CSL-Daily'/(condition+'_poses.npz')) as f:
            data[condition]={'source':[f[f'source{i:04d}'].copy() for i in range(128)],'attacked':[f[f'attacked{i:04d}'].copy() for i in range(128)]}
    assert all(np.array_equal(a,b) for a,b in zip(data['correct_caption']['source'],data['rolled_wrong_caption']['source']))
    model=make_back_translation_model(PRIOR/'checkpoints/CSL-Daily');model.eval()
    # Forbid the model.forward pathway used by the target-conditioned loss.
    def forbidden_forward(*args,**kwargs):raise AssertionError('teacher-forced forward is forbidden during this replay')
    model.forward=forbidden_forward
    original_run=model.run_batch;calls=[]
    def guarded_run(*args,**kwargs):
        batch=kwargs['batch'];assert set(vars(batch))=={'sgn','sgn_mask','sgn_lengths'}
        calls.append({'batch_size':int(batch.sgn.shape[0]),'input_fields':sorted(vars(batch))})
        return original_run(*args,**kwargs)
    model.run_batch=guarded_run
    poses={'source':data['correct_caption']['source'],'canonical_source':[project(x) for x in data['correct_caption']['source']]}
    for condition in data:
        poses[condition]=data[condition]['attacked'];poses['canonical_'+condition]=[project(x) for x in data[condition]['attacked']]
    decoded={}
    for condition,xs in poses.items():
        decoded[condition]=back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=32)
        print('FRESH_DECODE',condition,len(decoded[condition]),flush=True)
    assert sha(ck)==ckhash
    output={'checkpoint_sha256':ckhash,'prior_manifest_sha256':prior_hash,'teacher_forced_forward_forbidden':True,'references_loaded_only_after_all_decoding':True,'inference_calls':calls,'fresh_decoded_sentences':sum(map(len,decoded.values())),'conditions':{}}
    scores={k:json.loads((PRIOR/'CSL-Daily'/(k+'_scores.json')).read_text()) for k in data}
    r=scores['correct_caption'];refs=r['references']
    assert decoded['source']==r['baseline_actual']['hypotheses'] and decoded['canonical_source']==r['canonical_before_actual']['hypotheses']
    for condition,record in scores.items():
        assert refs==record['references']
        assert decoded[condition]==record['raw_after_actual']['hypotheses']
        assert decoded['canonical_'+condition]==record['canonical_after_actual']['hypotheses']==decoded['canonical_source']
        actual=metric(refs,decoded[condition]);target=metric(record['targets'],decoded[condition])
        assert abs(actual['bleu4']-record['raw_after_actual']['scores']['bleu']['bleu4'])<1e-10
        assert abs(target['bleu4']-record['raw_after_target_scores']['bleu']['bleu4'])<1e-10
        output['conditions'][condition]={'actual_reference':actual,'target_reference':target,'hypotheses_match_frozen_outputs':True,'canonical_hypotheses_match_source':True}
    output['baseline_raw']=metric(refs,decoded['source']);output['baseline_canonical']=metric(refs,decoded['canonical_source'])
    assert abs(output['baseline_raw']['bleu4']-r['baseline_actual']['scores']['bleu']['bleu4'])<1e-10
    output['examples']=[{'clip_id':r['clip_ids'][i],'reference':refs[i],'before':decoded['source'][i],'correct_attack':decoded['correct_caption'][i],'wrong_target':scores['rolled_wrong_caption']['targets'][i],'wrong_attack':decoded['rolled_wrong_caption'][i],'canonical':decoded['canonical_source'][i]} for i in range(8)]
    # All outputs preserved; printed examples fixed to the first eight, not chosen by success.
    (ROOT/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'decoded.json').write_text(json.dumps(decoded,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'native_inference_source.py').write_text(inspect.getsource(back_translate))
    shutil.copy2('scripts/recheck_csl_bleu.py',ROOT/'scripts/recheck_csl_bleu.py')
    print('RECHECK_COMPLETE',output['baseline_raw']['bleu4'],output['conditions']['correct_caption']['actual_reference']['bleu4'],flush=True)
if __name__=='__main__':main()
