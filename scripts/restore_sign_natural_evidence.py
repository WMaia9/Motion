"""Recheck frozen real-motion hypotheses without fitting or rerunning readers."""
import json
import math
import hashlib
from collections import Counter
from pathlib import Path

R=Path('reports/cvpr_sign_natural_restoration_20261014')
SOURCES={
    'PHOENIX-2014T':Path('reports/cvpr_evaluation_confirmation_20261011/heldout/PHOENIX-2014T/full_real_scores.json'),
    'How2Sign':Path('reports/cvpr_evaluation_confirmation_20261011/heldout/How2Sign/full_real_scores.json'),
    'CSL-Daily':Path('reports/cvpr_representation_paper_20261011/CSL-Daily/full_real_scores.json')}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bleu(refs,hyps,character=False):
    counts=[0]*4;totals=[0]*4;nr=nh=0
    for ref,hyp in zip(refs,hyps):
        a=list(ref.replace(' ','')) if character else ref.split()
        b=list(hyp.replace(' ','')) if character else hyp.split()
        nr+=len(a);nh+=len(b)
        for i in range(4):
            n=i+1;r=Counter(tuple(a[j:j+n]) for j in range(len(a)-n+1));h=Counter(tuple(b[j:j+n]) for j in range(len(b)-n+1))
            counts[i]+=sum(min(c,r[g]) for g,c in h.items());totals[i]+=sum(h.values())
    assert all(c>0 for c in counts) and all(c>0 for c in totals) and nh>0
    score=100*math.exp(min(0.,1-nr/nh)+sum(math.log(c/t) for c,t in zip(counts,totals))/4)
    return score,{'clipped_counts':counts,'total_counts':totals,'hypothesis_tokens':nh,'reference_tokens':nr}


def main():
    assert not (R/'manifest_sha256.json').exists();R.mkdir(exist_ok=True)
    rows={};inputs={};manifests={}
    for corpus,path in SOURCES.items():
        source_root=Path(*path.parts[:2]);manifest=source_root/'manifest_sha256.json'
        original=json.loads(manifest.read_text());assert sha(path)==original[str(path.relative_to(source_root))]
        v=json.loads(path.read_text());a=v['raw']['hypotheses'];b=v['canonical']['hypotheses'];refs=v['references'];ids=v['clip_ids']
        assert len(a)==len(b)==len(refs)==len(ids)==v['clips']
        changed=sum(x!=y for x,y in zip(a,b));assert changed==v['raw_canonical_changed_sentences']
        scores={};stats={}
        for label,h in [('raw',a),('canonical',b)]:
            score,s=bleu(refs,h,corpus=='CSL-Daily');expected=v[label]['scores']['bleu']['bleu4']
            assert abs(score-expected)<1e-10,(corpus,label,score,expected)
            scores[label]=score;stats[label]=s
        rows[corpus]={'clips':len(a),'changed_hypotheses':changed,'changed_fraction':changed/len(a),'BLEU4':scores,'canonical_minus_raw_BLEU4':scores['canonical']-scores['raw'],'token_convention':'characters preserving punctuation' if corpus=='CSL-Daily' else 'words','score_sufficient_statistics':stats,'source_text_supplied_to_layer':False,'attack_or_learning':False,'reader_fitting_or_new_inference':False}
        inputs[str(path)]=sha(path);manifests[str(manifest)]=sha(manifest)
    expected={'PHOENIX-2014T':(642,0),'How2Sign':(2183,1438),'CSL-Daily':(1162,464)}
    for name,(n,k) in expected.items():assert (rows[name]['clips'],rows[name]['changed_hypotheses'])==(n,k)
    record={'passed':True,'results':rows,'inputs_sha256':inputs,'source_manifest_sha256':manifests,'scope':'real-motion native TEST under shared released-loader preprocessing; no attack, text-conditioned layer, fitting or new inference','shared_preprocessing':'original producers canonicalize left-hand channels60:150 before both raw and canonical full-motion decoding; raw is therefore audit input, not a claim of untouched release bytes','geometry':'raw/canonical use the same native rotation converter; prior independent invariance checks preserved in source archives','no_accuracy_improvement_claim':True,'independent_counter_and_brevity_penalty_recomputation':True,'analysis_source_sha256':sha(__file__)}
    (R/'analysis.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    (R/'verification.json').write_text(json.dumps({'passed':True,'all_source_file_and_manifest_hashes_verified':True,'all_hypothesis_pairs_counted':sum(r['clips'] for r in rows.values()),'all_six_corpus_BLEUs_recomputed_from_saved_strings':True,'source_sha256':sha(__file__)},indent=2)+'\n')
    print('SIGN_NATURAL_EVIDENCE_VERIFIED', {n:(r['changed_hypotheses'],r['clips'],r['BLEU4']) for n,r in rows.items()},flush=True)


if __name__=='__main__':main()
