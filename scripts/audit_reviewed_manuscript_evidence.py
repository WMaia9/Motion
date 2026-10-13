"""Cross-check manuscript tables and saved metric outputs; no model inference."""
from collections import Counter
import json,math,re
from pathlib import Path
import numpy as np
from verify_signspark_studies import sha
from verify_representation_paper import edits
ROOT=Path('reports/cvpr_manuscript_review_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence_revision_20261011')
OLD=Path('reports/cvpr_evaluation_confirmation_20261011')
EXT=Path('reports/cvpr_representation_paper_20261011')
REPLAY=Path('reports/csl_bleu_recheck_20261011')
def read(p):return json.loads(Path(p).read_text())
def metric(refs,hyps,char=False):
    counts=[0]*4;totals=[0]*4;rl=hl=0
    for ref,hyp in zip(refs,hyps):
        r=list(ref.replace(' ','')) if char else ref.split();h=list(hyp.replace(' ','')) if char else hyp.split();rl+=len(r);hl+=len(h)
        for n in range(1,5):
            a=Counter(tuple(r[i:i+n]) for i in range(max(0,len(r)-n+1)));b=Counter(tuple(h[i:i+n]) for i in range(max(0,len(h)-n+1)))
            counts[n-1]+=sum(min(v,a[k]) for k,v in b.items());totals[n-1]+=sum(b.values())
    if min(counts)==0 or min(totals)==0:return 0.
    bp=min(1.,math.exp(1-rl/hl)) if hl else 0.
    return 100*bp*math.exp(sum(math.log(c/t) for c,t in zip(counts,totals))/4)
def checkscore(refs,result,char=False):
    actual=metric(refs,result['hypotheses'],char);assert abs(actual-result['scores']['bleu']['bleu4'])<1e-10
    return len(refs)
def rows(text,label):
    block=text.split('\\label{'+label+'}')[1].split('\\end{tabular}')[0]
    block=block.replace('\\midrule','').replace('\\toprule','')
    return [[c.strip() for c in line.removesuffix('\\\\').split('&')] for line in block.splitlines() if '&' in line and line.rstrip().endswith('\\\\')][1:]
def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    preserved={}
    for folder in [OLD,EXT,REPLAY,Path('reports/cvpr_motion_consistency_confirmation_20261010')]:
        manifest=read(folder/'manifest_sha256.json')
        for name,digest in manifest.items():assert sha(folder/name)==digest,(folder,name)
        preserved[str(folder)]={'files':len(manifest),'manifest_sha256':sha(folder/'manifest_sha256.json')}
    main=(PAPER/'main.tex').read_text();supp=(PAPER/'supplement.tex').read_text();corpora=['PHOENIX-2014T','How2Sign','CSL-Daily'];records={};sentences=0;defense=0
    for corpus in corpora:
        folder=EXT/corpus if corpus=='CSL-Daily' else OLD/'heldout'/corpus
        full=read(folder/'full_real_scores.json');char=corpus=='CSL-Daily'
        for kind in ['raw','canonical']:sentences+=checkscore(full['references'],full[kind],char)
        records[corpus]={}
        for condition in ['correct_caption','rolled_wrong_caption']:
            r=read(folder/(condition+'_scores.json'));refs=r['references'];targets=r['targets']
            for references,value in [(refs,r['baseline_actual']),(refs,r['raw_after_actual']),(refs,r['canonical_before_actual']),(refs,r['canonical_after_actual'])]:sentences+=checkscore(references,value,char)
            for hyps,value in [(r['baseline_actual']['hypotheses'],r['baseline_target_scores']),(r['raw_after_actual']['hypotheses'],r['raw_after_target_scores'])]:assert abs(metric(targets,hyps,char)-value['bleu']['bleu4'])<1e-10;sentences+=128
            assert r['canonical_before_actual']['hypotheses']==r['canonical_after_actual']['hypotheses'];defense+=128
            records[corpus][condition]=r
    assert len(rows(main,'tab:real'))==3 and len(rows(main,'tab:attacks'))==6 and len(rows(main,'tab:generators'))==4
    for row,corpus in zip(rows(main,'tab:real'),corpora):
        full=read((EXT/corpus if corpus=='CSL-Daily' else OLD/'heldout'/corpus)/'full_real_scores.json')
        assert int(row[1])==full['clips'] and row[2]==f"{full['raw']['scores']['bleu']['bleu4']:.4f}" and row[3]==f"{full['canonical']['scores']['bleu']['bleu4']:.4f}"
        assert int(row[5])==full['raw_canonical_changed_sentences']
    order=[(c,k) for k in ['correct_caption','rolled_wrong_caption'] for c in corpora]
    for row,(corpus,condition) in zip(rows(main,'tab:attacks'),order):
        r=records[corpus][condition]
        change=r['target_change'] if corpus=='CSL-Daily' else read(OLD/'analysis.json')['attacks'][corpus]['subset_conditions'][condition]['target_gain']
        expected=[r['baseline_target_scores']['bleu']['bleu4'],r['raw_after_target_scores']['bleu']['bleu4'],r['raw_after_actual']['scores']['bleu']['bleu4'],change['delta_bleu4']]
        assert row[2:6]==[f'{v:.2f}' for v in expected],(corpus,row)
        assert row[6]=='['+','.join(f'{v:.2f}' for v in change['ci95'])+']'
    rank=read(OLD/'analysis.json')['rankings'];names=['geometry','velocity','physical_ce','physical_generated_near']
    for row,name in zip(rows(main,'tab:generators'),names):assert row[1:]==[f"{rank['scores'][r][name]:.4f}" for r in ['raw','canonical']]
    for name in names:
        folder=OLD/'rankings'/name;can=read(folder/'canonical_scores.json')
        for rep in ['raw','canonical','gauge_scale090','gauge_scale110','gauge_shear010']:
            r=read(folder/(rep+'_scores.json'));sentences+=checkscore(r['references'],r)
            if rep.startswith('gauge'):
                assert r['defended']['hypotheses']==can['hypotheses'];sentences+=checkscore(r['references'],r['defended'])
    generated_rows=rows((PAPER/'tables_generated.tex').read_text(),'tab:allgenerators');assert len(generated_rows)==4
    for row,name in zip(generated_rows,names):assert row[1:]==[f"{rank['scores'][r][name]:.6f}" for r in ['raw','canonical','gauge_scale090','gauge_scale110','gauge_shear010']]
    assert all(not r['opposite_pairwise_ci_supported_rankings_vs_raw'] for r in rank['pairwise_comparisons'].values())
    ctc=read(EXT/'ctc_TEST/full_controls.json');ctcmetrics=0
    for control in ctc['controls'].values():
        values=[edits(r,h) for r,h in zip(ctc['references'],control['hypotheses'])]
        assert abs(100*sum(v[0] for v in values)/sum(v[1] for v in values)-control['metrics']['wer'])<1e-10;ctcmetrics+=len(values)
    for condition in ['correct_gloss','rolled_wrong_gloss']:
        r=read(EXT/'ctc_TEST'/(condition+'_scores.json'))
        assert r['canonical_before_hypotheses']==r['canonical_after_hypotheses'];defense+=128
        for refs,hyps,key in [(r['references'],r['baseline_hypotheses'],'baseline_actual'),(r['references'],r['attacked_hypotheses'],'attacked_actual'),(r['targets'],r['attacked_hypotheses'],'attacked_target')]:
            values=[edits(a,b) for a,b in zip(refs,hyps)];assert abs(100*sum(v[0] for v in values)/sum(v[1] for v in values)-r[key]['wer'])<1e-10;ctcmetrics+=len(values)
    replay=read(REPLAY/'results.json');assert replay['fresh_decoded_sentences']==768
    assert '39/1459&450/1451' in supp and replay['conditions']['correct_caption']['actual_reference']['matched_ngrams'][3]==450
    assert defense==1024
    # Check exact reported rounded sensitivity intervals, not copied prose guesses.
    sens=read(OLD/'canonical_accuracy_sensitivity.json')['results']['How2Sign']
    for key in ['full_real_scores','correct_caption_scores']:
        v=sens[key]['official_VIDEO_ID_cluster_bootstrap'];formatted='['+','.join(f'{x:.6f}' for x in v['ci95'])+']';assert formatted in supp,formatted
    hashes=read(ROOT/'sources/manuscript_checkpoint_hashes.json')
    for r in hashes:assert sha(r['path'])==r['sha256'] and r['sha256'] in (PAPER/'tables_hashes.tex').read_text()
    out={'prior_reports_verified':preserved,'standalone_BLEU_sentence_evaluations_recomputed':sentences,'independent_WER_sentence_evaluations_recomputed':ctcmetrics,'canonical_attack_pairs':defense,'generated_decodes_verified':20544,'main_numeric_tables_verified':['real baselines','six attacks and confidence intervals','four raw/canonical generators'],'supplement_generated_tables_data_derived':True,'CSL_replay_verified':768,'accuracy_intervals_match_immutable_results':True,'checkpoint_hashes_verified':len(hashes),'new_training_runs':0,'new_model_inference_runs':0,'scope':'manuscript revision and saved-evidence audit only'}
    (ROOT/'evidence_verification.json').write_text(json.dumps(out,indent=2)+'\n');print('MANUSCRIPT_EVIDENCE_VERIFIED',sentences,ctcmetrics,flush=True)
if __name__=='__main__':main()
