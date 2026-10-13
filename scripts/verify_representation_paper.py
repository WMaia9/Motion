"""Independent saved-output audit; no fitting, decoding, or frozen mutations."""
import ast,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from verify_signspark_studies import sha,matrices
from analyze_signspark_geometry_dev import statistics,b4
from run_signspark_geometry_dev import load_config,_build_split,project

ROOT=Path('reports/cvpr_representation_paper_20261011')
PRIOR=Path('reports/cvpr_evaluation_confirmation_20261011')
def read(p):return json.loads(Path(p).read_text())

def edits(a,b):
    # Full matrix dynamic programming, independent of producer rolling-row DP.
    a=a.split();b=b.split();d=np.empty((len(a)+1,len(b)+1),dtype=np.int64)
    d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
    for i,x in enumerate(a,1):
        for j,y in enumerate(b,1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(x!=y))
    return int(d[-1,-1]),len(a)

def wer(refs,hyps,expected):
    values=[edits(r,h) for r,h in zip(refs,hyps)];errors,lengths=map(list,zip(*values))
    score=100*sum(errors)/sum(lengths)
    assert errors==expected['errors'] and lengths==expected['reference_lengths']
    assert abs(score-expected['wer'])<1e-10
    return score

def chinese(xs):return [' '.join(list(x.replace(' ',''))) for x in xs]
def bleu(refs,hyps,expected):
    score=b4(statistics(chinese(refs),chinese(hyps)).sum(0))
    assert abs(score-expected['bleu']['bleu4'])<1e-10
    return score

def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8)
    registered=read(ROOT/'sources/registered_inputs.json')
    assert sha(ROOT/'PROTOCOLO.md')==registered['protocol_sha256']
    assert sha(registered['ctc_checkpoint_path'])==registered['ctc_checkpoint_sha256']
    assert sha(PRIOR/'manifest_sha256.json')==registered['previous_confirmation_manifest_sha256']
    preserved={}
    for frozen in [PRIOR,Path('reports/cvpr_motion_consistency_confirmation_20261010')]:
        manifest=read(frozen/'manifest_sha256.json')
        for name,digest in manifest.items():assert sha(frozen/name)==digest,(frozen,name)
        preserved[str(frozen)]={'files':len(manifest),'manifest_sha256':sha(frozen/'manifest_sha256.json')}
    for item in read(ROOT/'sources/CSL_downloads.json'):
        path=ROOT/item['file'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    module=ast.parse(Path('external/SignSparK/tools/visualize.py').read_text())
    function=next(f for f in module.body if isinstance(f,ast.FunctionDef) and f.name=='rot6d_to_matrix')
    scope={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),'<official converter>','exec'),scope)
    official=scope['rot6d_to_matrix'];converter_difference=0.;checks={};metric_sentences=0
    for corpus,conditions in [('ctc_TEST',['correct_gloss','rolled_wrong_gloss']),('CSL-Daily',['correct_caption','rolled_wrong_caption'])]:
        folder=ROOT/corpus;selection=read(folder/'selection.json');assert len(selection)==128
        full=read(folder/('full_controls.json' if corpus=='ctc_TEST' else 'full_real_scores.json'))
        assert len(full['clip_ids'])==full['clips'] and len(set(full['clip_ids']))==full['clips']
        chosen=np.sort(np.random.default_rng(2027).choice(full['clips'],128,replace=False)).tolist()
        assert chosen==[r['test_index'] for r in selection]
        assert [full['clip_ids'][i] for i in chosen]==[r['clip_id'] for r in selection]
        assert [full['references'][i] for i in chosen]==[r['reference'] for r in selection]
        if corpus=='ctc_TEST':
            from run_evaluation_heldout import dataset
            native_entries,_=dataset('PHOENIX-2014T')
        else:
            dc=dict(load_config(ROOT/'checkpoints/CSL-Daily/config.yaml')['data'])
            dc.update(base_path=str((ROOT/'data/lmdb').resolve()),index_cache=False,max_sgn_length=300)
            native_data=_build_split(dc,'test',None);native_entries=[native_data[i] for i in chosen]
        source_poses=[]
        for j,i in enumerate(chosen):
            e=native_entries[i] if corpus=='ctc_TEST' else native_entries[j]
            assert e.sequence==selection[j]['clip_id']
            x=e.sgn.numpy().copy()
            if corpus!='ctc_TEST':x[:,60:150]=project(x)[:,60:150]
            source_poses.append(x)
        if corpus=='ctc_TEST':
            assert chosen==read(PRIOR/'test_audit.json')['PHOENIX-2014T']['attack_indices']
            for control in full['controls'].values():wer(full['references'],control['hypotheses'],control['metrics']);metric_sentences+=full['clips']
        else:
            for kind in ['raw','canonical']:bleu(full['references'],full[kind]['hypotheses'],full[kind]['scores']);metric_sentences+=full['clips']
        records={}
        for condition in conditions:
            r=read(folder/(condition+'_scores.json'));refs=r['references'];targets=r['targets']
            assert r['clip_ids']==[e['clip_id'] for e in selection] and refs==[e['reference'] for e in selection]
            assert targets==(refs if condition.startswith('correct') else refs[1:]+refs[:1])
            if corpus=='ctc_TEST':
                before=r['baseline_hypotheses'];after=r['attacked_hypotheses'];canbefore=r['canonical_before_hypotheses'];canafter=r['canonical_after_hypotheses']
                for references,hyps,key in [(refs,before,'baseline_actual'),(targets,before,'baseline_target'),(refs,after,'attacked_actual'),(targets,after,'attacked_target'),(refs,canafter,'canonical_actual')]:wer(references,hyps,r[key]);metric_sentences+=128
                vocab=torch.load(registered['ctc_checkpoint_path'],map_location='cpu',weights_only=False)['vocab']
                assert sum(w not in vocab for t in targets for w in t.split())==r['target_unknown_tokens']==0
                infeasible=sum(len(t.split())+sum(a==b for a,b in zip(t.split(),t.split()[1:]))>(e['frames']+3)//4 for t,e in zip(targets,selection))
                assert infeasible==r['ctc_infeasible_targets']
                losslogs=r['optimization_logs'];losskey='losses'
            else:
                before=r['baseline_actual']['hypotheses'];after=r['raw_after_actual']['hypotheses'];canbefore=r['canonical_before_actual']['hypotheses'];canafter=r['canonical_after_actual']['hypotheses']
                for references,hyps,score in [(refs,before,r['baseline_actual']['scores']),(targets,before,r['baseline_target_scores']),(refs,after,r['raw_after_actual']['scores']),(targets,after,r['raw_after_target_scores']),(refs,canbefore,r['canonical_before_actual']['scores']),(refs,canafter,r['canonical_after_actual']['scores'])]:bleu(references,hyps,score);metric_sentences+=128
                losslogs=r['losslogs'];losskey='cross_entropy'
            assert len(before)==len(after)==len(canbefore)==len(canafter)==128
            assert canbefore==canafter and r['canonical_changed_sentences']==0
            assert sum(a!=b for a,b in zip(refs,targets))==r['different_targets']
            assert len(losslogs)==4 and all(len(log[losskey])==80 and np.isfinite(log[losskey]).all() for log in losslogs)
            maximum_rotation=0.;reconstruction=0.;bound=0.;minnorm=float('inf')
            with np.load(folder/(condition+'_poses.npz')) as data:
                assert len(data.files)==384
                for i,e in enumerate(selection):
                    x=data[f'source{i:04d}'];y=data[f'attacked{i:04d}'];u=data[f'gauge{i:04d}']
                    assert np.array_equal(x,source_poses[i])
                    assert x.shape==y.shape==(e['frames'],240) and 0<len(x)<=300 and u.shape==(len(x),40,3)
                    assert all(np.isfinite(v).all() for v in [x,y,u])
                    assert np.array_equal(x[:,60:150],y[:,60:150]) and (u[:,10:25]==0).all()
                    d=x.reshape(-1,40,6);z=.35*np.tanh(u)
                    rebuilt=np.concatenate((np.exp(z[...,0:1])*d[...,:3],np.exp(z[...,1:2])*d[...,3:]+z[...,2:3]*d[...,:3]),-1).reshape(y.shape)
                    reconstruction=max(reconstruction,float(abs(rebuilt-y).max()));bound=max(bound,float(abs(z).max()))
                    mx=matrices(x);my=matrices(y)
                    for value,matrix in [(x,mx),(y,my)]:
                        converted=official(torch.from_numpy(value.reshape(-1,40,6).astype(np.float64))).numpy()
                        converter_difference=max(converter_difference,float(abs(converted-matrix).max()))
                        q=value.reshape(-1,40,6).astype(np.float64);a=q[...,:3];b=q[...,3:];perp=b-(a*b).sum(-1,keepdims=True)/(a*a).sum(-1,keepdims=True)*a
                        minnorm=min(minnorm,float(np.linalg.norm(a,axis=-1).min()),float(np.linalg.norm(perp,axis=-1).min()))
                    rotation=float(np.degrees(Rotation.from_matrix((mx.transpose(0,1,3,2)@my).reshape(-1,3,3)).magnitude()).max())
                    maximum_rotation=max(maximum_rotation,rotation)
            assert reconstruction<2e-6 and bound<=.350001 and minnorm>1e-12 and maximum_rotation<.01
            assert abs(maximum_rotation-r['max_rotation_error_deg'])<1e-8
            records[condition]={'clips':128,'canonical_changes':0,'rotation_max_deg':maximum_rotation,'coefficient_bound':bound,'reconstruction_max_abs':reconstruction,'minimum_nondegenerate_norm':minnorm,'left_preserved_bitwise':True,'different_targets':r['different_targets']}
        checks[corpus]=records;print('VERIFIED_NEW_ATTACKS',corpus,flush=True)
    assert converter_difference<1e-12
    native=read(ROOT/'sources/native_loader_contract.json')
    # Inspect all diagnostic records without relying on the public card match.
    assert native['post_result_contract_diagnostic'] and native['no_hyperparameter_tuning'] and native['attack_outputs_not_replaced']
    for corpus,record in native['corpora'].items():
        folder=ROOT/corpus if corpus=='CSL-Daily' else PRIOR/'heldout'/corpus
        full=read(folder/'full_real_scores.json')
        assert record['clip_ids']==full['clip_ids'] and record['references']==full['references']
        assert record['native_loader_hypotheses']==full['raw']['hypotheses']
        assert record['changed_by_left_projection']==0 and record['prepared_baseline_replay_exact']
        assert abs(record['native_loader_scores']['bleu']['bleu4']-full['raw']['scores']['bleu']['bleu4'])<1e-10
    empty={'full':sum(not x.split() for x in read(ROOT/'ctc_TEST/full_controls.json')['references']),'subset':sum(not e['reference'].split() for e in read(ROOT/'ctc_TEST/selection.json'))}
    assert empty=={'full':2,'subset':1}
    out={'attacks':checks,'new_attack_pairs_verified':512,'all_attack_pairs_including_prior':1024,'metric_sentences_recomputed':metric_sentences,'ctc_metric_verification':'independent full matrix edit-distance DP','official_converter_max_abs_difference':converter_difference,'empty_gloss_annotations_retained':empty,'prior_frozen_reports':preserved,'new_training_runs':0,'rendered_meshes':False,'independent_published_BT_replication':False,'public_card_ASL_CSL_exact_match':False}
    (ROOT/'independent_verification.json').write_text(json.dumps(out,indent=2)+'\n');print('REPRESENTATION_VERIFIED',metric_sentences,flush=True)
if __name__=='__main__':main()
