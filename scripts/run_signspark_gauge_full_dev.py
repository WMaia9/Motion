"""Full DEV targeted audit, matching native sample loader's left-hand contract."""
import copy,json,os,time,ast
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
from run_signspark_gauge_attack import transform,SignCollator,Batch,PAD_TOKEN
from run_signspark_geometry_dev import ROOT,matrices,project,load_config,_build_split,make_back_translation_model,back_translate,score_translations
from signspark_bt.data.sample_dumps import flip_left_hand,load_sample_dump

OUT=Path('reports/cvpr_signspark_gauge_full_dev_20261010')

def check_dump(folder,label,poses,refs):
    count=8;n=max(len(x) for x in poses[:count]);body=np.zeros((1,count,60,n),np.float32);hand=np.zeros((1,count*2,90,n),np.float32)
    for i,x in enumerate(poses[:count]):
        body[0,i,:,:len(x)]=x[:,:60].T;hand[0,2*i,:,:len(x)]=flip_left_hand(x[:,60:150]).T;hand[0,2*i+1,:,:len(x)]=x[:,150:].T
    common={'lengths':np.asarray([len(x) for x in poses[:count]]),'text':np.asarray(refs[:count]),'video_names':np.asarray([f'contract_example_{i}' for i in range(count)])}
    bpath=folder/f'{label}_body_dump.npy';hpath=folder/f'{label}_hand_dump.npy';np.save(bpath,dict(pred_poses=body,**common));np.save(hpath,dict(pred_poses=hand,**common))
    restored,_,_=load_sample_dump(str(bpath),str(hpath));error=max(float(abs(x.numpy()-y).max()) for x,y in zip(restored,poses[:count]));assert error<1e-5
    return {'clips':count,'native_sample_loader_max_abs':error}

def run(corpus):
    started=time.time();torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    os.environ['DATA_ROOT']=str((ROOT/'data').resolve());cfg=load_config(ROOT/'checkpoints'/corpus/'config.yaml');dc=dict(cfg['data']);dc.pop('pseudo_gloss_dir',None)
    data=_build_split(dc,'dev',None);examples=[data[i] for i in range(len(data))];xs=[e.sgn.numpy().copy() for e in examples];refs=[' '.join(e.txt) for e in examples]
    for e,x in zip(examples,xs):x[:,60:150]=project(x)[:,60:150];e.sgn=torch.from_numpy(x)
    folder=OUT/corpus;folder.mkdir(exist_ok=True)
    (folder/'selection.json').write_text(json.dumps([{'dev_index':i,'clip_id':e.sequence,'reference':r,'frames':len(x)} for i,(e,r,x) in enumerate(zip(examples,refs,xs))],indent=2,ensure_ascii=False))
    checks={'original':check_dump(folder,'original',xs,refs)}
    # Official pure converter, with no proprietary body-model import.
    tree=ast.parse(Path('external/SignSparK/tools/visualize.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rot6d_to_matrix')
    namespace={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'official_visualize_helper','exec'),namespace);official=namespace['rot6d_to_matrix']
    model=make_back_translation_model(ROOT/'checkpoints'/corpus);model.do_recognition=False
    for parameter in model.parameters():parameter.requires_grad_(False)
    baseline=back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=32);canonical_before=back_translate(model,[torch.from_numpy(project(x)) for x in xs],batch_size=32)
    summary={'clips':len(xs),'raw_original_scores':score_translations(refs,baseline,dc.get('version',''),dc.get('level','word')),'canonical_original_scores':score_translations(refs,canonical_before,dc.get('version',''),dc.get('level','word')),'conditions':{}}
    collator=SignCollator(model.txt_vocab,None,240);pad=model.txt_vocab.stoi[PAD_TOKEN];jointmask=torch.ones(1,1,40,1,device='cuda');jointmask[:,:,10:25]=0
    for condition in ['correct_caption','rolled_wrong_caption']:
        targetrefs=refs if condition=='correct_caption' else refs[1:]+refs[:1];attacked=[];coefficients=[];losslogs=[]
        for start in range(0,len(xs),32):
            chunk=[copy.copy(e) for e in examples[start:start+32]];b=len(chunk)
            for e,target in zip(chunk,targetrefs[start:start+32]):e.txt=target.split()
            batch=Batch(collator(chunk),txt_pad_index=pad,sgn_dim=240,use_cuda=True);original=batch.sgn.detach().reshape(b,-1,40,6);u=torch.zeros((*original.shape[:-1],3),device='cuda',requires_grad=True);optimizer=torch.optim.Adam([u],lr=.1);lossvalues=[]
            for step in range(80):
                batch.sgn=transform(original,u*jointmask).reshape(b,-1,240)
                output,_=model.forward(batch.sgn,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask);logits=output[0]
                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad);assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();assert torch.isfinite(u.grad).all();optimizer.step();lossvalues.append(float(loss.detach()))
            y=transform(original,u*jointmask).detach().cpu().numpy().reshape(b,-1,240);saved=(u*jointmask).detach().cpu().numpy()
            for i,e in enumerate(chunk):attacked.append(y[i,:len(e.sgn)].copy());coefficients.append(saved[i,:len(e.sgn)].copy())
            losslogs.append({'batch_start':start,'cross_entropy':lossvalues});print(corpus,condition,start,'/',len(xs),'CE',round(lossvalues[0],4),'->',round(lossvalues[-1],4),flush=True)
        maxangle=0.;feature_delta=0.;official_difference=0.
        for x,y in zip(xs,attacked):
            assert np.isfinite(y).all();assert np.array_equal(x[:,60:150],y[:,60:150]);rx=official(torch.from_numpy(x).reshape(-1,6)).numpy().astype(np.float64);ry=official(torch.from_numpy(y).reshape(-1,6)).numpy().astype(np.float64)
            assert (np.linalg.det(ry)>.99).all();maxangle=max(maxangle,float(np.degrees(Rotation.from_matrix(rx.transpose(0,2,1)@ry).magnitude()).max()));official_difference=max(official_difference,float(abs(ry-matrices(y).reshape(-1,3,3)).max()));feature_delta=max(feature_delta,float(abs(x-y).max()))
        assert maxangle<.01;checks[condition]=check_dump(folder,condition,attacked,refs)
        raw_after=back_translate(model,[torch.from_numpy(x) for x in attacked],batch_size=32);canonical_after=back_translate(model,[torch.from_numpy(project(x)) for x in attacked],batch_size=32)
        result={'actual_reference_scores':score_translations(refs,raw_after,dc.get('version',''),dc.get('level','word')),'target_scores':score_translations(targetrefs,raw_after,dc.get('version',''),dc.get('level','word')),'baseline_target_scores':score_translations(targetrefs,baseline,dc.get('version',''),dc.get('level','word')),'canonical_actual_scores':score_translations(refs,canonical_after,dc.get('version',''),dc.get('level','word')),'canonical_target_scores':score_translations(targetrefs,canonical_after,dc.get('version',''),dc.get('level','word')),'raw_changed_sentences':sum(a!=b for a,b in zip(raw_after,baseline)),'canonical_changed_sentences':sum(a!=b for a,b in zip(canonical_after,canonical_before)),'max_rotation_error_deg':maxangle,'max_feature_abs_delta':feature_delta,'official_helper_vs_independent_max_abs':official_difference,'different_target_references':sum(a!=b for a,b in zip(refs,targetrefs)),'raw_original_hypotheses':baseline,'canonical_original_hypotheses':canonical_before,'raw_after_hypotheses':raw_after,'canonical_after_hypotheses':canonical_after,'targets':targetrefs,'losslogs':losslogs}
        (folder/f'{condition}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));np.savez_compressed(folder/f'{condition}_poses.npz',**{f'{prefix}{i:04d}':x for prefix,arrays in [('source',xs),('attacked',attacked),('gauge',coefficients)] for i,x in enumerate(arrays)})
        summary['conditions'][condition]={k:v for k,v in result.items() if 'hypotheses' not in k and k not in ('targets','losslogs')};(folder/'summary.json').write_text(json.dumps(summary,indent=2));(folder/'contract_checks.json').write_text(json.dumps(checks,indent=2));print('FINISHED',corpus,condition,json.dumps(summary['conditions'][condition]),flush=True)
    summary['seconds']=time.time()-started;(folder/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',corpus,summary['seconds'],flush=True)

if __name__=='__main__':
    for corpus in ['PHOENIX-2014T','How2Sign']:run(corpus)
