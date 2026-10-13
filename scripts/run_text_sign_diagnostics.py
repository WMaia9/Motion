"""CFG diagnosis and first physical-guidance prototype, same held-out DEV."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from run_signspark_geometry_dev import project,matrices,load_config,_build_split,make_back_translation_model,back_translate,score_translations
from signspark_bt.data.sample_dumps import load_sample_dump
from analyze_signspark_geometry_dev import statistics
from run_motion_ctc_confirmation import Reader,decode,metrics

ROOT=Path('reports/cvpr_text_to_sign_diagnostics_20261010').resolve()
PREV=Path('reports/cvpr_motion_consistency_confirmation_20261010').resolve()
BASE=Path('reports/cvpr_signspark_baseline_20261010').resolve()
CONDITIONS=[('cfg0',0,'standard'),('cfg1',1,'standard'),('cfg3',3,'standard'),('cfg5',5,'standard'),('cfg3_shuffled',3,'shuffled'),('cfg3_horizontal',3,'horizontal')]

def prepare():
    for stream in ['body','hand']:
        p=ROOT/'checkpoints'/stream;p.mkdir(parents=True,exist_ok=True);w=p/'ema_0.9999_200000.pt'
        if not w.exists():w.symlink_to(PREV/'checkpoints'/stream/w.name)
    for corpus in ['PHOENIX-2014T','How2Sign']:
        folder=ROOT/corpus;folder.mkdir(exist_ok=True)
        (folder/'selection.json').write_text((PREV/'generated'/corpus/'selection.json').read_text())
        p=json.loads((PREV/'generated'/corpus/'text_duration_paths.json').read_text());d=np.load(p['body'],allow_pickle=True).item();texts=d['text'][0].tolist();mapping={a:b for a,b in zip(texts,texts[1:]+texts[:1])};assert len(mapping)==128
        (folder/'text_mapping.json').write_text(json.dumps(mapping,indent=2,ensure_ascii=False))

def generate(corpus):
    for name,scale,mode in CONDITIONS:
        note=corpus.replace('-','')+'_'+name;paths={}
        for stream in ['body','hand']:
            matches=list((ROOT/'checkpoints'/stream).glob(f'*/seed102_clampstep0_{note}.npy'))
            if matches:assert len(matches)==1;paths[stream]=str(matches[0]);continue
            args=[str(Path.cwd()/'.venv_signspark_confirmation/bin/python'),'-u',str(Path('scripts/sample_text_sign_diagnostic.py').resolve()),f'data_v2={stream}','data_v2/common@common=inference',f'eval.model_path={ROOT}/checkpoints/{stream}/ema_0.9999_200000.pt',f'dev_data=[{corpus}]',f'base_path={PREV}/selected_data/{corpus}/lmdb','eval.split=dev','eval.batch_size=16','eval.ode_stepnum=50','eval.max_samples=-1','eval.candidate_num=1','eval.keyframes_mask_all=True','flip_left_hand=True','seed=102',f'eval.note={note}',f'eval.text_guidance_scale={scale}',f'text_enc_ver={PREV}/text_encoder',f'hydra.run.dir={ROOT}/hydra/{note}_{stream}']
            (ROOT/'logs'/(note+'_'+stream+'_command.json')).write_text(json.dumps(args,indent=2));env=dict(os.environ,WANDB_MODE='disabled',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',SIGN_GUIDANCE_MODE=mode,SIGN_TEXT_MAPPING=str(ROOT/corpus/'text_mapping.json'),SIGN_DIAGNOSTIC_STATS=str(ROOT/'logs'/(note+'_'+stream+'_stats.json')))
            with (ROOT/'logs'/(note+'_'+stream+'.log')).open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            matches=list((ROOT/'checkpoints'/stream).glob(f'*/seed102_clampstep0_{note}.npy'));assert len(matches)==1;paths[stream]=str(matches[0]);print('GENERATED',corpus,name,stream,flush=True)
        (ROOT/corpus/(name+'_paths.json')).write_text(json.dumps(paths,indent=2))

def score(corpus):
    cfg=load_config(BASE/'checkpoints'/corpus/'config.yaml');dc=dict(cfg['data']);dc.pop('pseudo_gloss_dir',None);dc['base_path']=str(PREV/'data/lmdb');dc['index_cache']=False
    native=_build_split(dc,'dev',None);texts={e.sequence:' '.join(e.txt) for e in native.examples};glosses={e.sequence:' '.join(e.gls) for e in native.examples};model=make_back_translation_model(BASE/'checkpoints'/corpus)
    original=json.loads((PREV/'generated'/corpus/'text_duration_scores.json').read_text());names=original['clip_ids'];refs=[texts[n] for n in names];summary={};new_poses={}
    for name,scale,mode in CONDITIONS:
        paths=json.loads((ROOT/corpus/(name+'_paths.json')).read_text());ts,_,ids=load_sample_dump(paths['body'],paths['hand']);gt,_,gtids=load_sample_dump(paths['body'],paths['hand'],poses='gt_poses');assert ids==names==gtids
        b=np.load(paths['body'],allow_pickle=True).item();h=np.load(paths['hand'],allow_pickle=True).item();assert not np.any(b['keyframes']) and not np.any(h['keyframes']);assert np.array_equal(b['video_names'][0],h['video_names'][0][::2]);assert np.array_equal(b['lengths'][0],h['lengths'][0][::2]);assert np.isfinite(b['pred_poses']).all() and np.isfinite(h['pred_poses']).all()
        xs=[x[:300].numpy().copy() for x in ts];ys=[project(x) for x in xs];new_poses[name]=xs;hyp=back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=32);can=back_translate(model,[torch.from_numpy(x) for x in ys],batch_size=32)
        geometry=[]
        for x,y,g in zip(xs,ys,gt):
            rm=matrices(x);real=matrices(g[:300].numpy());angle=np.degrees(Rotation.from_matrix((rm.transpose(0,1,3,2)@real).reshape(-1,3,3)).magnitude());canangle=np.degrees(Rotation.from_matrix((rm.transpose(0,1,3,2)@matrices(y)).reshape(-1,3,3)).magnitude());temporal=np.degrees(Rotation.from_matrix((rm[:-1].transpose(0,1,3,2)@rm[1:]).reshape(-1,3,3)).magnitude())
            geometry.append({'mean_reference_rotation_error_deg':float(angle.mean()),'mean_frame_rotation_change_deg':float(temporal.mean()) if len(x)>1 else 0.,'canonical_max_angle_deg':float(canangle.max())})
        stat=statistics(refs,hyp).sum(0).tolist();result={'condition':name,'guidance_scale':scale,'ode_steps':50,'clips':128,'clip_ids':names,'references':refs,'hypotheses':hyp,'canonical_hypotheses':can,'raw_scores':score_translations(refs,hyp,dc.get('version',''),dc.get('level','word')),'canonical_scores':score_translations(refs,can,dc.get('version',''),dc.get('level','word')),'raw_canonical_changed_sentences':sum(a!=b for a,b in zip(hyp,can)),'ngram_matches_1_to_4':stat[:4],'hypothesis_ngram_totals_1_to_4':stat[4:8],'hypothesis_reference_token_totals':stat[8:10],'geometry_per_clip':geometry,'geometry_clip_mean':{k:float(np.mean([g[k] for g in geometry])) for k in geometry[0]},'canonical_max_angle_deg':max(g['canonical_max_angle_deg'] for g in geometry)}
        if mode=='shuffled':result['shuffled_target_scores']=score_translations(refs[1:]+refs[:1],hyp,dc.get('version',''),dc.get('level','word'))
        (ROOT/corpus/(name+'_scores.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False));np.savez_compressed(ROOT/corpus/(name+'_poses.npz'),**{f'clip{i:04d}':x for i,x in enumerate(xs)})
        summary[name]={k:v for k,v in result.items() if k not in ['clip_ids','references','hypotheses','canonical_hypotheses','geometry_per_clip']};print('SCORED',corpus,name,result['raw_scores']['bleu']['bleu4'],result['canonical_scores']['bleu']['bleu4'],stat[:4],flush=True)
    del model;torch.cuda.empty_cache()
    if corpus=='PHOENIX-2014T':
        ck=torch.load(PREV/'ctc_reader/best.pt',map_location='cpu',weights_only=False);s=ck['state_dict'];reader=Reader(len(ck['vocab']),s['mean'].numpy(),s['std'].numpy()).cuda();reader.load_state_dict(s,strict=True);gr=[glosses[n] for n in names]
        p=json.loads((PREV/'generated'/corpus/'text_duration_paths.json').read_text());old,_,_=load_sample_dump(p['body'],p['hand']);new_poses['baseline20_noCFG']=[x[:300].numpy() for x in old]
        independent={}
        for name,xs in new_poses.items():
            hy=decode(reader,xs,ck['vocab']);hc=decode(reader,[project(x) for x in xs],ck['vocab']);independent[name]={'raw':metrics(gr,hy),'canonical':metrics(gr,hc),'hypotheses':hy,'canonical_hypotheses':hc,'references':gr};print('INDEPENDENT_CTC',name,independent[name]['raw']['wer'],flush=True)
        (ROOT/corpus/'independent_ctc.json').write_text(json.dumps(independent,indent=2,ensure_ascii=False));del reader;torch.cuda.empty_cache()
    summary['baseline20_noCFG']={k:original[k] for k in ['raw_scores','canonical_scores','changed_sentences']};(ROOT/corpus/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    import sys
    if sys.argv[-1]=='prepare':prepare()
    else:
        for corpus in ['PHOENIX-2014T','How2Sign']:generate(corpus);score(corpus)
