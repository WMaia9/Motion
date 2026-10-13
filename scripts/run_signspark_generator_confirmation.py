"""Preselected official generator outputs, natural raw/canonical comparisons."""
import io, json, os, pickle, subprocess, sys, time
from pathlib import Path
import lmdb, numpy as np, torch
from scipy.spatial.transform import Rotation
from run_signspark_geometry_dev import project, matrices, load_config, _build_split, back_translate, make_back_translation_model, score_translations
from signspark_bt.data.sample_dumps import load_sample_dump

ROOT=Path('reports/cvpr_motion_consistency_confirmation_20261010').resolve()
BASE=Path('reports/cvpr_signspark_baseline_20261010').resolve()

def prepare():
    for corpus in ['PHOENIX-2014T','How2Sign']:
        folder=ROOT/'generated'/corpus;folder.mkdir(parents=True,exist_ok=True)
        src=ROOT/'data/lmdb/dev'/(corpus+'_reopt_dev.lmdb')
        with lmdb.open(str(src),readonly=True,lock=False,readahead=False) as db:
            with db.begin() as t:
                meta=pickle.loads(t.get(b'__meta__'));indices=np.sort(np.random.default_rng(2027).choice(len(meta['clip_ids']),128,replace=False));names=[meta['clip_ids'][i] for i in indices]
                selection=[]
                dst=ROOT/'selected_data'/corpus/'lmdb/dev'/(corpus+'_reopt_dev.lmdb');dst.mkdir(parents=True,exist_ok=True)
                with lmdb.open(str(dst),map_size=1<<30) as out:
                    with out.begin(write=True) as u:
                        u.put(b'__meta__',pickle.dumps(dict(clip_ids=names,num_clips=len(names))))
                        for i,name in zip(indices,names):
                            blob=t.get(name.encode());u.put(name.encode(),blob)
                            with np.load(io.BytesIO(blob),allow_pickle=True) as z:selection.append({'dev_index':int(i),'clip_id':name,'frames':len(z['segment']),'reference':str(z['translation'][0])})
        (folder/'selection.json').write_text(json.dumps(selection,indent=2,ensure_ascii=False))
    (ROOT/'selection_prepared.json').write_text(json.dumps({'rng':2027,'count_per_dataset':128,'time_host_utc':time.time(),'protocol':'PROTOCOLO.md'},indent=2))

def generate():
    for corpus in ['PHOENIX-2014T','How2Sign']:
        for mode in ['text_duration','oracle_keyframes']:
            paths={};note=corpus.replace('-','')+'_'+mode
            for stream in ['body','hand']:
                paths[stream]=None;matches=list((ROOT/'checkpoints'/stream).glob(f'*/seed102_clampstep0_{note}.npy'))
                if matches:assert len(matches)==1;paths[stream]=str(matches[0]);continue
                args=[str(Path.cwd()/'.venv_signspark_confirmation/bin/python'),'-u',str(Path('scripts/sample_signspark_confirmation.py').resolve()),f'data_v2={stream}','data_v2/common@common=inference',f'eval.model_path={ROOT}/checkpoints/{stream}/ema_0.9999_200000.pt',f'dev_data=[{corpus}]',f'base_path={ROOT}/selected_data/{corpus}/lmdb','eval.split=dev','eval.batch_size=16','eval.ode_stepnum=20','eval.max_samples=-1','eval.candidate_num=1',f'eval.keyframes_mask_all={"True" if mode=="text_duration" else "False"}','flip_left_hand=True','seed=102',f'eval.note={note}',f'text_enc_ver={ROOT}/text_encoder',f'hydra.run.dir={ROOT}/hydra/{note}_{stream}']
                (ROOT/'logs'/f'{note}_{stream}_command.json').write_text(json.dumps(args,indent=2))
                env=dict(os.environ,WANDB_MODE='disabled',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
                with (ROOT/'logs'/f'{note}_{stream}.log').open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                matches=list((ROOT/'checkpoints'/stream).glob(f'*/seed102_clampstep0_{note}.npy'));assert len(matches)==1;paths[stream]=str(matches[0]);print('GENERATED',corpus,mode,stream,flush=True)
            (ROOT/'generated'/corpus/(mode+'_paths.json')).write_text(json.dumps(paths,indent=2))

def score():
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    for corpus in ['PHOENIX-2014T','How2Sign']:
        folder=ROOT/'generated'/corpus;cfg=load_config(BASE/'checkpoints'/corpus/'config.yaml');dc=dict(cfg['data']);dc.pop('pseudo_gloss_dir',None);dc['base_path']=str(ROOT/'data/lmdb');dc['index_cache']=False
        native=_build_split(dc,'dev',None);references={e.sequence:' '.join(e.txt) for e in native.examples};model=make_back_translation_model(BASE/'checkpoints'/corpus)
        summary={}
        for mode in ['text_duration','oracle_keyframes']:
            paths=json.loads((folder/(mode+'_paths.json')).read_text());tensors,_,names=load_sample_dump(paths['body'],paths['hand']);assert len(tensors)==128
            hand=np.load(paths['hand'],allow_pickle=True).item();body=np.load(paths['body'],allow_pickle=True).item();assert np.array_equal(body['video_names'][0],hand['video_names'][0][::2]);assert np.array_equal(body['lengths'][0],hand['lengths'][0][::2]);assert np.isfinite(body['pred_poses']).all() and np.isfinite(hand['pred_poses']).all()
            if mode=='text_duration':assert not np.any(body['keyframes']) and not np.any(hand['keyframes'])
            xs=[x[:300].numpy().copy() for x in tensors];ys=[project(x) for x in xs];refs=[references[n] for n in names];raw=back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=32);canon=back_translate(model,[torch.from_numpy(x) for x in ys],batch_size=32)
            gt,_,gtnames=load_sample_dump(paths['body'],paths['hand'],poses='gt_poses');assert names==gtnames
            gt_h=back_translate(model,[x[:300] for x in gt],batch_size=32)
            errors=[]
            for x,y in zip(xs,ys):
                rr=matrices(x).transpose(0,1,3,2)@matrices(y);errors.append(float(np.degrees(Rotation.from_matrix(rr.reshape(-1,3,3)).magnitude()).max()))
            result={'corpus':corpus,'mode':mode,'clips':128,'raw_scores':score_translations(refs,raw,dc.get('version',''),dc.get('level','word')),'canonical_scores':score_translations(refs,canon,dc.get('version',''),dc.get('level','word')),'ground_truth_scores':score_translations(refs,gt_h,dc.get('version',''),dc.get('level','word')),'changed_sentences':sum(a!=b for a,b in zip(raw,canon)),'max_angle_deg':max(errors),'per_clip_angle_deg':errors,'clip_ids':names,'references':refs,'raw_hypotheses':raw,'canonical_hypotheses':canon,'ground_truth_hypotheses':gt_h,'keyframe_elements_body':int(np.count_nonzero(body['keyframes'])),'keyframe_elements_hand':int(np.count_nonzero(hand['keyframes']))}
            assert result['max_angle_deg']<.01;(folder/(mode+'_scores.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False));np.savez_compressed(folder/(mode+'_loaded_poses.npz'),**{f'{p}{i:04d}':x for p,a in [('raw',xs),('canonical',ys)] for i,x in enumerate(a)})
            summary[mode]={k:v for k,v in result.items() if k not in ['references','raw_hypotheses','canonical_hypotheses','ground_truth_hypotheses','clip_ids','per_clip_angle_deg']};print('SCORED',corpus,mode,json.dumps(summary[mode]),flush=True)
        (folder/'summary.json').write_text(json.dumps(summary,indent=2));del model;torch.cuda.empty_cache()

if __name__=='__main__':
    if sys.argv[-1]=='prepare':prepare()
    elif sys.argv[-1]=='score':score()
    else:generate();score()
