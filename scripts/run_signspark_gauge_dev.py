"""Render-parameter equivalence and fixed-reader sensitivity; exploratory."""
import argparse,ast,json,os,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from run_signspark_geometry_dev import ROOT,matrices,load_config,_build_split,make_back_translation_model,back_translate,score_translations

def gauge(x,condition):
    d=x.reshape(-1,40,6).copy()
    if condition=='gauge_scale090':return x*.9
    if condition=='gauge_scale110':return x*1.1
    if condition=='gauge_shear010':d[...,3:]+=.1*d[...,:3]
    if condition in ('gauge_only_identity_rotations','allidentity_canonical'):
        a,b=d[...,:3],d[...,3:]
        na=np.linalg.norm(a,axis=-1);h=(a*b).sum(-1)/np.maximum(na,1e-12)
        q=np.sqrt(np.maximum((b*b).sum(-1)-h*h,0))
        d[:]=0
        if condition=='allidentity_canonical':d[...,0]=1;d[...,4]=1
        else:d[...,0]=na;d[...,3]=h;d[...,4]=q
    return d.reshape(x.shape)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args()
    torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    os.environ['DATA_ROOT']=str((ROOT/'data').resolve());dest=ROOT/args.dataset
    cfg=load_config(ROOT/'checkpoints'/args.dataset/'config.yaml');data_cfg=dict(cfg['data']);data_cfg.pop('pseudo_gloss_dir',None)
    dataset=_build_split(data_cfg,'dev',None);xs=[dataset[i].sgn.numpy() for i in range(len(dataset))];refs=[' '.join(e.txt) for e in dataset.examples]
    # Extract and execute only the pinned upstream pure rotation helper, avoiding
    # mesh dependencies. AST source is archived and its outputs checked directly.
    source=Path('external/SignSparK/tools/visualize.py').read_text()
    tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rot6d_to_matrix')
    helper=ast.get_source_segment(source,fn);(ROOT/'sources'/'upstream_rot6d_helper.py').write_text(helper+'\n')
    namespace={'torch':torch};exec(compile(helper,'upstream_rot6d_helper','exec'),namespace);official=namespace['rot6d_to_matrix']
    model=make_back_translation_model(ROOT/'checkpoints'/args.dataset);order=np.argsort([len(x) for x in xs],kind='stable')
    original=json.loads((dest/'original.json').read_text())['hypotheses'];conditions=json.loads((ROOT/'gauge_extension_plan.json').read_text())['conditions'];summary={}
    for condition in conditions:
        started=time.time();ys=[gauge(x,condition).astype(np.float32) for x in xs];max_angle=0.;min_det=1.;min_hand_norm=1.;helper_diff=0.
        for x,y in zip(xs,ys):
            assert np.isfinite(y).all()
            ox=official(torch.from_numpy(x).reshape(-1,6)).numpy().astype(np.float64)
            oy=official(torch.from_numpy(y).reshape(-1,6)).numpy().astype(np.float64)
            det=np.linalg.det(oy);min_det=min(min_det,float(det.min()));assert (det>.99).all()
            helper_diff=max(helper_diff,float(abs(oy-matrices(y).reshape(-1,3,3)).max()))
            if condition.startswith('gauge_scale') or condition=='gauge_shear010':
                max_angle=max(max_angle,float(np.degrees(Rotation.from_matrix(ox.transpose(0,2,1)@oy).magnitude()).max()))
            else:max_angle=max(max_angle,float(np.degrees(Rotation.from_matrix(oy).magnitude()).max()))
        assert max_angle<.001,(condition,max_angle)
        hypotheses=['']*len(xs)
        for start in range(0,len(order),32):
            ix=order[start:start+32];pred=back_translate(model,[torch.from_numpy(ys[i]) for i in ix],batch_size=32)
            for i,h in zip(ix,pred):hypotheses[i]=h
        result={'scores':score_translations(refs,hypotheses,cfg['data'].get('version',''),cfg['data'].get('level','word')),'hypotheses':hypotheses,'same_as_original':sum(a==b for a,b in zip(original,hypotheses)),'maximum_rotation_error_deg':max_angle,'minimum_rotation_determinant':min_det,'official_helper_vs_independent_max_abs':helper_diff,'error_reference':'source rotations' if condition.startswith('gauge_scale') or condition=='gauge_shear010' else 'identity rotations','seconds':time.time()-started}
        (dest/f'{condition}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));summary[condition]={k:v for k,v in result.items() if k!='hypotheses'}
        (dest/'gauge_summary.json').write_text(json.dumps(summary,indent=2));print(args.dataset,condition,json.dumps(summary[condition]),flush=True)

if __name__=='__main__':main()
