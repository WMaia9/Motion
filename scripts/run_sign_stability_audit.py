"""Replay the exact native real-sign audit and decode norm/physical controls."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.sign_stability_controls import transform,CONDITIONS,SEEDS
from run_signspark_geometry_dev import project,load_config,_build_split,make_back_translation_model,back_translate,score_translations
from motion_release_audit_common import sha,read,write

R=Path('reports/cvpr_sign_stability_20261014')
SETTINGS={
 'How2Sign':{'checkpoint':Path('reports/cvpr_signspark_baseline_20261010/checkpoints/How2Sign'),'data':Path('reports/cvpr_evaluation_confirmation_20261011/data/lmdb'),'baseline':Path('reports/cvpr_evaluation_confirmation_20261011/heldout/How2Sign/full_real_scores.json'),'corpus_index':0},
 'CSL-Daily':{'checkpoint':Path('reports/cvpr_representation_paper_20261011/checkpoints/CSL-Daily'),'data':Path('reports/cvpr_representation_paper_20261011/data/lmdb'),'baseline':Path('reports/cvpr_representation_paper_20261011/CSL-Daily/full_real_scores.json'),'corpus_index':1}}


def protocol():
    inputs={}
    for values in SETTINGS.values():
        for p in [values['baseline'],*values['checkpoint'].glob('*.yaml'),*values['checkpoint'].glob('*.ckpt'),*values['checkpoint'].glob('*.vocab')]:inputs[str(p)]=sha(p)
    group=Path('reports/cvpr_evaluation_confirmation_20261011/sources/how2sign_test.csv');inputs[str(group)]=sha(group)
    return {'corpora':list(SETTINGS),'counts':{'How2Sign':2183,'CSL-Daily':1162},'conditions':CONDITIONS,'no_caption_or_evaluator_gradient_in_interventions':True,
      'inputs_sha256':inputs,'scientific_sources_sha256':{str(p):sha(p) for p in [Path(__file__),Path('scripts/sign_stability_controls.py'),Path('scripts/run_signspark_geometry_dev.py'),Path('external/SignSparK_BT/signspark_bt/inference.py')]},
      'source_manifest_sha256':{str(r):sha(r/'manifest_sha256.json') for r in [Path('reports/cvpr_evaluation_confirmation_20261011'),Path('reports/cvpr_representation_paper_20261011')]},
      'source_inputs':'exact native full TEST IDs/order/crops; shared left-hand projection60:150 before every condition',
      'decoder':'unchanged author weights/config/vocab/beam; eval; batch32, original order, no recognition; full raw/canonical replay must match frozen strings',
      'budget':'each clip B=||canonical(x)-x||_2 over all valid240 features; no corpus-average RMS approximation; no evaluator normalizer added',
      'Gaussian_support':'IID Gaussian on original body/right-hand150 channels; left90 protected; each clip rescaled to B; three preset fields averaged per clip for primary comparison',
      'Gaussian_all':'secondary full240 Gaussian; first preset seed, rescaled per clip to B',
      'Gaussian_seeds':SEEDS,'field_index':'seed+1000000*corpus_index+clip_index; no score-selected seeds',
      'physical_small':'1degree about per-joint isotropic axes fixed in time; rotate both original3-vectors together; preserve norms/dot products; left90 protected',
      'physical_matched':'same axes, per-clip angle exactly solved from B in [0,pi]; original redundant coefficients retained; mark unreachable clips explicitly, never silently drop; report angle distribution, not assume all small',
      'input_norm_tolerance':'5e-6*max(B,1) for float32 outputs, realized values retained',
      'primary_endpoints':'metric-token changed fraction and mean normalized Levenshtein distance versus raw; wordsHow2Sign, charactersCSL preserving punctuation; raw string changed fraction and unnormalized edit count also retained',
      'primary_interval_family':20,'primary_family_detail':'canonical, Gaussian-support seed-average, Gaussian-all, physical1deg, physicalmatched x2corpora x2endpoints; paired cluster bootstrap,20k draws, Bonferroni family20',
      'contrast_interval_family':8,'contrast_family_detail':'canonical minus Gaussian-support-average or physicalmatched x2corpora x2endpoints; separate8endpoint paired bootstrap family',
      'bootstrap_groups':'official VIDEO_ID149How2Sign, clipCSL; same weights across conditions; no independent-seed or untouchedTEST population claim',
      'empty_hypotheses':'retained; normalized edit divisor=max(lengthraw,lengthvariant,1)',
      'BLEU':'all original references evaluated after decoding; scores and per-seed outputs retained; no reference strings passed to decoder',
      'no_training_new_generators_or_reward_fits':True}


def prepare():
    assert not (R/'manifest_sha256.json').exists();R.mkdir(exist_ok=True)
    p=R/'protocol.json';record=protocol()
    if p.exists():assert read(p)==record
    else:write(p,record)
    print('SIGN_STABILITY_REGISTERED',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--corpus',choices=SETTINGS);args=p.parse_args()
    if args.prepare:prepare();return
    assert args.corpus and not (R/'manifest_sha256.json').exists();registered=read(R/'protocol.json');assert registered==protocol()
    cfg=SETTINGS[args.corpus];folder=R/args.corpus;folder.mkdir(exist_ok=True)
    config=load_config(cfg['checkpoint']/'config.yaml');dc=dict(config['data']);dc.pop('pseudo_gloss_dir',None)
    dc.update(base_path=str(cfg['data'].resolve()),index_cache=False,max_sgn_length=300)
    data=_build_split(dc,'test',None);entries=[data[i] for i in range(len(data))];old=read(cfg['baseline'])
    assert [e.sequence for e in entries]==old['clip_ids']
    assert [' '.join(e.txt) for e in entries]==old['references']
    xs=[]
    for e in entries:
        x=e.sgn.numpy().copy();x[:,60:150]=project(x)[:,60:150];xs.append(x)
    source=folder/'source_raw.npz'
    if not source.exists():np.savez_compressed(source,**{f'clip{i:04d}':x for i,x in enumerate(xs)})
    write(folder/'inputs.json',{'clip_ids':old['clip_ids'],'references':old['references'],'lengths':[len(x) for x in xs],'source_raw_sha256':sha(source),'data_config':dc,'original_baseline_sha256':sha(cfg['baseline'])})
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=make_back_translation_model(cfg['checkpoint']);model.do_recognition=False
    for parameter in model.parameters():parameter.requires_grad_(False)
    write(folder/'native_decoder.json',{'beam_size':model.beam_size,'beam_alpha':model.beam_alpha,'max_output_len':model.max_output_len,'eval':not model.training,'recognition':False,'no_reference_or_caption_input':True})
    for condition in CONDITIONS:
        destination=folder/(condition+'.json')
        if destination.exists():continue
        begun=time.monotonic();hypotheses=[];parameters=[];fixtures={};feasible=[]
        for start in range(0,len(xs),32):
            ys=[]
            for i in range(start,min(start+32,len(xs))):
                y,parameter=transform(xs[i],condition,i,cfg['corpus_index']);parameters.append(parameter);feasible.append(parameter['feasible'])
                if y is None:y=xs[i].copy() # explicit infeasible identity fallback; masked from matched inference claims
                ys.append(torch.from_numpy(y))
                if i<8 or i>=len(xs)-8:fixtures[f'clip{i:04d}']=y
            hypotheses.extend(back_translate(model,ys,batch_size=32))
            if start%320==0:print('SIGN_STABILITY_BATCH',args.corpus,condition,start,len(xs),round(time.monotonic()-begun,2),flush=True)
        assert len(hypotheses)==len(xs)
        if condition in ['raw','canonical']:assert hypotheses==old[condition]['hypotheses'],(args.corpus,condition,'baseline replay differs')
        score=score_translations(old['references'],hypotheses,dc.get('version',''),dc.get('level','word'))
        record={'corpus':args.corpus,'condition':condition,'hypotheses':hypotheses,'scores':score,'parameters':parameters,'feasible':feasible,'seconds':time.monotonic()-begun}
        np.savez_compressed(folder/(condition+'_fixtures.npz'),**fixtures);write(destination,record)
        print('SIGN_STABILITY_CONDITION_DONE',args.corpus,condition,score['bleu']['bleu4'],record['seconds'],flush=True)
    write(folder/'complete.json',{'all8_conditions_full_nativeTEST':True,'raw_and_canonical_hypotheses_match_frozen_baselines':True,'no_inference_caption_input':True})
    print('SIGN_STABILITY_CORPUS_COMPLETE',args.corpus,flush=True)


if __name__=='__main__':main()
