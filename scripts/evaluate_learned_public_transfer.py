"""Zero-fit transfer of the learned output layer to saved public SLP motions."""
from pathlib import Path
import json
import numpy as np
import torch
import run_learned_equivalence as study
from run_evaluation_heldout import geometry
from verify_signspark_studies import sha

SOURCE = Path('reports/cvpr_motion_consistency_confirmation_20261010/generated/PHOENIX-2014T')


@torch.no_grad()
def main():
    audit,stoi,generator = study.register()
    source = SOURCE/'text_duration_scores.json';result = study.read(source)
    path = SOURCE/'text_duration_loaded_poses.npz'
    with np.load(path) as z:
        keys = sorted(k for k in z.files if k.startswith('raw'))
        assert len(keys) == result['clips'] == 128
        xs = [study.base.project(z[k]) for k in keys]
    records = [{'clip_id':name,'frames':len(x),'text':ref.split()} for name,x,ref in zip(result['clip_ids'],xs,result['references'])]
    refs = result['references'];ids = result['clip_ids']
    study.write('public_transfer/protocol.json',{'source_pose_file':str(path),'source_sha256':sha(path),
        'source_scores_file':str(source),'source_scores_sha256':sha(source),
        'generator':'published SignSparK checkpoint and official sampling implementation, local text+duration replay',
        'selection':'fixed 128-clip DEV subset from earlier native generation study',
        'no_new_fitting':True,'local_output_layer_training_generator':'local frozen DDPM',
        'historical_public_generator_raw_bleu4':result['raw_scores']['bleu']['bleu4'],
        'baseline_convention':'canonicalize the public output to match the layer training convention',
        'not_a_reproduction_of_published_SOTA_scores':True})
    teacher = study.base.make_back_translation_model(study.base.BASE/'checkpoints/PHOENIX-2014T')
    teacher.eval();teacher.do_recognition=False
    baseline = study.score(teacher,xs,refs,ids);study.write('public_transfer/baseline.json',baseline)
    for seed in study.SEEDS:
        model = study.OutputLayer(generator).cuda()
        model.load_state_dict(torch.load(study.ROOT/f'training/seed{seed}/text_raw.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True);model.eval()
        outputs=[];coefficients=[]
        for start in range(0,len(records),32):
            group=records[start:start+32];lengths=torch.tensor([r['frames'] for r in group],device='cuda')
            pose=torch.zeros(len(group),int(lengths.max()),240,device='cuda')
            for j,x in enumerate(xs[start:start+32]):pose[j,:len(x)]=torch.from_numpy(x).cuda()
            y,c=model(pose,study.token_tensor(group,stoi),lengths)
            for j,n in enumerate(lengths.tolist()):outputs.append(y[j,:n].cpu().numpy().copy());coefficients.append(c[j,:n].cpu().numpy().copy())
        raw = study.score(teacher,outputs,refs,ids);canonical = study.score(teacher,[study.base.project(x) for x in outputs],refs,ids)
        assert baseline['hypotheses']==canonical['hypotheses']
        out={'seed':seed,'clips':128,'raw':raw,'canonical':canonical,'canonical_changed_hypotheses':0,
             'eval_optimization_steps':0,**geometry(xs,outputs)}
        study.write(f'public_transfer/seed{seed}.json',out)
        np.savez_compressed(study.ROOT/'public_transfer'/f'seed{seed}_poses.npz',
            **{f'{tag}{i:04d}':x for tag,values in [('source',xs),('attacked',outputs),('coefficient',coefficients)] for i,x in enumerate(values)})
        print('PUBLIC_GENERATOR_TRANSFER',seed,baseline['scores']['bleu']['bleu4'],raw['scores']['bleu']['bleu4'],canonical['scores']['bleu']['bleu4'],flush=True)
        del model;torch.cuda.empty_cache()
    study.write('public_transfer/complete.json',{'seeds':study.SEEDS,'clips':128,'no_new_fitting':True,'canonical_hypotheses_unchanged':True})


if __name__=='__main__':main()
