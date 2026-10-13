"""Post-result contract diagnostic; preserve all pre-registered attack outputs."""
import json
from pathlib import Path
import numpy as np
import torch
from run_signspark_geometry_dev import load_config,_build_split,project,make_back_translation_model,back_translate,score_translations

ROOT=Path('reports/cvpr_representation_paper_20261011')
OLD=Path('reports/cvpr_evaluation_confirmation_20261011')
BASE=Path('reports/cvpr_signspark_baseline_20261010')


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(42)
    result={'post_result_contract_diagnostic':True,'reason':'CSL prepared baseline differs from public rounded6.98; left-stream projection was pre-registered for dump compatibility',
        'attack_outputs_not_replaced':True,'no_hyperparameter_tuning':True,'corpora':{}}
    for corpus,expected in [('PHOENIX-2014T',15.00),('How2Sign',3.53),('CSL-Daily',6.98)]:
        local=ROOT if corpus=='CSL-Daily' else OLD
        ck=(ROOT/'checkpoints' if corpus=='CSL-Daily' else BASE/'checkpoints')/corpus
        cfg=load_config(ck/'config.yaml');dc=dict(cfg['data']);dc.pop('pseudo_gloss_dir',None)
        dc.update(base_path=str((local/'data/lmdb').resolve()),index_cache=False,max_sgn_length=300)
        d=_build_split(dc,'test',None);e=[d[i] for i in range(len(d))];xs=[q.sgn.numpy().copy() for q in e];refs=[' '.join(q.txt) for q in e]
        prepared=[]
        for x in xs:
            y=x.copy();y[:,60:150]=project(x)[:,60:150];prepared.append(y)
        sourcefolder=ROOT/corpus if corpus=='CSL-Daily' else OLD/'heldout'/corpus
        archived=json.loads((sourcefolder/'full_real_scores.json').read_text())
        assert [q.sequence for q in e]==archived['clip_ids'] and refs==archived['references']
        model=make_back_translation_model(ck)
        h=back_translate(model,[torch.from_numpy(x) for x in xs],batch_size=32);scores=score_translations(refs,h,dc['version'],dc['level'])
        replay=back_translate(model,[torch.from_numpy(x) for x in prepared],batch_size=32)
        assert replay==archived['raw']['hypotheses']
        r={'clips':len(e),'clip_ids':[q.sequence for q in e],'references':refs,'native_loader_hypotheses':h,'native_loader_scores':scores,
            'prepared_left_canonical_bleu4':archived['raw']['scores']['bleu']['bleu4'],
            'changed_by_left_projection':sum(a!=b for a,b in zip(h,replay)),
            'max_left_projection_feature_delta':max(float(abs(a[:,60:150]-b[:,60:150]).max()) for a,b in zip(xs,prepared)),
            'public_rounded_bleu4':expected,'native_difference_from_public_rounded':scores['bleu']['bleu4']-expected,
            'prepared_baseline_replay_exact':True,'native_baseline_does_not_replace_attack_baseline':True}
        result['corpora'][corpus]=r;print('NATIVE_CONTRACT',corpus,scores['bleu']['bleu4'],r['changed_by_left_projection'],flush=True)
        del model;torch.cuda.empty_cache()
    (ROOT/'sources/native_loader_contract.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
