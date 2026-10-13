"""Validate a separate public XYZ backtranslator in its official DEV contract."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

REPO=Path('external/SLRTP_Evaluation_20261011').resolve()
sys.path.insert(0,str(REPO))
from back_translation.back_translate import make_back_translation_model,back_translate
from external_metrics.sacrebleu import raw_corpus_bleu

ROOT=Path('reports/cvpr_evaluation_confirmation_20261011')
DATA=ROOT/'external_bt/extracted/SLRTP-Sign-Production-Evaluation-Data'


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(42)
    native=torch.load(DATA/'data/dev.pt',map_location='cpu',weights_only=True)
    predictions=torch.load(DATA/'data/PT_baseline_dev.pt',map_location='cpu',weights_only=True)
    keys=list(native);references=[native[k]['text'] for k in keys]
    model=make_back_translation_model(DATA/'backTranslation_PHIX_model')
    assert len(native)==515 and set(native)==set(predictions)
    results={}
    for name in ['ground_truth','progressive_transformer']:
        source=[native[k]['poses_3d'] for k in keys] if name=='ground_truth' else [predictions[k]['poses_3d'] if isinstance(predictions[k],dict) else predictions[k] for k in keys]
        assert all(x.ndim==3 and x.shape[1:]==(178,3) and torch.isfinite(x).all() for x in source)
        # Official GT is25fps. PT baseline is already half-rate (--fps12).
        poses=[(x[::2] if name=='ground_truth' else x).float().cpu() for x in source]
        hypotheses=back_translate(model,poses)
        bleu=raw_corpus_bleu(hypotheses,[references]);scores={f'bleu{i+1}':v for i,v in enumerate(bleu.scores)}
        result={'clips':len(keys),'clip_ids':keys,'references':references,'hypotheses':hypotheses,'bleu':scores,
            'input_format':'T x 178 x 3 XYZ; GT25fps decimated2; PT already12fps','source_frames':list(map(len,source)),
            'input_frames':list(map(len,poses)),'official_beam_size':model.beam_size,'official_beam_alpha':model.beam_alpha,
            'official_max_output_length':model.max_output_len,'6D_gauge_attack_executed':False}
        (ROOT/'external_bt'/(name+'_scores.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        results[name]={'bleu':scores,'clips':len(keys)};print('EXTERNAL_BT_SCORE',name,json.dumps(scores),flush=True)
    expected=13.378651856913775;actual=results['ground_truth']['bleu']['bleu4']
    contract={'repository':str(REPO),'commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'native_input_dimensions':534,'6D_input_dimensions_audited_elsewhere':240,
        'official_DEV_BLEU4':expected,'reproduced_DEV_BLEU4':actual,'absolute_difference':abs(expected-actual),
        'native_format_validation_passed':abs(expected-actual)<1e-8,'same_BT_checkpoint_as_SignSparK':False,
        'same_published_architecture_family_SignLanguageTransformers':True,
        '6D_gauge_vulnerability_replication':False,'reason':'XYZ input does not expose the redundant 6D coordinates; no licensed mesh/validated corresponding conversion available',
        'TEST_scored_with_external_BT':False,'results':results}
    assert contract['native_format_validation_passed']
    (ROOT/'external_bt/contract_and_reproduction.json').write_text(json.dumps(contract,indent=2)+'\n')
    print('EXTERNAL_BT_NATIVE_VALIDATED',json.dumps(contract),flush=True)


if __name__=='__main__':main()
