"""Official VIDEO_ID sensitivity for ASL heldout attacks; no tuning."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from analyze_signspark_geometry_dev import statistics,b4
from run_evaluation_heldout import ROOT


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    path=Path('/mnt/datasets/sign_datasets/raw/how2sign/How2Sign/sentence_level/test/text/en/raw_text/how2sign_test.csv')
    with path.open() as f:lookup={r['SENTENCE_NAME']:r['VIDEO_ID'] for r in csv.DictReader(f,delimiter='\t')}
    shutil.copy2(path,ROOT/'sources/how2sign_test.csv')
    result={'group_field':'official VIDEO_ID, not signer','post_result_cluster_sensitivity':True,'seed':42,'replicates':2000,'conditions':{}}
    folder=ROOT/'heldout/How2Sign'
    for name in ['correct_caption','rolled_wrong_caption']:
        r=json.loads((folder/(name+'_scores.json')).read_text())
        groups=[lookup[key] for key in r['clip_ids']];unique=sorted(set(groups));indices=np.asarray([unique.index(g) for g in groups])
        weights=np.random.default_rng(42).multinomial(len(unique),np.ones(len(unique))/len(unique),size=2000)
        def boot(refs,hyps):
            stat=statistics(refs,hyps);grouped=np.zeros((len(unique),10),np.int64);np.add.at(grouped,indices,stat)
            return b4(stat.sum(0)),np.asarray([b4(row) for row in weights@grouped])
        before,bb=boot(r['targets'],r['baseline_actual']['hypotheses'])
        after,ba=boot(r['targets'],r['raw_after_actual']['hypotheses'])
        result['conditions'][name]={'clips':128,'groups':len(unique),'delta_target_bleu4':after-before,
            'ci95':np.percentile(ba-bb,[2.5,97.5]).tolist(),
            'bootstrap_groups':dict(zip(r['clip_ids'],groups)),
            'wrong_targets_not_reassigned_within_bootstrap':name=='rolled_wrong_caption'}
    (ROOT/'cluster_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ASL_CLUSTER_SENSITIVITY',json.dumps(result),flush=True)


if __name__=='__main__':main()
