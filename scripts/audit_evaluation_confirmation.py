"""Read-only audit of the frozen report, suitable for subsequent reruns."""
import argparse,json
from pathlib import Path
from verify_signspark_studies import sha
from analyze_signspark_geometry_dev import statistics,b4


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('reports/cvpr_evaluation_confirmation_20261011'))
    args=parser.parse_args();root=args.root
    manifest=json.loads((root/'manifest_sha256.json').read_text())
    for filename,digest in manifest.items():assert sha(root/filename)==digest,filename
    verified=0;pairs=0
    for folder in (root/'rankings').iterdir():
        if not folder.is_dir():continue
        can=json.loads((folder/'canonical_scores.json').read_text())
        for path in folder.glob('*_scores.json'):
            r=json.loads(path.read_text());refs=r['references']
            assert abs(b4(statistics(refs,r['hypotheses']).sum(0))-r['scores']['bleu']['bleu4'])<1e-10
            verified+=len(refs)
            if 'defended' in r:
                assert r['defended']['hypotheses']==can['hypotheses'];pairs+=len(refs)
                assert abs(b4(statistics(refs,r['defended']['hypotheses']).sum(0))-r['defended']['scores']['bleu']['bleu4'])<1e-10
                verified+=len(refs)
    for folder in (root/'heldout').iterdir():
        for condition in ['correct_caption','rolled_wrong_caption']:
            r=json.loads((folder/(condition+'_scores.json')).read_text())
            assert r['canonical_before_actual']['hypotheses']==r['canonical_after_actual']['hypotheses'];pairs+=r['clips']
            actual=b4(statistics(r['references'],r['raw_after_actual']['hypotheses']).sum(0))
            target=b4(statistics(r['targets'],r['raw_after_actual']['hypotheses']).sum(0))
            assert abs(actual-r['raw_after_actual']['scores']['bleu']['bleu4'])<1e-10
            assert abs(target-r['raw_after_target_scores']['bleu']['bleu4'])<1e-10
    for name in ['ground_truth','progressive_transformer']:
        r=json.loads((root/'external_bt'/(name+'_scores.json')).read_text())
        assert abs(b4(statistics(r['references'],r['hypotheses']).sum(0))-r['bleu']['bleu4'])<1e-10
    print(json.dumps({'report':str(root),'verified_files':len(manifest),'manifest_sha256':sha(root/'manifest_sha256.json'),
        'generated_decodes_BLEU_recomputed':verified,'canonical_hypothesis_pairs_verified':pairs,'passed':True},indent=2))


if __name__=='__main__':main()
