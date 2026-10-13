"""Isolate native MDM namespaces from the evaluator preparation process."""
import sys,json,hashlib
from pathlib import Path
R=Path('reports/cvpr_humanml3d_blind_fid_20261012')
def main():
 assert not (R/'manifest_sha256.json').exists() and not (R/'test_mdm.npz').exists()
 p=json.loads((R/'protocol.json').read_text())
 for path,h in p['scientific_sources_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 import generate_humanml_mdm as generator
 generator.ROOT=R;sys.argv=['generate_humanml_mdm.py','test'];generator.main()
 (R/'generation_recovery.json').write_text(json.dumps({'reason':'Run native MDM in a fresh process to avoid evaluator utils namespace collision. No change to model, checkpoint, sampler or registered settings.','driver_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sampling_producer_unchanged':True,'new_TEST_evaluator_scores_not_observed':True},indent=2)+'\n')
if __name__=='__main__':main()
