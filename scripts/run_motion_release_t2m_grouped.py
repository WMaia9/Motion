"""Correct native conditional batching; prior incompatible attempt archived."""
from pathlib import Path
import motion_release_native_adapters as serial
from motion_release_t2m_grouped import NativeGenerator
import run_motion_release_audit as worker
from motion_release_audit_common import R,read,write,sha

def register(method):
    assert method=='t2m'
    files=[Path(__file__),*[Path('scripts')/p for p in ['motion_release_t2m_grouped.py','motion_release_fast_adapters.py','motion_release_native_adapters.py','run_motion_release_audit.py','motion_release_audit_common.py']]]
    record={'method':method,'source_sha256':{str(p):sha(p) for p in files},'original_protocol_conditions_unchanged':True,'native_conditional_support':'group by BOTH motion duration and caption length; unmasked native attention sees no additional padding','excluded_attempt':'invalid_attempts/t2m_motion_length_only_groups','generation_rng_seed_offset':200000,'reference_draw_seed_offset':100000,'no_score_based_hyperparameter_selection':True}
    p=R/'producer_t2m_v3.json'
    if p.exists():assert read(p)==record
    else:write(p,record)
    for p,h in read(R/'protocol.json')['producer_sha256'].items():assert sha(p)==h
if __name__=='__main__':
    serial.NativeGenerator=NativeGenerator;worker.register=register;worker.main()
