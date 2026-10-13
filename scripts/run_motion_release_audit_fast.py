"""Registered component corrections/vectorization, preserving original producers."""
from pathlib import Path
import motion_release_native_adapters as serial
from motion_release_fast_adapters import NativeGenerator
import run_motion_release_audit as worker
from motion_release_audit_common import R,read,write,sha

def register(method):
    files=[Path(__file__),Path('scripts/run_motion_release_audit.py'),Path('scripts/motion_release_audit_common.py'),Path('scripts/motion_release_native_adapters.py'),Path('scripts/motion_release_fast_adapters.py')]
    record={'method':method,'source_sha256':{str(p):sha(p) for p in files},'original_registration':'producer_'+method+'.json','reference_draw_seed_offset':100000,'generation_rng_seed_offset':200000,'conditions_unchanged':True,'vectorized_native_conditional_sampling':'T2M/GPT only; different RNG draw order; native components and equal-length decoder boundaries','MotionLCM_correction':'original and current native configs require Sentence-T5, not CLIP; initial incompatible outputs excluded and archived'}
    p=R/f'producer_{method}_v2.json'
    if p.exists():assert read(p)==record
    else:write(p,record)
    for p,h in read(R/'protocol.json')['producer_sha256'].items():assert sha(p)==h

if __name__=='__main__':
    serial.NativeGenerator=NativeGenerator;worker.register=register;worker.main()
