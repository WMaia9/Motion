"""Compile native float32 inference; preserve the1000-step DDPM and all weights."""
from pathlib import Path
import sys
import torch
from motion_release_audit_common import write,read,sha,seed
from motiondiffuse_vectorized_adapter import VectorizedMotionDiffuse
import run_motiondiffuse_vectorized_audit as worker
R=Path('reports/cvpr_motiondiffuse_compiled_20261013')

class CompiledNative(VectorizedMotionDiffuse):
 def __init__(self,dataset):
  super().__init__(dataset);eager=self.model.forward
  compiled=torch.compile(eager,backend='inductor',mode='reduce-overhead',fullgraph=True)
  def forward(*args,**kwargs):
   torch.compiler.cudagraph_mark_step_begin()
   return compiled(*args,**kwargs).clone()
  with torch.no_grad():
   L=torch.tensor([40,64,100,196]*32,device='cuda');x=torch.randn((128,196,self.dim),device='cuda');proj,tokens=self.model.encode_text(['a person walks forward']*128,torch.device('cuda'));maximum=0.
   for step in [0,500,999]:
    tt=torch.full((128,),step,device='cuda');a=eager(x,tt,length=L,xf_proj=proj,xf_out=tokens);b=forward(x,tt,length=L,xf_proj=proj,xf_out=tokens);err=float(abs(a-b).max());assert err<5e-4;maximum=max(err,maximum)
  self.model.forward=forward
  self.parity.update(compiled_native_forward=True,compiler='torch2.11inductor reduce-overhead, float32 TF32disabled',compiled_forward_max_difference=maximum,compiled_parity_timesteps=[0,500,999],compiled_parity_batch=128,native_layers_and_parameters_unchanged=True)
  print('NATIVE_COMPILED_PARITY',dataset,self.parity,flush=True)

def main():
 dataset=sys.argv[1];assert dataset in ['humanml','kit'];R.mkdir(exist_ok=True);files=[Path(__file__),Path('scripts/run_motiondiffuse_vectorized_audit.py'),Path('scripts/motiondiffuse_vectorized_adapter.py'),Path('scripts/additional_motion_adapters.py')]
 registration={'dataset':dataset,'change':'native float32 forward compiled; operator fusion may change floatingpoint rounding, no layer/state/distribution substitution','diffusion_steps':1000,'TF32':False,'native_forward_parity_threshold':.0005,'all20draws_and_both_raw_C_conditions':True,'previous_incomplete_eager_attempts_excluded':True,'source_sha256':{str(f):sha(f) for f in files}}
 p=R/(dataset+'_compiled_registration.json')
 if p.exists():assert read(p)==registration
 else:write(p,registration)
 worker.R=R;worker.VectorizedMotionDiffuse=CompiledNative;worker.main()
if __name__=='__main__':main()
