"""Same MLD sampler/components with larger inference batches; metrics stay32.

This changes random draw order and float32 batching, not the conditional
sampler, guidance, weights, crop, output count, or metric candidate pool.
Completed native-batch32 repetitions are retained without selection.
"""
from pathlib import Path
import numpy as np
import torch
import motion_release_native_adapters as original
from motion_release_native_adapters import NativeGenerator as Base
import run_motion_release_audit as worker
from motion_release_audit_common import R,read,write,sha

class NativeGenerator(Base):
    @torch.no_grad()
    def generate(self,cohort):
        assert self.method=='mld';out=np.zeros_like(cohort['raw']);L=cohort['lengths'].copy()
        for i in range(0,len(out),128):
            caps=cohort['captions'][i:i+128];lengths=L[i:i+128].tolist();n=len(caps);ids=self.tokenizer(['']*n+caps,padding='max_length',truncation=True,max_length=self.tokenizer.model_max_length,return_tensors='pt').input_ids.cuda();E=self.text.get_text_features(ids).unsqueeze(1)
            z=torch.randn((n,1,256),device='cuda')*self.scheduler.init_noise_sigma;self.scheduler.set_timesteps(50)
            for t in self.scheduler.timesteps.to('cuda'):
                eps=self.denoiser(sample=torch.cat([z,z]),timestep=t,encoder_hidden_states=E,lengths=lengths*2)[0];a,b=eps.chunk(2);z=self.scheduler.step(a+7.5*(b-a),t,z,eta=0.).prev_sample
            x=self.vae.decode(z.permute(1,0,2),lengths).cpu().numpy()*self.std+self.mean
            for j,l in enumerate(lengths):out[i+j,:l]=x[j,:l]
            print('GENERATE_MLD_BATCH128',i+n,len(out),flush=True)
        return out,L

def register(method):
    assert method=='mld';p=R/'producer_mld_batch128.json'
    sources=[Path(__file__),Path('scripts/motion_release_native_adapters.py'),Path('scripts/run_motion_release_audit.py'),Path('scripts/motion_release_audit_common.py')]
    record={'method':method,'sources_sha256':{str(x):sha(x) for x in sources},'initial_valid_batch32_repetitions':[0,1],'remaining_generation_batch_size':128,'evaluation_batch_size':32,'output_count_and_full_reference_unchanged':True,'conditional_sampler_and_parameters_unchanged':True,'RNG_draw_order_and_float32_batching_change':True,'purpose':'execution throughput; no model or metric condition removed or selected'}
    if p.exists():assert read(p)==record
    else:write(p,record)
    for x,h in read(R/'protocol.json')['producer_sha256'].items():assert sha(x)==h
if __name__=='__main__':
    original.NativeGenerator=NativeGenerator;worker.register=register;worker.main()
