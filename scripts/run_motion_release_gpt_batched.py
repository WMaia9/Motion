"""Larger native-forward GPT sampling batches; evaluation candidates remain32."""
from pathlib import Path
import numpy as np
import torch
import motion_release_native_adapters as original
from motion_release_fast_adapters import NativeGenerator as Base
import run_motion_release_audit as worker
from motion_release_audit_common import R,read,write,sha

class NativeGenerator(Base):
    @torch.no_grad()
    def generate(self,cohort):
        assert self.method=='t2mgpt';out=np.zeros_like(cohort['raw']);L=cohort['lengths'].copy();N=len(out)
        for i in range(0,N,128):
            E=self.text.encode_text(self.clip.tokenize(cohort['captions'][i:i+128],truncate=True).cuda()).float();tokens=self.sample_gpt(E)
            for size in sorted({x.shape[1] for x in tokens}):
                jj=[j for j,x in enumerate(tokens) if x.shape[1]==size];indices=torch.cat([tokens[j] for j in jj]);vq=self.vae.vqvae;emb=vq.quantizer.dequantize(indices).permute(0,2,1).contiguous();x=vq.postprocess(vq.decoder(emb)).cpu().numpy()
                for j,values in zip(jj,x):L[i+j]=min(len(values),196);out[i+j,:L[i+j]]=values[:196]*self.std+self.mean
            print('GENERATE_GPT_BATCH128',min(i+128,N),N,flush=True)
        assert (L>0).all() and np.isfinite(out).all()
        if self.failures:write(R/'t2mgpt_native_sampling_failures_batch128.json',self.failures)
        return out,L

def register(method):
    assert method=='t2mgpt';p=R/'producer_t2mgpt_batch128.json';sources=[Path(__file__),*[Path('scripts')/x for x in ['motion_release_fast_adapters.py','motion_release_native_adapters.py','run_motion_release_audit.py','motion_release_audit_common.py']]]
    record={'method':method,'sources_sha256':{str(x):sha(x) for x in sources},'remaining_generation_batch_size':128,'evaluation_batch_size':32,'conditional_sampling_law_and_EOS_unchanged':True,'RNG_draw_order_and_float32_batching_change':True,'purpose':'GPU throughput, no condition or valid repetition selected/removed'}
    if p.exists():
        saved=read(p)
        for key in ['initial_valid_batch32_repetitions','initial_generated_batch32_repetitions','revision_history']:saved.pop(key,None)
        assert saved==record
    else:
        record['initial_valid_batch32_repetitions']=[int(x.parent.name.split('_')[1]) for x in sorted((R/method).glob('rep_*/metrics.json'))];write(p,record)
    for x,h in read(R/'protocol.json')['producer_sha256'].items():assert sha(x)==h

if __name__=='__main__':
    original.NativeGenerator=NativeGenerator;worker.register=register;worker.main()
