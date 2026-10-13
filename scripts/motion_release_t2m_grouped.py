"""Native T2M batching with both duration AND caption length held fixed.

Its attention does not mask packed-sequence padding. Grouping by duration
alone changes the conditional law for shorter captions, so is excluded.
"""
import numpy as np
import torch
from motion_release_fast_adapters import NativeGenerator as FastGenerator

class NativeGenerator(FastGenerator):
    def t2m(self):
        super().t2m();self.settings['sampling']='native length estimator; equal (motion duration, caption length) groups. This preserves the unmasked native attention support and movement-decoder boundaries. RNG draw order differs from serial sampling.'

    @torch.no_grad()
    def generate(self,cohort):
        if self.method!='t2m':return super().generate(cohort)
        N=len(cohort['raw']);units=np.zeros(N,dtype=int)
        for i in range(0,N,256):
            idx=np.arange(i,min(i+256,N));order=np.argsort(-cohort['sent_lengths'][idx],kind='stable');chosen=idx[order]
            W=torch.tensor(cohort['words'][chosen],device='cuda');P=torch.tensor(cohort['pos'][chosen],device='cuda');S=torch.tensor(cohort['sent_lengths'][chosen]);probs=self.length_estimator(W,P,S).softmax(-1);u=torch.multinomial(probs,1,replacement=True)[:,0]
            for _ in range(2):
                retry=u<10
                if retry.any():u[retry]=torch.multinomial(probs[retry],1,replacement=True)[:,0]
            units[chosen]=u.cpu().numpy()
        out=np.zeros_like(cohort['raw']);L=units*4;assert (L>0).all() and (L<=196).all()
        groups={}
        for i,(size,caption_length) in enumerate(zip(units,cohort['sent_lengths'])):groups.setdefault((int(size),int(caption_length)),[]).append(i)
        count=0
        for (size,caplen),indices in sorted(groups.items()):
            for k in range(0,len(indices),32):
                ix=np.array(indices[k:k+32]);W=torch.tensor(cohort['words'][ix],device='cuda');P=torch.tensor(cohort['pos'][ix],device='cuda');S=torch.full((len(ix),),caplen,dtype=torch.long);lengths=torch.full((len(ix),),size*4,dtype=torch.long)
                x=self.trainer.generate(W,P,S,lengths,size,263)[0].cpu().numpy()
                for j,values in zip(ix,x):out[j,:L[j]]=values*self.std+self.mean
                count+=len(ix)
            if count//256>(count-len(indices))//256:print('GENERATE_GROUPED_T2M',count,N,flush=True)
        return out,L
