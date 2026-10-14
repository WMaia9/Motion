"""Native MotionDiffuse with an elementwise-equivalent frame mask.

The upstream mask loops over frames and copies each CUDA length to Python at
every denoising step. The replacement constructs exactly that0/1 mask on device;
native layers, diffusion1000 steps, normalization and weights are unchanged.
"""
import types
import numpy as np
import torch
from additional_motion_adapters import NativeAdditional

class VectorizedMotionDiffuse(NativeAdditional):
    def __init__(self,dataset):
        super().__init__('motiondiffuse',dataset)
        native=self.model.generate_src_mask
        L=torch.tensor([24,36,40,64,100,196],device='cuda');T=196
        def vectorized(instance,T,length):
            return (torch.arange(T,device=length.device)[None,:]<length.reshape(-1,1)).float()
        a=native(T,L).to('cuda');b=vectorized(self.model,T,L);assert torch.equal(a,b)
        with torch.no_grad():
            captions=['a person walks forward']*len(L);proj,tokens=self.model.encode_text(captions,torch.device('cuda'));x=torch.randn((len(L),T,self.dim),device='cuda');t=torch.tensor([0,100,250,500,750,999],device='cuda');before=self.model(x,t,length=L,xf_proj=proj,xf_out=tokens)
            self.model.generate_src_mask=types.MethodType(vectorized,self.model);after=self.model(x,t,length=L,xf_proj=proj,xf_out=tokens)
        error=float(abs(before-after).max());assert error==0
        self.parity={'native_mask_bitwise_equal':True,'native_forward_max_abs_difference':error,'native_diffusion_steps':1000,'no_layer_or_weight_replacement':True,'length_cases':L.tolist()}

    def generate(self,cohort,batch=128):
        # The author's evaluation code clips prescribed lengths to this floor.
        cohort['lengths']=np.maximum(cohort['lengths'],40 if self.dataset=='humanml' else 24)
        return super().generate(cohort,batch=batch)
