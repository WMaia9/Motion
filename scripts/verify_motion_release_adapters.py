"""Synthetic native-function parity; no TEST score or checkpoint selection."""
import argparse,ast,sys,types,inspect
from pathlib import Path
import numpy as np
import torch
from motion_release_audit_common import R,seed,sha,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['mld','motionlcm_v1','t2mgpt','t2m']);method=parser.parse_args().method;seed(9981)
    if method=='t2m':
        if 'spacy' not in sys.modules:sys.modules['spacy']=types.ModuleType('spacy')
        from motion_release_t2m_grouped import NativeGenerator
    else:
        from motion_release_fast_adapters import NativeGenerator
    g=NativeGenerator(method);errors={}
    with torch.no_grad():
        if method=='t2mgpt':
            E=g.text.encode_text(g.clip.tokenize(['a person walks forward','a person raises an arm']).cuda()).float()
            for size in [5,20]:
                idx=torch.randint(512,(2,size),device='cuda');batch=g.trans(idx,E);single=torch.cat([g.trans(idx[i:i+1],E[i:i+1]) for i in range(2)]);errors[f'logits_{size}']=float((batch-single).abs().max())
                vq=g.vae.vqvae;B=vq.postprocess(vq.decoder(vq.quantizer.dequantize(idx).permute(0,2,1).contiguous()));S=torch.cat([g.vae.forward_decoder(idx[i:i+1]) for i in range(2)]);errors[f'decoder_{size}']=float((B-S).abs().max())
            original=torch.distributions.Categorical.sample
            try:
                torch.distributions.Categorical.sample=lambda self:torch.argmax(self.probs,dim=-1)
                batched=g.sample_gpt(E)
                for i in range(2):
                    try:native=g.trans.sample(E[i:i+1],False)
                    except UnboundLocalError:native=torch.ones((1,1),device='cuda',dtype=torch.long)
                    assert torch.equal(batched[i],native),'native EOS/truncation parity'
            finally:torch.distributions.Categorical.sample=original
            errors['greedy_native_EOS_tokens_exact']=True
        elif method=='t2m':
            import networks.modules as modules
            original=modules.reparameterize
            try:
                modules.reparameterize=lambda mu,logvar:mu
                W=torch.randn((2,22,300),device='cuda')*.1;P=torch.randn((2,22,15),device='cuda')*.1;S=torch.tensor([12,12]);L=torch.tensor([48,48]);batch=g.trainer.generate(W,P,S,L,12,263)[0];single=torch.cat([g.trainer.generate(W[i:i+1],P[i:i+1],S[i:i+1],L[i:i+1],12,263)[0] for i in range(2)]);errors['equal_caption_and_motion_length']=float((batch-single).abs().max())
                S=torch.tensor([12,8]);batch=g.trainer.generate(W,P,S,L,12,263)[0];single=torch.cat([g.trainer.generate(W[i:i+1],P[i:i+1],S[i:i+1],L[i:i+1],12,263)[0] for i in range(2)]);errors['excluded_unequal_caption_support_difference']=float((batch-single).abs().max());assert errors['excluded_unequal_caption_support_difference']>1e-4
            finally:modules.reparameterize=original
        else:
            source=Path('external/MLD' if method=='mld' else 'external/MotionLCM')/'mld/models/modeltype/mld.py';tree=ast.parse(source.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MLD');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_diffusion_reverse')
            import tqdm
            ns={'torch':torch,'inspect':inspect,'tqdm':tqdm,'get_guidance_scale_embedding':getattr(g,'guidance_embedding',None),'Optional':__import__('typing').Optional};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
            g.cfg=types.SimpleNamespace(model=types.SimpleNamespace(scheduler=types.SimpleNamespace(num_inference_timesteps=50,num_inference_steps=1,eta=0.)),DATASET=types.SimpleNamespace(NFEATS=263));g.do_classifier_free_guidance=method=='mld';g.vae_type='actor';g.latent_dim=[1,256];g.guidance_scale=7.5;g.is_controlnet=False;g.guess_mode=False
            caps=['a person walks forward','a person raises an arm'];L=[64,80]
            if method=='mld':
                ids=g.tokenizer(['','']+caps,padding='max_length',truncation=True,max_length=g.tokenizer.model_max_length,return_tensors='pt').input_ids.cuda();E=g.text.get_text_features(ids).unsqueeze(1);seed(2234);native_z=ns[fn.name](g,E,L)
            else:
                E=g.text.encode(caps,show_progress_bar=False,convert_to_tensor=True,batch_size=2).clone().unsqueeze(1);seed(2234);z=torch.randn((2,1,256),device='cuda');native_z=ns[fn.name](g,z,E)
            mask=torch.arange(80,device='cuda')[None,:]<torch.tensor(L,device='cuda')[:,None];native_x=g.vae.decode(native_z,L if method=='mld' else mask).cpu().numpy();expected=np.zeros((2,196,263),np.float32)
            for j,l in enumerate(L):expected[j,:l]=native_x[j,:l]*g.std+g.mean
            c={'raw':np.zeros((2,196,263),np.float32),'lengths':np.array(L),'captions':caps};seed(2234);actual,_=g.generate(c);errors['native_reverse_and_decoder_unnormalized_output']=float(abs(expected-actual).max());errors['native_reverse_source_sha256']=sha(source)
    numerical=[v for k,v in errors.items() if isinstance(v,float) and not k.startswith('excluded')];tolerance=1e-3 if method=='t2mgpt' else 1e-4;assert max(numerical)<tolerance,errors
    write(R/f'adapter_parity_{method}.json',{'passed':True,'synthetic_inputs_only':True,'errors':errors,'float32_batch_operation_tolerance':tolerance,'batched_and_single_precision_not_bitwise_identical':method in ['t2mgpt','t2m'],'producer_sha256':sha(__file__),'raw_array_pipeline_source':'scripts/run_motion_release_audit.py','Torch':torch.__version__});print('NATIVE_ADAPTER_PARITY',method,errors,flush=True)
if __name__=='__main__':main()
