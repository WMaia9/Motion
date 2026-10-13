"""Correct MotionLCM text encoder and batch native autoregressive components.

The original serial inference producer remains intact. GPT sampling and T2M
generation use identical native forward functions and conditional laws, with
different RNG draw order. Equal-length groups preserve decoder boundaries.
"""
import ast,sys,importlib.machinery
from pathlib import Path
import numpy as np
import torch
from motion_release_native_adapters import NativeGenerator as SerialGenerator
from motion_release_audit_common import R,sha,write

class NativeGenerator(SerialGenerator):
    def latent(self):
        if self.method!='motionlcm_v1':return super().latent()
        sys.path[:0]=[str(Path(p).resolve()) for p in ['external/motion_lcm_python','external/MotionLCM']]
        if 'spacy' in sys.modules and sys.modules['spacy'].__spec__ is None:sys.modules['spacy'].__spec__=importlib.machinery.ModuleSpec('spacy',loader=None)
        from sentence_transformers import SentenceTransformer
        from mld.models.architectures.mld_vae import MldVae
        from mld.models.architectures.mld_denoiser import MldDenoiser
        from diffusers import LCMScheduler
        textroot=Path('models/motion_release_audit/sentence-t5-large')
        self.text=SentenceTransformer(str(textroot),device='cuda');self.text.eval()
        self.vae=MldVae(263,latent_dim=[1,256]);self.denoiser=MldDenoiser(latent_dim=[1,256],time_cond_proj_dim=256,controlnet_embed_dim=256,controlnet_act_fn=None)
        self.scheduler=LCMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',clip_sample=False,set_alpha_to_one=False,original_inference_steps=50)
        ckpt=Path('models/motion_release_audit/extracted/experiments_t2m/motionlcm_humanml/motionlcm_humanml_v1.ckpt');state=torch.load(ckpt,map_location='cpu',weights_only=False)['state_dict']
        for key,model in [('vae',self.vae),('denoiser',self.denoiser)]:
            model.load_state_dict({k[len(key)+1:]:v for k,v in state.items() if k.startswith(key+'.')},strict=True);model.cuda().eval()
        helper=Path('external/MotionLCM/mld/utils/utils.py');function=next(x for x in ast.parse(helper.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='get_guidance_scale_embedding');ns={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),str(helper),'exec'),ns);self.guidance_embedding=ns[function.name]
        self.paths=[ckpt,helper,*[p for p in textroot.rglob('*') if p.is_file() and '.cache' not in p.parts]]
        self.settings={'text_encoder':'sentence-transformers/sentence-t5-large, native SentenceTransformer.encode; required by original April2024 config and current configs_v1','steps':1,'guidance':7.5,'vae_scale_factor':1.,'latent_shape':[1,256],'normalization':'HumanML3D Mean.npy/Std.npy','initial_CLIP_attempt_excluded_as_configuration_error':True,'diffusers':'0.25.0'}

    def gpt(self):
        super().gpt();self.settings['sampling']='native trans.forward batched32, independent categorical draws, per-row EOS, native last-iteration truncation; RNG draw order changes';self.settings['decoder']='native VQ convolution decoder batched by equal token length; logits/decoder parity checked on synthetic inputs'

    def t2m(self):
        super().t2m();self.settings['sampling']='native length estimator batched256, per-row categorical and two retry rule; native CompTrainerV6.generate in equal-length/caption-sorted groups, RNG draw order changes'

    def sample_gpt(self,E):
        N=len(E);xs=[];active=torch.ones(N,device='cuda',dtype=torch.bool);lengths=torch.zeros(N,device='cuda',dtype=torch.long)
        for k in range(self.trans.block_size):
            logits=self.trans.forward(xs if k else [],E)[:,-1,:];idx=torch.distributions.Categorical(logits.softmax(-1)).sample()
            ended=active & (idx==self.trans.num_vq);lengths[ended]=k;active=active & ~ended
            if not active.any():break
            idx=idx.clamp_max(self.trans.num_vq-1)[:,None];xs=idx if k==0 else torch.cat([xs,idx],1)
            if k==self.trans.block_size-1:lengths[active]=k
        samples=[]
        for j,l in enumerate(lengths.tolist()):
            if l==0:
                self.failures.append({'synthetic_or_generated_row':j,'exception':'native first-token EOS fallback index1'});samples.append(torch.ones((1,1),device='cuda',dtype=torch.long))
            else:samples.append(xs[j:j+1,:l])
        return samples

    @torch.no_grad()
    def generate(self,cohort):
        if self.method not in ['motionlcm_v1','t2mgpt','t2m']:return super().generate(cohort)
        out=np.zeros_like(cohort['raw']);L=cohort['lengths'].copy();N=len(out)
        if self.method=='motionlcm_v1':
            for i in range(0,N,32):
                caps=cohort['captions'][i:i+32];lengths=L[i:i+32];n=len(caps);E=self.text.encode(caps,show_progress_bar=False,convert_to_tensor=True,batch_size=n).clone().unsqueeze(1)
                z=torch.randn((n,1,256),device='cuda')*self.scheduler.init_noise_sigma;self.scheduler.set_timesteps(1);cond=self.guidance_embedding(torch.full((n,),6.5),256).cuda()
                for t in self.scheduler.timesteps.to('cuda'):
                    eps=self.denoiser(sample=self.scheduler.scale_model_input(z,t),timestep=t,timestep_cond=cond,encoder_hidden_states=E,controlnet_residuals=None)[0];z=self.scheduler.step(eps,t,z).prev_sample
                mask=torch.arange(max(lengths),device='cuda')[None,:]<torch.tensor(lengths,device='cuda')[:,None];x=self.vae.decode(z,mask).cpu().numpy()*self.std+self.mean
                for j,l in enumerate(lengths):out[i+j,:l]=x[j,:l]
                if i%256==0:print('GENERATE',self.method,i,N,flush=True)
        elif self.method=='t2mgpt':
            for i in range(0,N,32):
                E=self.text.encode_text(self.clip.tokenize(cohort['captions'][i:i+32],truncate=True).cuda()).float();tokens=self.sample_gpt(E)
                for size in sorted({x.shape[1] for x in tokens}):
                    jj=[j for j,x in enumerate(tokens) if x.shape[1]==size];indices=torch.cat([tokens[j] for j in jj]);vq=self.vae.vqvae
                    emb=vq.quantizer.dequantize(indices).permute(0,2,1).contiguous();x=vq.postprocess(vq.decoder(emb)).cpu().numpy()
                    for j,values in zip(jj,x):L[i+j]=min(len(values),196);out[i+j,:L[i+j]]=values[:196]*self.std+self.mean
                if i%256==0:print('GENERATE',self.method,i,N,flush=True)
        else:
            for i in range(0,N,256):
                idx=np.arange(i,min(i+256,N));order=np.argsort(-cohort['sent_lengths'][idx],kind='stable');sorted_idx=idx[order]
                W=torch.tensor(cohort['words'][sorted_idx],device='cuda');P=torch.tensor(cohort['pos'][sorted_idx],device='cuda');S=torch.tensor(cohort['sent_lengths'][sorted_idx]);probs=self.length_estimator(W,P,S).softmax(-1);units=torch.multinomial(probs,1,replacement=True)[:,0]
                for _ in range(2):
                    retry=units<10
                    if retry.any():units[retry]=torch.multinomial(probs[retry],1,replacement=True)[:,0]
                for size in sorted(units.unique().tolist()):
                    positions=torch.where(units==size)[0];chosen=sorted_idx[positions.cpu().numpy()];lengths=torch.full((len(positions),),size*4,dtype=torch.long)
                    if size==0:raise RuntimeError('native length estimator emitted zero duration; no silent sample omission')
                    x=self.trainer.generate(W[positions],P[positions],S[positions.cpu()],lengths,size,263)[0].cpu().numpy()
                    for j,values in zip(chosen,x):L[j]=min(len(values),196);out[j,:L[j]]=values[:196]*self.std+self.mean
                print('GENERATE',self.method,min(i+256,N),N,flush=True)
        assert (L>0).all() and np.isfinite(out).all()
        if self.failures:write(R/f'{self.method}_native_sampling_failures_v2.json',self.failures)
        return out,L
