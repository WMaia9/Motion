"""Public inference components, strict checkpoint loading, no reference poses.

Separate interpreter per method is required: the two ``mld`` packages and the
three ``models/utils`` packages are incompatible Python namespaces. We omit
visualization, Lightning trainers, and multimodality-only extra draws.
"""
import json,sys,types,importlib.machinery,ast
from pathlib import Path
import numpy as np
import torch
from motion_release_audit_common import R,A,sha,write

class NativeGenerator:
    def __init__(self,method):
        self.method=method;self.paths=[];self.failures=[]
        self.mean=np.load('external/HumanML3D/HumanML3D/Mean.npy');self.std=np.load('external/HumanML3D/HumanML3D/Std.npy')
        if method in ['mld','motionlcm_v1']:self.latent()
        elif method=='t2mgpt':self.gpt()
        elif method=='t2m':self.t2m()
        write(R/f'adapter_{method}.json',{'method':method,'inputs_sha256':{str(p):sha(p) for p in self.paths},'no_reference_poses':True,'strict_component_checkpoint_loading':True,'settings':self.settings,'source_sha256':sha(__file__)})

    def latent(self):
        sys.path.insert(0,str(Path('external/motion_lcm_python').resolve()))
        # Native pretokenized dataset never invokes spaCy. Transformers checks
        # optional-module specifications, so give the already inert stub one.
        if 'spacy' in sys.modules and sys.modules['spacy'].__spec__ is None:sys.modules['spacy'].__spec__=importlib.machinery.ModuleSpec('spacy',loader=None)
        from transformers import AutoModel,AutoTokenizer
        textroot=Path('models/motion_release_audit/clip-vit-large-patch14')
        self.tokenizer=AutoTokenizer.from_pretrained(textroot,local_files_only=True)
        self.text=AutoModel.from_pretrained(textroot,local_files_only=True).cuda().eval()
        for p in self.text.parameters():p.requires_grad_(False)
        self.paths.extend(p for p in textroot.iterdir() if p.is_file())
        if self.method=='mld':
            sys.path.insert(0,str(Path('external/MLD').resolve()))
            from mld.models.architectures.mld_vae import MldVae
            from mld.models.architectures.mld_denoiser import MldDenoiser
            from diffusers import DDIMScheduler
            abl=types.SimpleNamespace(MLP_DIST=False,PE_TYPE='mld',DIFF_PE_TYPE='mld',SKIP_CONNECT=True,VAE_TYPE='actor')
            self.vae=MldVae(abl,263,latent_dim=[1,256],arch='encoder_decoder')
            self.denoiser=MldDenoiser(abl,num_layers=9)
            self.scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',clip_sample=False,set_alpha_to_one=False,steps_offset=1)
            ckpt=Path('models/motion_release_audit/mld.ckpt');self.steps=50
        else:
            sys.path.insert(0,str(Path('external/MotionLCM').resolve()))
            from mld.models.architectures.mld_vae import MldVae
            from mld.models.architectures.mld_denoiser import MldDenoiser
            from diffusers import LCMScheduler
            self.vae=MldVae(263,latent_dim=[1,256])
            self.denoiser=MldDenoiser(latent_dim=[1,256],time_cond_proj_dim=256,controlnet_embed_dim=256,controlnet_act_fn=None)
            self.scheduler=LCMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',clip_sample=False,set_alpha_to_one=False,original_inference_steps=50)
            ckpt=Path('models/motion_release_audit/extracted/experiments_t2m/motionlcm_humanml/motionlcm_humanml_v1.ckpt');self.steps=1
            helper=Path('external/MotionLCM/mld/utils/utils.py');tree=ast.parse(helper.read_text());function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='get_guidance_scale_embedding');namespace={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),str(helper),'exec'),namespace);self.guidance_embedding=namespace[function.name];self.paths.append(helper)
        self.paths.append(ckpt);weights=torch.load(ckpt,map_location='cpu',weights_only=False)['state_dict']
        for key,model in [('vae',self.vae),('denoiser',self.denoiser)]:
            model.load_state_dict({k[len(key)+1:]:v for k,v in weights.items() if k.startswith(key+'.')},strict=True);model.cuda().eval()
        self.settings={'steps':self.steps,'CFG':7.5,'latent_shape':[1,256],'CLIP':'openai/clip-vit-large-patch14','normalization':'HumanML3D Mean.npy/Std.npy','native_reverse_loop':'MLD DDIM or MotionLCM_v1 LCM, no controlnet','diffusers':'0.25.0'}

    def gpt(self):
        sys.path[:0]=[str(Path(p).resolve()) for p in ['external/T2M_GPT','external/humanml_python','external/CLIP']]
        import clip
        from models.vqvae import HumanVQVAE
        from models.t2m_trans import Text2Motion_Transformer
        args=types.SimpleNamespace(dataname='t2m',quantizer='ema_reset',mu=.99)
        self.vae=HumanVQVAE(args,512,512,512,2,2,512,3,3).cuda().eval()
        self.trans=Text2Motion_Transformer(num_vq=512,embed_dim=1024,clip_dim=512,block_size=51,num_layers=9,n_head=16,drop_out_rate=.1,fc_rate=4).cuda().eval()
        root=Path('models/motion_release_audit/extracted');v=root/'VQVAE/net_last.pth';t=root/'VQTransformer_corruption05/net_best_fid.pth'
        self.vae.load_state_dict(torch.load(v,map_location='cpu',weights_only=False)['net'],strict=True)
        self.trans.load_state_dict(torch.load(t,map_location='cpu',weights_only=False)['trans'],strict=True)
        self.text,_=clip.load('ViT-B/32',device='cuda',jit=False,download_root='models/humanml3d_audit/clip');self.text.eval();self.clip=clip
        meta=Path('models/motion_release_audit/gpt_evaluator/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta')
        self.mean=np.load(meta/'mean.npy');self.std=np.load(meta/'std.npy')
        self.paths=[v,t,meta/'mean.npy',meta/'std.npy']
        self.settings={'sampling':'unmodified trans.sample(..., True), batch1, EOS native, block_size51','decoder':'author-recommended VQVAE/net_last.pth','normalization':'author-linked VQVAEV3 meta; downloaded native model metadata','sampling_exception':'native fallback one token of index1, recorded per clip','multimodality_only_extra_draws':False}

    def t2m(self):
        sys.path.insert(0,str(Path('external/HumanML3D_Evaluator').resolve()))
        from motion_loaders.comp_v6_model_dataset import build_models
        from networks.trainers import CompTrainerV6
        root=A/'t2m/Comp_v6_KLD01';opt={}
        for line in (root/'opt.txt').read_text().splitlines():
            if ': ' not in line:continue
            k,v=line.split(': ',1)
            try:v=json.loads(v.lower()) if v in ['True','False'] else json.loads(v)
            except json.JSONDecodeError:pass
            opt[k]=v
        opt.update(is_train=False,device=torch.device('cuda'),dim_word=300,dim_pos_ohot=15,dim_pose=263,num_classes=50,checkpoints_dir=str(A),dataset_name='t2m')
        opt=types.SimpleNamespace(**opt)
        # Trusted author checkpoint includes training metadata, requires
        # explicit legacy load on Torch >=2.6. Restore global loader afterward.
        load=torch.load
        def legacy(*a,**kw):kw.setdefault('weights_only',False);return load(*a,**kw)
        torch.load=legacy
        try:
            te,pr,dec,att,me,md,self.length_estimator=build_models(opt)
            self.trainer=CompTrainerV6(opt,te,pr,dec,att,md,mov_enc=me);self.trainer.load(str(root/'model/latest.tar'));self.trainer.eval_mode();self.trainer.to(opt.device)
        finally:torch.load=load
        self.mean=np.load(root/'meta/mean.npy');self.std=np.load(root/'meta/std.npy')
        self.paths=[root/'model/latest.tar',A/'t2m/length_est_bigru/model/latest.tar',root/'opt.txt',root/'meta/mean.npy',root/'meta/std.npy']
        self.settings={'sampling':'native CompTrainerV6.generate, batch1; length-estimator categorical with two native retries below10units','unit_length':4,'native_length_estimator':True,'normalization':'Comp_v6_KLD01 meta','multimodality_only_extra_draws':False}

    @torch.no_grad()
    def generate(self,cohort):
        out=np.zeros_like(cohort['raw']);L=cohort['lengths'].copy()
        if self.method in ['mld','motionlcm_v1']:
            for i in range(0,len(out),32):
                caps=cohort['captions'][i:i+32];lengths=L[i:i+32].tolist();n=len(caps)
                txt=caps if self.method!='mld' else ['']*n+caps
                ids=self.tokenizer(txt,padding='max_length',truncation=True,max_length=self.tokenizer.model_max_length,return_tensors='pt').input_ids.cuda()
                E=self.text.get_text_features(ids).unsqueeze(1)
                z=torch.randn((n,1,256),device='cuda')*self.scheduler.init_noise_sigma;self.scheduler.set_timesteps(self.steps)
                cond=None
                if self.method!='mld':
                    cond=self.guidance_embedding(torch.full((n,),6.5),256).cuda()
                for t in self.scheduler.timesteps.to('cuda'):
                    if self.method=='mld':
                        eps=self.denoiser(sample=torch.cat([z,z]),timestep=t,encoder_hidden_states=E,lengths=lengths*2)[0];a,b=eps.chunk(2);eps=a+7.5*(b-a)
                        z=self.scheduler.step(eps,t,z,eta=0.).prev_sample
                    else:
                        eps=self.denoiser(sample=self.scheduler.scale_model_input(z,t),timestep=t,timestep_cond=cond,encoder_hidden_states=E,controlnet_residuals=None)[0]
                        z=self.scheduler.step(eps,t,z).prev_sample
                mask=torch.arange(max(lengths),device='cuda')[None,:]<torch.tensor(lengths,device='cuda')[:,None]
                x=self.vae.decode(z.permute(1,0,2) if self.method=='mld' else z,lengths if self.method=='mld' else mask).cpu().numpy()
                x=x*self.std+self.mean
                for j,l in enumerate(lengths):out[i+j,:l]=x[j,:l]
                if i%256==0:print('GENERATE',self.method,i,len(out),flush=True)
        else:
            for i in range(len(out)):
                if self.method=='t2mgpt':
                    E=self.text.encode_text(self.clip.tokenize([cohort['captions'][i]],truncate=True).cuda()).float()
                    try:idx=self.trans.sample(E,True)
                    except Exception as e:
                        self.failures.append({'clip':cohort['clip_ids'][i],'exception':repr(e)});idx=torch.ones((1,1),device='cuda',dtype=torch.long)
                    x=self.vae.forward_decoder(idx)[0].cpu().numpy()
                else:
                    words=torch.tensor(cohort['words'][i:i+1],device='cuda');pos=torch.tensor(cohort['pos'][i:i+1],device='cuda');sl=torch.tensor(cohort['sent_lengths'][i:i+1])
                    prob=self.length_estimator(words,pos,sl).softmax(-1).squeeze()
                    units=torch.multinomial(prob,1,replacement=True)
                    for _ in range(2):
                        if units<10:units=torch.multinomial(prob,1,replacement=True)
                    lengths=(units*4).cpu();x=self.trainer.generate(words,pos,sl,lengths,int(units.item()),263)[0][0].cpu().numpy()
                L[i]=min(len(x),196);out[i,:L[i]]=x[:196]*self.std+self.mean
                if i%64==0:print('GENERATE',self.method,i,len(out),flush=True)
        assert (L>0).all()
        if self.failures:write(R/f'{self.method}_native_sampling_failures.json',self.failures)
        return out,L
