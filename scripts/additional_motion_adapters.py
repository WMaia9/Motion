"""Native release components; each upstream namespace needs its own process.

No trained component is substituted. Mesh conversion is unavailable and is not
used by these feature-space inference paths. Dataset lengths condition sampling.
"""
from pathlib import Path
import sys,types
import numpy as np
import torch
from motion_release_audit_common import read

ASSETS=Path('models/generalization_audit')

def clip_cache():
    sys.path[:0]=[str(Path(p).resolve()) for p in ['external/humanml_python','external/CLIP']]
    import clip
    original=clip.load
    def load(*args,**kwargs):
        kwargs['download_root']='models/humanml3d_audit/clip'
        return original(*args,**kwargs)
    clip.load=load

def state(path):
    # Author archives contain metadata in addition to state tensors.
    return torch.load(path,map_location='cpu',weights_only=False)

class NativeAdditional:
    def __init__(self,method,dataset):
        self.method=method;self.dataset=dataset;self.dim=263 if dataset=='humanml' else 251
        if method=='momask':self.momask()
        elif method=='motiondiffuse':self.motiondiffuse()
        elif method=='dip':
            assert dataset=='humanml';self.dip()
        else:raise ValueError(method)

    def momask(self):
        clip_cache();sys.path.insert(0,str(Path('external/MoMask').resolve()))
        from utils.get_opt import get_opt
        from models.vq.model import RVQVAE
        from models.mask_transformer.transformer import MaskTransformer,ResidualTransformer
        root=ASSETS/('momask' if self.dataset=='humanml' else 'momask_kit')
        maskroot=next(root.glob('t2m_nlayer*'));resroot=next(root.glob('tres_nlayer*'))
        mo=get_opt(str(maskroot/'opt.txt'),torch.device('cuda'));vqroot=root/mo.vq_name
        vo=get_opt(str(vqroot/'opt.txt'),torch.device('cuda'));ro=get_opt(str(resroot/'opt.txt'),torch.device('cuda'))
        assert mo.dim_pose==vo.dim_pose==self.dim and ro.vq_name==mo.vq_name
        self.vq=RVQVAE(vo,vo.dim_pose,vo.nb_code,vo.code_dim,vo.output_emb_width,vo.down_t,vo.stride_t,vo.width,vo.depth,vo.dilation_growth_rate,vo.vq_act,vo.vq_norm)
        ck=state(vqroot/'model/net_best_fid.tar');self.vq.load_state_dict(ck['vq_model'] if 'vq_model' in ck else ck['net'],strict=True)
        mo.num_tokens=vo.nb_code;mo.num_quantizers=vo.num_quantizers;mo.code_dim=vo.code_dim
        self.mask=MaskTransformer(code_dim=mo.code_dim,cond_mode='text',latent_dim=mo.latent_dim,ff_size=mo.ff_size,num_layers=mo.n_layers,num_heads=mo.n_heads,dropout=mo.dropout,clip_dim=512,cond_drop_prob=mo.cond_drop_prob,clip_version='ViT-B/32',opt=mo)
        ck=state(maskroot/'model/latest.tar');missing,extra=self.mask.load_state_dict(ck['t2m_transformer'] if 't2m_transformer' in ck else ck['trans'],strict=False)
        assert not extra and all(k.startswith('clip_model.') for k in missing)
        ro.num_quantizers=vo.num_quantizers;ro.num_tokens=vo.nb_code
        self.res=ResidualTransformer(code_dim=vo.code_dim,cond_mode='text',latent_dim=ro.latent_dim,ff_size=ro.ff_size,num_layers=ro.n_layers,num_heads=ro.n_heads,dropout=ro.dropout,clip_dim=512,shared_codebook=vo.shared_codebook,cond_drop_prob=ro.cond_drop_prob,share_weight=ro.share_weight,clip_version='ViT-B/32',opt=ro)
        ck=state(resroot/'model/net_best_fid.tar');missing,extra=self.res.load_state_dict(ck['res_transformer'],strict=False)
        assert not extra and all(k.startswith('clip_model.') for k in missing)
        for model in [self.mask,self.res,self.vq]:model.to('cuda').eval()
        self.mean=np.load(vqroot/'meta/mean.npy');self.std=np.load(vqroot/'meta/std.npy')
        self.files=[p for p in root.rglob('*') if p.is_file()]

    def motiondiffuse(self):
        clip_cache();sys.path.insert(0,str(Path('external/MotionDiffuse/text2motion').resolve()))
        from utils.get_opt import get_opt
        from models.transformer import MotionTransformer
        from models.gaussian_diffusion import GaussianDiffusion,get_named_beta_schedule,ModelMeanType,ModelVarType,LossType
        root=ASSETS/('motiondiffuse_humanml/t2m/t2m_motiondiffuse' if self.dataset=='humanml' else 'motiondiffuse_kit/kit/kit_motiondiffuse')
        cfg=get_opt(str(root/'opt.txt'),torch.device('cuda'));assert cfg.dim_pose==self.dim and cfg.diffusion_steps==1000
        self.model=MotionTransformer(input_feats=self.dim,num_frames=196,num_layers=cfg.num_layers,latent_dim=cfg.latent_dim,no_clip=cfg.no_clip,no_eff=cfg.no_eff)
        checkpoint=state(root/'model/latest.tar');self.model.load_state_dict(checkpoint['encoder'],strict=True);self.model.to('cuda').eval()
        self.diffusion=GaussianDiffusion(betas=get_named_beta_schedule('linear',1000),model_mean_type=ModelMeanType.EPSILON,model_var_type=ModelVarType.FIXED_SMALL,loss_type=LossType.MSE)
        self.mean=np.load(root/'meta/mean.npy');self.std=np.load(root/'meta/std.npy');self.files=[p for p in root.rglob('*') if p.is_file()]

    def dip(self):
        clip_cache()
        sys.path[:0]=[str(Path(p).resolve()) for p in ['external/humanml_python','external/MDM']]
        stub=types.ModuleType('model.rotation2xyz')
        class NoMesh:
            def __init__(self,*a,**k):self.smpl_model=torch.nn.Identity()
            def __call__(self,*a,**k):raise RuntimeError('No licensed mesh model')
        stub.Rotation2xyz=NoMesh;sys.modules['model.rotation2xyz']=stub
        from model.BERT import BERT_encoder
        native_load=BERT_encoder.load_bert
        BERT_encoder.load_bert=lambda p:native_load(str(ASSETS/'distilbert-base-uncased'))
        from utils.model_util import create_model_and_diffusion,load_saved_model
        from utils.sampler_util import ClassifierFreeSampleModel,AutoRegressiveSampler
        root=ASSETS/'dip/checkpoints/dip/DiP_no-target_10steps_context20_predict40'
        args=types.SimpleNamespace(**read(root/'args.json'));args.autoregressive=True
        self.model,self.diffusion=create_model_and_diffusion(args,types.SimpleNamespace(dataset=types.SimpleNamespace(num_actions=1)))
        load_saved_model(self.model,str(root/'model000200000.pt'),use_avg=True);self.model.to('cuda');self.model.eval()
        self.guided=ClassifierFreeSampleModel(self.model);self.sampler=AutoRegressiveSampler(args,self.diffusion.p_sample_loop,required_frames=196)
        self.mean=np.load('external/HumanML3D/HumanML3D/Mean.npy');self.std=np.load('external/HumanML3D/HumanML3D/Std.npy');self.files=[root/'args.json',root/'model000200000.pt']

    @torch.no_grad()
    def generate(self,cohort,batch=128):
        N=len(cohort['lengths']);out=np.zeros((N,196,self.dim),np.float32)
        for start in range(0,N,batch):
            L=torch.tensor(cohort['lengths'][start:start+batch],device='cuda');captions=cohort['captions'][start:start+batch];n=len(L)
            if self.method=='momask':
                mids=self.mask.generate(captions,L//4,timesteps=18,cond_scale=4,temperature=1,topk_filter_thres=.9,gsample=False)
                mids=self.res.generate(mids,captions,L//4,temperature=1,cond_scale=5)
                x=self.vq.forward_decoder(mids).cpu().numpy()
            elif self.method=='motiondiffuse':
                proj,tokens=self.model.encode_text(captions,torch.device('cuda'))
                x=self.diffusion.p_sample_loop(self.model,(n,int(L.max()),self.dim),clip_denoised=False,progress=False,model_kwargs={'xf_proj':proj,'xf_out':tokens,'length':L}).cpu().numpy()
            else:
                prefix=(cohort['raw'][start:start+batch,:20]-self.mean)/self.std
                prefix=torch.tensor(prefix,device='cuda',dtype=torch.float32).permute(0,2,1)[:,:,None]
                y={'text':captions,'prefix':prefix,'lengths':torch.full_like(L,40),'mask':torch.ones((n,1,1,40),device='cuda',dtype=torch.bool),'scale':torch.full((n,),7.5,device='cuda')}
                y['text_embed']=self.model.encode_text(captions)
                x=self.sampler.sample(self.guided,(n,self.dim,1,40),clip_denoised=False,model_kwargs={'y':y},skip_timesteps=0,init_image=None,progress=False,dump_steps=None,noise=None,const_noise=False)[:,:,0].permute(0,2,1).cpu().numpy()
            x=x*self.std+self.mean
            for j,length in enumerate(L.cpu().tolist()):out[start+j,:length]=x[j,:length]
            assert np.isfinite(out[start:start+n]).all()
            print('ADDITIONAL_GENERATE',self.method,self.dataset,start+n,N,flush=True)
        return out
