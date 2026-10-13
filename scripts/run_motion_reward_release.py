"""Paired audit of publicly released MotionRFT weights and their linked base.

Supplemental five-seed diagnostic on fixed1024-output native cohorts, separate
from the primary five-method20-repetition audit. All endpoints are retained.
Inference uses native component classes and the actual native reverse method;
the reward model and reference poses are unnecessary for released inference.
"""
import ast,argparse,sys,types,time,inspect,importlib.machinery
from pathlib import Path
import numpy as np
import torch
from typing import Optional
from motion_release_audit_common import R as PRIMARY,A,read,write,sha,seed,evaluator,score_embeddings,canonical,metric
from run_motion_release_audit import geometry
R=Path('reports/cvpr_motion_reward_release_20261013')
BASE=Path('models/motion_release_audit/motionrft_base/pretrain/mld/mld_humanml_v1.ckpt')
TUNED=Path('models/motion_release_audit/motionrft_checkpoint/RFT_MLD/checkpoints/rft_mld/best.ckpt')
TEXT=Path('models/motion_release_audit/sentence-t5-large')
SOURCE=Path('external/MotionRFT/RFT_MLD/mld/models/modeltype/mld.py')

def register():
    R.mkdir(exist_ok=True);files=[BASE,TUNED,SOURCE,Path(__file__),Path('scripts/motion_release_audit_common.py'),Path('scripts/run_motion_release_audit.py')]
    p={'scope':'Prospective supplemental released reward-trained model audit and author-linked base; five matched generation seeds, not a full20repetition leaderboard reproduction','registered_after_negative_own_reward_pilot':True,'checkpoints':{'base':str(BASE),'reward_finetuned':str(TUNED)},'HF_revisions':{'txf0620/MotionRFT_Models':'2931fda962875d8c258cc57f5d5446c4487f2812','txf0620/MotionRFT_Checkpoints':'61457e92aeb68a86778be1601569fe89a3fe04fc'},'replications':5,'generation_seeds':[2101,2102,2103,2104,2105],'conditioning':'native1024-output cohorts0-4 from primary MDM audit; captions and duration only','reference':'full4640 native TEST draws0-4 from primary audit','sampler':'native MotionRFT reverse method, DDIM50, CFG7.5, epsilon, scale_factor1, Sentence-T5-large','generation_batch':128,'metric_candidate_batch':32,'conditions':['raw','canonical'],'readers':['common_MDM_checkpoint','official_GPT_checkpoint'],'not_proof_of_intentional_hacking_or_vulnerability_of_MotionReward':True,'inputs_and_sources_sha256':{str(f):sha(f) for f in files}}
    path=R/'protocol.json'
    if path.exists():assert read(path)==p
    else:write(path,p)
    return p

class NativeRelease:
    def __init__(self,checkpoint):
        sys.path[:0]=[str(Path(p).resolve()) for p in ['external/motion_lcm_python','external/MotionRFT/RFT_MLD']]
        if 'spacy' in sys.modules and sys.modules['spacy'].__spec__ is None:sys.modules['spacy'].__spec__=importlib.machinery.ModuleSpec('spacy',loader=None)
        from sentence_transformers import SentenceTransformer
        from diffusers import DDIMScheduler
        from mld.models.architectures.mld_vae import MldVae
        from mld.models.architectures.mld_denoiser import MldDenoiser
        self.text=SentenceTransformer(str(TEXT),device='cuda');self.text.eval();self.vae=MldVae(263,latent_dim=[1,256]);self.denoiser=MldDenoiser(latent_dim=[1,256],controlnet_embed_dim=256,controlnet_act_fn=None)
        record=torch.load(checkpoint,map_location='cpu',weights_only=False);state=record['state_dict']
        for key,model in [('vae',self.vae),('denoiser',self.denoiser)]:model.load_state_dict({k[len(key)+1:]:v for k,v in state.items() if k.startswith(key+'.')},strict=True);model.cuda().eval()
        if checkpoint==TUNED:write(R/'release_checkpoint_metadata.json',{'author_stored_metrics_not_our_results':record.get('metrics'),'author_ft_config':record.get('ft_config'),'checkpoint_sha256':sha(checkpoint)})
        del state,record
        self.scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',prediction_type='epsilon',clip_sample=False,set_alpha_to_one=False,steps_offset=1);self.alphas=torch.sqrt(self.scheduler.alphas_cumprod);self.sigmas=torch.sqrt(1-self.scheduler.alphas_cumprod);self.guidance_scale=7.5;self.do_classifier_free_guidance=True;self.is_controlnet=False;self.cfg=types.SimpleNamespace(model=types.SimpleNamespace(scheduler=types.SimpleNamespace(num_inference_steps=50,eta=0.)))
        helper=Path('external/MotionRFT/RFT_MLD/mld/utils/utils.py');fn=next(x for x in ast.parse(helper.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='extract_into_tensor');ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(helper),'exec'),ns)
        functions=next(x for x in ast.parse(SOURCE.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='MLD').body;environment={'torch':torch,'Optional':Optional,'inspect':inspect,'extract_into_tensor':ns['extract_into_tensor']}
        for name in ['predicted_origin','_diffusion_reverse']:
            f=next(x for x in functions if isinstance(x,ast.FunctionDef) and x.name==name);exec(compile(ast.Module(body=[f],type_ignores=[]),str(SOURCE),'exec'),environment);setattr(self,name,types.MethodType(environment[name],self))
        self.mean=np.load('external/HumanML3D/HumanML3D/Mean.npy');self.std=np.load('external/HumanML3D/HumanML3D/Std.npy')

    @torch.no_grad()
    def generate(self,cohort):
        out=np.zeros((len(cohort['lengths']),196,263),np.float32)
        for i in range(0,len(out),128):
            L=cohort['lengths'][i:i+128];caps=cohort['captions'][i:i+128];n=len(caps);E=self.text.encode(caps+['']*n,show_progress_bar=False,convert_to_tensor=True,batch_size=2*n).clone().unsqueeze(1)
            # Native TEST method draws and discards one initial noise tensor.
            torch.randn((n,1,256),device='cuda');z=torch.randn((n,1,256),device='cuda');z=self._diffusion_reverse(z,E,controlnet_cond=None);mask=torch.arange(max(L),device='cuda')[None,:]<torch.tensor(L,device='cuda')[:,None];x=self.vae.decode(z,mask).cpu().numpy()*self.std+self.mean
            for j,l in enumerate(L):out[i+j,:l]=x[j,:l]
            print('RFT_RELEASE_GENERATE',i+n,len(out),flush=True)
        return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-primary',action='store_true');parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args();p=register()
    if args.wait_for_primary:
        deadline=time.monotonic()+14400
        while not read(PRIMARY/'gpu_queue.json').get('all_workers_exited',False):
            if time.monotonic()>deadline:raise RuntimeError('primary GPU queue incomplete; no supplemental inference')
            time.sleep(20)
    seed(434);wrapper,mean,std=evaluator();second=__import__('copy').deepcopy(wrapper);state=torch.load('models/motion_release_audit/gpt_evaluator/t2m/text_mot_match/model/finest.tar',map_location='cpu',weights_only=False)
    for key,model in [('movement_encoder',second.movement_encoder),('motion_encoder',second.motion_encoder),('text_encoder',second.text_encoder)]:model.load_state_dict(state[key],strict=True);model.eval()
    for method,checkpoint in [('base',BASE),('reward_finetuned',TUNED)]:
        gen=NativeRelease(checkpoint)
        if args.preflight_only:
            c={'lengths':np.array([64,80]),'captions':['a person walks forward','a person raises an arm']};x=gen.generate(c);assert np.isfinite(x).all();write(R/f'preflight_{method}.json',{'synthetic_prompts_only':True,'shape':list(x.shape),'finite':True});del gen;torch.cuda.empty_cache();continue
        for rep,s in enumerate(p['generation_seeds']):
            folder=R/method/f'rep_{rep:02d}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'metrics.json').exists():continue
            source=PRIMARY/'mdm50'/f'rep_{rep:02d}';z=np.load(source/'conditioning.npz');cohort={k:z[k].tolist() if k in ['captions','tokens','clip_ids'] else z[k].copy() for k in z.files};seed(s);raw=gen.generate(cohort);np.save(folder/'generated_raw.npy',raw);scores={};checks={};ref=np.load(source/'reference_embeddings.npy');ref2=np.load(PRIMARY/'secondary_gpt_reader'/f'reference_rep_{rep:02d}.npy')
            for condition,x in [('raw',raw),('canonical',canonical(raw))]:
                checks[condition]=geometry(raw,x,cohort['lengths']);scores[condition]={}
                for name,reader,reference in [('common_MDM_checkpoint',wrapper,ref),('official_GPT_checkpoint',second,ref2)]:T,E=score_embeddings(reader,x,cohort,mean,std);scores[condition][name]=metric(T,E,reference);np.save(folder/f'{condition}_{name}_embeddings.npy',E)
            write(folder/'metrics.json',{'method':method,'rep':rep,'seed':s,'results':scores,'geometry':checks,'source_weights_sha256':sha(checkpoint)});print('RFT_RELEASE_SCORED',method,rep,scores,flush=True)
        del gen;torch.cuda.empty_cache()
    if not args.preflight_only:write(R/'complete.json',{'two_checkpoints':True,'generation_seeds':5,'readers':2,'all_raw_canonical_endpoints_retained':True})
if __name__=='__main__':main()
