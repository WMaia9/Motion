"""Native MDM50 sampling with no mesh conversion or reference poses."""
import argparse,hashlib,json,sys,types
from pathlib import Path
import numpy as np
import torch
from scipy import linalg

ROOT=Path('reports/cvpr_humanml3d_audit_20261012');A=Path('models/humanml3d_audit/extracted')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('split',choices=['val','test']);args0=parser.parse_args();split=args0.split
 assert not (ROOT/'manifest_sha256.json').exists();assert not (ROOT/f'{split}_mdm.npz').exists()
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42);np.random.seed(42)
 sys.path[:0]=[str(Path(p).resolve()) for p in ['external/humanml_python','external/CLIP','external/MDM']]
 import clip
 native_load=clip.load
 def load_clip(*a,**kw):kw['download_root']='models/humanml3d_audit/clip';return native_load(*a,**kw)
 clip.load=load_clip
 # MDM eagerly creates an unused SMPL converter for hml_vec. No licensed body
 # model is available. Fail if that converter is ever called, and preserve all
 # denoiser/CLIP/diffusion operations and checkpoint parameters unchanged.
 stub=types.ModuleType('model.rotation2xyz')
 class NoMeshConversion:
  def __init__(self,*a,**k):self.smpl_model=torch.nn.Identity()
  def __call__(self,*a,**k):raise RuntimeError('No mesh conversion permitted in this audit')
 stub.Rotation2xyz=NoMeshConversion;sys.modules['model.rotation2xyz']=stub
 from utils.model_util import create_model_and_diffusion,load_model_wo_clip
 from model.cfg_sampler import ClassifierFreeSampleModel
 cfg=json.loads((A/'humanml_enc_512_50steps/args.json').read_text());args=types.SimpleNamespace(**cfg);data=types.SimpleNamespace(dataset=types.SimpleNamespace(num_actions=1))
 model,diffusion=create_model_and_diffusion(args,data)
 state=torch.load(A/'humanml_enc_512_50steps/model000750000.pt',map_location='cpu',weights_only=True)
 missing,unexpected=model.load_state_dict(state,strict=False)
 assert not unexpected and all(n.startswith('clip_model.') for n in missing),(missing,unexpected)
 model.to('cuda');model.eval();guided=ClassifierFreeSampleModel(model)
 selection=json.loads((ROOT/f'{split}_selection.json').read_text());N=len(selection)
 mean=np.load('external/HumanML3D/HumanML3D/Mean.npy');std=np.load('external/HumanML3D/HumanML3D/Std.npy')
 out=[]
 with torch.no_grad():
  for start in range(0,N,32):
   items=selection[start:start+32];n=len(items);lengths=torch.tensor([x['frames'] for x in items],device='cuda');mask=torch.arange(196,device='cuda')[None,:]<lengths[:,None]
   y={'text':[x['caption'] for x in items],'lengths':lengths,'mask':mask[:,None,None,:],'scale':torch.full((n,),2.5,device='cuda')};y['text_embed']=model.encode_text(y['text'])
   sample=diffusion.p_sample_loop(guided,(n,263,1,196),clip_denoised=False,model_kwargs={'y':y},skip_timesteps=0,init_image=None,progress=False,dump_steps=None,noise=None,const_noise=False)
   raw=sample[:, :,0,:].permute(0,2,1).cpu().numpy()*std+mean
   raw[~mask.cpu().numpy()]=0;out.append(raw.astype(np.float32));print('MDM_GENERATED',split,start+n,N,flush=True)
 np.savez_compressed(ROOT/f'{split}_mdm.npz',raw=np.concatenate(out),lengths=np.array([x['frames'] for x in selection]))
 (ROOT/f'{split}_generation.json').write_text(json.dumps({'clips':N,'source_code_sha256':digest(Path(__file__)),'no_reference_poses':True,'checkpoint_missing_only_clip_model':missing,'checkpoint_unexpected':unexpected,'official_native_model_and_diffusion':True,'unused_eager_SMPL_constructor_disabled':True,'SMPL_converter_calls':0,'mesh_quality_not_claimed':True,'steps':50,'CFG':2.5,'seed':42},indent=2)+'\n')
if __name__=='__main__':main()
