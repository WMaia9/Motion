"""Official SignSparK sampler with optional physical CFG / text controls."""
import os,sys,json,runpy,atexit
from pathlib import Path
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0';os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'external/SignSparK'));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from sign_physical_guidance import horizontal
import classifier_free
from signspark.unet_model import Wilor_UNET

mode=os.getenv('SIGN_GUIDANCE_MODE','standard');mapping=json.loads(Path(os.environ['SIGN_TEXT_MAPPING']).read_text()) if mode=='shuffled' else {}
base_encode=Wilor_UNET.encode_text
def cached_encode(self,raw):
    actual=[mapping.get(t,t) for t in raw];key=tuple(actual)
    if not hasattr(self,'_diagnostic_text_cache'):self._diagnostic_text_cache={}
    if key not in self._diagnostic_text_cache:self._diagnostic_text_cache[key]=base_encode(self,actual)
    return self._diagnostic_text_cache[key]
Wilor_UNET.encode_text=cached_encode

summary={'mode':mode,'threshold_time':.8,'calls':0,'late_calls':0,'valid_extra_energy':0.,'valid_vertical_energy':0.,'degenerate_joint_frames':0}
if mode=='horizontal':
    Base=classifier_free.ClassifierFreeSampleModel
    class PhysicalCFG(Base):
        def forward(self,x,timesteps,y=None,obs_x0=None,obs_mask=None,**kwargs):
            import copy
            yu=copy.deepcopy(y);yu['uncond']=True
            cond=self.model(x,timesteps,y,obs_x0,obs_mask,**kwargs);uncond=self.model(x,timesteps,yu,obs_x0,obs_mask,**kwargs);delta=cond-uncond;summary['calls']+=1
            if float(timesteps[0])>=.8:
                d=x.transpose(1,2).reshape(*x.shape[:1],x.shape[-1],-1,6);v=delta.transpose(1,2).reshape(d.shape);h,valid=horizontal(d,v)
                mask=y['mask'].to(x.device).transpose(1,2).unsqueeze(-1);mask=mask.expand_as(valid);vertical=v-h
                summary['late_calls']+=1;summary['valid_extra_energy']+=float((v.square()*mask).sum());summary['valid_vertical_energy']+=float((vertical.square()*mask).sum());summary['degenerate_joint_frames']+=int((~valid&mask).sum())
                delta=h.reshape(x.shape[0],x.shape[-1],x.shape[1]).transpose(1,2)
                # Keep the ordinary conditional flow intact. Only extra CFG changes.
                return cond+(self.text_scale.view(-1,1,1)-1)*delta
            # Exactly the official floating-point expression before the intervention.
            return uncond+self.text_scale.view(-1,1,1)*delta
    classifier_free.ClassifierFreeSampleModel=PhysicalCFG

def finish():
    if summary['valid_extra_energy']:summary['late_vertical_energy_fraction']=summary['valid_vertical_energy']/summary['valid_extra_energy']
    Path(os.environ['SIGN_DIAGNOSTIC_STATS']).write_text(json.dumps(summary,indent=2))
atexit.register(finish)
runpy.run_path(str(ROOT/'scripts/sample_signspark_confirmation.py'),run_name='__main__')
