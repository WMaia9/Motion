"""Official MARDM essential-feature evaluators; strict released weights.

The original constructor downloads an unused CLIP visual tower. Only token and
position embeddings are retained by MotionCLIP, and both are supplied in its
released state. A constructor-only placeholder provides these exact shapes;
strict state loading overwrites every parameter before any inference.
"""
from pathlib import Path
import sys,types
import numpy as np
import torch
from motion_release_audit_common import read,sha
SOURCE=Path('external/MARDM');MODELS=Path('models/mardm_audit')

class EssentialEvaluators:
    def __init__(self,device='cuda'):
        sys.path[:0]=[str(Path(p).resolve()) for p in ['external/MARDM','external/CLIP','external/humanml_python']]
        import clip
        from utils.evaluators import Evaluators
        original=clip.load
        def construct(*args,**kwargs):
            return types.SimpleNamespace(token_embedding=torch.nn.Embedding(49408,512),positional_embedding=torch.nn.Parameter(torch.zeros(77,512))),None
        # Author paths are relative to checkpoints/t2m. Construct in a dedicated
        # local view without changing the upstream checkout or any old archive.
        import os
        view=MODELS/'native_view';(view/'checkpoints').mkdir(parents=True,exist_ok=True)
        target=view/'checkpoints/t2m'
        if not target.exists():target.symlink_to(MODELS.absolute(),target_is_directory=True)
        cwd=Path.cwd();clip.load=construct
        try:
            os.chdir(view);self.wrapper=Evaluators('t2m',torch.device(device))
        finally:os.chdir(cwd);clip.load=original
        self.device=torch.device(device);self.mean=np.load(SOURCE/'utils/eval_mean_std/t2m/eval_mean.npy');self.std=np.load(SOURCE/'utils/eval_mean_std/t2m/eval_std.npy');assert self.mean.shape==self.std.shape==(67,)
        for module in [self.wrapper.text_encoder,self.wrapper.motion_encoder,self.wrapper.movement_encoder,self.wrapper.contrast_model]:
            module.eval()
            for p in module.parameters():p.requires_grad_(False)
        state=torch.load(MODELS/'text_mot_match_clip/model/finest.tar',map_location='cpu',weights_only=True)['contrast_model']
        for name,parameter in self.wrapper.contrast_model.state_dict().items():assert torch.equal(parameter.cpu(),state[name]),name
        assert self.wrapper.movement_encoder.main[0].weight.shape[1]==67
        assert self.wrapper.contrast_model.motion_encoder.embed_motion.weight.shape[1]==67

    @torch.no_grad()
    def encode(self,raw,cohort):
        tg=[];eg=[];tc=[];ec=[]
        for i in range(0,len(raw),32):
            order=np.argsort(-cohort['sent_lengths'][i:i+32],kind='stable');ll=cohort['lengths'][i:i+32][order];motion=(np.asarray(raw[i:i+32][order,:196,:67],dtype=np.float64)-self.mean)/self.std
            motion[np.arange(196)[None,:]>=ll[:,None]]=0
            captions=np.asarray(cohort['captions'])[i:i+32][order].tolist()
            (t,e),(ct,ce)=self.wrapper.get_co_embeddings(torch.tensor(cohort['words'][i:i+32][order]),torch.tensor(cohort['pos'][i:i+32][order]),torch.tensor(cohort['sent_lengths'][i:i+32][order]),captions,torch.tensor(motion),torch.tensor(ll))
            tg.append(t.cpu().numpy());eg.append(e.cpu().numpy());tc.append(ct.cpu().numpy());ec.append(ce.cpu().numpy())
        return {'gru':(np.concatenate(tg),np.concatenate(eg)),'clip':(np.concatenate(tc),np.concatenate(ec))}
