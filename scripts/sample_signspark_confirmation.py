"""Official sampler with pinned local CLIP assets; initialization-only shortcut."""
import os, runpy, sys
from pathlib import Path
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0';os.environ['TOKENIZERS_PARALLELISM']='false'
import torch, transformers
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
ROOT=Path(__file__).resolve().parents[1]
orig=transformers.AutoModel.from_pretrained
def local_base(name,*args,**kwargs):
    if str(name) in ('xlm-roberta-large','FacebookAI/xlm-roberta-large'):
        # M-CLIP immediately loads its entire transformer state, then the EMA
        # checkpoint is loaded strictly by the unchanged official sample.py.
        cfg=transformers.AutoConfig.from_pretrained(ROOT/'reports/cvpr_motion_consistency_confirmation_20261010/text_encoder/base_config',local_files_only=True)
        return transformers.AutoModel.from_config(cfg)
    return orig(name,*args,**kwargs)
transformers.AutoModel.from_pretrained=local_base
repo=ROOT/'external/SignSparK';sys.path.insert(0,str(repo))
runpy.run_path(str(repo/'sample.py'),run_name='__main__')
