"""Pinned public text encoders and author-linked evaluator metadata."""
import argparse,zipfile
from pathlib import Path
import gdown
from huggingface_hub import snapshot_download,hf_hub_download
from motion_release_audit_common import R,sha,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reward-release',action='store_true');args=parser.parse_args();root=Path('models/motion_release_audit');root.mkdir(exist_ok=True);records=[]
    assets=[('clip-vit-large-patch14','openai/clip-vit-large-patch14','32bd64288804d66eefd0ccbe215aa642df71cc41',['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','preprocessor_config.json']),('sentence-t5-large','sentence-transformers/sentence-t5-large','1b36eb48a1df42a07ffd02234d25abfbf3e3f3cb',None)]
    for name,repo,rev,patterns in assets:
        snapshot_download(repo,revision=rev,local_dir=root/name,allow_patterns=patterns,ignore_patterns=['onnx/*','openvino/*','*.onnx','tf_model.h5','flax_model.msgpack','model.safetensors']);records.append({'repo':repo,'revision':rev,'path':str(root/name)})
    archive=root/'t2mgpt_evaluator.zip'
    if not archive.exists():gdown.download(id='1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-',output=str(archive),quiet=True,resume=True)
    destination=root/'gpt_evaluator';destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for member in z.infolist():assert (destination/member.filename).resolve().is_relative_to(destination.resolve()),member.filename
        z.extractall(destination)
    records.append({'author_script':'external/T2M_GPT/dataset/prepare/download_extractor.sh','Drive_id':'1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-','archive_sha256':sha(archive),'bytes':archive.stat().st_size})
    if args.reward_release:
        items=[('txf0620/MotionRFT_Models','2931fda962875d8c258cc57f5d5446c4487f2812','pretrain/mld/mld_humanml_v1.ckpt','motionrft_base'),('txf0620/MotionRFT_Checkpoints','61457e92aeb68a86778be1601569fe89a3fe04fc','RFT_MLD/checkpoints/rft_mld/best.ckpt','motionrft_checkpoint')]
        for repo,rev,file,subdir in items:
            p=hf_hub_download(repo_id=repo,revision=rev,filename=file,local_dir=root/subdir);records.append({'repo':repo,'revision':rev,'file':file,'sha256':sha(p)})
    write(R/'additional_public_downloads.json',{'assets':records,'source_sha256':sha(__file__),'no_environment_changes':True,'author_cleanup_scripts_not_executed':True})
if __name__=='__main__':main()
