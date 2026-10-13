"""Learn a bounded rotation-preserving output layer on a frozen generator.

All fitting uses generated TRAIN motions and source text only. Evaluation is
DEV-only, feed-forward, with no per-clip optimization and no reference poses.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import run_physical_semantic_ddpm as base
from run_signspark_gauge_attack import SignCollator, Batch, PAD_TOKEN
from run_reviewer_attack_ablation import transformed
from run_evaluation_heldout import geometry
from analyze_signspark_geometry_dev import statistics, b4
from verify_signspark_studies import sha

ROOT = Path('reports/cvpr_learned_equivalence_20261011')
GENERATOR = base.ROOT/'geometry.pt'
SEEDS = [42,123,777]
ARMS = ['text_raw', 'text_canonical', 'motion_raw']
EPOCHS = 20
BOUND = .35


def read(p): return json.loads(Path(p).read_text())
def write(name, value):
    p = ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def context():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest_sha256.json').exists()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32 = False
    audit = read(base.ROOT/'data_audit.json');stoi = {w:i for i,w in enumerate(audit['vocab'])}
    model = base.Denoiser(len(stoi)).cuda()
    model.load_state_dict(torch.load(GENERATOR,map_location='cpu',weights_only=False)['state_dict'],strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return audit,stoi,model


def register():
    audit,stoi,model = context()
    paths = [GENERATOR,base.ROOT/'data_audit.json',base.ROOT/'normalization.npz',
             base.BASE/'checkpoints/PHOENIX-2014T/best.ckpt']
    protocol = {'corpus':'PHOENIX-2014T','seeds':SEEDS,'arms':ARMS,'epochs':EPOCHS,
        'batch_size':64,'lr':.001,'weight_decay':.0001,'coefficient_bound':BOUND,
        'left_hand_unchanged':True,'selection':'last epoch; no DEV/BLEU checkpoint selection',
        'loss':'mean per-clip TRAIN caption CE through frozen BT; no physical/MSE term needed',
        'generation':'frozen DDPM, DDIM50, CFG3, sampling seed102; source text and duration only',
        'adapter':'temporal Conv/MHA; frozen generator text encoder; zero-initialized coefficient head',
        'frame_cap':300,'adapter_token_cap':64,'caption_drop':False,
        'no_new_TEST_use':True,'eval_optimization_steps':0,'clip_ID_is_model_input':False,
        'TRAIN_clips':audit['train_clips'],'DEV_clips':audit['dev_clips'],
        'causal_scope':'purpose-built learned output layer; not spontaneous exploitation by ordinary DDPM fine-tuning',
        'inputs':{str(p):sha(p) for p in paths},'code_sha256':sha(Path(__file__))}
    path = ROOT/'protocol.json'
    if path.exists():assert read(path) == protocol, 'protocol changed after registration'
    else:write('protocol.json',protocol)
    print('LEARNED_PROTOCOL_REGISTERED',flush=True)
    return audit,stoi,model


def entries(split):
    cfg = {'dataset':'PHOENIX-2014T','base_path':str((base.PREV/'data/lmdb').resolve()),
           'index_cache':False,'max_sgn_length':300}
    dataset = base._build_split(cfg,split,None)
    # Metadata only; __getitem__ / reference feature tensors are never used here.
    return [{'clip_id':e.sequence,'frames':e.num_frames,'text':list(e.txt)} for e in dataset.examples]


def token_tensor(records, stoi):
    text = np.zeros((len(records),64),np.int64)
    for j,e in enumerate(records):
        words = [stoi.get(w,1) for w in e['text']][:64] or [2]
        text[j,:len(words)] = words
    return torch.tensor(text,device='cuda')


@torch.no_grad()
def prepare():
    audit,stoi,model = register()
    train,dev = entries('train'),entries('dev')
    assert [e['clip_id'] for e in train] == audit['train_ids']
    assert [e['clip_id'] for e in dev] == audit['dev_ids']
    assert not set(audit['train_ids']) & set(audit['dev_ids'])
    training_text = {' '.join(e['text']) for e in train}
    write('data_audit.json',{'train':train,'dev':dev,'train_dev_clip_overlap':0,
        'DEV_exact_caption_seen_in_TRAIN':[(' '.join(e['text']) in training_text) for e in dev],
        'reference_pose_tensors_read_by_generator':False})
    with np.load(base.ROOT/'normalization.npz') as z:mean,std = z['mean'],z['std']
    mt = torch.tensor(mean,device='cuda');st = torch.tensor(std,device='cuda');alpha = base.alpha_schedule()
    times = np.linspace(999,0,50).round().astype(int)
    for split,records in [('train',train),('dev',dev)]:
        folder = ROOT/'generated'/split;folder.mkdir(parents=True,exist_ok=True)
        generator = torch.Generator(device='cuda').manual_seed(102)
        for chunk_start in range(0,len(records),128):
            end = min(chunk_start+128,len(records));path = folder/f'poses_{chunk_start:05d}_{end:05d}.npz'
            exists = path.exists();saved = {}
            for start in range(chunk_start,end,16):
                group = records[start:min(start+16,end)]
                lengths = torch.tensor([e['frames'] for e in group],device='cuda')
                xt = torch.randn((len(group),int(lengths.max()),240),device='cuda',generator=generator)
                if exists:continue  # Consume the identical noise stream for exact restart.
                tokens = token_tensor(group,stoi);null = torch.zeros_like(tokens);null[:,0] = 2
                for k,ti in enumerate(times):
                    timestep = torch.full((len(group),),int(ti),device='cuda')
                    conditional = model(xt,timestep,tokens,lengths);unconditional = model(xt,timestep,null,lengths)
                    predicted = unconditional+3*(conditional-unconditional)
                    at = alpha[ti];next_t = times[k+1] if k+1 < len(times) else -1
                    an = alpha[next_t] if next_t >= 0 else torch.tensor(1.,device='cuda')
                    noise = (xt-at.sqrt()*predicted)/(1-at).sqrt()
                    xt = an.sqrt()*predicted+(1-an).sqrt()*noise
                physical = (xt*st+mt).cpu().numpy()
                for j,n in enumerate(lengths.tolist()):
                    pose = base.project(physical[j,:n])
                    assert np.isfinite(pose).all()
                    saved[f'clip{start+j:05d}'] = pose
            if not exists:np.savez_compressed(path,**saved)
            print('FROZEN_GENERATION',split,end,len(records),flush=True)
    write('generated/complete.json',{'train':len(train),'dev':len(dev),
        'all_poses_are_canonical_representatives':True,'source_pose_tensors_read':False})


class OutputLayer(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.token = copy.deepcopy(generator.token);self.token_pos = copy.deepcopy(generator.token_pos)
        self.text = copy.deepcopy(generator.text)
        for module in [self.token,self.token_pos,self.text]:
            for p in module.parameters():p.requires_grad_(False)
        self.input = nn.Conv1d(240,128,1);self.position = nn.Embedding(300,128)
        self.attention = nn.MultiheadAttention(128,4,batch_first=True,dropout=0.)
        self.convs = nn.ModuleList([nn.Conv1d(128,128,3,padding=d,dilation=d) for d in [1,2,4]])
        self.norms = nn.ModuleList([nn.GroupNorm(8,128) for _ in range(3)])
        self.output = nn.Conv1d(128,120,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        mask = torch.ones(1,1,40,1);mask[:,:,10:25] = 0
        self.register_buffer('joint_mask',mask)

    def forward(self,pose,tokens,lengths):
        self.text.eval()
        padding = tokens == 0
        with torch.no_grad():
            z = self.token(tokens)+self.token_pos(torch.arange(tokens.shape[1],device=pose.device))[None]
            z = self.text(z,src_key_padding_mask=padding)
        valid = torch.arange(pose.shape[1],device=pose.device)[None] < lengths[:,None]
        h = self.input(pose.transpose(1,2))+self.position(torch.arange(pose.shape[1],device=pose.device)).T[None]
        query = h.transpose(1,2)
        context,_ = self.attention(query,z,z,key_padding_mask=padding,need_weights=False)
        h = (h+context.transpose(1,2))*valid[:,None]
        for conv,norm in zip(self.convs,self.norms):h = (h+conv(F.silu(norm(h))))*valid[:,None]
        c = BOUND*torch.tanh(self.output(h).transpose(1,2).reshape(*pose.shape[:-1],40,3))
        c = c*self.joint_mask*valid[:,:,None,None]
        y = transformed(pose.reshape(*pose.shape[:-1],40,6),c).reshape(pose.shape)
        return y,c


def cache(split):
    records = read(ROOT/'data_audit.json')[split];poses = {}
    for path in sorted((ROOT/'generated'/split).glob('poses_*.npz')):
        with np.load(path) as z:poses.update({k:z[k].copy() for k in z.files})
    assert len(poses) == len(records)
    return records,[poses[f'clip{i:05d}'] for i in range(len(records))]


def pack(records,poses,ids,stoi,collator,pad):
    examples = []
    for i in ids:
        examples.append(type('AdapterExample',(),{'sequence':records[i]['clip_id'],'signer':'unknown',
            'sgn':torch.from_numpy(poses[i]),'txt':records[i]['text'],'gls':[]})())
    batch = Batch(collator(examples),txt_pad_index=pad,sgn_dim=240,use_cuda=True)
    tokens = token_tensor([records[i] for i in ids],stoi)
    return batch,tokens


def train():
    audit,stoi,generator = register();records,poses = cache('train')
    teacher = base.make_back_translation_model(base.BASE/'checkpoints/PHOENIX-2014T')
    teacher.eval();teacher.do_recognition = False
    for p in teacher.parameters():p.requires_grad_(False)
    collator = SignCollator(teacher.txt_vocab,None,240);pad = teacher.txt_vocab.stoi[PAD_TOKEN]
    for seed in SEEDS:
        matched = []
        for arm in ARMS:
            folder = ROOT/f'training/seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            path = folder/(arm+'.pt')
            if path.exists():matched.append(read(folder/(arm+'_batches.json')));continue
            base.seed(seed);model = OutputLayer(generator).cuda();model.train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
            history = [];batch_order = [];gradient = None
            for epoch in range(EPOCHS):
                started = time.time();order = np.random.permutation(len(records));ordered = []
                for start in range(0,len(order),256):ordered.extend(sorted(order[start:start+256],key=lambda i:len(poses[i])))
                groups = [ordered[p:p+64] for p in range(0,len(ordered),64)];random.shuffle(groups)
                losses = [];bounds = []
                for ids in groups:
                    if epoch == 0:batch_order.append(list(map(int,ids)))
                    batch,tokens = pack(records,poses,ids,stoi,collator,pad)
                    if arm == 'motion_raw':tokens = torch.zeros_like(tokens);tokens[:,0] = 2
                    output,coefficients = model(batch.sgn,tokens,batch.sgn_lengths.to(batch.sgn.device))
                    consumed = base.physical(output) if arm == 'text_canonical' else output
                    prediction,_ = teacher.forward(consumed,batch.sgn_mask,batch.sgn_lengths,batch.txt_input,batch.txt_mask)
                    logits = prediction[0]
                    per_token = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch.txt.reshape(-1),ignore_index=pad,reduction='none').reshape(batch.txt.shape)
                    loss = (per_token.sum(1)/(batch.txt != pad).sum(1)).mean()
                    if gradient is None:
                        g = torch.autograd.grad(loss,coefficients,retain_graph=True)[0]
                        gradient = {'coefficient_gradient_norm':float(g.detach().norm()),
                            'coefficient_gradient_max':float(g.detach().abs().max()),
                            'generator_frozen':True,'teacher_frozen':True}
                    assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step()
                    losses.append(float(loss.detach()));bounds.append(float(coefficients.detach().abs().mean()))
                record = {'epoch':epoch+1,'train_per_clip_ce':float(np.mean(losses)),
                          'mean_absolute_coefficient':float(np.mean(bounds)),'seconds':time.time()-started}
                history.append(record);write(f'training/seed{seed}/{arm}_history.json',history)
                print('LEARNED_OUTPUT_TRAIN',seed,arm,json.dumps(record),flush=True)
            torch.save({'state_dict':model.state_dict(),'seed':seed,'arm':arm,'epoch':EPOCHS,
                'selection':'last epoch'},path)
            write(f'training/seed{seed}/{arm}_gradient.json',gradient)
            write(f'training/seed{seed}/{arm}_batches.json',batch_order);matched.append(batch_order)
            del model;torch.cuda.empty_cache()
        assert matched[0] == matched[1] == matched[2], 'batch orders not matched'
    write('training/complete.json',{'seeds':SEEDS,'arms':ARMS,'epochs':EPOCHS,'all_first_epoch_batches_matched':True})


def score(teacher,poses,refs,ids):
    hyps = base.back_translate(teacher,[torch.from_numpy(x) for x in poses],batch_size=32)
    scores = base.score_translations(refs,hyps,'phoenix_2014_trans','word')
    assert abs(b4(statistics(refs,hyps).sum(0))-scores['bleu']['bleu4']) < 1e-10
    return {'clip_ids':ids,'references':refs,'hypotheses':hyps,'scores':scores}


@torch.no_grad()
def evaluate():
    audit,stoi,generator = register();records,poses = cache('dev')
    refs = [' '.join(e['text']) for e in records];ids = [e['clip_id'] for e in records]
    teacher = base.make_back_translation_model(base.BASE/'checkpoints/PHOENIX-2014T');teacher.eval();teacher.do_recognition = False
    baseline = score(teacher,poses,refs,ids);write('evaluation/baseline.json',baseline)
    for seed in SEEDS:
        for arm in ARMS:
            model = OutputLayer(generator).cuda();model.load_state_dict(torch.load(ROOT/f'training/seed{seed}/{arm}.pt',map_location='cpu',weights_only=False)['state_dict'],strict=True);model.eval()
            conditions = ['source_text','rolled_adapter_text'] if arm == 'text_raw' else ['source_text']
            for condition in conditions:
                outputs = [];coefficients = []
                for start in range(0,len(records),32):
                    group = records[start:start+32];length = torch.tensor([len(x) for x in poses[start:start+32]],device='cuda')
                    x = torch.zeros(len(group),int(length.max()),240,device='cuda')
                    for j,pose in enumerate(poses[start:start+32]):x[j,:len(pose)] = torch.from_numpy(pose).cuda()
                    conditioning = group if condition == 'source_text' else [records[(start+j+1)%len(records)] for j in range(len(group))]
                    tokens = token_tensor(conditioning,stoi)
                    if arm == 'motion_raw':tokens = torch.zeros_like(tokens);tokens[:,0] = 2
                    y,c = model(x,tokens,length)
                    for j,n in enumerate(length.tolist()):outputs.append(y[j,:n].cpu().numpy().copy());coefficients.append(c[j,:n].cpu().numpy().copy())
                checks = geometry(poses,outputs)
                raw = score(teacher,outputs,refs,ids);canonical = score(teacher,[base.project(x) for x in outputs],refs,ids)
                assert baseline['hypotheses'] == canonical['hypotheses'], 'numerical canonical hypotheses changed'
                targets = refs if condition == 'source_text' else refs[1:]+refs[:1]
                result = {'seed':seed,'arm':arm,'condition':condition,'clips':len(records),'raw':raw,'canonical':canonical,
                    'adapter_target_references':targets,'raw_adapter_target_bleu4':base.score_translations(targets,raw['hypotheses'],'phoenix_2014_trans','word')['bleu']['bleu4'],
                    'no_eval_optimization':True,'canonical_changed_hypotheses':0,**checks}
                tag = f'seed{seed}_{arm}_{condition}'
                write('evaluation/'+tag+'.json',result)
                np.savez_compressed(ROOT/'evaluation'/(tag+'_poses.npz'),**{f'{prefix}{i:04d}':x for prefix,arrays in [('attacked',outputs),('coefficient',coefficients)] for i,x in enumerate(arrays)})
                print('LEARNED_OUTPUT_SCORE',tag,raw['scores']['bleu']['bleu4'],canonical['scores']['bleu']['bleu4'],result['raw_adapter_target_bleu4'],flush=True)
            del model;torch.cuda.empty_cache()
    write('evaluation/complete.json',{'DEV_clips':len(records),'new_TEST_use':False,'conditions':12,'canonical_hypotheses_unchanged':True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','train','evaluate'])
    args = parser.parse_args();globals()[args.phase]()
