"""Readable vector figures from frozen predictions; no annotations over bars."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PAPER=Path('paper/cvpr2027_training_objectives_revision_20261012');ROOT=Path('reports/cvpr_second_review_20261012')
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavusans'})
def read(p):return json.loads(Path(p).read_text())
def save(fig,name):
 fig.savefig(PAPER/'figures'/(name+'.pdf'),bbox_inches='tight',pad_inches=.07);fig.savefig(PAPER/'figures'/(name+'.png'),dpi=200,bbox_inches='tight',pad_inches=.07);plt.close(fig)
def main():
 assert not (ROOT/'manifest_sha256.json').exists();d=read('reports/cvpr_learned_equivalence_20261011/analysis.json');names=['text_raw','motion_raw','text_canonical'];raw=[d['systems'][n]['mean_bleu4'] for n in names];base=d['baseline_canonical_bleu4'];error=np.array([[v-base-a,b-(v-base)] for n,v in zip(names,raw) for a,b in [d['systems'][n]['gain_ci95_conditional_on_seeds']]]).T
 fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained');x=np.arange(3);ax.bar(x-.17,raw,.32,color='#c65d27',label='Raw');ax.errorbar(x-.17,raw,yerr=error,fmt='none',color='black',capsize=4);ax.bar(x+.17,[base]*3,.32,color='#27856f',label='Canonical')
 for j,n in enumerate(names):ax.scatter(np.full(3,j-.17),d['systems'][n]['scores_per_seed'],s=25,color='white',edgecolor='black',zorder=3)
 ax.set_xticks(x,['Text + motion\nRaw loss','Motion only\nRaw loss','Text + motion\nCanonical loss'],fontsize=16);ax.tick_params(labelsize=16);ax.set_ylabel('BLEU-4',fontsize=18);ax.set_ylim(0,17);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(loc='upper right',fontsize=16);save(fig,'learned_results')
 test=read('reports/cvpr_second_review_confirmation_20261012/analysis.json');full=read('reports/cvpr_learned_confirmation_20261012/analysis.json');print('OLD_FULL_KEYS',list(full),flush=True)
 # The original .35 confirmation uses the same fixed generator TEST motion.
 learn=full['learned'] if 'learned' in full else full
 print('OLD_LEARN_KEYS',list(learn),flush=True)
 baseline=test['budget']['baseline_canonical'];bounds=[.01,.05,.1,.35];data=[]
 for bound in bounds[:-1]:
  a=test['budget']['arms'][f'bound{bound:g}'];data.append({'gain':a['gain'],'ci':a['paired_ci95'],'seeds':np.array(a['raw_per_seed'])-baseline,'rms':a['RMS_to_canonical_per_seed']})
 # Recompute RMS independently from saved outputs; score/gain CI are already frozen.
 from run_signspark_geometry_dev import project
 oldroot=Path('reports/cvpr_learned_confirmation_20261012/learned');scores=[];rms=[]
 for s in [42,123,777]:
  r=read(oldroot/f'seed{s}_text_raw_source_text.json');scores.append(r['raw']['scores']['bleu']['bleu4'])
  with np.load(oldroot/f'seed{s}_text_raw_source_text_poses.npz') as z:rms.append(float(np.mean([np.sqrt(np.mean((z[f'attacked{i:04d}']-project(z[f'attacked{i:04d}']))**2)) for i in range(642)])))
 exact=learn['systems']['text_raw']['gain_ci95_conditional_on_seeds']
 data.append({'gain':float(np.mean(scores)-baseline),'ci':exact,'seeds':np.array(scores)-baseline,'rms':rms})
 fig,axes=plt.subplots(1,2,figsize=(14,3.7),layout='constrained');x=np.arange(4)
 gains=np.array([a['gain'] for a in data]);err=np.array([[a['gain']-a['ci'][0],a['ci'][1]-a['gain']] for a in data]).T
 axes[0].bar(x,gains,.6,color='#c65d27');axes[0].errorbar(x,gains,yerr=err,fmt='none',capsize=5,color='black')
 for j,a in enumerate(data):axes[0].scatter(np.full(3,j),a['seeds'],color='white',edgecolor='black',zorder=3,s=35)
 axes[0].axhline(0,color='gray',linewidth=1);axes[0].set_ylabel('Raw BLEU gain',fontsize=18)
 means=[np.mean(a['rms']) for a in data];axes[1].bar(x,means,.6,color='#245f91')
 for j,a in enumerate(data):axes[1].scatter(np.full(3,j),a['rms'],color='white',edgecolor='black',zorder=3,s=35)
 natural=read('reports/cvpr_second_review_confirmation_20261012/budget/natural_generator.json')['mean_RMS'];axes[1].axhline(natural,color='#27856f',linestyle='--',linewidth=2,label='Natural raw generator');axes[1].legend(fontsize=16,loc='upper left');axes[1].set_ylabel('RMS to canonical encoding',fontsize=18)
 for ax in axes:ax.set_xticks(x,['0.01','0.05','0.10','0.35']);ax.set_xlabel('Coefficient bound',fontsize=18);ax.tick_params(labelsize=17);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 save(fig,'learned_budget');(ROOT/'figure_data.json').write_text(json.dumps({'bounds':bounds,'gain':gains.tolist(),'CI': [a['ci'] for a in data],'RMS':[a['rms'] for a in data],'natural_raw_RMS':natural,'RMS035_from_frozen_arrays':True,'original_primary_CI_from_frozen_JSON':exact},indent=2)+'\n')
 print('NEW_FIGURES_READY')
if __name__=='__main__':main()
