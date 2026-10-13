"""Data-backed opening figure and publication graphics for the revised paper."""
import json
import shutil
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from verify_signspark_studies import sha

ROOT=Path('reports/cvpr_paper_story_20261011')
PAPER=Path('paper/cvpr2027_learned_equivalence_20261011')
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42,
                     'mathtext.fontset':'dejavusans'})
COLORS={'blue':'#245f91','orange':'#c65d27','green':'#27856f','gray':'#566574'}


def read(p):return json.loads(Path(p).read_text())
def save(fig,name):
    folder=PAPER/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight',pad_inches=.06)
    fig.savefig(folder/(name+'.png'),dpi=200,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)


def teaser():
    source=Path('reports/cvpr_evaluation_confirmation_20261011/heldout/PHOENIX-2014T')
    scores=read(source/'rolled_wrong_caption_scores.json');i=108
    assert scores['references'][i]=='guten abend liebe zuschauer .'
    assert scores['baseline_actual']['hypotheses'][i]==scores['references'][i]
    assert scores['raw_after_actual']['hypotheses'][i]==scores['targets'][i]
    assert scores['canonical_before_actual']['hypotheses'][i]==scores['canonical_after_actual']['hypotheses'][i]
    clip=scores['clip_ids'][i];images=sorted((Path('features/fullFrame-210x260px/test')/clip).glob('images*.png'))
    assert len(images)>=3
    chosen=[images[int(q*(len(images)-1))] for q in [.2,.5,.8]]
    assets=PAPER/'figures/source_frames';assets.mkdir(exist_ok=True)
    for image in chosen:shutil.copy2(image,assets/image.name)
    fig=plt.figure(figsize=(14,4.5));ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
    ax.text(.02,.95,'Different captions from the same joint rotations',fontsize=22,weight='bold',va='top')
    ax.text(.02,.835,'Observed PHOENIX source clip',fontsize=17,weight='bold')
    for j,image in enumerate(chosen):
        a=fig.add_axes([.02+j*.105,.355,.10,.45]);a.imshow(plt.imread(image));a.set_axis_off()
    ax.text(.02,.31,'Reference: “Good evening, dear viewers.”',fontsize=15,va='top')
    ax.text(.02,.20,'Joint rotations preserved\nthroughout the feature intervention',fontsize=16,color=COLORS['green'],va='top')
    rows=[(.62,COLORS['blue'],'Original encoding',scores['baseline_actual']['hypotheses'][i]),
          (.36,COLORS['orange'],'Equivalent encoding — wrong-caption attack',scores['raw_after_actual']['hypotheses'][i]),
          (.10,COLORS['green'],'Canonicalize before reading — both encodings',scores['canonical_after_actual']['hypotheses'][i])]
    for y,color,title,hypothesis in rows:
        box=FancyBboxPatch((.37,y-.02),.61,.225,boxstyle='round,pad=.012',facecolor='white',edgecolor=color,linewidth=2)
        ax.add_patch(box);ax.text(.385,y+.16,title,fontsize=17,color=color,weight='bold',va='top')
        display='\n'.join(textwrap.wrap(hypothesis,width=57))
        ax.text(.385,y+.093,display,fontsize=17,va='top')
        if 'wrong-caption' in title:ax.text(.385,y+.015,'“Mostly overcast or foggy during the day.”',fontsize=15,fontstyle='italic',color=color,va='top')
    save(fig,'teaser')
    record={'source_clip':clip,'TEST_subset_index':i,'original_reference':scores['references'][i],
        'original_hypothesis':scores['baseline_actual']['hypotheses'][i],
        'wrong_target':scores['targets'][i],'attacked_hypothesis':scores['raw_after_actual']['hypotheses'][i],
        'canonical_before':scores['canonical_before_actual']['hypotheses'][i],
        'canonical_after':scores['canonical_after_actual']['hypotheses'][i],
        'source_scores_sha256':sha(source/'rolled_wrong_caption_scores.json'),
        'RGB_source_frames':{str(p):sha(p) for p in chosen},
        'selection':'illustrative exact-match greeting example; chosen after outputs, not used for statistical selection',
        'translation':'reference English translation supplied for exposition; metrics use original German',
        'image_scope':'RGB source frames identify the clip; not renderings of the attacked representation'}
    (ROOT/'sources/teaser_example.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')


def method():
    fig,ax=plt.subplots(figsize=(14,4.2));ax.set(xlim=(-.025,1.015),ylim=(-.09,1.12));ax.set_axis_off()
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor='white',edgecolor=color,linewidth=1.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=18)
    def arrow(start,end,color=COLORS['gray'],style='-'):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=18,linewidth=1.8,color=color,linestyle=style))
    box(.01,.52,.18,.26,'Source text\nand duration',COLORS['gray'])
    box(.245,.52,.18,.26,'Frozen\ngenerator',COLORS['blue'])
    box(.49,.52,.20,.26,'Learned bounded\noutput layer',COLORS['orange'])
    box(.76,.52,.20,.26,'Raw-feature\nBT reader',COLORS['orange'])
    arrow((.19,.65),(.235,.65));arrow((.425,.65),(.48,.65));arrow((.69,.65),(.75,.65))
    ax.plot([.10,.10,.56,.56],[.795,.94,.94,.795],color=COLORS['gray'],linewidth=1.5)
    arrow((.56,.84),(.56,.795),COLORS['gray'])
    ax.text(.30,.975,'Source-text conditioning',fontsize=16,color=COLORS['gray'],ha='center')
    ax.text(.458,.85,'x',fontsize=20,ha='center');ax.text(.721,.85,'y',fontsize=20,ha='center')
    box(.755,.06,.21,.26,'Canonicalize\nthen read',COLORS['green'])
    arrow((.71,.65),(.71,.19),COLORS['green']);arrow((.71,.19),(.745,.19),COLORS['green'])
    ax.text(.38,.17,r'$R(y)=R(x)$',color=COLORS['green'],fontsize=25,ha='center')
    ax.text(.38,.05,'Identical represented rotations',fontsize=17,color=COLORS['green'],ha='center')
    arrow((.86,.81),(.59,.81),COLORS['orange'],'--')
    ax.text(.73,1.09,'TRAIN caption loss updates the output layer',fontsize=17,color=COLORS['orange'],ha='center')
    ax.text(.86,.385,'Score can change',fontsize=17,color=COLORS['orange'],ha='center')
    ax.text(.86,-.045,'Prediction unchanged',fontsize=17,color=COLORS['green'],ha='center')
    save(fig,'learned_method')


def learned():
    r=read('reports/cvpr_learned_equivalence_20261011/analysis.json')
    names=['text_raw','motion_raw','text_canonical'];labels=['Text + motion\nRaw loss','Motion only\nRaw loss','Text + motion\nCanonical loss']
    fig,ax=plt.subplots(figsize=(7.2,4.8),layout='constrained');x=np.arange(3)
    raw=[r['systems'][name]['mean_bleu4'] for name in names];base=r['baseline_canonical_bleu4']
    ci=[r['systems'][name]['gain_ci95_conditional_on_seeds'] for name in names]
    error=np.asarray([[v-base-a,b-(v-base)] for v,(a,b) in zip(raw,ci)]).T
    ax.bar(x-.17,raw,.32,color=COLORS['orange'],label='Raw evaluation')
    ax.errorbar(x-.17,raw,yerr=error,fmt='none',color='black',capsize=4)
    ax.bar(x+.17,[base]*3,.32,color=COLORS['green'],label='Canonical evaluation')
    for j,name in enumerate(names):ax.scatter(np.full(3,j-.17),r['systems'][name]['scores_per_seed'],s=18,color='white',edgecolor='black',zorder=3)
    ax.set_xticks(x,labels,rotation=0,ha='center',fontsize=15);ax.set(ylabel='BLEU-4',ylim=(0,17))
    ax.legend(loc='center right',fontsize=15);ax.grid(axis='y',alpha=.2)
    ax.tick_params(labelsize=15);ax.yaxis.label.set_size(17)
    ax.set_title('PHOENIX DEV: learned representation, frozen motion',fontsize=16)
    ax.text(.01,.975,'Bars: mean of 3 layer-training seeds; dots: individual seeds\nWhiskers: paired 95% CI for gain from the fixed baseline',transform=ax.transAxes,va='top',fontsize=13)
    save(fig,'learned_results')


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    teaser();method();learned()
    print('PAPER_STORY_FIGURES_READY',flush=True)


if __name__=='__main__':main()
