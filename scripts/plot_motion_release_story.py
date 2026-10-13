"""HumanML3D teaser from actual saved motion, no mesh/body model required."""
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from motion_release_audit_common import read,write,sha
R=Path('reports/cvpr_motion_release_review_20261012')
P=Path('paper/current');AUDIT=Path('reports/cvpr_humanml3d_audit_20261012')
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42,'ps.fonttype':42})

def main():
    sys.path.insert(0,str(Path('external/MDM').resolve()))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
    i=0;clip=read(AUDIT/'test_selection.json')[i];n=clip['frames']
    with np.load(AUDIT/'test_mdm.npz') as z:x=z['raw'][i,:n].copy()
    with np.load(AUDIT/'test/mdm/target_caption.npz') as z:y=z['raw'][i,:n].copy()
    a=recover_from_ric(torch.tensor(x),22).numpy();b=recover_from_ric(torch.tensor(y),22).numpy();assert np.array_equal(a,b)
    scores=read(AUDIT/'test_analysis.json')['results']['mdm']['target_caption'];before=scores['R1_before'];after=scores['R1_after']
    fig=plt.figure(figsize=(13.6,4.05));canvas=fig.add_axes([0,0,1,1]);canvas.set_axis_off();canvas.text(.02,.97,'Same decoded movement. Different reported scores.',fontsize=23,fontweight='bold',va='top')
    canvas.text(.025,.84,'“'+clip['caption']+'”',fontsize=15,color='#526174')
    for j,q in enumerate([.2,.5,.8]):
        t=int(q*(n-1));ax=fig.add_axes([.015+j*.16,.20,.16,.59],projection='3d');xyz=a[t];center=xyz[0].copy();center[1]=.95;xyz=xyz-center
        for chain in t2m_kinematic_chain:
            v=xyz[chain];w=(b[t]-center)[chain];ax.plot(v[:,0],v[:,2],v[:,1],color='#265f91',lw=2.5);ax.plot(w[:,0],w[:,2],w[:,1],color='#d17a39',lw=1.,ls='--')
        ax.set(xlim=(-.8,.8),ylim=(-.8,.8),zlim=(-1.,1.));ax.view_init(elev=14,azim=-55);ax.set_axis_off();ax.set_box_aspect((1,1,1.5));ax.set_title(f'{q:.0%} of clip',fontsize=11,pad=-6)
    canvas.text(.025,.14,'Original and recoded joint trajectories overlap exactly',fontsize=12,color='#267a64')
    for y0,color,title,value in [(.53,'#265f91','Original 6D features',f'R-precision top-1: {before:.3f}'),(.26,'#c9692d','Equivalent 6D features',f'R-precision top-1: {after:.3f}')]:
        canvas.add_patch(FancyBboxPatch((.54,y0),.425,.21,boxstyle='round,pad=.012',edgecolor=color,facecolor='white',linewidth=1.7));canvas.text(.56,y0+.16,title,color=color,fontsize=16,fontweight='bold',va='top');canvas.text(.56,y0+.09,value,fontsize=21,va='top')
    canvas.text(.555,.14,'Scores aggregate 1,024 MDM clips; caption-targeted attack',fontsize=11,color='#526174');canvas.text(.555,.075,'Canonicalize first: both encodings give the same embedding',fontsize=12,color='#267a64')
    target=P/'figures/teaser.pdf';fig.savefig(target,bbox_inches='tight',pad_inches=.04);fig.savefig(R/'teaser_preview.png',dpi=150,bbox_inches='tight',pad_inches=.04);plt.close(fig)
    write(R/'teaser_evidence.json',{'source_clip':clip['clip_id'],'illustration_index':i,'selection':'first fixed TEST item, not chosen by an attack result','caption':clip['caption'],'rendering':'native RIC skeleton from saved MDM features, no avatar/mesh model','RIC_positions_bitwise_equal':True,'scores_are_cohort_aggregate_not_per_clip':True,'cohort_clips':1024,'R1_before':before,'R1_after':after,'inputs_sha256':{str(p):sha(p) for p in [AUDIT/'test_mdm.npz',AUDIT/'test/mdm/target_caption.npz',AUDIT/'test_analysis.json']},'figure_sha256':sha(target),'producer_sha256':sha(__file__)})
    print('HUMANML_TEASER_VERIFIED',flush=True)
if __name__=='__main__':
    R.mkdir(exist_ok=True);main()
