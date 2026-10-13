"""Vector schematic of the audited physical/evaluation contract."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path('paper/current/figures');D=Path('reports/cvpr_motion_release_review_20261012')
def main():
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','font.size':9})
    fig,ax=plt.subplots(figsize=(7.1,1.65));ax.set_xlim(0,10);ax.set_ylim(0,3.3);ax.axis('off')
    def box(x,y,w,h,title,detail,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',linewidth=1,edgecolor=color,facecolor='white'))
        ax.text(x+w/2,y+h*.68,title,ha='center',va='center',fontsize=9,weight='bold',color=color)
        ax.text(x+w/2,y+h*.29,detail,ha='center',va='center',fontsize=7)
    def arrow(a,b,color='#59646d',style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color=color,linewidth=1.1,linestyle=style))
    box(.12,1.45,2.05,1.02,'Motion features','Positions + rotations\nRoot + velocity + contacts','#2479aa')
    box(3.05,1.45,2.3,1.02,'6D recoding','Positive scales + shear','#c77a24')
    box(6.35,2.17,3.25,.89,'Raw learned evaluator','Different coordinates → different scores','#c77a24')
    box(6.35,.19,3.25,.89,'Canonical evaluator','Same coordinates → same embedding','#2479aa')
    arrow((2.24,1.95),(2.96,1.95));arrow((5.43,2.1),(6.25,2.57),'#c77a24');arrow((5.43,1.78),(6.25,.69),'#2479aa')
    ax.text(5.73,.93,'C',color='#2479aa',fontsize=10,weight='bold');arrow((2.3,2.65),(5.22,2.65),'#5a8450');ax.text(3.76,3.03,'Native rotations / RIC / FK remain fixed',ha='center',fontsize=8,color='#5a8450')
    ax.text(3.45,.60,'Fixed encoding, attack, or natural output',ha='center',fontsize=8);ax.text(3.45,.23,'Only 6D rotation channels change; other features stay fixed',ha='center',fontsize=7)
    fig.tight_layout(pad=.2);fig.savefig(P/'motion_consistency_contract.pdf',bbox_inches='tight');fig.savefig(D/'motion_consistency_contract.png',dpi=180,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
