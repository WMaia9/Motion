"""Vector figures from completed frozen/read-only scores."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path('reports/cvpr_representation_paper_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence')
OLD=Path('reports/cvpr_evaluation_confirmation_20261011')


def save(fig,name):
    fig.savefig(PAPER/'figures'/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(PAPER/'figures'/(name+'.png'),bbox_inches='tight',dpi=200);plt.close(fig)


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.3),layout='constrained')
    labels=['PHOENIX','How2Sign','CSL-Daily*'];before=[];after=[];wrong=[];actual=[]
    for corpus in ['PHOENIX-2014T','How2Sign','CSL-Daily']:
        folder=ROOT/corpus if corpus=='CSL-Daily' else OLD/'heldout'/corpus
        a=json.loads((folder/'correct_caption_scores.json').read_text());b=json.loads((folder/'rolled_wrong_caption_scores.json').read_text())
        before.append(a['baseline_target_scores']['bleu']['bleu4']);after.append(a['raw_after_target_scores']['bleu']['bleu4'])
        wrong.append(b['raw_after_target_scores']['bleu']['bleu4']);actual.append(b['raw_after_actual']['scores']['bleu']['bleu4'])
    x=np.arange(3)
    for ax,left,right,l1,l2 in [(axes[0],before,after,'Before','After; target known'),(axes[1],wrong,actual,'After vs shifted target','After vs true reference')]:
        ax.bar(x-.19,left,.36,label=l1,color='#397ca6');ax.bar(x+.19,right,.36,label=l2,color='#cb6447')
        ax.set_xticks(x,labels);ax.set_ylim(0,105);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper right',fontsize=7)
    axes[0].set_title('Correct-caption audit');axes[1].set_title('Shifted-caption audit');axes[0].set_ylabel('Native BLEU-4')
    save(fig,'attacks')
    with (OLD/'rankings/summary.csv').open() as f:rows=list(csv.DictReader(f))
    values={(r['system'],r['representation']):float(r['bleu4']) for r in rows}
    kinds=['raw','canonical','gauge_scale090','gauge_scale110','gauge_shear010']
    names={'geometry':'Geometry','velocity':'Physical velocity','physical_ce':'Physical CE','physical_generated_near':'Generated near negative'}
    fig,ax=plt.subplots(figsize=(6.7,2.6),layout='constrained')
    for name,label in names.items():ax.plot(range(5),[values[name,k] for k in kinds],marker='o',label=label,linewidth=1.3,markersize=4)
    ax.set_xticks(range(5),['Raw','Canonical','Scale 0.9','Scale 1.1','Shear 0.1']);ax.set_ylabel('Native BLEU-4');ax.grid(alpha=.2);ax.legend(loc='lower right',ncol=2)
    save(fig,'rankings')
    fig,ax=plt.subplots(figsize=(6.7,2.3));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',linewidth=.8,edgecolor=color,facecolor=color+'15'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    box(.1,.35,3.1,2.3,'Two encodings of one motion\n\n'+r'$(a,b)$'+'\n'+r'$(\alpha a,\;\beta b+\gamma a)$'+'\n\n'+r'$\alpha,\beta>0$; timing fixed','#397ca6')
    box(4.0,2.0,2.25,.7,'Rotation\nconversion','#34835c')
    box(4.0,1.15,2.25,.7,'Raw reader','#cb6447')
    box(4.0,.3,2.25,.7,'Canonicalize,\nthen read','#34835c')
    box(7.1,2.0,2.75,.7,'Same joint rotations','#34835c')
    box(7.1,1.15,2.75,.7,'Predictions can differ','#cb6447')
    box(7.1,.3,2.75,.7,'Same prediction','#34835c')
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','linewidth':.9,'color':'#34495e'})
    for y in [2.35,1.5,.65]:
        arrow((3.3,y),(3.9,y));arrow((6.35,y),(7.0,y))
    save(fig,'mechanism')
    print('PAPER_FIGURES_COMPLETE',flush=True)


if __name__=='__main__':main()
