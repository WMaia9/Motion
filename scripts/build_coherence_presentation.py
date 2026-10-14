"""Build current-paper tables/figures from immutable direction controls."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from freeze_motion_release_studies import read,write,sha

D=Path('reports/cvpr_coherence_review_20261014')
P=Path('paper/current')
C=Path('reports/cvpr_coherent_blocks_20261014')
OLD=Path('reports/cvpr_matched_blocks_20261013')
S=Path('reports/cvpr_sign_natural_restoration_20261014')


def checked(path):
    root=Path(*path.parts[:2]);m=read(root/'manifest_sha256.json')
    assert sha(path)==m[str(path.relative_to(root))],path
    return read(path)


def table(name,caption,label,header,rows):
    text='\\begin{table*}[t]\n\\centering\\small\n'
    text+=f'\\caption{{{caption}}}\\label{{{label}}}\n'
    text+='\\begin{tabular}{@{}llrrrr@{}}\\toprule\n'+header+'\\\\\\midrule\n'
    text+='\n'.join(rows)+'\n\\bottomrule\\end{tabular}\\end{table*}\n'
    (P/(name+'.tex')).write_text(text)


def main():
    assert not (D/'manifest_sha256.json').exists()
    assert checked(C/'verification.json')['passed'] and checked(S/'verification.json')['passed']
    old={r:checked(OLD/r/'analysis.json')['conditions'] for r in ['guo','tmr']}
    new={r:checked(C/r/'analysis.json') for r in ['guo','tmr']}
    inputs=[C/r/'analysis.json' for r in ['guo','tmr']]+[OLD/r/'analysis.json' for r in ['guo','tmr']]+[S/'analysis.json']
    assets=[]
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'Liberation Sans','font.size':9})
    blocks=['null','rotation','velocity','contact','position']
    labels=['Rotation fiber','Changed rotations','Velocities','Contacts','Positions']
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65))
    for mode,marker,color,offset in [('random','o','#767676',-.18),('fixed_plus','^','#27769b',0.),('fixed_minus','s','#c66a2d',.18)]:
        for ax,reader,endpoint,factor in [(axes[0],'guo','FID',1.),(axes[1],'tmr','R1',100.)]:
            values=[];lo=[];hi=[]
            for block in blocks:
                if block=='null':
                    name={'random':'null_random','fixed_plus':'null_plus','fixed_minus':'null_minus'}[mode]
                    q=old[reader][name+'_r3'][endpoint];ci=q['family28_CI95']
                elif mode=='random':q=old[reader][block+'_r3'][endpoint];ci=q['family28_CI95']
                else:q=new[reader]['conditions'][block+'_'+mode+'_r3'][endpoint];ci=q['family32_CI95']
                value=q['mean_change']*factor;a,b=np.asarray(ci)*factor
                values.append(value);lo.append(max(0.,value-a));hi.append(max(0.,b-value))
            ax.errorbar(values,np.arange(5)+offset,xerr=[lo,hi],fmt=marker,color=color,markersize=4,capsize=2,label=mode.replace('_',' '))
    axes[0].set_yticks(np.arange(5),labels);axes[1].set_yticks(np.arange(5),[])
    axes[0].set_xscale('symlog',linthresh=1e-4)
    axes[0].set_xlim(-1e-5,1.5)
    axes[0].set_xticks([0.,.001,.01,.1,1.],['0',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$','1'])
    axes[0].set_xlabel(r'Guo $\Delta$ FID');axes[1].set_xlabel(r'TMR $\Delta$ R1 (percentage points)')
    for ax in axes:ax.invert_yaxis();ax.axvline(0,color='black',lw=.6,alpha=.5);ax.grid(axis='x',alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.58,-.005))
    fig.subplots_adjust(left=.20,right=.99,bottom=.27,top=.96,wspace=.13)
    file=P/'figures/matched_blocks.pdf';fig.savefig(file);plt.close(fig);assets.append(file)
    # Retain all original radii/random/null controls with the new budget symbol.
    conditions=['null_plus','null_minus','null_random','rotation','velocity','contact','position']
    names=['Null: fixed +','Null: fixed −','Null: random','Changed rotations','Velocities','Contacts','Positions']
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65))
    for radius,marker,color,offset in [(1,'o','#27769b',-.1),(3,'s','#c66a2d',.1)]:
        for ax,reader,endpoint,factor in [(axes[0],'guo','FID',1.),(axes[1],'tmr','R1',100.)]:
            values=[];lo=[];hi=[]
            for condition in conditions:
                q=old[reader][f'{condition}_r{radius}'][endpoint];value=q['mean_change']*factor;a,b=np.asarray(q['family28_CI95'])*factor
                values.append(value);lo.append(max(0.,value-a));hi.append(max(0.,b-value))
            ax.errorbar(values,np.arange(7)+offset,xerr=[lo,hi],fmt=marker,color=color,markersize=3,capsize=2,label=rf'$\kappa={radius}$')
    axes[0].set_yticks(np.arange(7),names);axes[1].set_yticks(np.arange(7),[])
    axes[0].set_xscale('symlog',linthresh=1e-5);axes[0].set_xticks([0.,1e-4,.001,.01,.1],['0',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'])
    axes[0].set_xlim(-1e-6,.14)
    axes[0].set_xlabel(r'Guo $\Delta$ FID');axes[1].set_xlabel(r'TMR $\Delta$ R1 (percentage points)')
    for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    axes[0].legend(loc='upper left',fontsize=8,frameon=False)
    fig.subplots_adjust(left=.20,right=.99,bottom=.21,top=.97,wspace=.12)
    file=P/'figures/matched_blocks_random.pdf';fig.savefig(file);plt.close(fig);assets.append(file)
    for reader in ['guo','tmr']:
        first='FID' if reader=='guo' else 'matched_cosine'
        for contrast in [False,True]:
            rows=[];records=new[reader]['fixed_minus_random' if contrast else 'conditions']
            for radius in [1,3]:
                for block in ['rotation','velocity','contact','position']:
                    for sign in ['plus','minus']:
                        q=records[f'{block}_fixed_{sign}_r{radius}'];a=q[first];b=q['R1'];lo,hi=a['family32_CI95'];rl,rh=np.asarray(b['family32_CI95'])*100
                        text={'rotation':'Rotations','velocity':'Velocities','contact':'Contacts','position':'Positions'}[block]+(' $+$' if sign=='plus' else ' $-$')
                        rows.append(f"{radius}&{text}&{a['mean_change']:+.5f}&$[{lo:+.5f},{hi:+.5f}]$&{100*b['mean_change']:+.3f}&$[{rl:+.3f},{rh:+.3f}]$\\\\")
            kind='contrast' if contrast else 'fixed';filename=f'table_coherent_{reader}_{kind}'
            caption=rf'\textbf{{{reader.upper()} {"fixed minus random" if contrast else "fixed directions"}, all twenty draws.}} '
            caption+=r'Both signs and radii; paired $t_{19}$ intervals adjust 32 endpoints in this '+('secondary contrast' if contrast else 'primary')+' family. '
            caption+='Directions are fixed in time, but held-fixed feature blocks may be physically inconsistent. Guo excludes contacts; TMR reads all 263 channels.'
            header=rf'$\kappa$&Intervention&$\Delta$ {"FID" if reader=="guo" else "cosine"}&95\% interval&$\Delta$ R1 (pp)&95\% interval'
            table(filename,caption,'tab:'+filename,header,rows);assets.append(P/(filename+'.tex'))
    # The restored aggregate is verified directly from all saved hypotheses.
    natural=checked(S/'analysis.json')['results'];rows=[]
    for name,label in [('PHOENIX-2014T','PHOENIX'),('How2Sign','How2Sign'),('CSL-Daily','CSL-Daily')]:
        v=natural[name];rows.append(f"{label}&{v['changed_hypotheses']:,}/{v['clips']:,}&{100*v['changed_fraction']:.1f}\\%&{v['BLEU4']['raw']:.4f}&{v['BLEU4']['canonical']:.4f}&{v['canonical_minus_raw_BLEU4']:+.4f}\\\\")
    table('table_sign_natural',r'\textbf{Natural dependence in real sign inputs, no attack.} Full native TEST with common loader preprocessing and frozen raw reader; no source-conditioned layer, optimization or new decoding. The shared left-hand preprocessing precedes both paths. Hypothesis counts and six BLEUs are independently recomputed; the CSL score is character-level.', 'tab:signnatural',r'Corpus&Changed / clips&Changed \%&Raw BLEU&$\C$ BLEU&$\C-$raw',rows);assets.append(P/'table_sign_natural.tex')
    write(D/'presentation_evidence.json',{'passed':True,'inputs_sha256':{str(p):sha(p) for p in inputs},'generated_assets':{str(p):sha(p) for p in assets},'plot_scope':'kappa3 primary; both radii and all prospective contrasts supplemental','old_family28_and_new_family32_separate':True,'builder_sha256':sha(__file__)})
    print('COHERENCE_PRESENTATION_BUILT',len(assets),flush=True)


if __name__=='__main__':main()
