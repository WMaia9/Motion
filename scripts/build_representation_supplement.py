"""Build supplementary tables from audited files, without touching old reports."""
import json
from pathlib import Path
ROOT=Path('reports/cvpr_representation_paper_20261011')
OLD=Path('reports/cvpr_evaluation_confirmation_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence')
def read(p):return json.loads(p.read_text())
def ci(r):return f"{r['delta_bleu4']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"
def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    a=read(OLD/'analysis.json')['rankings'];full=read(ROOT/'CSL-Daily/full_real_scores.json')
    names={'geometry':'Geometry','velocity':'Velocity','physical_ce':'Physical CE','physical_generated_near':'Generated near'}
    score_rows=[]
    for name,label in names.items():
        score_rows.append(label+' & '+' & '.join(f"{a['scores'][r][name]:.6f}" for r in ['raw','canonical','gauge_scale090','gauge_scale110','gauge_shear010'])+r'\\')
    pair_rows=[]
    for pair,r in a['pairwise_comparisons'].items():
        left,right=pair.split(' minus ');gap=r['raw_to_canonical_change_in_pairwise_gap']
        pair_rows.append(names[left]+' -- '+names[right]+' & '+ci(r['raw'])+' & '+ci(r['canonical'])+' & '+f"{gap['delta_gap']:.4f} [{gap['ci95'][0]:.4f}, {gap['ci95'][1]:.4f}]"+r'\\')
    text=r'''\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[review]{cvpr}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\usepackage{graphicx,amsmath,amssymb,booktabs}
\usepackage[breaklinks,colorlinks]{hyperref}
\def\paperID{DRAFT}\def\confName{CVPR}\def\confYear{2027}
\title{Same Motion, Different Scores: Supplementary Material}
\author{Anonymous authors}
\begin{document}\maketitle
\setcounter{section}{0}\renewcommand{\thesection}{S\arabic{section}}
\section{Complete equivalence argument}
Let $a\ne0$ and $b$ not be collinear with $a$. Write $u=a/\|a\|$ and $b_\perp=b-(u^Tb)u$. The conversion gives $v=b_\perp/\|b_\perp\|$ and the row rotation matrix $[u;v;u\times v]$. Under $a'=\alpha a$, $b'=\beta b+\gamma a$, with positive $\alpha,\beta$, the normalized first row remains $u$. Consequently
\begin{align}
b'_\perp&=b'-(u^Tb')u\\
&=\beta b+\gamma a-\beta(u^Tb)u-\gamma\|a\|u\\
&=\beta b_\perp.
\end{align}
Normalizing preserves $v$. The third row is also fixed. Negative scales are excluded: they can change rows and the rotation.

Conversely, any feature with this rotation must have $a=\alpha u$, $\alpha>0$, and $b=\gamma u+\beta v$, $\beta>0$. The parameters are uniquely $\alpha=\|a\|$, $\beta=\|b_\perp\|$, and $\gamma=u^Tb$. Thus each admissible fiber has three independent coordinates. This characterizes the known map; it is not a claim of new mathematics.

The transformations form a group on this domain. Applying $(\alpha_1,\beta_1,\gamma_1)$ and then $(\alpha_2,\beta_2,\gamma_2)$ gives
\begin{equation}
(\alpha_2\alpha_1,\ \beta_2\beta_1,\ \beta_2\gamma_1+\gamma_2\alpha_1).
\end{equation}
The identity is $(1,1,0)$, and the inverse has parameters $(\alpha^{-1},\beta^{-1},-\gamma/(\alpha\beta))$.

Canonicalization $C(a,b)=(u,v)$ is constant on each fiber and idempotent. Therefore $f(C(G(x)))=f(C(x))$ for any deterministic reader $f$ in exact arithmetic. For a stochastic reader, the statement concerns a fixed randomness realization or equal output distributions with appropriately coupled randomness, not necessarily equal independent samples. Our audited decoding is deterministic in each fixed environment.

Differentiating along scales and shear produces the three vertical directions $(a,0)$, $(0,b)$, and $(0,a)$; the canonicalization Jacobian annihilates them. On 4,096 nondegenerate double-precision random examples, complete-fiber reconstruction error is at most $4.61\times10^{-15}$, canonical invariance error $4.89\times10^{-15}$, and vertical JVP error $2.95\times10^{-14}$. Four invalid zero/collinear cases are rejected by the provided utility. These numerical checks support the implementation, separately from the algebraic proof.

\section{Protocol and reader contracts}
The audit perturbs features, not evaluator weights. The two registered stages preserve protocol hashes before their corresponding inference. Additional CSL assets are pinned before opening them; the CTC checkpoint was previously selected on development data. There is no TEST-driven training, checkpoint choice, or attack-budget search in these stages. Historical project experiments did use public test splits in other representations; we do not claim project-wide untouched holdouts.

\paragraph{Joint layout.} Each frame has 240 features: ten body rotations, fifteen left-hand rotations, and fifteen right-hand rotations, six coordinates each. The left-hand slice is dimensions 60--149 inclusive. It is canonically prepared for the generation-dump interface, and its perturbation parameters are fixed to zero. Changing the other streams already suffices for the reported effects. A direct native-loader diagnostic confirms zero changed full-test hypotheses after this left-stream preparation on all three corpora. Maximum feature difference is $1.19\times10^{-7}$.

\paragraph{Attack budget.} For each 128-clip subset, four batches of 32 are optimized with 80 Adam steps, learning rate 0.1. Feature sequences are capped at 300 frames. Each frame/joint has three zero-initialized unconstrained parameters $q$. We set $z=0.35\tanh(q)$ and use $(\alpha,\beta,\gamma)=(e^{z_1},e^{z_2},z_3)$. Scales lie in $[e^{-0.35},e^{0.35}]$ and absolute shear is at most 0.35. All checkpoint parameters are frozen. Before/after inference uses the same decoder settings. All source arrays, optimized arrays, coefficients, losses and hypotheses are preserved.

\paragraph{Text and gloss targets.} BT uses native teacher-forced cross-entropy including native padding handling. Correct targets are actual references; shifted targets are the next reference in the sorted subset, circularly. CSL has three equal-caption collisions in the shifted condition, retained in the primary result. Its optimization uses the native jieba word vocabulary; evaluation removes spaces and then inserts spaces between each character, exactly as the official CSL metric. Word BLEU and character BLEU should not be compared as the same scale across corpora.

The CTC recognizer uses two stride-two temporal convolutions, a two-layer bidirectional GRU with 256 hidden units per direction, and a gloss classifier. Fixed training normalization buffers are part of the checkpoint. Best development checkpoint is from epoch 21 of a 40-epoch run. Greedy CTC decoding collapses repetitions and removes the blank. CuDNN is disabled consistently in this audit to support eval-mode recurrent input gradients. Two full-test references and one subset reference have empty gloss annotations. They remain included in micro-WER; the zero correct-target WER includes 127 nonempty references and one empty reference. Shifted targets contain no unknown vocabulary tokens and one length/repetition-infeasible target, retained with the registered zero-infinity CTC setting. The attack is not reported as 128 fully recognized nonempty gloss sentences.

\paragraph{Missing dependency amendment.} The initial CSL run stopped before its first score because jieba was absent. The failure log and pre-inference amendment record installation of jieba 0.42.1 in the existing virtual environment. The producer was rerun without changing its hyperparameters or model. There is no discarded scored CSL run.

\section{Full generated-motion comparison}
Four frozen local DDPM checkpoints are sampled with text and the real sequence duration capped at 300, DDIM 50, guidance 3 and seed 102. None uses reference poses, glosses or anchors in the audited sampler. A removal of historical reference-tensor noise allocation reproduces 16 development samples bitwise. Each model supplies 642 motions. Representations are raw, canonical, joint scales 0.9/1.1, and shear 0.1; fixed scale/shear conditions leave the left stream unchanged. The numerical input changes but rotations do not.

\begin{table*}[t]\centering\small
\begin{tabular}{lrrrrr}\toprule
Variant & Raw & Canonical & Scale 0.9 & Scale 1.1 & Shear 0.1\\\midrule
SCORE_ROWS
\bottomrule\end{tabular}
\caption{All twenty native BLEU-4 scores on the same four local generated-motion sets, PHOENIX TEST, real-duration conditioning.}\end{table*}

\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{llll}\toprule
Pair (first minus second)& Raw gap [95\% CI] & Canonical gap [95\% CI] & Gap change [95\% CI]\\\midrule
PAIR_ROWS
\bottomrule\end{tabular}
\caption{Paired bootstrap, 2,000 resamples, seed 42. All six point signs flip, but no pair has opposite rankings supported by both confidence intervals. These are exploratory comparisons without multiplicity correction or training-seed uncertainty.}\end{table*}

\section{Defense and linguistic accuracy}
Canonicalization yields identical hypotheses for all 1,024 source/attack pairs (512 preceding PHOENIX/How2Sign BT, 256 CSL BT, 256 local CTC). It also preserves canonical predictions in all 7,704 generated fixed-gauge pairs. This tests the stated equivalence channel, not robustness to arbitrary perturbation or improved translation.

Whole real-test CSL canonicalization changes 464 of 1,162 hypotheses. BLEU changes by CSL_DELTA, CI CSL_CI. For How2Sign, the full split changes 1,438 of 2,183 hypotheses, with aggregate BLEU change $+0.0062$. The registered 128-clip subset instead loses $0.7240$ BLEU. Post-result resampling by official VIDEO\_ID gives intervals $[-0.1717,0.1932]$ over all 149 video groups and $[-1.7700,-0.0092]$ over the 88 subset groups. VIDEO\_ID is not a verified signer identifier. The original shifted-target assignment stays fixed during cluster resampling.

An invariant interface may change a reader's accuracy because the training distribution was not canonical. A consistently wrong reader would satisfy the equivalence criterion. Our interpretation therefore rests on two separate statements: channel closure is justified algebraically and observed in these pairs; linguistic quality is neither certified nor generally improved.

\section{Native baseline and external validity}
Full native real-test BLEU-4 is 14.9995004460 for PHOENIX, 3.4778651050 for How2Sign, and 6.8907036392 for CSL. The card lists 15.00, 3.53 and 6.98. Only PHOENIX agrees after rounding. The latter two discrepancies remain unresolved. Native-loader decoding and prepared-baseline replay give identical hypotheses, excluding our left-stream preparation as their cause in the tested environment. We do not infer a cause, select alternate decoding settings on TEST, or label these exact model-card reproductions.

The separately released SLRTP2025 evaluator takes 178 XYZ joints (534 dimensions), not 240 rotation coordinates. The provided development archive contains 515 actual clips. We reproduce its real-motion BLEU-4 exactly at 13.378651856913775 with the published checkpoint and the native frame-rate convention. Its Progressive Transformer artifact has zero matched four-grams, so BLEU-4 is zero although lower-order matches are present (1,161/88/8). Predictions are nonempty. This is a native contract check, not a SOTA generator replication or the same feature-equivalence vulnerability.

Primary-source inspection found a related 360-dimensional rotation/face BT in M$^3$T. The official repository explicitly marks code and weights as unreleased at the audit date. Bounded repository/model-hub searches did not supply another validated compatible independently published reader. Three SignSparK-BT language-specific weights share one architecture family; the local CTC differs in architecture but is not an independently published translation system. Human body models required for a validated mesh/XYZ conversion were not available, and no licensed assets were bypassed.

\section{Artifacts and verification}
The artifact bundle preserves two immutable stages: the previous TEST/generation report and the additional CSL/CTC/paper report. Their JSON files expose references, targets, predictions and metric summaries; NPZ files expose source/attacked features and gauge coefficients. SHA256 manifests record the files, inputs and pinned sources. The extension's independent verifier recomputes WER through a full edit-distance matrix, independently from the producer's rolling-row implementation; native BLEU is recomputed from n-gram sufficient statistics. Every new source is compared with the native prepared LMDB data, and every attacked rotation is checked against both NumPy and the official converter in double precision. The prior frozen stages are fully hash-checked.

The included utility applies canonicalization, admissible equivalence transforms and hypothesis agreement. It rejects degenerate inputs instead of silently treating them as rotations. Environment records identify the installed tokenizer amendment, CUDA/PyTorch versions and checkpoint hashes. Exact floating-point reproduction can depend on hardware and library versions; the invariance proof alone does not guarantee bitwise equality in all environments.

No human assessment, rendered mesh, independently published compatible BT attack, multi-seed generator experiment, new training in this extension, or conference submission was performed. The artifacts support a reproducible evaluation audit with these explicit scope limits.
\end{document}
'''
    text=text.replace('SCORE_ROWS','\n'.join(score_rows)).replace('PAIR_ROWS','\n'.join(pair_rows))
    d=full['canonical_accuracy_change'];text=text.replace('CSL_DELTA',f"${d['delta_bleu4']:+.4f}$").replace('CSL_CI',f"$[{d['ci95'][0]:.4f},{d['ci95'][1]:.4f}]$")
    (PAPER/'supplement.tex').write_text(text);print('SUPPLEMENT_WRITTEN',flush=True)
if __name__=='__main__':main()
