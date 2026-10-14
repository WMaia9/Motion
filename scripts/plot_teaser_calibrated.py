"""Teaser: one movement, one joint written two ways, and the evaluator's score.

Uses the first fixed HumanML3D test clip (same clip as before, not selected on
results) and the caption-targeted attack against the T2M-GPT-distributed T2M
checkpoint (reports/cvpr_teaser_audit_20261014/calibrated_attack).  Body frames
are procedural renders of native RIC joint positions; the panel-b vectors are the
actual saved 6D vectors of one joint, drawn exactly in the plane they span.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_generalization_teaser import render  # noqa: E402

OLD = Path('reports/cvpr_humanml3d_audit_20261012')
CAL = Path('reports/cvpr_teaser_audit_20261014/calibrated_attack')
OUT = Path('paper/current/figures/teaser.pdf')
B, O, G, GR = '#2a6fb0', '#c8642b', '#3b7d4f', '#5a6573'


def coords(vec, u, v):
    return np.array([vec @ u, vec @ v])


def main():
    sys.path.insert(0, str(Path('external/MDM').resolve()))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
    item = json.loads((OLD / 'test_selection.json').read_text())[0]
    n = item['frames']
    with np.load(OLD / 'test_mdm.npz') as z:
        X = z['raw'][0, :n].astype(np.float64)
    with np.load(CAL / 'mdm_target_caption.npz') as z:
        Y = z['raw'][0, :n].astype(np.float64)
    pos0 = recover_from_ric(torch.tensor(X, dtype=torch.float32), 22).numpy()
    pos1 = recover_from_ric(torch.tensor(Y, dtype=torch.float32), 22).numpy()
    assert np.array_equal(pos0, pos1)
    frames = [int(q * (n - 1)) for q in (.2, .5, .8)]
    imgs = [render(pos0[t], t2m_kinematic_chain) for t in frames]

    # joint/frame with the largest raw change at the middle frame (for visibility)
    t = frames[1]
    d0 = X[t, 67:193].reshape(21, 6); d1 = Y[t, 67:193].reshape(21, 6)
    j = int(np.argmax(np.abs(d1 - d0).max(1)))
    a0, b0, a1, b1 = d0[j, :3], d0[j, 3:], d1[j, :3], d1[j, 3:]
    u = a0 / np.linalg.norm(a0); v = b0 - (u @ b0) * u; v /= np.linalg.norm(v)
    u1 = a1 / np.linalg.norm(a1); v1 = b1 - (u1 @ b1) * u1; v1 /= np.linalg.norm(v1)
    assert np.abs(u - u1).max() < 1e-5 and np.abs(v - v1).max() < 1e-5

    res = json.loads((CAL / 'results.json').read_text())['mdm']
    r_orig, r_att, r_can = res['identity']['actual_R1'], res['target_caption']['actual_R1'], res['target_caption']['canonical_actual_R1']

    plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'Liberation Sans', 'font.size': 9})
    fig = plt.figure(figsize=(7.0, 2.25))
    # (a) the movement
    fig.text(0.01, 0.97, 'a  One movement', fontsize=10, fontweight='bold', va='top')
    fig.text(0.01, 0.88, '“' + item['caption'].rstrip('.') + '”', fontsize=7.5, color=GR, va='top', style='italic')
    for k, im in enumerate(imgs):
        ax = fig.add_axes([0.005 + k * 0.105, 0.12, 0.105, 0.70]); ax.imshow(im); ax.set_axis_off()
    fig.text(0.165, 0.05, 'identical joint positions and rotations\nin both versions', fontsize=7, color=G, ha='center', va='bottom')
    # (b) one joint written two ways
    fig.text(0.345, 0.97, 'b  One joint, written two ways', fontsize=10, fontweight='bold', va='top')
    lim = max(np.abs(np.r_[coords(a1, u, v), coords(b1, u, v), coords(a0, u, v), coords(b0, u, v)]).max(), 1.0) * 1.15
    for k, (a, b, col, title) in enumerate([(a0, b0, B, 'original numbers'), (a1, b1, O, 'rewritten numbers')]):
        ax = fig.add_axes([0.345 + k * 0.155, 0.17, 0.145, 0.62])
        for vec, lab in [(u, ''), (v, '')]:
            c = coords(vec, u, v); ax.annotate('', xy=c, xytext=(0, 0), arrowprops=dict(arrowstyle='-', color=G, lw=1, ls=(0, (2, 2))))
        for vec, lab in [(a, '$a$'), (b, '$b$')]:
            c = coords(vec, u, v)
            ax.annotate('', xy=c, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8, mutation_scale=9))
            ax.text(*(c * 1.08), lab, color=col, fontsize=9, ha='center', va='center')
        ax.set_xlim(-0.25 * lim, lim); ax.set_ylim(-0.25 * lim, lim); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#c9ced4')
        ax.set_title(title, fontsize=7.5, color=col, pad=2)
    fig.text(0.5, 0.05, 'six different numbers, same rotation (dotted axes)', fontsize=7, color=G, ha='center', va='bottom')
    # (c) the score
    fig.text(0.68, 0.97, 'c  Evaluator score', fontsize=10, fontweight='bold', va='top')
    ax = fig.add_axes([0.73, 0.2, 0.26, 0.6])
    vals = [r_orig, r_att, r_can]
    bars = ax.bar([0, 1, 2], vals, color=[B, O, G], width=0.62)
    for x, val in enumerate(vals):
        ax.text(x, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['original', 'rewritten\n(attack)', 'rewritten\n+ our fix'], fontsize=7)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, .5, 1]); ax.tick_params(axis='y', labelsize=7)
    ax.set_ylabel('R-Precision (top-1)', fontsize=7.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT)
    meta = {'clip_id': item['clip_id'], 'caption': item['caption'], 'frames': frames, 'joint_index_in_rotation_block': j,
            'frame_for_vectors': t, 'R1': {'original': r_orig, 'rewritten': r_att, 'rewritten_canonicalized': r_can},
            'positions_bitwise_equal': True, 'canonical_axes_max_diff': float(max(np.abs(u - u1).max(), np.abs(v - v1).max())),
            'vectors_original': [a0.tolist(), b0.tolist()], 'vectors_rewritten': [a1.tolist(), b1.tolist()]}
    (CAL.parent / 'teaser_evidence.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta['R1']), 'joint', j, 'frame', t)


if __name__ == '__main__':
    main()
