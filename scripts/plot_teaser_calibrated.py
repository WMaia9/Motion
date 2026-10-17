"""Teaser: one movement, and how one of its joint rotations is read from two different sets of six numbers.

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


JOINTS = ['pelvis', 'left hip', 'right hip', 'spine', 'left knee', 'right knee', 'spine', 'left ankle',
          'right ankle', 'chest', 'left foot', 'right foot', 'neck', 'left collar', 'right collar', 'head',
          'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist']
SHOWN = [4, 5, 18, 19]  # knees and elbows: joints a reader can see in the render


def coords(vec, u, v):
    return np.array([vec @ u, vec @ v])


def render_marked(positions, chains, mark):
    """render() from plot_generalization_teaser with one joint drawn as an orange sphere on top."""
    from plot_generalization_teaser import capsule, sphere
    xyz = positions[:, [0, 2, 1]].copy(); xyz[:, :2] -= xyz[0, :2]; color = '#4d7897'
    fig = plt.figure(figsize=(2.2, 2.8), dpi=150); ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.computed_zorder = False
    for chain in chains:
        for parent, joint in zip(chain[:-1], chain[1:]):
            r = .068 if joint in [1, 2, 4, 5] else .047 if joint in [7, 8, 10, 11] else .042 if joint in [16, 17, 18, 19, 20, 21] else .075
            capsule(ax, xyz[parent], xyz[joint], r, color)
    capsule(ax, xyz[0], xyz[9], .105, color); sphere(ax, xyz[15], .105, color)
    a, b = np.meshgrid(np.linspace(-.65, .65, 2), np.linspace(-.65, .65, 2))
    ax.plot_surface(a, b, np.full_like(a, float(xyz[:, 2].min()) - .035), color='#e5e9ec', alpha=.6, shade=False, zorder=-1)
    if mark is not None:
        sphere(ax, xyz[mark], .085, O)
    ax.set(xlim=(-.65, .65), ylim=(-.65, .65), zlim=(-.1, 2.)); ax.view_init(elev=12, azim=-58); ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1.62)); fig.canvas.draw(); pixels = np.asarray(fig.canvas.buffer_rgba()).copy(); plt.close(fig)
    visible = np.any(pixels[:, :, :3] < 245, axis=-1); rows, cols = np.where(visible); m = 8
    return pixels[max(0, rows.min() - m):rows.max() + m + 1, max(0, cols.min() - m):cols.max() + m + 1].copy()


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
    t = frames[1]

    # among visible joints, the one whose six numbers change most at the middle frame
    d0 = X[t, 67:193].reshape(21, 6); d1 = Y[t, 67:193].reshape(21, 6)
    joint = max(SHOWN, key=lambda q: np.abs(d1[q - 1] - d0[q - 1]).max()); j = joint - 1
    a0, b0, a1, b1 = d0[j, :3], d0[j, 3:], d1[j, :3], d1[j, 3:]
    u = a0 / np.linalg.norm(a0); v = b0 - (u @ b0) * u; v /= np.linalg.norm(v)
    u1 = a1 / np.linalg.norm(a1); v1 = b1 - (u1 @ b1) * u1; v1 /= np.linalg.norm(v1)
    assert np.abs(u - u1).max() < 1e-5 and np.abs(v - v1).max() < 1e-5
    imgs = [render_marked(pos0[f], t2m_kinematic_chain, joint if f == t else None) for f in frames]
    res = json.loads((CAL / 'results.json').read_text())['mdm']
    r_orig, r_att, r_can = res['identity']['actual_R1'], res['target_caption']['actual_R1'], res['target_caption']['canonical_actual_R1']

    plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'Liberation Sans', 'font.size': 9})
    fig = plt.figure(figsize=(7.0, 2.2))
    fig.text(0.01, 0.97, 'a  One movement', fontsize=10, fontweight='bold', va='top')
    fig.text(0.01, 0.875, '“' + item['caption'].rstrip('.') + '”', fontsize=7.5, color=GR, va='top', style='italic')
    for k, im in enumerate(imgs):
        ax = fig.add_axes([0.01 + k * 0.125, 0.13, 0.125, 0.69]); ax.imshow(im); ax.set_axis_off()
    fig.text(0.20, 0.035, f'orange: the {JOINTS[joint]}, shown in panel b', fontsize=7, color=O, ha='center', va='bottom')

    fig.text(0.42, 0.97, f'b  How the {JOINTS[joint]} rotation is read from six numbers', fontsize=10, fontweight='bold', va='top')
    pts = [coords(x, u, v) for x in (a0, b0, a1, b1)]
    lo = min(min(p.min() for p in pts), 0) - 0.22; hi = max(max(p.max() for p in pts), 1) + 0.25
    for k, (a, b, col, title, la, lb) in enumerate([(a0, b0, B, 'stored numbers', '$a$', '$b$'),
                                                    (a1, b1, O, 'rewritten numbers', "$a'$", "$b'$")]):
        ax = fig.add_axes([0.43 + k * 0.29, 0.15, 0.27, 0.66])
        ca, cb = coords(a, u, v), coords(b, u, v)
        if k == 1:  # the part of b' along a', which the conversion removes
            ax.plot([cb[0], 0], [cb[1], cb[1]], color='#7d8790', lw=0.9, ls=(0, (1.5, 1.5)))
        for vec, lab, off in [((1, 0), '$u$', (0.0, -0.14)), ((0, 1), '$v$', (-0.14, 0.0))]:
            ax.annotate('', xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle='-', color=G, lw=2.6, alpha=0.45))
            ax.text(vec[0] + off[0], vec[1] + off[1], lab, color=G, fontsize=8.5, ha='center', va='center')
        for c, lab, off in [(ca, la, (0.0, 0.11)), (cb, lb, (0.14, 0.04) if cb[0] >= 0 else (-0.15, -0.08))]:
            ax.annotate('', xy=c, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6, mutation_scale=9))
            ax.text(c[0] + off[0], c[1] + off[1], lab, color=col, fontsize=9.5, ha='center', va='center')
        sa, sb = np.linalg.norm(a), coords(b, u, v)
        note = ('unit length, at right angles:\n$u=a$, $v=b$' if k == 0 else
                f"$a'={sa / np.linalg.norm(a0):.2f}\\,a$\n$b'={sb[1]:.2f}\\,v {'-' if sb[0] < 0 else '+'} {abs(sb[0]):.2f}\\,u$")
        ax.text(0.97, 0.97, note, transform=ax.transAxes, fontsize=7, color=col, ha='right', va='top', linespacing=1.4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color('#c9ced4')
        ax.set_title(title, fontsize=7.5, color=col, pad=2)
    fig.text(0.715, 0.035, 'green: $u$ = direction of $a$; $v$ = part of $b$ at right angles to $a$. Same $u, v$: same rotation.', fontsize=7, color=G, ha='center', va='bottom')
    fig.savefig(OUT)
    meta = {'clip_id': item['clip_id'], 'caption': item['caption'], 'frames': frames, 'joint': JOINTS[joint],
            'joint_index_in_rotation_block': j, 'frame_for_vectors': t,
            'R1': {'original': r_orig, 'rewritten': r_att, 'rewritten_canonicalized': r_can},
            'positions_bitwise_equal': True, 'canonical_axes_max_diff': float(max(np.abs(u - u1).max(), np.abs(v - v1).max())),
            'vectors_original': [a0.tolist(), b0.tolist()], 'vectors_rewritten': [a1.tolist(), b1.tolist()]}
    (CAL.parent / 'teaser_evidence.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta['R1']), JOINTS[joint], 'frame', t, 'plane coords', [p.round(3).tolist() for p in pts])


if __name__ == '__main__':
    main()
