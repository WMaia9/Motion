"""Measure physical changes of the saved generic comparator, without new attacks."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from motion_release_audit_common import read, write, sha
from score_humanml_blind_fid import matrices

R = Path('reports/cvpr_tmr1024_controls_20261013')


def main():
    assert not (R / 'manifest_sha256.json').exists()
    assert read(R / 'verification.json')['passed']
    sys.path.insert(0, str(Path('external/MDM').resolve()))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    torch.set_num_threads(2)
    rows = {}
    for source in ['real', 'mdm']:
        p = R / source; z = np.load(p / 'source.npz'); a = np.load(p / 'generic_L2.npz'); X, Y, L = z['raw'], a['raw'], z['lengths']
        angles = []; displacement_sum = 0.; displacement_count = 0
        for start in range(0, len(X), 32):
            valid = np.arange(196)[None, :] < L[start:start + 32, None]
            x = X[start:start + 32, :, 67:193].reshape(-1, 196, 21, 6)[valid]
            y = Y[start:start + 32, :, 67:193].reshape(-1, 196, 21, 6)[valid]
            relative = matrices(x).swapaxes(-1, -2) @ matrices(y)
            angles.append(np.degrees(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()))
        for i in range(len(X)):
            px = recover_from_ric(torch.tensor(X[i, :L[i]]), 22).numpy()
            py = recover_from_ric(torch.tensor(Y[i, :L[i]]), 22).numpy()
            distance = np.linalg.norm(py - px, axis=-1)
            displacement_sum += float(distance.astype(float).sum()); displacement_count += distance.size
        angles = np.concatenate(angles)
        rows[source] = {'clips': len(X), 'mean_rotation_change_deg': float(angles.mean()), 'max_rotation_change_deg': float(angles.max()), 'mean_RIC_position_displacement_m': displacement_sum / displacement_count, 'means_weight_all_valid_joint_frames': True, 'physically_inconsistent_cross_feature_attack': True, 'saved_source_sha256': sha(p / 'source.npz'), 'saved_generic_sha256': sha(p / 'generic_L2.npz')}
        print('TMR1024_GENERIC_GEOMETRY', source, rows[source], flush=True)
    write(R / 'generic_geometry.json', {'passed': True, 'conditions': rows, 'no_new_attack_or_training': True, 'measurement_source_sha256': sha(__file__)})


if __name__ == '__main__':main()
