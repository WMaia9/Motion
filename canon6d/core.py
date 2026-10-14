"""NumPy implementation of 6D canonicalization for motion features."""
from __future__ import annotations

import numpy as np

#: Feature widths with a known 6D rotation block: HumanML3D (263) and native KIT-ML (251).
SUPPORTED_WIDTHS = (263, 251)


def rotation_block(width: int) -> slice:
    """Channels holding the 6D rotations in HumanML3D/KIT-ML feature vectors.

    The layout is root (4) | positions (3J) | rotations (6J) | velocities (3J+3) | contacts (4),
    with J = 21 joints for HumanML3D (channels 67:193) and J = 20 for KIT-ML (64:184).
    """
    if width not in SUPPORTED_WIDTHS:
        raise ValueError(f'unsupported feature width {width}; expected one of {SUPPORTED_WIDTHS}')
    joints = (width - 11) // 12
    start = 4 + 3 * joints
    return slice(start, start + 6 * joints)


def canonicalize6d(d, valid=None, eps: float = 1e-12):
    """Map each 6D vector [a, b] to its canonical form [u, v].

    u = a / |a| and v is the unit component of b orthogonal to u. The represented
    rotation is unchanged; every equivalent encoding maps to the same output.

    Args:
        d: floating array of shape [..., 6].
        valid: optional boolean mask broadcastable to d.shape[:-1]; entries outside
            the mask (e.g. padding) are copied unchanged.
        eps: relative collinearity tolerance.

    Raises:
        ValueError: on non-finite input, a zero first vector, or (near-)collinear vectors,
            for which no rotation is defined.
    """
    d = np.asarray(d)
    if d.shape[-1] != 6 or not np.issubdtype(d.dtype, np.floating):
        raise ValueError('expected a floating array with last dimension 6')
    mask = np.ones(d.shape[:-1], bool) if valid is None else np.broadcast_to(valid, d.shape[:-1])
    x = d[mask].astype(np.float64)
    a, b = x[:, :3], x[:, 3:]
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or not np.isfinite(na).all() or np.any(na == 0):
        raise ValueError('non-finite input or zero first vector')
    u = a / na
    w = b - (u * b).sum(-1, keepdims=True) * u
    nw = np.linalg.norm(w, axis=-1, keepdims=True)
    if not np.isfinite(nw).all():
        raise ValueError('norm overflow')
    if np.any(nw <= eps * np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-300)):
        raise ValueError('collinear rotation vectors')
    y = d.copy()
    y[mask] = np.concatenate([u, w / nw], axis=-1).astype(d.dtype)
    return y


def canonicalize(features, lengths=None):
    """Canonicalize the rotation channels of HumanML3D (263) or KIT-ML (251) features.

    Pass raw, unnormalized features, then apply the evaluator's own normalization.
    All non-rotation channels and padded frames are preserved bit for bit.

    Args:
        features: array [N, T, D] (or [T, D] without `lengths`).
        lengths: optional integer array [N] of valid frame counts.
    """
    x = np.asarray(features)
    block = rotation_block(x.shape[-1])
    valid = None
    if lengths is not None:
        lengths = np.asarray(lengths)
        if (x.ndim != 3 or lengths.shape != (len(x),) or not np.issubdtype(lengths.dtype, np.integer)
                or np.any(lengths < 0) or np.any(lengths > x.shape[1])):
            raise ValueError('expected integer lengths [N] within [0, T]')
        valid = (np.arange(x.shape[1])[None, :] < lengths[:, None])[..., None]
    y = x.copy()
    d = x[..., block].reshape(*x.shape[:-1], -1, 6)
    y[..., block] = canonicalize6d(d, valid).reshape(*x.shape[:-1], -1)
    return y


def recode(d, log_alpha, log_beta, eta):
    """Rewrite 6D vectors without changing their rotation: (a, b) -> (e^log_alpha a, e^log_beta b + eta a).

    Useful for auditing an evaluator: compare its scores on `d` and `recode(d, ...)`.
    Coefficients broadcast against d.shape[:-1].
    """
    d = np.asarray(d)
    a, b = d[..., :3], d[..., 3:]
    la, lb, e = (np.asarray(c)[..., None] for c in (log_alpha, log_beta, eta))
    return np.concatenate([np.exp(la) * a, np.exp(lb) * b + e * a], axis=-1).astype(d.dtype)


def rotation_matrix(d):
    """Rotation matrices [..., 3, 3] with the canonical vectors as the first two columns."""
    c = canonicalize6d(np.asarray(d, dtype=np.float64))
    u, v = c[..., :3], c[..., 3:]
    return np.stack([u, v, np.cross(u, v)], axis=-1)
