"""Canonicalize unnormalized continuous6D motion without changing rotations.

The layout is [first3-vector,second3-vector]; the converter may use those as
rows or columns. Interleaved matrix storage must be reordered by the caller.
Inputs must be finite and nondegenerate on valid frames; a relative 1e-12
collinearity guard rejects numerically unstable vectors. Padded frames are
copied unchanged. This closes only the6D encoding fiber, not physical attacks.
"""
import numpy as np

def canonicalize6d(d, valid=None):
    """Return orthonormal 6D representatives; reject invalid/near-collinear inputs."""
    d=np.asarray(d)
    if d.shape[-1]!=6 or not np.issubdtype(d.dtype,np.floating):
        raise ValueError('Expected floating [...,6] vectors')
    mask=np.ones(d.shape[:-1],bool) if valid is None else np.broadcast_to(valid,d.shape[:-1])
    x=d[mask].astype(np.float64);a,b=x[:,:3],x[:,3:]
    n=np.linalg.norm(a,axis=-1,keepdims=True)
    if not np.isfinite(x).all() or not np.isfinite(n).all() or np.any(n==0):raise ValueError('Invalid first-vector norm or nonfinite input')
    u=a/n;v=b-(u*b).sum(-1,keepdims=True)*u
    n=np.linalg.norm(v,axis=-1,keepdims=True)
    if not np.isfinite(n).all():raise ValueError('Orthogonal-vector norm overflow')
    # Relative degeneracy guard avoids pretending to define a rotation at
    # collinearity; it does not impose a canonical-only input domain.
    if np.any(n<=1e-12*np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-300)):
        raise ValueError('Collinear rotation vectors')
    y=d.copy();y[mask]=np.concatenate([u,v/n],axis=-1).astype(d.dtype);return y

def canonicalize_motion(features, lengths=None):
    """HumanML3D263/KIT251: canonicalize rotations before evaluator normalization.

    Features have shape[N,T,D] (or[T,D] without lengths). Lengths are[N].
    All other channels, including contacts and padding, are bitwise preserved.
    """
    x=np.asarray(features);width=x.shape[-1]
    if width not in (263,251):raise ValueError('Expected HumanML3D263 or nativeKIT251')
    joints=(width-11)//12;start=4+3*joints;stop=start+6*joints
    valid=None
    if lengths is not None:
        lengths=np.asarray(lengths)
        if x.ndim!=3 or lengths.shape!=(len(x),) or not np.issubdtype(lengths.dtype,np.integer) or np.any(lengths<0) or np.any(lengths>x.shape[1]):
            raise ValueError('Expected integer valid lengths[N] within[0,T]')
        valid=(np.arange(x.shape[1])[None,:]<lengths[:,None])[...,None]
    y=x.copy();d=x[...,start:stop].reshape(*x.shape[:-1],joints,6)
    y[...,start:stop]=canonicalize6d(d,valid).reshape(*x.shape[:-1],6*joints)
    return y
