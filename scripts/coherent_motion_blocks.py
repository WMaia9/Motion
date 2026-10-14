"""Score-independent fixed directions for the equal-L2 block audit.

The same local unit axis (1,1,1)/sqrt(3) is fixed for every joint and valid frame.
Both signs are retained. Temporal coherence does not imply consistency between
feature blocks: only one block changes. These are input-sensitivity controls.
"""
import numpy as np
from .matched_motion_blocks import basis, BLOCKS

AXIS = np.ones(3, dtype=np.float64) / np.sqrt(3.)
CONDITIONS = [f'{block}_fixed_{sign}' for block in ['rotation','velocity','contact','position'] for sign in ['plus','minus']]


def intervene(raw, lengths, std, condition, radius):
    block, _, sign_name = condition.split('_')
    assert condition in CONDITIONS
    sign = 1. if sign_name == 'plus' else -1.
    x = raw.astype(np.float64)
    L = np.asarray(lengths)
    mask = np.arange(x.shape[1])[None, :] < L[:, None]
    scale = np.asarray(std, dtype=np.float64)
    assert scale.shape == (263,) and np.all(scale > 0)
    target = radius * np.sqrt(L)
    if block in BLOCKS:
        begin, end = BLOCKS[block]
        direction = np.ones(end-begin) if block == 'contact' else np.tile(AXIS, (end-begin)//3)
        direction *= sign
        # A constant direction in unnormalized coordinates, not std-scaled
        # Gaussian noise; scalar adapts only to the declared input norm.
        unit_norm = np.sqrt(L) * np.linalg.norm(direction / scale[begin:end])
        scalar = target / unit_norm
        y = x.copy()
        y[..., begin:end] += scalar[:,None,None] * direction
    else:
        begin,end=67,193
        R=basis(raw)
        a,b,c=sign*AXIS
        K=np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])
        K2=K@K
        def changed(angle):
            t=angle[:,None,None,None,None]
            Q=np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K2
            U=R@Q
            return np.concatenate([U[...,0],U[...,1]],axis=-1).reshape(*x.shape[:2],126)
        fields=[]
        for A in [R,R@K,R@K2]:
            fields.append(np.concatenate([A[...,0],A[...,1]],axis=-1).reshape(*x.shape[:2],126))
        fields[0]-=x[...,begin:end]
        D,A,B=[f/scale[begin:end]*mask[...,None] for f in fields]
        dd,aa,bb=[np.sum(f*f,axis=(1,2)) for f in [D,A,B]]
        da=np.sum(D*A,axis=(1,2));db=np.sum(D*B,axis=(1,2));ab=np.sum(A*B,axis=(1,2))
        def norm(angle):
            sn=np.sin(angle);cs=1-np.cos(angle)
            return np.sqrt(np.maximum(dd+aa*sn**2+bb*cs**2+2*da*sn+2*db*cs+2*ab*sn*cs,0))
        lo=np.zeros(len(x));hi=np.full(len(x),np.pi/2)
        assert np.all(norm(lo)<target) and np.all(norm(hi)>target), 'Budget unreachable in registered rotation range'
        for _ in range(28):
            mid=(lo+hi)/2;below=norm(mid)<target
            lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        scalar=(lo+hi)/2
        y=x.copy();y[...,begin:end]=changed(scalar)
    y=y.astype(raw.dtype)
    y[...,:begin]=raw[...,:begin];y[...,end:]=raw[...,end:];y[~mask]=raw[~mask]
    actual=np.linalg.norm((((y.astype(float)-x)/scale)*mask[...,None]).reshape(len(x),-1),axis=1)
    # A constant contact offset applied to binary float32 values has coarser
    # relative quantization than a field of random continuous offsets. Retain
    # that architectural control with an explicitly bounded input-norm error.
    tolerance=2e-5 if block=='contact' else 5e-6
    assert np.max(abs(actual-target)/target)<tolerance,(condition,float(np.max(abs(actual-target)/target)))
    assert np.isfinite(y).all()
    return y,scalar,actual
