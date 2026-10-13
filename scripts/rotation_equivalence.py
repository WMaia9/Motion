"""Rotation-equivalence utilities; do not claim linguistic or privacy guarantees."""
import numpy as np
import torch


def canonicalize_6d(features):
    """Project each consecutive 6D block; reject invalid frames/joints."""
    if not features.shape or features.shape[-1]%6:raise ValueError('last dimension must be a multiple of 6')
    if isinstance(features,torch.Tensor):
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):raise ValueError('finite floating-point features required')
    elif not np.issubdtype(features.dtype,np.floating) or not np.isfinite(features).all():raise ValueError('finite floating-point features required')
    shape=features.shape;d=features.reshape(*shape[:-1],-1,6)
    if isinstance(features,torch.Tensor):
        a=d[...,:3];b=d[...,3:];an=torch.linalg.vector_norm(a,dim=-1,keepdim=True)
        if bool((an<=1e-12).any()):raise ValueError('zero first rotation vector')
        u=a/an;v=b-(u*b).sum(-1,keepdim=True)*u;vn=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
        if bool((vn<=1e-12).any()):raise ValueError('collinear rotation vectors')
        return torch.cat((u,v/vn),-1).reshape(shape)
    d=np.asarray(d,dtype=np.float64);a=d[...,:3];b=d[...,3:];an=np.linalg.norm(a,axis=-1,keepdims=True)
    if (an<=1e-12).any():raise ValueError('zero first rotation vector')
    u=a/an;v=b-(u*b).sum(-1,keepdims=True)*u;vn=np.linalg.norm(v,axis=-1,keepdims=True)
    if (vn<=1e-12).any():raise ValueError('collinear rotation vectors')
    return np.concatenate((u,v/vn),-1).reshape(shape).astype(features.dtype)


def gauge_transform(features,alpha,beta,gamma):
    """Positive scale/shear on 6D blocks; accepts scalar or array coefficients."""
    if not features.shape or features.shape[-1]%6:raise ValueError('last dimension must be a multiple of 6')
    if isinstance(features,torch.Tensor):
        alpha,beta,gamma=[torch.as_tensor(c,dtype=features.dtype,device=features.device) for c in (alpha,beta,gamma)]
        if not all(bool(torch.isfinite(c).all()) for c in [alpha,beta,gamma]):raise ValueError('finite coefficients required')
        if bool((alpha<=0).any()) or bool((beta<=0).any()):raise ValueError('scales must be positive')
        concat=torch.cat
    else:
        alpha,beta,gamma=[np.asarray(c) for c in (alpha,beta,gamma)]
        if not all(np.isfinite(c).all() for c in [alpha,beta,gamma]):raise ValueError('finite coefficients required')
        if np.any(alpha<=0) or np.any(beta<=0):raise ValueError('scales must be positive')
        concat=np.concatenate
    shape=features.shape;d=features.reshape(*shape[:-1],-1,6);a=d[...,:3]
    return concat((alpha*a,beta*d[...,3:]+gamma*a),-1).reshape(shape)


def equivalence_hypothesis_agreement(before,after):
    if len(before)!=len(after) or not before:raise ValueError('nonempty equal-length hypotheses required')
    return sum(a==b for a,b in zip(before,after))/len(before)
