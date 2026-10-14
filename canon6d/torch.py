"""Differentiable PyTorch canonicalization, e.g. for learned rewards during training.

Canonicalizing before a learned reward means its gradients cannot push a generator
along directions that change no rotation.
"""
import torch

from .core import rotation_block


def canonicalize6d(d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """[..., 6] -> canonical [u, v]; differentiable, no validity checks."""
    a, b = d[..., :3], d[..., 3:]
    u = torch.nn.functional.normalize(a, dim=-1, eps=eps)
    v = torch.nn.functional.normalize(b - (u * b).sum(-1, keepdim=True) * u, dim=-1, eps=eps)
    return torch.cat([u, v], dim=-1)


def canonicalize(features: torch.Tensor) -> torch.Tensor:
    """Canonicalize the rotation channels of raw HumanML3D/KIT-ML features [..., D]."""
    block = rotation_block(features.shape[-1])
    rot = features[..., block].reshape(*features.shape[:-1], -1, 6)
    out = features.clone()
    out[..., block] = canonicalize6d(rot).reshape(*features.shape[:-1], -1)
    return out
