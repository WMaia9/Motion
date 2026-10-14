"""canon6d: score motion by its rotations, not by how they are written.

The continuous 6D rotation representation stores a rotation as two 3D vectors
(a, b). Positive rescalings of either vector and adding a multiple of `a` to `b`
leave the rotation unchanged. Learned motion evaluators that read the raw
vectors can respond to these redundant coordinates. Canonicalizing before an
evaluator's normalization maps every equivalent encoding to a single input.

    >>> from canon6d import canonicalize
    >>> x = canonicalize(raw_features, lengths)   # HumanML3D (263) or KIT-ML (251)
"""
from .core import (canonicalize, canonicalize6d, recode, rotation_block,
                   rotation_matrix)

__all__ = ['canonicalize', 'canonicalize6d', 'recode', 'rotation_block', 'rotation_matrix']
__version__ = '1.0.0'
