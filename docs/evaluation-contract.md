# Rotation representation contract

An evaluator of movement should assign the same result to different encodings
of the same represented movement. This property does not establish accuracy.
The mathematical statement assumes nondegenerate exact divisions. Native
normalization clamps must remain inactive; equivalence in floating point
requires separate native matrix/FK checks. Nonfinite computed norms are rejected.

1. State the native converter, its row/column convention, and valid input domain.
2. Undo the generator's normalization before handling rotation features.
3. Canonicalize the two consecutive 3-vectors by native Gram–Schmidt. HumanML3D
   uses channels 67:193; standard KIT-ML uses 64:184. The separate TMR KIT AMASS
   representation has 263 channels and is not the native KIT 251 representation.
4. Apply the evaluator's own normalization, mask padding, then evaluate.
5. Use that same input contract for fitting an evaluator and at inference.
6. Report raw/canonical results on real and generated data with paired draws.

[`tools/canonicalize_motion.py`](../tools/canonicalize_motion.py) is a portable
NumPy implementation. Pass valid lengths for padded data. Undefined rotations
(zero or collinear vectors) raise an error, with a relative 1e-12 collinearity
guard for numerical stability; silently choosing a rotation would
invalidate the physical equivalence guarantee. Protected feature channels and
padded frames are copied bitwise. Gradients require an analogous native Torch
implementation; this utility is for evaluation/export.

The guarantee is limited to the continuous 6D rotation fiber. It does not cover
changes in positions, velocities, timing, facial expression or semantic content.
Native geometry tests and accuracy measurements remain necessary.
