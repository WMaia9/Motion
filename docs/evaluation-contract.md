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

[`canon6d`](../canon6d/) implements steps 2–4: `canon6d.canonicalize(raw, lengths)` for
evaluation (NumPy), and `canon6d.torch.canonicalize` when the evaluator is used as a
differentiable reward. Pass valid lengths for padded data. Undefined rotations (zero
or collinear vectors) raise an error, with a relative 1e-12 collinearity guard;
silently choosing a rotation would break the equivalence guarantee. Non-rotation
channels and padded frames are copied bit for bit. The original evaluation code used
in the paper is [`tools/canonicalize_motion.py`](../tools/canonicalize_motion.py); the
tests check that both give identical outputs.

The guarantee is limited to the continuous 6D rotation fiber. It does not cover
changes in positions, velocities, timing, facial expression or semantic content.
Native geometry tests and accuracy measurements remain necessary.
