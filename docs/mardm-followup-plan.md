# Controlled follow-up to MARDM

This finite follow-up tests rotation-encoding consistency and positions the paper
against [MARDM](https://arxiv.org/abs/2411.16575v2). It adds no generator training,
reward tuning, generation, or model-family seeds. The working manuscript remains
in [paper/current](../paper/current/README.md).

## Matched block interventions

Use all twenty saved real HumanML3D cohorts, each with 4,640 eligible native
entries. Preserve captions, candidate pools, crop lengths and masks. For each
clip with L valid frames, impose

`norm((output - input) / native_reader_std) = radius * sqrt(L)`.

Register radii 1 and 3 before scoring. Match total norm per clip, not per-coordinate
amplitude. Each reader uses its own fixed training normalization, so cross-reader
numerical budget scales are not interchangeable. Compare:

- Exact rotation equivalence: both fixed directions and an independent Gaussian
  coefficient field, with positive exponential scales and shear.
- Actual local SO(3) rotation changes, with other feature blocks fixed.
- Local velocities (66), contacts (4) and joint positions (63); root features fixed.

Guo consumes 259 features and excludes contacts. The contact intervention is a
negative control with zero consumed Guo budget, despite its matched pre-selection
263-feature norm. TMR consumes all 263 features. Continuous contact noise can leave
the binary domain. Non-null controls can be inconsistent with unchanged blocks;
they measure input sensitivity and are distinct from coherent angular calibration.

Report all fourteen conditions per reader. Use paired t19 intervals with 28
endpoints per reader: FID/R1 for Guo, R1/matched cosine for TMR. Paired-reference FID
is zero by construction and is not a generation-quality score. Canonical closure
is a numerical diagnostic, not another selected significance endpoint.

## Natural mechanism check

Reuse frozen signed FID gaps and valid-frame raw-to-canonical rotation RMS from
all eight releases. Show four diffusion/consistency releases, two VQ releases,
the original T2M VAE, and prefix-conditioned DiP separately. Model settings and
training regimes are confounded; family means are descriptive. Correlations use
eight release means, exact permutation tests, and adjustment for signed and
absolute gaps. Twenty draws are not twenty independent model-family replicas.

Canonicalization tests within-rotation noncanonicality. It cannot refute MARDM's
full mechanism involving velocities, contacts, representation distribution and
diffusion training. A nonsignificant RMS correlation is not proof of independence.

## Essential-reader comparison

Use the author-linked MARDM GRU and CLIP essential-feature checkpoints with strict
state loading, native layers and their 67-feature normalization. Both ignore the
rotation block by construction. Confirm this with repeated native forward passes.
Compare within-reader rankings of the same eight saved releases over twenty
draws, using the same real cohort as reference for all readers. Recompute Guo
scores with this common reference; do not silently reuse the old independently
drawn reference scores. CLIP latent FID is descriptive, not a leaderboard metric.

The native constructor's unused CLIP visual-tower download is replaced only
while constructing token/position embedding shapes. The public checkpoint then
strictly overwrites every parameter; all released contrast-model state tensors
are checked bitwise before inference. No forward layer or fitted weight changes.

## Positioning and presentation

MARDM already diagnoses sensitivity to redundant feature blocks. Its additive
and replacement noise changes represented rotations and is not an equal-L2
exact-equivalence experiment. Our narrower contribution preserves both positions
and rotations and provides a defense retaining rotational information. For the
rotation-only sign-language features in this study, removing rotations removes
the entire movement input. Neither invariance nor changed rankings proves better
linguistic or human-perceived motion evaluation.

Expand targeted TMR controls to the full fixed 1,024-clip cohort, preserving every
target and generic-L2 control. Label blind transfer as transfer between checkpoints
of the same architecture. Downgrade unchanged accuracy on already canonical real
motion to an implementation sanity check. Preserve the separate matched adaptation
experiment on noncanonical sign-language inputs.

## Completion and external process

Require independent score/geometry checks, source and checkpoint fingerprints,
all twenty draws, final PDF layout and visual review, and immutable archives.
Keep every outcome in the main paper or supplement. Once this list is complete,
freeze scientific scope. Authors, registration and two human cold reviewers are
deferred to the user; this workflow neither submits nor contacts reviewers.

The [official 2027 dates](https://cvpr.thecvf.com/Conferences/2027/Dates) list
November 10 registration, November 16 paper and November 23 supplement (AoE).
The 2027 AuthorGuidelines endpoint returned 404 on October 13; the unmodified
available official 2026 kit remains provisional until final 2027 guidance appears.
