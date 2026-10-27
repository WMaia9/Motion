# Storage cleanup — 22 October 2026

The Rotation workspace remains at `/mnt/datasets/PHOENIX-2014-T`.
The manuscript and delivered PDFs remain in `paper/current/` and
`paper/current/reviewed/`; no manuscript revision was made by this cleanup.

## Measured disk usage

- Before: **465.50 GiB** allocated.
- After: **266.39 GiB** allocated.
- Freed: **199.11 GiB**.

These are physical allocations measured with `du -s -B1`, not the sum of
logical array sizes. Directory metadata and cleanup receipts account for the
small difference between measured savings and the per-file totals below.

| Operation | Reclaimed GiB |
|---|---:|
| Share 306 identical frozen copies with hardlinks | 110.56 |
| Remove obsolete RGB train/dev frames and legacy keypoints | 38.69 |
| Remove 13 ZIP downloads with byte-identical extracted payloads | 6.79 |
| Remove the HumanML3D RAR after verifying its full extracted payload | 4.00 |
| Release zero-only padding blocks in 313 saved arrays | 39.04 |

## Preserved

- Current manuscript, PDFs, source ZIP, code and curated evidence exports.
- Every file in the 65 frozen report directories and their original manifests.
- Native HumanML3D/KIT-ML motion, native sign-language rotation features,
  fitted checkpoints, public generator/evaluator weights and Python environments.
- All PHOENIX RGB TEST frames, including those used by the qualitative figure.

Only the obsolete `features/fullFrame-210x260px/train/`,
`features/fullFrame-210x260px/dev/` and `keypoints/` extraction inputs were
removed. The current Rotation pipeline uses the native motion features instead.
All useful ZIP members were compared byte-for-byte with their extracted files.
The RAR's complete uncompressed stream matched the extracted files by SHA256.

## Verification

All 760 previously tracked files were checked against a pre-cleanup snapshot;
only the intended storage documentation was subsequently updated. In particular,
the user's existing edit to `paper/current/main.tex` was preserved byte-for-byte.
All 8 `make test` library tests passed. Deduplicated files were verified by SHA256
before linking; sparsified arrays were verified before and after, with unchanged
logical sizes. Every report path remained available.

The complete integrity scan checked **15,683 report paths**.
**15,682 match their frozen fingerprints**. One pre-existing mismatch remains:

- `reports/cvpr_teaser_audit_20261014/teaser_evidence.json`: modified on
  16 October, after its 14 October manifest and before this cleanup. The current
  file was retained. Its frozen version remains in
  `results/cvpr_teaser_audit_20261014/teaser_evidence.json`. Neither version nor
  the original manifest was rewritten to conceal the discrepancy.

## Copying and maintenance

Frozen hardlinked files must never be edited in place. Write new experiments to
new directories, as required by the existing reproduction instructions.
Use `rsync -aHS` when copying the full workspace to preserve hardlinks and sparse
allocation; otherwise copies can consume substantially more disk space.

Detailed local receipts, removed-file inventories, scripts and verification
logs are in `maintenance/cleanup_20261022/`. The overall machine-readable receipt
is `cleanup_summary.json`; these operational files stay outside Git.
