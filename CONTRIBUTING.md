# Contributing

Run `make check` before committing. It requires only Python and Git. Keep generated
arrays, model weights, datasets, environments and compiler output outside Git.

## Scientific provenance

The existing producers in `scripts/` retain their original bytes and paths.
Registered experiments record source hashes; a formatter or refactor of these
files would create a different experimental source version. For a new experiment,
use a new output directory and record its code, configuration, inputs and seeds.
Never rerun a producer against a frozen result directory.

Protocols and full archive inventories in `results/` are read-only exports.
Their fingerprints are listed in `results/index.json`. Add a new export rather
than silently replacing an outcome. The full local reports remain authoritative
for weights, poses, hypotheses and independent geometry/metric verification.

## Manuscript

`paper/current/` is the single active manuscript directory. Edit its sources
in place and keep this path across all future revisions; do not create dated
working paper directories. Its last reviewed input/PDF fingerprints are in
`provenance.json`. After intentional source edits, review the new version,
archive its evidence under a fresh frozen report, and update the reviewed PDFs
and provenance in the same `paper/current/` location. Keep historical snapshots
intact. `make paper` builds into `.build/paper/` without overwriting reviewed PDFs.

Author names, submission ID and public citation metadata should be completed by
the authors when the draft is ready for the intended submission/release.

## New maintenance code

Use descriptive names, four-space indentation and UTF-8/LF files. Repository
maintenance commands must work without datasets or GPU imports unless their
purpose explicitly requires the full archive. Avoid importing training modules
from lightweight checks. Do not reset existing external Git checkouts.
