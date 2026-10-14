# Contributing

Run `make check` and `make test` before committing. Both require only Python, Git and NumPy. Keep generated
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
intact. `make paper` compiles in `.build/paper/` and copies the PDFs to `paper/current/reviewed/`;
run `make check` afterwards and update `provenance.json` for the new reviewed version.

Author names, submission ID and public citation metadata should be completed by
the authors when the draft is ready for the intended submission/release.

## The `canon6d` library

`canon6d/` is the public, installable implementation of the paper's fix. Keep its API
small and stable, add a test in `tests/` for every behavior change, and keep it identical
in output to `tools/canonicalize_motion.py`, which is the frozen version used in the
experiments (a test enforces this).

## New maintenance code

Use descriptive names, four-space indentation and UTF-8/LF files. Repository
maintenance commands must work without datasets or GPU imports unless their
purpose explicitly requires the full archive. Avoid importing training modules
from lightweight checks. Do not reset existing external Git checkouts.
