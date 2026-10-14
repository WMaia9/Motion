# Current manuscript — the single working location

**Same Motion, Different Scores: Auditing Representation Consistency**

Read and edit the active manuscript in **this directory**. Every future revision
keeps the same `paper/current/` path. Do not create another dated working paper
directory. Dated directories under `paper/` are historical local copies, and
frozen snapshots under `reports/` preserve evidence rather than serve as working
manuscripts.

- [Main PDF](reviewed/main.pdf): eight content pages plus two reference pages.
- [Supplement PDF](reviewed/supplement.pdf): twenty-six content pages plus two reference pages.
- `main.tex` / `supplement.tex`: standalone sources, with local tables and figures.
- `provenance.json`: reviewed file fingerprints, upstream style revision and origin.

The reviewed PDFs come from the frozen direction-control delivery selected by
`configs/reviewed_delivery.json`. All 38 final PDF pages were visually inspected;
layout, references, embedded fonts and evidence fingerprints were checked.
The [current review](../../docs/coherence-review.md) explains the completed
fixed/random direction controls, restored natural sign evidence and scientific limits. The
[editorial review](../../docs/editorial-review.md) records the earlier writing
study and its primary awarded-paper references.
This README is the repository-facing build guide. The manuscript source ZIP and
full scientific artifacts remain in the local frozen archive.

Build from repository root with `make paper`. PDFs and logs are written to
`.build/paper/`, leaving `reviewed/` unchanged. Tectonic must be available on PATH
or in the workspace's ignored `.tools/tectonic` path. A standard LaTeX installation
can also compile the standalone sources using the included unmodified style/BST.

`reviewed/` contains the last reviewed PDFs; `.build/paper/` contains the latest
local compilation. Source edits intentionally invalidate the reviewed hashes
until a new review snapshot and provenance update are completed. Keep the old
frozen snapshots intact during that update.

The kit is the available official 2026 version. Target 2027 conformity is provisional
until its final rules can be checked. Author metadata and ID remain anonymous/DRAFT.
No external submission or publication has occurred.

Numbers derive from the frozen evidence summarized in
[results/](../../results/README.md); scientific limits are in
[research-status.md](../../docs/research-status.md).
