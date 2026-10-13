# Current manuscript — the single working location

**Same Motion, Different Scores: Auditing Representation Consistency**

Read and edit the active manuscript in **this directory**. Every future revision
keeps the same `paper/current/` path. Do not create another dated working paper
directory. Dated directories under `paper/` are historical local copies, and
frozen snapshots under `reports/` preserve evidence rather than serve as working
manuscripts.

- [Main PDF](reviewed/main.pdf): eight content pages plus two reference pages.
- [Supplement PDF](reviewed/supplement.pdf): fifteen content pages plus one reference page.
- `main.tex` / `supplement.tex`: standalone sources, with local tables and figures.
- `provenance.json`: reviewed file fingerprints, upstream style revision and origin.

All scientific source/figure/PDF bytes were copied from the frozen 13 October 2026
released-motion audit delivery selected by `configs/reviewed_delivery.json`. This is
the provenance of the current reviewed version, not a second working location.
All 26 pages were visually reviewed. This README is the repository-facing build guide. The full original
manuscript ZIP and its frozen evidence remain in the local archive.

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
