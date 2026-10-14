# Manuscript

The active manuscript lives here and keeps this path across revisions.

| File | Contents |
|---|---|
| `main.tex`, `supplement.tex` | Paper and supplementary material |
| `table_*.tex`, `figures/` | Tables and figures included by the two documents |
| `references.bib` | Bibliography |
| `cvpr.sty`, `ieeenat_fullname*.bst` | Unmodified official CVPR author-kit files |
| `reviewed/` | Last reviewed PDFs |
| `provenance.json` | SHA256 fingerprints of the reviewed sources, checked by `make check` |

Build from the repository root with `make paper` (Tectonic), or upload the directory to Overleaf (pdfLaTeX, main document `main.tex`; switch the main document to `supplement.tex` for the supplement). Every number in the text comes from the frozen evidence listed in [`docs/reproduction.md`](../../docs/reproduction.md).

The CVPR 2027 author kit was not yet available; the 2026 kit is used and must be rechecked against the final 2027 rules. Authors and paper ID are anonymous placeholders.
