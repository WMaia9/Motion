# Same Motion, Different Scores

Research code and evidence for **Auditing Representation Consistency**,
an anonymous CVPR 2027 draft. The project studies how redundant 6D rotation
coordinates can change sign-language backtranslation and HumanML3D evaluation
scores while preserving represented motion.

[Paper](paper/current/reviewed/main.pdf) ·
[Supplement](paper/current/reviewed/supplement.pdf) ·
[Results](docs/research-status.md) ·
[Reproduction](docs/reproduction.md)

The active manuscript always lives in **`paper/current/`**. Edit its
`main.tex`, `supplement.tex`, tables and figures there. New revisions keep this
same path; dated manuscript directories are historical local copies.

The [current review](docs/coherence-review.md) adds time-fixed directions in
every non-null block and restores natural real-input sign-language evidence.
All twenty Guo/TMR draws and independent checks are complete. The paper stays
in the same working directory; the
[previous MARDM round](docs/mardm-followup-review.md) is an immutable record.
`reviewed/` contains the checked eight-page paper and its supplement.

## Results and scope

The paired audit completes **20 native TEST repetitions for five public
HumanML3D generators**: MDM50, MLD, MotionLCM-v1, T2M-GPT and T2M.
Canonicalizing identical outputs changes all five primary FID means, with
family-adjusted intervals excluding zero. T2M-GPT has a raw-minus-canonical FID gap of **+0.034** and
MotionLCM-v1 **−0.031**; the primary mean ranking stays unchanged.
Two official evaluator checkpoints share an architecture and have different
weights. This establishes natural representation dependence, not intentional
score manipulation. See the [completed review](docs/motion-release-review.md).

| Conditional learned layer, full TEST | Source raw BLEU | Layer raw BLEU | Canonical prediction |
|---|---:|---:|---|
| PHOENIX, generated motion | 6.39 | 15.21 | Identical for every clip |
| How2Sign, real motion | 3.48 | 6.11 | Identical for every clip |
| CSL-Daily, real motion | 6.89 | 7.43 | Identical for every clip |

Each primary layer uses three training seeds and receives source text. These are
controlled evaluation attacks, not improvements in signing. Language-specific
BLEU values use different token conventions.
PHOENIX layer inputs are canonicalized generated arrays; How2Sign/CSL inputs
retain the released real-motion features.

An adaptive per-clip circle attack preserves canonical distance and squared 6D
norm per joint/frame. On full PHOENIX TEST, BLEU rises **6.62 → 10.72**, gain
**4.09**, four-comparison-adjusted interval **[3.04, 5.24]**; RMS detector AUC
is **0.5003**. This targets declared statistics, not every possible detector.

The original ordinary-generator endpoint remains inconclusive after extension
to ten seeds: gap **0.20**, seed+clip interval **[−0.43, 0.80]**. Stronger-CE
stress tests show positive representation gaps, but neither monotonicity nor
caption-coordinate transmission is confirmed. Native public-generator transfer
remains inconclusive, and the public BT readers share one architecture family.

The HumanML3D extension uses an independent public evaluator and released MDM50
weights, with no model fitting. On 1,024 fixed TEST clips, target-aware attacks
raise R-precision from **0.397 to 0.993** on MDM and **0.462 to 0.996** on real
motion, preserving rotations and positions. These attacks worsen FID.

A separate attack uses only TRAIN motion statistics, without caption content or
TEST reference embeddings. On the first 960-output TEST cohort it reduces FID
**0.384 → 0.210** at bound 0.1; bound 0.35 instead worsens FID to **1.085**.
Two further generation/selection cohorts repeat the reduction (39–53% overall);
the cohorts overlap in source groups. The post-hoc grouped sensitivity conditions
on the original generated cohort. All bounds
and controls remain reported. The processed-data mirror's origin and numerical
differences are explicit; full leaderboard reproduction is not claimed.
See the [HumanML3D study and review](docs/humanml-review.md).

The completed extension adds MoMask, prefix-conditioned DiP and MotionDiffuse
to HumanML3D, plus MoMask and MotionDiffuse on native KIT. All five additional
FID gaps have adjusted intervals excluding zero; their signs vary, and their
R-precision gaps remain inconclusive. TMR supplies a different evaluator
architecture: on 1,024 fixed MDM clips, target-aware R1 changes **0.578 → 0.993**
without changing represented motion; canonical R1 stays **0.580** before and
after. Real-motion R1 changes **0.688 → 1.000**. Wrong-target, canonical-gradient
and generic adversarial controls are retained.

Equal L2 does not match temporal structure. With time-fixed directions in all
blocks, Guo FID changes reach **1.189 for positions** and **0.476 for velocities**,
above the **0.117 rotation-fiber maximum** at the larger budget. Random velocity
and position noise also exceed random fiber coefficients. The fiber's distinction
is preserving rotations and every nonrotation feature; there is no claim that
it is the most influential feature block.

Without an attack or layer, canonicalizing real native-preprocessed inputs changes
**1,438/2,183 How2Sign** and **464/1,162 CSL** decoded sentences; PHOENIX changes
**0/642**. How2Sign BLEU changes only 3.4779 → 3.4840. The shared original
left-hand preprocessing and native token conventions are explicit. This exposes
natural encoding dependence, without establishing better or worse semantics.

Public MARDM essential readers are invariant by excluding rotation features.
Our defense retains rotations, which matters for the rotation-only sign inputs.
Neither reader rankings nor the eight-release RMS analysis refute MARDM’s full
mechanism. See the [complete analysis](docs/coherence-review.md).

Earlier replication adds 23 fine-tunes and adaptive/mechanism controls. Positive
and negative outcomes are preserved. The bibliography contains 52 entries; 40 are cited in the main paper.
The failed MDM reward pilot and eight-fit DEV calibration do not establish
exploitation during successful reward training. Released MotionRFT retrieval
gains survive canonicalization.
See the [independent critique analysis](docs/critical-review.md)
and the [scientific status](docs/research-status.md).

## Quick start

The lightweight checks require Python 3.10+ and Git; no GPU, datasets or Python
packages need to be installed.

```bash
make check
make summary
```

For the research environment, use Python 3.10 and the observed dependencies in
[`requirements/`](requirements/README.md). Main inference and public CFM replay
use separate environments. Existing scientific environments are not modified.

```bash
make env PYTHON=venv_mska/bin/python
make vendor-check
```

`make vendor` clones missing external repositories at the exact revisions in
[`configs/external_sources.json`](configs/external_sources.json). Existing
checkouts are checked without resetting their contents.

## Repository layout

| Directory | Contents |
|---|---|
| `paper/current/` | Standalone LaTeX, figures, reviewed PDFs and provenance |
| `scripts/` | Scientific producers, analyses and verification helpers |
| `tools/` | Repository checks, manuscript build and compact archive export |
| `results/` | Aggregate result JSONs, protocols and archive inventories |
| `configs/` | Pinned external source revisions |
| `requirements/` | Separate dependency pins and observed environment snapshots |
| `docs/` | Reproduction, scientific status, data layout and reviewer response |
| `.github/workflows/` | Lightweight checks on push and pull request |

Datasets, checkpoints, full `reports/`, external checkouts, virtual environments,
old manuscript versions and temporary build files are local-only and ignored by
Git. Their existing filesystem paths are preserved. The compact result export
contains metadata; it is not a substitute for the full research archive.

## Manuscript

```bash
make paper
```

Install Tectonic on `PATH`, or use the compiler already present in the original
workspace. Build output goes to `.build/paper/`; reviewed PDFs remain unchanged.
The manuscript uses the unmodified available CVPR kit from 2026. Final 2027
format rules must be checked when available. The draft is anonymous and retains
`DRAFT` as its submission ID; no submission has been made.

## Verification and contribution

`make check` validates Python syntax, manuscript inputs, reference keys, exported
evidence fingerprints, commit size and exclusions of local artifacts. It runs
on a lightweight checkout. `make verify-archive` checks every full-archive SHA256
when the local `reports/` directory is available.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing scientific producers or
reviewed snapshots. Dataset/model licenses and third-party attribution are
recorded in [NOTICE.md](NOTICE.md).
