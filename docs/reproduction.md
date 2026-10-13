# Reproduction

## Lightweight checkout

A clone contains scientific code, the reviewed manuscript and aggregate evidence.
It does not contain the full local result archive, native datasets or model weights.
Python 3.10+ and Git are sufficient for:

```bash
make check
make summary
```

The checker compiles Python source without importing training modules, validates
LaTeX inputs/reference keys, verifies exported evidence and reviewed-file hashes,
and checks that local datasets, environments and model artifacts are excluded
from the Git tree. CI runs these checks without a GPU.

## Manuscript build

The single active source location is `paper/current/`. Edit sources there and
keep that path across revisions. Dated paper directories are historical copies;
frozen reports are archives, not alternative editing locations.

Install Tectonic on PATH. In the original workspace the existing local compiler
is supported at ignored `.tools/tectonic`. Run `make paper` to build `.build/paper/main.pdf` and
`.build/paper/supplement.pdf`. These generated files are ignored. The reviewed
PDFs in `paper/current/reviewed/` remain unchanged.

A new build can change PDF metadata and byte hashes. Scientific source/figure
fingerprints and the exact reviewed PDF bytes are recorded separately in
`paper/current/provenance.json`.

## Research environment and source dependencies

Use Python 3.10 with a CUDA 12.8-compatible PyTorch environment as recorded in
`requirements/main.txt`. Public CFM replay uses the separate profile
`requirements/replay.txt`. The JSON snapshots record all installed distributions
on the experiment host; they are observational records, not portable lockfiles.

External code is pinned in `configs/external_sources.json`:

```bash
make vendor
make vendor-check
```

This clones missing repositories and checks revisions. Existing checkouts are
not reset. It does not install dependencies, fetch model weights or acquire
licensed data. For the full original archive, restore the paths in
[data-and-artifacts.md](data-and-artifacts.md).

## Scientific pipelines

The latest delivery is selected explicitly by
`configs/reviewed_delivery.json`. `tools/export_repository_assets.py` reads that
frozen delivery and verifies every copied fingerprint. It requires the complete
local archive, writes only the curated export, and never reruns training.

## Completed public-motion release audit

The full review is in [motion-release-review.md](motion-release-review.md).
Optional inference packages are isolated from the main environment:

```bash
venv_mska/bin/python -m pip install --no-deps \
  --target external/motion_lcm_python \
  -r requirements/motion-release-extra.txt
```

Do this only in a new reproduction checkout if dependencies are missing. The
completed host keeps its observed main/replay environments unchanged. The
CLIP optional target remains separate. PyTorch and SciPy must import before
vendor modules that install legacy NumPy aliases. Public implementations expose
incompatible `utils` and `mld` namespaces; run each method in a fresh process.

| Completed study | Drivers | Analysis and verification |
|---|---|---|
| Five releases, native 20 repeats | `prepare_motion_release_audit.py`, `run_motion_release_audit.py`, `run_motion_release_audit_fast.py`, `run_motion_release_mld_batched.py`, `run_motion_release_gpt_batched.py`, `run_motion_release_t2m_grouped.py` | `analyze_motion_prevalence.py --secondary`, `verify_motion_prevalence.py`, `complete_motion_release_verification.py extras` |
| Second official evaluator weights | `score_motion_release_second_reader.py` | Same full endpoint/metric checks |
| Two further blind cohorts | `prepare_humanml_blind_repetitions.py`, `run_humanml_blind_repetitions.py` (separate generate/attack/score processes) | Native verification saved per cohort; combined source/overlap review |
| MDM ordinary reward pilot | `run_humanml_reward_pilot.py`, `score_humanml_reward_pilot.py` | `verify_humanml_reward_pilot.py` |
| Released MotionRFT and linked base | `run_motion_reward_release.py` | `complete_motion_release_verification.py reward` |

Download helpers are `download_motion_release_audit.py` and
`download_motion_release_additional.py --reward-release`. They identify the
public assets and pinned HF revisions; preparation registers hashes before
inference. The download helper is an acquisition recipe; the executed download
records and post-run asset inventory separately identify the actual bytes used.
A quota failure leaves MoMask without an evaluated checkpoint. It is not counted
among the five methods. Licensed meshes are not required for native RIC/FK.

The primary protocol uses 20 seeds, batch-32 metric candidate pools and a
4,640-entry native reference draw. MDM's nominal 1,000 cap emits 1,024; other
methods emit 4,640. This is a paired native-loader audit, not a universal
leaderboard protocol for all methods. Source arrays, generated durations,
caption/crop draws, separate reference draws, and embeddings are retained.
Canonicalization alters unnormalized channels `67:193` only.

Generation batches are 32/128 for MLD and GPT, with changes in draw order and
float32 operations explicitly recorded. Repetitions 0–9 of GPT already had
batch-32 generated arrays, including repetition 9 scored after the throughput
amendment. T2M groups both duration and caption length; grouping only duration
failed native attention parity and is excluded as an invalid implementation,
not a discarded valid result. MotionLCM-v1 uses Sentence-T5-large. Archived
`producer_revisions/` retains executed sources, failed preflights and the
mechanical GPT resume-guard correction, which changes no inference operation.

The additional verifier disables TF32, matching the producers. Its initial
text-embedding recomputation without matching this flag differed in a near-tied
retrieval decision; the failure log is preserved and no generated array or
reported endpoint was changed. The corrected independent checks pass. Scientific
checks verify native converters/FK, protected channels and saved metric values;
repository CI only checks compact artifacts and source syntax.

All four completed studies were frozen with `freeze_motion_release_studies.py`.
Their producer roots and registered hashes are fixed. **Do not execute writing
or analysis drivers against these frozen directories.** To regenerate, copy
an archived import closure to a separate checkout, change all study ROOTs to
fresh directories, and register the changed sources before preparation or
inference. Existing results are never overwritten as part of reproduction.
The same rule applies to figure/table scripts: they target the active manuscript
and a new, unfrozen presentation delivery, not an immutable past report.

The critical-review round adds these separately frozen studies:

The HumanML3D extension uses a separate public T2M evaluator and the released
MDM50 checkpoint, without fitting models. `configs/external_sources.json` also
pins these author repositories and CLIP. Optional CLIP imports were installed
with `--no-deps --target external/humanml_python` from
`requirements/humanml-extra.txt`; the observed main/replay environments were not
changed. Sampling disables MDM's eager, unused mesh constructor and fails if
mesh conversion is called. The denoiser, CLIP, diffusion and weights are preserved.

| HumanML3D study | Producers | Analysis / verification |
|---|---|---|
| Target-aware and fixed-encoding audit | `prepare_humanml_audit.py`, `generate_humanml_mdm.py`, `run_humanml_audit.py` | `analyze_humanml_audit.py`, `verify_humanml_audit.py` |
| Caption-blind FID audit | `prepare_humanml_blind_fid.py`, `resume_humanml_blind_generation.py`, `run_humanml_blind_fid.py` | `score_humanml_blind_fid.py`, `bootstrap_humanml_blind_fid.py` |

The second preparation originally hit a cached `utils` namespace collision;
the separate generation driver runs the unchanged native sampling producer in
a fresh process. Its protocol is already registered before generation, so do
not rerun preparation over the frozen output. The complete failure/recovery
logs are kept in the archive. For a new study, use fresh roots and record the
new source versions as described below.

The author-linked Drive motion archive exceeded its public quota. The processed
vectors came from a public third-party HF mirror pinned by revision and LFS
SHA256. Its byte count agrees with the author-linked inventory; the author Git
example agrees numerically within 1.94e-5 but differs in dtype/bytes. Full archive
identity against original Drive bytes was not established. The original targeted/blind studies audit fixed eligible subsets. The new
release audit uses the full native reference loader and twenty repetitions,
with each method's documented generation count; it still does not claim exact
reproduction of each published leaderboard number.

The blind attack uses TRAIN motion statistics and generated arrays. Its integrity
checks hash held-out files but do not deserialize or encode their reference
content. Attack outputs are saved and hashed before reference scoring. Its FID
bootstrap is explicitly post-hoc and conditional on one generated cohort.

| Frozen study | Producer | Analysis | Fits / conditions |
|---|---|---|---|
| Seed replication and CE stress grid | `run_seed_dose_response.py` | `analyze_review_replication.py seed_dose` | 23 fits; 32 fixed checkpoints on each of DEV and TEST |
| Adaptive fixed-statistic DEV attacks | `run_adaptive_distance_attack.py` | `analyze_review_replication.py adaptive` | 16 attack/control conditions on 64 DEV clips |
| Full TEST adaptive confirmation | `run_adaptive_distance_confirmation.py` | `analyze_adaptive_confirmation.py` | 12 fixed attack/control conditions on 642 clips |
| Stress-grid coordinate transplant | `run_dose_coordinate_transplant.py` | `analyze_dose_transplant.py` | Eight DEV conditions, three seeds each |
| Post-hoc detector and cluster checks | `audit_adaptive_detectors.py`, `audit_phoenix_cluster_sensitivity.py` | Same scripts | No fitting of generators or attacks; detector thresholds fitted only on DEV |

The seed driver first registered the complete grid. The execution partition and
`run_seed_dose_worker.py` then assigned disjoint fits to three worker processes;
the driver was paused during that partition and resumed afterward. Do not run
workers and an unpaused driver over the same output paths. The execution records,
completed worker inventories and pre-inference checkpoint hashes are exported.
All planned conditions were retained; there was no best-score checkpoint selection.

Earlier studies remain available and unchanged:

| Frozen study | Producer | Analysis | Fits / conditions |
|---|---|---|---|
| Ordinary objectives | `run_quotient_training.py` | `analyze_quotient_training.py` | 42 fits; fixed 5/20-epoch endpoints |
| Real-motion layers | `run_real_motion_layers.py` | `analyze_real_motion_layers.py` | 10 fits |
| Smaller bounds | `run_learned_budget.py` | `analyze_learned_budget.py` | 9 fits |
| Fixed distance+norm | `run_constant_noncanonicity_layer.py` | `analyze_constant_noncanonicity.py` | 9 fits |
| Fixed distance sphere | `run_constant_distance_sphere.py` | `analyze_constant_distance_sphere.py` | 9 fits |
| Fixed TEST | `run_second_review_confirmation.py` | `analyze_second_review_results.py confirmation` | 28 inference conditions |
| Public retrieval | `run_retrieval_transfer.py`, `confirm_retrieval_layer_transfer.py` | `analyze_second_review_results.py retrieval` | No new fitting |
| Coordinate transplant | `run_ordinary_coordinate_transplant.py` | `analyze_second_review_results.py transplant` | 9 diagnostic conditions |

The original producer scripts have fixed ROOT paths and input hashes. They are
preserved to identify the exact experiment versions, not presented as a generic
training API. For a new run, copy the archived source closure to a separate
checkout, choose fresh output ROOTs and register a new protocol. Restore the
necessary base models/data first; retain declared endpoints and all conditions.
Do not call writing helpers against frozen directories. Protocol-source hashes
will change if code or output paths change; record that new version explicitly.

The original environment used eight Torch threads, TF32 disabled,
`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4`, and the seeds in each protocol.
The public replay's effective seed is 102 despite the passed `eval.seed=123`;
this behavior is documented in the experiment rather than silently corrected.

Full-archive verification:

```bash
make verify-archive PYTHON=venv_mska/bin/python
```

This reads original manifests and checks every archived file. `make check`
validates only the compact export and repository. It does not rerun model fits,
recompute missing pose geometry or substitute for independent full-archive checks.
