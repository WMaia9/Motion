# Compact evidence export

This directory contains byte-for-byte copies of aggregate result JSONs, protocols,
verification records and SHA256 inventories from 40 frozen research reports.
`index.json` identifies each exported file and its original full-archive manifest.

**The full manifests list omitted archive files too.** Their presence here is
provenance, not a claim that this Git checkout includes the full datasets, weights,
pose arrays or decoded hypothesis collections. `make check` verifies the exported
subset; `make verify-archive` verifies the complete local `reports/` tree.

Core result sources:

| Study | Aggregate results |
|---|---|
| Current motion release manuscript | [summary](cvpr_motion_release_review_20261012/summary.json), [verification](cvpr_motion_release_review_20261012/verification.json) |
| Five public generators, native 20 repetitions | [analysis](cvpr_motion_prevalence_20261012/analysis.json), [verification](cvpr_motion_prevalence_20261012/verification.json) |
| Caption-blind repetitions | [analysis](cvpr_humanml_blind_repetitions_20261012/analysis.json), [verification](cvpr_humanml_blind_repetitions_20261012/verification.json) |
| Ordinary MDM reward pilot | [summary](cvpr_humanml_reward_pilot_20261012/summary.json), [verification](cvpr_humanml_reward_pilot_20261012/verification.json) |
| Released MotionRFT/base diagnostic | [analysis](cvpr_motion_reward_release_20261013/analysis.json), [verification](cvpr_motion_reward_release_20261013/verification.json) |
| Previous HumanML3D manuscript delivery | [summary](cvpr_humanml_review_20261012/summary.json), [verification](cvpr_humanml_review_20261012/verification.json) |
| Public HumanML3D/MDM evaluator audit | [analysis](cvpr_humanml3d_audit_20261012/analysis.json), [verification](cvpr_humanml3d_audit_20261012/verification.json) |
| Caption-blind TRAIN-targeted FID audit | [analysis](cvpr_humanml3d_blind_fid_20261012/analysis.json), [conditional sensitivity](cvpr_humanml3d_blind_fid_20261012/FID_bootstrap_sensitivity.json) |
| Latest critical review and delivery | [summary](cvpr_critical_review_20261012/summary.json), [verification](cvpr_critical_review_20261012/verification.json) |
| Adaptive fixed-statistic full TEST attacks | [analysis](cvpr_adaptive_distance_confirmation_20261012/analysis.json) |
| Adaptive fixed-statistic DEV pilot and controls | [analysis](cvpr_adaptive_distance_attack_20261012/analysis.json) |
| Seven new seeds, ten-seed pool and full CE stress grid | [analysis](cvpr_seed_dose_response_20261012/analysis.json) |
| Stress-grid coordinate-transplant diagnostic | [analysis](cvpr_dose_transplant_20261012/analysis.json) |
| Original fixed-motion PHOENIX layer TEST | [analysis](cvpr_learned_confirmation_20261012/analysis.json) |
| Ordinary denoiser DEV sweep | [analysis](cvpr_quotient_training_20261012/analysis.json) |
| New fixed TEST, real-motion layers and bounds | [analysis](cvpr_second_review_confirmation_20261012/analysis.json) |
| Real-motion How2Sign/CSL DEV | [analysis](cvpr_real_motion_layers_20261012/analysis.json) |
| Smaller learned bounds DEV | [analysis](cvpr_learned_budget_20261012/analysis.json) |
| Fixed-norm/distance negative control | [analysis](cvpr_constant_noncanonicity_20261012/analysis.json) |
| Fixed-distance sphere negative control | [analysis](cvpr_constant_distance_sphere_20261012/analysis.json) |
| Public retrieval replay/transfer | [analysis](cvpr_retrieval_transfer_20261012/analysis.json) |
| Ordinary coordinate-transplant diagnostic | [analysis](cvpr_ordinary_transplant_20261012/analysis.json) |

The export preserves original metadata paths and timestamps. Absolute local paths
inside historical protocols identify recorded inputs; they are not portable
download locations. Restoring the full archive requires the artifacts described
in [data-and-artifacts.md](../docs/data-and-artifacts.md).
