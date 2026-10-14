# Compact evidence export

This directory contains byte-for-byte copies of aggregate result JSONs, protocols,
verification records and SHA256 inventories from the frozen reports listed in
`index.json`.
`index.json` identifies each exported file and its original full-archive manifest.

**The full manifests list omitted archive files too.** Their presence here is
provenance, not a claim that this Git checkout includes the full datasets, weights,
pose arrays or decoded hypothesis collections. `make check` verifies the exported
subset; `make verify-archive` verifies the complete local `reports/` tree.

Core result sources:

| Study | Aggregate results |
|---|---|
| Current sign-stability manuscript | [summary](cvpr_sign_stability_review_20261014/summary.json), [claim audit](cvpr_sign_stability_review_20261014/claim_audit.json) |
| Matched Gaussian/physical real-sign controls | [analysis](cvpr_sign_stability_20261014/analysis.json), [verification](cvpr_sign_stability_20261014/verification.json) |
| Same-sign within-rotation FID comparison | [ratio arithmetic and sources](cvpr_sign_stability_20261014/rotation_ratio.json) |
| Previous direction-control manuscript | [summary](cvpr_coherence_review_20261014/summary.json), [claim audit](cvpr_coherence_review_20261014/claim_audit.json) |
| Time-fixed directions, 20 repeats per reader | [Guo](cvpr_coherent_blocks_20261014/guo/analysis.json), [TMR](cvpr_coherent_blocks_20261014/tmr/analysis.json), [verification](cvpr_coherent_blocks_20261014/verification.json) |
| Natural real sign inputs, restored without new inference | [analysis](cvpr_sign_natural_restoration_20261014/analysis.json), [verification](cvpr_sign_natural_restoration_20261014/verification.json) |
| Previous MARDM follow-up manuscript | [summary](cvpr_mardm_followup_20261013/summary.json), [claim audit](cvpr_mardm_followup_20261013/claim_audit.json) |
| Equal-L2 block controls, 20 repeats per reader | [Guo](cvpr_matched_blocks_20261013/guo/analysis.json), [TMR](cvpr_matched_blocks_20261013/tmr/analysis.json), [verification](cvpr_matched_blocks_20261013/verification.json) |
| Rotation RMS and family analysis, eight releases | [analysis](cvpr_rotation_mechanism_20261013/analysis.json), [verification](cvpr_rotation_mechanism_20261013/verification.json) |
| Public MARDM essential readers, 20 repeats | [analysis](cvpr_mardm_readers_20261013/analysis.json), [verification](cvpr_mardm_readers_20261013/verification.json) |
| TMR targeted controls, 1,024 clips per source | [analysis](cvpr_tmr1024_controls_20261013/analysis.json), [verification](cvpr_tmr1024_controls_20261013/verification.json) |
| Previous editorial manuscript | [summary](cvpr_editorial_review_20261013/summary.json), [claim audit](cvpr_editorial_review_20261013/editorial_claim_audit.json) |
| Previous generalization manuscript | [summary](cvpr_generalization_review_20261013/summary.json), [verification](cvpr_generalization_review_20261013/verification.json) |
| MotionDiffuse HumanML/KIT, 20 repeats each | [analysis](cvpr_motiondiffuse_compiled_20261013/analysis.json), [verification](cvpr_motiondiffuse_compiled_20261013/verification.json) |
| TMR, another evaluator architecture | [analysis](cvpr_tmr_audit_20261013/analysis.json), [verification](cvpr_tmr_audit_20261013/verification.json) |
| Earlier 128-clip TMR attack pilot | [analysis](cvpr_tmr_attack_20261013/analysis.json), [verification](cvpr_tmr_attack_20261013/verification.json) |
| Coherent physical angular calibration | [analysis](cvpr_physical_calibration_20261013/analysis.json), [verification](cvpr_physical_calibration_20261013/verification.json) |
| Finite reward calibration on DEV | [all fits and selection](cvpr_reward_dev_search_20261013/selection.json), [verification](cvpr_reward_dev_search_20261013/verification.json) |
| Conditional reward validation, skipped without qualified DEV fit | [analysis](cvpr_reward_dev_validation_20261013/analysis.json), [verification](cvpr_reward_dev_validation_20261013/verification.json) |
| Blind MLD and checkpoint transfer | [analysis](cvpr_blind_transfer_20261013/cross_reader_analysis.json), [verification](cvpr_blind_transfer_20261013/verification.json) |
| Already-canonical real-motion sanity check | [analysis](cvpr_motion_defense_cost_20261013/analysis.json), [verification](cvpr_motion_defense_cost_20261013/verification.json) |
| Additional MoMask/DiP and KIT, 20 repeats | [analysis](cvpr_additional_motion_audit_20261013/analysis.json), [verification](cvpr_additional_motion_audit_20261013/verification.json) |
| Previous motion release manuscript | [summary](cvpr_motion_release_review_20261012/summary.json), [verification](cvpr_motion_release_review_20261012/verification.json) |
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

[Metadata clarifications](metadata-notes.json) document label corrections separately
from byte-for-byte historical exports. A MotionDiffuse exclusion flag was mislabeled
in the archived aggregate; both registered compiled protocols excluded incomplete
eager attempts, and the reported metrics use all twenty completed compiled draws.
