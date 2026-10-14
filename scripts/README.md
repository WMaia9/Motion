# Scientific scripts

These files identify the code versions used by the frozen experiments. They
retain their existing module names and relative paths because scripts import
each other and protocols record source SHA256s. Do not apply a tree-wide
formatter or run producers against an existing frozen ROOT.

## Latest additions (Oct 14, 2026)

- `run_teaser_attack_calibrated.py`, `verify_teaser_attack.py`: caption-targeted attack on the T2M-GPT checkpoint (Fig. 1, Tab. 2) and an independent recomputation of both checkpoints' scores and geometry.
- `plot_teaser_calibrated.py`: the Fig. 1 teaser.
- `run_blackbox_fid_attack.py`: query-only attack (scalar FID per query, no gradients).
- `analyze_ranking_reversal.py`: paired test of ranking changes between all release pairs.

For a map from each paper result to its scripts, see [docs/reproduction.md](../docs/reproduction.md).

## HumanML3D audit round

- `download_humanml_audit.py`, `download_humanml_mirror.py`: author-linked assets and a transparently recorded processed-data mirror fallback.
- `prepare_humanml_audit.py`, `generate_humanml_mdm.py`, `run_humanml_audit.py`: fixed HumanML3D/MDM subsets and all target-aware/fixed controls.
- `humanml_audit_common.py`: native evaluator adapter and unnormalized rotation-only transformations.
- `analyze_humanml_audit.py`, `verify_humanml_audit.py`: grouped uncertainty, native metric/encoder parity and independent physical checks.
- `prepare_humanml_blind_fid.py`, `resume_humanml_blind_generation.py`, `run_humanml_blind_fid.py`: fresh-group caption-blind distribution targets and attack outputs sealed before scoring.
- `score_humanml_blind_fid.py`, `bootstrap_humanml_blind_fid.py`: native scoring/geometry checks and explicitly post-hoc conditional FID sensitivity.

## Seed replication and adaptive-statistic round

- `run_seed_dose_response.py`, `run_seed_dose_worker.py`: registered seven-seed replication, matched controls and three-seed CE stress grid.
- `run_adaptive_distance_attack.py`: DEV per-clip attacks preserving canonical distance, and optionally 6D norm.
- `run_adaptive_distance_confirmation.py`: fixed full TEST confirmation of those attacks and controls.
- `analyze_review_replication.py`, `analyze_adaptive_confirmation.py`: paired, seed-aware and family-adjusted uncertainty.
- `run_dose_coordinate_transplant.py`, `analyze_dose_transplant.py`: full-grid DEV diagnostic of caption influence through redundant coordinates.
- `audit_adaptive_detectors.py`, `audit_phoenix_cluster_sensitivity.py`: explicitly post-hoc scalar detector and source-prefix grouping checks.
- `freeze_review_replication.py`, `freeze_dose_transplant.py`: independent geometry/metric reconstruction and immutable manifests.

## Previous training-objective round

- `run_quotient_training.py`: matched ordinary-denoiser MSE/geodesic objectives.
- `run_real_motion_layers.py`: source-text-conditioned layers on real How2Sign/CSL.
- `run_learned_budget.py`: smaller learned coefficient bounds.
- `run_constant_noncanonicity_layer.py`: fixed-distance and fixed-norm control.
- `run_constant_distance_sphere.py`: fixed-distance sphere control.
- `run_second_review_confirmation.py`: fixed-model full TEST inference.
- `run_retrieval_transfer.py`, `confirm_retrieval_layer_transfer.py`: native public replay/transfer.
- `run_ordinary_coordinate_transplant.py`: exploratory redundant-coordinate transplant.
- `analyze_second_review_results.py`: paired uncertainty for confirmation/diagnostics.

## Shared and historical modules

`run_physical_semantic_ddpm.py` defines the local denoiser and TRAIN preprocessing;
`run_evaluation_heldout.py` supplies sampling without reference poses;
`run_signspark_geometry_dev.py` wraps the native reader/geometry contract.
`run_learned_equivalence.py` defines the original fixed-motion output layer.
Other `run_*` and `analyze_*` files support historical evidence or dependencies.
Their presence does not mean every pilot is part of the final positive claim.

`rotation_equivalence.py` provides NumPy/Torch canonicalization, admissible
transforms and hypothesis agreement. It does not establish linguistic quality,
privacy or cryptographic security.

`verify_*`, `freeze_*` and `audit_*` distinguish full scientific checks from
manuscript creation. Some audit/finalization scripts write reports and are meant
for an unfrozen output directory. Only `verify_frozen_research.py` is exposed by
`make verify-archive`, a read-only operation. Repository checks use
`tools/repository.py` and require no GPU/data imports.

`download_*` and `fetch_*` record data/model/source acquisition. Follow the
observed environments and locked external sources; do not overwrite them with
a different upstream dependency set.

One-off scripts that rearranged paragraphs, edited floating figures or performed
an earlier destructive cleanup are archived locally instead of remaining in the
active Git code tree. Exact snapshots remain in the frozen delivery.

See [reproduction.md](../docs/reproduction.md) for study/analysis pairings and
[data-and-artifacts.md](../docs/data-and-artifacts.md) for the complete local layout.
