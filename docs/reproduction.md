# Reproduction

This guide maps every result in the paper to the code that produced it, the analysis that summarized it and the independent check that verified it. Producers live in [`scripts/`](../scripts/README.md); full frozen outputs live in `reports/<study>/` in the original workspace, and compact exported metadata in [`results/`](../results/README.md).

## Two levels of reproduction

**Lightweight clone (no GPU, no data).** Python 3.10+:

```bash
make check     # repository policy, exported-evidence fingerprints, manuscript inputs, links
make test      # canon6d contract tests
make summary   # print the headline numbers from results/
```

**Full archive.** Restoring the original `reports/` tree, datasets and model weights (see [data-and-artifacts.md](data-and-artifacts.md)) allows a read-only check of every archived file:

```bash
make verify-archive PYTHON=venv_mska/bin/python
```

## Environment and external code

Use Python 3.10 with a CUDA 12.8 PyTorch environment as recorded in [`requirements/main.txt`](../requirements/main.txt); the public sign-language replay uses [`requirements/replay.txt`](../requirements/replay.txt). External repositories are pinned in [`configs/external_sources.json`](../configs/external_sources.json):

```bash
make vendor        # clone missing external repositories at their pinned revisions
make vendor-check  # verify revisions
```

Public generators expose conflicting `utils` modules; run each method in a fresh process. Experiments used eight Torch threads, TF32 disabled, and the seeds stored in each protocol.

## Rules for frozen studies

Each study directory records its protocol, source hashes and inputs before inference, and is frozen after independent verification. **Never run a producer against an existing frozen directory.** To repeat a study, copy its archived source closure to a separate checkout, point its output root to a fresh directory, register a new protocol, then run. All planned conditions are reported; no checkpoint or setting was chosen on test scores.

## Results by paper section

### Sec. 5.1 — Rewriting alone changes the score

| Result | Producer | Analysis / verification | Study |
|---|---|---|---|
| Fixed rewritings of real motion (FID 0.001 → 3.64) | `run_motion_release_audit.py` (real controls) | `analyze_motion_prevalence.py`, `verify_motion_prevalence.py` | `cvpr_motion_prevalence_20261012` |
| Physical calibration (Fig. 3) | `run_physical_rotation_calibration.py` | `verify_physical_calibration.py` | `cvpr_physical_calibration_20261013` |
| Equal-budget block controls (Fig. 5) | `run_matched_block_audit.py`, `run_coherent_block_audit.py` | `verify_matched_block_audit.py`, `verify_coherent_block_audit.py` | `cvpr_matched_blocks_20261013`, `cvpr_coherent_blocks_20261014` |

### Sec. 5.2 — Released generators

| Result | Producer | Analysis / verification | Study |
|---|---|---|---|
| Five releases, 20 native draws, two T2M checkpoints | `prepare_motion_release_audit.py`, `run_motion_release_audit*.py`, `score_motion_release_second_reader.py` | `analyze_motion_prevalence.py --secondary`, `verify_motion_prevalence.py` | `cvpr_motion_prevalence_20261012` |
| MoMask, DiP, MotionDiffuse, KIT-ML | `run_additional_motion_audit.py`, `run_motiondiffuse_compiled_audit.py` | `verify_additional_motion_audits.py` | `cvpr_additional_motion_audit_20261013`, `cvpr_motiondiffuse_compiled_20261013` |
| TMR retrieval on released outputs | `run_tmr_release_audit.py` | `verify_tmr_consistency_audits.py` | `cvpr_tmr_audit_20261013` |
| Ranking changes between all release pairs | — (reads frozen per-draw metrics) | `analyze_ranking_reversal.py` | `cvpr_ranking_reversal_20261014` |
| Gap vs. distance from canonical; model families | — | `analyze_rotation_mechanism.py`, `verify_rotation_mechanism.py` | `cvpr_rotation_mechanism_20261013` |
| Essential-feature (MARDM) readers | `run_mardm_reader_audit.py` | `verify_mardm_reader_audit.py` | `cvpr_mardm_readers_20261013` |

### Sec. 5.3 — Attacks on the evaluator

| Result | Producer | Analysis / verification | Study |
|---|---|---|---|
| Caption-targeted attack, T2M-GPT checkpoint (Fig. 1, Tab. 2) | `run_teaser_attack_calibrated.py` | `verify_teaser_attack.py` | `cvpr_teaser_audit_20261014` |
| Caption-targeted attack, MDM checkpoint (supplement) | `prepare_humanml_audit.py`, `generate_humanml_mdm.py`, `run_humanml_audit.py` | `analyze_humanml_audit.py`, `verify_humanml_audit.py` | `cvpr_humanml3d_audit_20261012` |
| Caption-targeted attack, TMR | `run_tmr_targeted_controls.py --clips 1024` | `verify_tmr1024_controls.py` | `cvpr_tmr1024_controls_20261013` |
| Caption-blind attack, three MDM cohorts | `run_humanml_blind_fid.py`, `run_humanml_blind_repetitions.py` | `score_humanml_blind_fid.py`, `bootstrap_humanml_blind_fid.py` | `cvpr_humanml3d_blind_fid_20261012`, `cvpr_humanml_blind_repetitions_20261012` |
| Caption-blind attack on MLD; transfer to the second checkpoint | `score_blind_cross_reader.py` | `verify_generalization_controls.py` | `cvpr_blind_transfer_20261013` |
| Query-only (black-box) attack | `run_blackbox_fid_attack.py` | built-in checks (geometry, canonical invariance) | `cvpr_blackbox_attack_20261014` |
| Teaser figure | `plot_teaser_calibrated.py` | asserts identical positions and rotation axes | `cvpr_teaser_audit_20261014` |

### Sec. 5.4 — Reward fine-tuning

| Result | Producer | Analysis / verification | Study |
|---|---|---|---|
| Reward fine-tuning of MDM (Discussion, supplement S6) | `reward_finetune_mdm.py` (protocol in `DEV_PROTOCOL.md`) | `eval_reward_finetune_test.py`, `analyze_reward_finetune_test.py` | `cvpr_reward_finetune_20261014` |
| Reward fine-tuning of MLD (supplement S6) | `reward_finetune_mld.py` | `reward_finetune_mld.py --test`, `analyze_reward_finetune_mld.py` | `cvpr_reward_finetune_20261014` |
| Released MotionRFT off-canonical distance (supplement S6) | `run_motion_reward_release.py` | `motionrft_release_rms.py` | `cvpr_motion_reward_release_20261013`, `cvpr_reward_finetune_20261014` |

### Sec. 5.5 — Sign language

| Result | Producer | Analysis / verification | Study |
|---|---|---|---|
| Canonicalizing real sign inputs | `restore_sign_natural_evidence.py` | independent string and BLEU recomputation | `cvpr_sign_natural_restoration_20261014` |
| Stability controls (Gaussian, physical) | `run_sign_stability_audit.py` | `analyze_sign_stability.py`, `verify_sign_stability.py` | `cvpr_sign_stability_20261014` |
| Learned layer, PHOENIX | `run_learned_equivalence.py`, `run_learned_confirmation.py` | `analyze_learned_confirmation.py` | `cvpr_learned_equivalence_20261011`, `cvpr_learned_confirmation_20261012` |
| Learned layers, How2Sign / CSL-Daily | `run_real_motion_layers.py` | `analyze_real_motion_layers.py` | `cvpr_real_motion_layers_20261012` |
| Per-clip wrong-sentence attacks, CTC recognizer | `run_signspark_gauge_attack.py`, `run_motion_ctc_confirmation.py` | `audit_evaluation_confirmation.py` | `cvpr_evaluation_confirmation_20261011` |
| Attack with fixed detector statistics | `run_adaptive_distance_confirmation.py` | `analyze_adaptive_confirmation.py`, `audit_adaptive_detectors.py` | `cvpr_adaptive_distance_confirmation_20261012` |

### Sec. 5.6 — Canonicalization

| Result | Producer | Analysis / verification | Study |
|---|---|---|---|
| Real-motion retrieval unchanged | `measure_motion_defense_cost.py` | `verify_generalization_controls.py` | `cvpr_motion_defense_cost_20261013` |
| Readers adapted to canonical sign inputs | `run_canonical_readers.py` | `analyze_canonical_readers.py` | `cvpr_canonical_readers_20261011` |
| Portable implementation | [`canon6d/`](../canon6d/) | [`tests/test_canon6d.py`](../tests/test_canon6d.py) | — |

## Data notes

- HumanML3D processed motion comes from a public mirror pinned by revision and SHA256, because the author-linked archive exceeded its download quota. Its size matches the author inventory; full byte identity with the original archive is not claimed.
- Two official T2M evaluator checkpoints exist. The one distributed with T2M-GPT reproduces the published real-motion scores (top-1 R-Precision 0.517 vs 0.511) and is used for the targeted attacks; the one distributed with MDM (0.462) is used for the remaining HumanML3D studies, with the other as a replication.
- Release audits are paired consistency audits under native inference settings, not reproductions of published leaderboard numbers.
- Public sign-language test splits were used during development; no study claims a pristine held-out set.

## Building the manuscript

`make paper` compiles `paper/current/main.tex` and `supplement.tex` with [Tectonic](https://tectonic-typesetting.github.io/) and writes the PDFs to `paper/current/reviewed/`. Temporary files go to `.build/paper/`.
