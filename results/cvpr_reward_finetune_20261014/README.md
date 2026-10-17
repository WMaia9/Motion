# Reward fine-tuning of MDM against the T2M evaluator (2026-10-14/15)

Question: does a generator fine-tuned on a raw-feature evaluator drift into the redundant 6D
coordinates, and what does that do to the evaluator's scores?  Paper Sec. 5.4, Tab. 3;
supplement Sec. S6, Tab. S8.

Scripts: `scripts/reward_finetune_mdm.py` (training, validation curves),
`scripts/eval_reward_finetune_test.py` (single test scoring, 2,048 clips, both T2M checkpoints),
`scripts/analyze_reward_finetune_test.py` (R1, FID, paired bootstrap over clips).
Protocol and amendments, dated before any test inference: `DEV_PROTOCOL.md`.
Validation summary of every run, including rejected settings: `dev_summary.json`.

Confirmation (3 seeds per version, KL-proxy weight 100, canonical regularizers, lr 1e-5, 2,000 steps):
reward evaluator raw-minus-canonical FID gap +0.467 [0.278, 0.658] for raw reward, -0.044
[-0.076, -0.015] for canonical reward, +0.003 [0.001, 0.005] for the no-reward control;
R1 gap +0.008 [0.002, 0.014] for raw reward. Off-canonical RMS 0.044 (raw reward) vs 0.019.
Second checkpoint, raw reward: FID gap +0.035 [-0.035, 0.109].

Replication with the T2M-GPT checkpoint as reward (runs `*_rgpt`, test keys `graw_*`/`gcanonical_*`;
in test/analysis.json the keys `reward`/`held` always denote the MDM-linked/T2M-GPT checkpoints):
off-canonical 0.040 (raw reward) vs 0.018; R1 gap by the reward checkpoint +0.009 [0.002, 0.016];
its FID gap -0.031 [-0.118, 0.058]; the MDM-linked checkpoint as judge: FID gap -0.051 [-0.074, -0.027].

Second generator, MLD (`scripts/reward_finetune_mld.py`, runs `mld_*`, test in `test_mld/`, analysis
`scripts/analyze_reward_finetune_mld.py`): selected penalty weight 0. Off-canonical 0.061 (raw reward)
vs 0.060 (canonical reward), 0.044 pretrained, 0.043 control; reward-checkpoint FID gap +0.031 [0.011, 0.051]
(raw) vs +0.033 [0.014, 0.053] (canonical), -0.019 pretrained. No raw-specific drift.
Released MotionRFT off-canonical RMS: `motionrft_release_rms.json` (`scripts/motionrft_release_rms.py`).
