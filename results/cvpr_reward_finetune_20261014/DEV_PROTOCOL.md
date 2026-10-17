# Dev calibration (validation split only; test not touched)

Arms: mse, raw, canonical. Fixed: MDM50, CFG 2.5, batch 32, AdamW, reward from MDM-linked T2M
evaluator (Euclidean InfoNCE), reward at random late step t_k in [0, tmax], held-out evaluator =
T2M-GPT checkpoint. 512 fixed val captions, 3 candidate orders.

Qualification (decided before seeing results): a reward configuration qualifies if the reward
evaluator's raw R1 on dev improves by >= 0.03 over step 0 while raw FID stays <= 1.5x its step-0
value. The chosen configuration is then run with all three arms x 3 seeds and scored once on test.

## Amendment 1 (2026-10-15, after the first dev round; test still untouched)

Round 1 (diffusion MSE on raw features) optimized the reward (R1 0.42 -> 0.57) but every reward arm
exceeded the FID criterion, so none qualified. Two changes for round 2, decided before any round-2 result:
1. `--reg canon`: the diffusion loss is computed on the canonicalized prediction, so the regularizer no
   longer penalizes the redundant 6D directions (round 1's MSE penalized exactly the channel under test).
2. A lower learning rate (3e-6) alongside 1e-5, to look for a configuration within the FID criterion.
Arms: raw and canonical reward, weight 1, seed 42, 2000 steps. Same qualification criterion.

## Amendment 2 (2026-10-15, after round 2; test still untouched)

Round 2 (fiber-free regularizer, lr 1e-5 and 3e-6) also failed the FID criterion. Round 3 adds the
standard remedy for reward over-optimization: a penalty keeping the fine-tuned model's x0 predictions
close to the frozen pretrained model (DPOK/ReFL-style KL proxy), computed after canonicalizing both, so
the 6D fiber stays free. Weights 10 and 100 (a smoke test showed weight 1 is negligible), raw and
canonical reward, --reg canon, lr 1e-5, weight 1, seed 42, 2000 steps. Same qualification criterion.

## Round 3 outcome and confirmation plan (2026-10-15; test still untouched)

KL weight 100 gives a healthy fine-tune. Canonical arm: reward R1 0.420 -> 0.477, raw FID 0.416 -> 0.285
(qualifies). Raw arm: reward R1 0.501 raw / 0.490 canonical, raw FID 0.660 (just above the 0.62 limit)
but canonical FID 0.244; the raw-FID excess is the encoding effect under study, so the configuration is
taken forward. Confirmation (fixed now, before any test inference): arms raw and canonical (KL 100,
--reg canon, lr 1e-5, weight 1) and an MSE-only control (--reg canon), seeds 42/123/777, 2000 steps,
final checkpoints only, each scored once on the HumanML3D test split with both T2M checkpoints.
Primary endpoints: raw-minus-canonical R1 and FID of the reward evaluator, and non-canonical RMS,
compared between the raw and canonical arms.

## Replication with the calibrated checkpoint (2026-10-15, fixed before training)

Same configuration as the confirmation (KL 100, --reg canon, lr 1e-5, weight 1, 2000 steps, seeds
42/123/777), with the roles of the two T2M checkpoints swapped: the T2M-GPT checkpoint (which
reproduces published real-motion scores) gives the reward, and the MDM-linked checkpoint is the
held-out judge. No setting is retuned. The no-reward control does not depend on the evaluator and
is reused. Final checkpoints are scored once on the same 2,048 test clips with the same scripts.
Endpoints as in the confirmation, with "reward evaluator" now meaning the T2M-GPT checkpoint.
