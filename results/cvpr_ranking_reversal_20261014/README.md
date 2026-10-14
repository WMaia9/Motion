# Ranking reversal test (2026-10-14)

Script: `scripts/analyze_ranking_reversal.py` (read-only over frozen per-draw metrics).
Paired t_19 over the 20 shared native draws, Bonferroni over 2 x pairs per reader/metric.

- Main T2M checkpoint, 8 HumanML3D releases: FID ranking identical raw vs canonical (Kendall tau = 1).
- Second checkpoint: MLD/MotionLCM swap in means; change of their gap 0.009 [0.005, 0.013],
  but both orderings unsupported. No reliable reversal in any pair, reader or metric.
