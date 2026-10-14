# Query-only (black-box) caption-blind attack (2026-10-14)

Script: `scripts/run_blackbox_fid_attack.py`. Attacker sees only the scalar FID against 2,048
training-motion embeddings (primary T2M); (1+1)-ES; budgets 10/30/100/300; bounds 0.1/0.35;
3 global or 63 per-joint coefficients. Best coefficients scored once on the withheld test
reference with the primary reader and the unqueried second checkpoint. Results: `results.json`.

Per-joint, b=0.1, 300 queries: queried test FID -20/-27/-11/-57% (MDM42/4242/7777/MLD);
second checkpoint -30/-4/-22/-41%. b=0.35: queried -29..-68%, second checkpoint worse.
Random coefficients of the same bound raise FID. R1 0.41-0.44 throughout.
