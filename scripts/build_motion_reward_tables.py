"""Supplement tables from complete reward results; no fitting or selection."""
import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
from build_motion_release_presentation import table
root=Path('reports/cvpr_humanml_reward_pilot_20261012');rows=[]
for seed in [42,123,777]:
 for arm,label in [('mse_only','MSE'),('raw_reward','Raw reward'),('canonical_reward','Projected reward')]:
  x=json.loads((root/f'seed_{seed}'/arm/'metrics.json').read_text())['results'];a=x['raw'];b=x['canonical']
  rows.append(f"{seed}&{label}&{a['FID']:.3f}&{b['FID']:.3f}&{a['R_precision'][0]:.4f}&{b['R_precision'][0]:.4f}\\\\")
table('table_motion_reward_pilot',r'\textbf{Ordinary MDM reward pilot: every fit.} Three training seeds, three arms, one common generation seed, 1,024 outputs and 4,640 real-reference embeddings. No added coefficient layer. The raw-reward arm does not inflate retrieval. This is a denoised-$x_0$ reward experiment, not a full rollout/RFT reproduction.','tab:rewardpilot','rlrrrr',r'Seed&Training arm&Raw FID&$C$ FID&Raw R1&$C$ R1',rows)
x=json.loads(Path('reports/cvpr_motion_reward_release_20261013/analysis.json').read_text());rows=[]
for reader,label in [('common_MDM_checkpoint','MDM reader'),('official_GPT_checkpoint','GPT reader')]:
 for method,name in [('base','Linked base'),('reward_finetuned','Reward-tuned')]:
  r=x['readers'][reader][method];f=r['FID'];q=r['R1'];a,b=f['paired_family_CI95'];c,d=q['paired_family_CI95']
  rows.append(f"{label}&{name}&{f['raw_mean']:.3f}&{f['canonical_mean']:.3f}&$[{a:+.4f},{b:+.4f}]$&{q['raw_mean']:.4f}&{q['canonical_mean']:.4f}&$[{c:+.4f},{d:+.4f}]$\\\\")
table('table_motion_reward_release',r'\textbf{Released MotionRFT and its author-linked base.} Five matched generation seeds; 1,024 outputs and 4,640 real-reference embeddings per repetition, two official readers. Eight raw$-C$ endpoints share one interval family. This supplemental diagnostic is separate from the 20-repetition audit and published leaderboard numbers. Retrieval gains survive canonicalization.','tab:rewardrelease','llrrrrrr',r'Reader&Release&Raw FID&$C$ FID&FID gap interval&Raw R1&$C$ R1&R1 gap interval',rows,wide=True)
