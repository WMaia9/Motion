"""Add recent direct evaluator/reward antecedents to the primary-source archive."""
import json
from concurrent.futures import ThreadPoolExecutor
from fetch_second_review_research import ROOT,SOURCES,fetch
ADDITIONAL=[
('fang2025','https://openaccess.thecvf.com/content/ICCV2025/html/Fang_One_Perturbation_is_Enough_On_Generating_Universal_Adversarial_Perturbations_against_ICCV_2025_paper.html','https://openaccess.thecvf.com/content/ICCV2025/papers/Fang_One_Perturbation_is_Enough_On_Generating_Universal_Adversarial_Perturbations_against_ICCV_2025_paper.pdf'),
('draft2024','https://proceedings.iclr.cc/paper_files/paper/2024/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html','https://proceedings.iclr.cc/paper_files/paper/2024/file/13e8be77982beb73d7ed0bbf122f9f3c-Paper-Conference.pdf'),
('solireward2026','https://openaccess.thecvf.com/content/CVPR2026/html/Lian_SoliReward_Mitigating_Susceptibility_to_Reward_Hacking_and_Annotation_Noise_in_CVPR_2026_paper.html','https://openaccess.thecvf.com/content/CVPR2026/papers/Lian_SoliReward_Mitigating_Susceptibility_to_Reward_Hacking_and_Annotation_Noise_in_CVPR_2026_paper.pdf')]
def main():
 assert not (ROOT/'manifest_sha256.json').exists();path=SOURCES/'research_catalog.json';records=json.loads(path.read_text());seen={d['id'] for d in records}
 with ThreadPoolExecutor(max_workers=3) as pool:records.extend(pool.map(fetch,[d for d in ADDITIONAL if d[0] not in seen]))
 path.write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
