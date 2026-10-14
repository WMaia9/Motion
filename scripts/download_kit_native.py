"""Official HumanML3D repository's KIT-ML vector release, no AMASS meshes."""
from pathlib import Path
import gdown,subprocess
from motion_release_audit_common import write,sha,read
R=Path('reports/cvpr_generalization_review_20261013');D=Path('models/generalization_audit/KIT-ML')
def main():
 D.mkdir(parents=True,exist_ok=True);records=[]
 for item in read(R/'kit_data_public_listing.json'):
  name=item['name']
  if name=='new_joints.rar':continue
  p=D/name
  if not p.exists():gdown.download(id=item['id'],output=str(p),quiet=True,resume=True)
  assert p.is_file();records.append({**item,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size});print('KIT_DOWNLOAD',name,p.stat().st_size,flush=True)
  if p.suffix=='.rar':
   subprocess.run(['unrar','t','-inul',str(p)],check=True);subprocess.run(['unrar','x','-o-','-inul',str(p),str(D)+'/'],check=True)
 write(R/'kit_downloads.json',{'source':'author-linked HumanML3D README KIT folder','files':records,'mesh_models_used':False});print('KIT_DATA_COMPLETE',flush=True)
if __name__=='__main__':main()
