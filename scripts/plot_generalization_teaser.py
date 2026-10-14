"""Procedural body rendering of saved joint positions; no licensed body asset."""
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from motion_release_audit_common import read,write,sha
P=Path('paper/current');R=Path('reports/cvpr_generalization_review_20261013');OLD=Path('reports/cvpr_humanml3d_audit_20261012')
plt.rcParams.update({'pdf.fonttype':42,'font.family':'Liberation Sans'})

def sphere(ax,p,r,color):
 u,v=np.meshgrid(np.linspace(0,2*np.pi,16),np.linspace(0,np.pi,10));x=p[0]+r*np.cos(u)*np.sin(v);y=p[1]+r*np.sin(u)*np.sin(v);z=p[2]+r*np.cos(v);ax.plot_surface(x,y,z,color=color,linewidth=0,antialiased=True,shade=True)

def capsule(ax,a,b,r,color):
 direction=b-a;length=np.linalg.norm(direction)
 if length<1e-8:return
 direction/=length;first=np.cross(direction,[1.,0.,0.])
 if np.linalg.norm(first)<1e-6:first=np.cross(direction,[0.,1.,0.])
 first/=np.linalg.norm(first);second=np.cross(direction,first);angles=np.linspace(0,2*np.pi,16);circle=r*(np.cos(angles)[:,None]*first+np.sin(angles)[:,None]*second);surface=np.stack([a+circle,b+circle]);ax.plot_surface(surface[:,:,0],surface[:,:,1],surface[:,:,2],color=color,linewidth=0,antialiased=True,shade=True)
 sphere(ax,a,r,color);sphere(ax,b,r,color)

def render(positions,chains):
 # World axis y is vertical; plotting coordinates are(x,z,y). The same
 # root-centered camera and procedural radii apply to both feature arrays.
 xyz=positions[:,[0,2,1]].copy();xyz[:,:2]-=xyz[0,:2];color='#4d7897';fig=plt.figure(figsize=(2.2,2.8),dpi=150);ax=fig.add_axes([0,0,1,1],projection='3d')
 for chain in chains:
  for parent,joint in zip(chain[:-1],chain[1:]):
   r=.068 if joint in [1,2,4,5] else .047 if joint in [7,8,10,11] else .042 if joint in [16,17,18,19,20,21] else .075
   capsule(ax,xyz[parent],xyz[joint],r,color)
 capsule(ax,xyz[0],xyz[9],.105,color);sphere(ax,xyz[15],.105,color)
 a,b=np.meshgrid(np.linspace(-.65,.65,2),np.linspace(-.65,.65,2));ax.plot_surface(a,b,np.full_like(a,float(xyz[:,2].min())-.035),color='#e5e9ec',alpha=.6,shade=False)
 ax.set(xlim=(-.65,.65),ylim=(-.65,.65),zlim=(-.1,2.));ax.view_init(elev=12,azim=-58);ax.set_axis_off();ax.set_box_aspect((1,1,1.62));fig.canvas.draw();pixels=np.asarray(fig.canvas.buffer_rgba()).copy();plt.close(fig)
 # Trim plotting whitespace equally; the body vertices remain unchanged.
 visible=np.any(pixels[:,:,:3]<245,axis=-1);rows,cols=np.where(visible);margin=8
 return pixels[max(0,rows.min()-margin):min(len(pixels),rows.max()+margin+1),max(0,cols.min()-margin):min(pixels.shape[1],cols.max()+margin+1)].copy()

def main():
 assert not (R/'manifest_sha256.json').exists(), 'Presentation delivery is frozen; register a new delivery path before writing'
 sys.path.insert(0,str(Path('external/MDM').resolve()))
 from data_loaders.humanml.scripts.motion_process import recover_from_ric
 from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
 item=read(OLD/'test_selection.json')[0];n=item['frames'];qual=R/'qualitative';qual.mkdir(exist_ok=True)
 with np.load(OLD/'test_mdm.npz') as z:X=z['raw'][0,:n].copy()
 with np.load(OLD/'test/mdm/target_caption.npz') as z:Y=z['raw'][0,:n].copy()
 original=recover_from_ric(torch.tensor(X),22).numpy();changed=recover_from_ric(torch.tensor(Y),22).numpy();assert np.array_equal(original,changed)
 images=[];records=[]
 for q in [.2,.5,.8]:
  t=int(q*(n-1));a=render(original[t],t2m_kinematic_chain);b=render(changed[t],t2m_kinematic_chain);assert np.array_equal(a,b)
  for name,pixels in [('original',a),('recoded',b)]:plt.imsave(qual/f'{name}_{t:03d}.png',pixels)
  images.append(a);records.append({'frame':t,'fraction':q,'body_render_pixels_bitwise_equal':True})
 fig=plt.figure(figsize=(13.6,4.55));canvas=fig.add_axes([0,0,1,1]);canvas.set_axis_off();canvas.text(.015,.98,'Same movement. Different reported scores.',fontsize=23,fontweight='bold',va='top');canvas.text(.015,.86,'“'+item['caption']+'”',fontsize=14,color='#526174')
 for row,label in enumerate(['Original','Recoded']):
  y=.435-row*.33;canvas.text(.015,y+.18,label,fontsize=11,color='#526174',va='center')
  for j,pixels in enumerate(images):
   ax=fig.add_axes([.095+j*.142,y,.142,.31]);ax.imshow(pixels);ax.set_axis_off()
   if row==0:ax.set_title(f'{[20,50,80][j]}% of clip',fontsize=9,pad=1)
 for y,color,title,score in [(.51,'#265f91','Original rotation encoding','R1 0.397'),(.26,'#c9692d','Equivalent rotation encoding','R1 0.993')]:
  canvas.add_patch(FancyBboxPatch((.555,y),.41,.2,boxstyle='round,pad=.012',edgecolor=color,facecolor='white',linewidth=1.7));canvas.text(.572,y+.153,title,fontsize=15,fontweight='bold',color=color,va='top');canvas.text(.572,y+.085,score,fontsize=21,va='top')
 canvas.text(.555,.15,'Caption-targeted attack · aggregate of 1,024 MDM clips',fontsize=10.5,color='#526174');canvas.text(.555,.078,'Canonicalize first: both embeddings become the same',fontsize=11.5,color='#267a64');canvas.text(.095,.035,'Identical procedural body renders from protected joint positions',fontsize=10,color='#267a64')
 target=P/'figures/teaser.pdf';fig.savefig(target,bbox_inches='tight',pad_inches=.03);fig.savefig(R/'teaser_preview.png',dpi=150,bbox_inches='tight',pad_inches=.03);plt.close(fig)
 write(R/'teaser_evidence.json',{'source_clip':item['clip_id'],'illustration_index':0,'caption':item['caption'],'selection':'first fixed TEST record, unchanged from previous teaser, no result selection','rendering':'procedural spheres/cylinders following native RIC joint positions, fixed camera/radii/colors; no SMPL/MANO model or licensed avatar asset','positions_bitwise_equal':True,'renders':records,'scores_scope':'aggregate1024 MDM outputs from original frozen target-aware study, not singleclip or TMR','inputs_sha256':{str(f):sha(f) for f in [OLD/'test_mdm.npz',OLD/'test/mdm/target_caption.npz',OLD/'test_analysis.json']},'figure_sha256':sha(target),'producer_sha256':sha(__file__)})
 print('PROCEDURAL_TEASER_VERIFIED',flush=True)
if __name__=='__main__':main()
