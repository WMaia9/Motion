"""Independent NumPy sums and falsifiable geometric invariants of path losses."""
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from run_motion_path_diffusion import ROOT, signatures
from run_physical_semantic_ddpm import physical


def numpy_signature(path):
    delta=np.diff(path,axis=0);area=np.zeros((40,path.shape[-1],path.shape[-1]))
    for j in range(len(delta)):
        for i in range(j):
            product=np.einsum('jd,je->jde',delta[i],delta[j])
            area+=.5*(product-product.transpose(0,2,1))
    triangle=np.triu_indices(path.shape[-1],1)
    return path[-1,...,-6:]-path[0,...,-6:],area[:,triangle[0],triangle[1]]


def check():
    generator=torch.Generator().manual_seed(2027)
    raw=torch.randn(4,9,240,generator=generator,dtype=torch.float64)
    canonical=physical(raw);lengths=torch.tensor([9,7,5,8]);errors={}
    for clock in [False,True]:
        first,area=signatures(canonical,lengths,clock);maximum=0.
        for i,n in enumerate(lengths.tolist()):
            path=canonical[i,:n].numpy().reshape(n,40,6)
            if clock:path=np.concatenate([np.broadcast_to(np.linspace(0,1,n)[:,None,None],(n,40,1)),path],axis=-1)
            nf,na=numpy_signature(path)
            maximum=max(maximum,float(abs(nf-first[i].numpy()).max()),float(abs(na-area[i].numpy()).max()))
        assert maximum<1e-11;errors['numpy_'+('time_' if clock else '')+'signature_max_error']=maximum
    params=torch.randn(4,9,40,3,generator=generator,dtype=torch.float64)
    d=raw.reshape(4,9,40,6)
    transformed=torch.cat([torch.exp(.2*params[...,0:1])*d[...,:3],
        torch.exp(.2*params[...,1:2])*d[...,3:]+.3*params[...,2:3]*d[...,:3]],-1).reshape(raw.shape)
    before=signatures(physical(raw),lengths,True);after=signatures(physical(transformed),lengths,True)
    gauge_error=max(float((a-b).abs().max()) for a,b in zip(before,after));assert gauge_error<1e-10
    direction=torch.cat([params[...,0:1]*d[...,:3],params[...,1:2]*d[...,3:]+params[...,2:3]*d[...,:3]],-1).reshape(raw.shape)
    def feature(value):
        first,area=signatures(physical(value),lengths,True)
        return torch.cat([first.flatten(1),area.flatten(1)],-1)
    _,jvp=torch.func.jvp(feature,(raw,),(direction,))
    jvp_error=float(jvp.abs().max());assert jvp_error<1e-10
    rotations=np.stack([np.eye(3),Rotation.from_rotvec([.4,0,0]).as_matrix(),
        Rotation.from_rotvec([0,.4,0]).as_matrix(),np.eye(3)])
    loop=np.tile(rotations[:,:2,:].reshape(4,1,6),(1,40,1)).reshape(1,4,240)
    loop=torch.as_tensor(loop,dtype=torch.float64);reversed_loop=loop.flip(1)
    lf,la=signatures(loop,torch.tensor([4]));rf,ra=signatures(reversed_loop,torch.tensor([4]))
    assert not lf.any() and not rf.any();assert (la+ra).abs().max()<1e-12
    assert (la-ra).norm()>1e-3
    assert abs(float(loop.diff(dim=1).square().sum())-float(reversed_loop.diff(dim=1).square().sum()))<1e-12
    path=canonical[0]
    refined=torch.stack([z for i in range(len(path)-1) for z in [path[i],(path[i]+path[i+1])/2]]+[path[-1]])
    original=signatures(path[None],torch.tensor([len(path)]));subdivided=signatures(refined[None],torch.tensor([len(refined)]))
    subdivision_error=max(float((a-b).abs().max()) for a,b in zip(original,subdivided));assert subdivision_error<1e-10
    return {**errors,'finite_gauge_invariance_max_error':gauge_error,
        'vertical_signature_jvp_max_error':jvp_error,'closed_reversed_loop_same_endpoints_and_velocity_energy':True,
        'closed_loop_area_difference_norm':float((la-ra).norm()),
        'polygonal_subdivision_max_error':subdivision_error,
        'subdivision_is_embedding_polyline_not_SO3_geodesic':True,
        'truncated_signature_does_not_uniquely_identify_signs':True,'dtype':'float64','seed':2027}


if __name__=='__main__':
    result=check();(ROOT/'sources/math_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PATH_MATH_VERIFIED',json.dumps(result),flush=True)
