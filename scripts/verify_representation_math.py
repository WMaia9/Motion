"""Check fiber reconstruction and vertical derivatives independently of readers."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.func import jvp
from rotation_equivalence import canonicalize_6d,gauge_transform

ROOT=Path('reports/cvpr_representation_paper_20261011')


def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    rng=np.random.default_rng(42);d=rng.normal(size=(4096,6));x=torch.from_numpy(d)
    c=canonicalize_6d(x);a=x[:,:3];b=x[:,3:];u=c[:,:3];v=c[:,3:]
    alpha=torch.linalg.vector_norm(a,dim=-1,keepdim=True);gamma=(u*b).sum(-1,keepdim=True);beta=(v*b).sum(-1,keepdim=True)
    reconstructed=gauge_transform(c,alpha[:,None],beta[:,None],gamma[:,None])
    residual=float(abs(reconstructed-x).max());assert residual<1e-12
    values=np.exp(rng.uniform(-.35,.35,size=(4096,1,2)));shear=rng.uniform(-.35,.35,size=(4096,1,1))
    transformed=gauge_transform(d,values[...,0:1],values[...,1:2],shear)
    invariance=float(abs(canonicalize_6d(transformed)-canonicalize_6d(d)).max());assert invariance<1e-12
    derivative={}
    zero=torch.zeros_like(a)
    for name,vector in [('scale_a',torch.cat((a,zero),-1)),('scale_b',torch.cat((zero,b),-1)),('shear',torch.cat((zero,a),-1))]:
        _,tangent=jvp(canonicalize_6d,(x,),(vector,));derivative[name]=float(abs(tangent).max());assert derivative[name]<1e-12
    checked=0
    for value in [np.zeros((1,6)),np.array([[1.,0,0,2,0,0]])]:
        try:canonicalize_6d(value)
        except ValueError:checked+=1
        else:raise AssertionError('degenerate input accepted')
    for values in [(np.zeros((1,1,1)),np.ones((1,1,1))),(np.ones((1,1,1)),-np.ones((1,1,1)))]:
        try:gauge_transform(d[:1],*values,np.zeros((1,1,1)))
        except ValueError:checked+=1
        else:raise AssertionError('nonpositive gauge accepted')
    result={'random_non_degenerate_examples':4096,'fiber_reconstruction_max_abs':residual,'canonical_invariance_max_abs':invariance,
        'vertical_JVP_max_abs':derivative,'invalid_input_rejection_checks':checked,'known_geometry_not_new_mathematics':True}
    (ROOT/'math_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
