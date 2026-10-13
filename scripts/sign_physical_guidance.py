"""Remove the differential rotation-null component of an ambient 6D vector."""
import torch

def horizontal(x,v,eps=1e-8):
    assert x.shape==v.shape and x.shape[-1]==6
    a=x[...,:3];na=a.norm(dim=-1,keepdim=True);e1=a/na.clamp_min(eps)
    b=x[...,3:];b=b-(e1*b).sum(-1,keepdim=True)*e1;nb=b.norm(dim=-1,keepdim=True);e2=b/nb.clamp_min(eps)
    p=v[...,:3];q=v[...,3:]
    p=p-(p*e1).sum(-1,keepdim=True)*e1
    q=q-(q*e1).sum(-1,keepdim=True)*e1-(q*e2).sum(-1,keepdim=True)*e2
    result=torch.cat([p,q],dim=-1);valid=(na>eps)&(nb>eps)
    # Undefined rotational derivative at degeneracies: leave field unchanged.
    return torch.where(valid,result,v),valid

def rotation(x):
    a=torch.nn.functional.normalize(x[...,:3],dim=-1);b=x[...,3:];b=torch.nn.functional.normalize(b-(a*b).sum(-1,keepdim=True)*a,dim=-1)
    return torch.stack([a,b,torch.cross(a,b,dim=-1)],dim=-2)

def check():
    gen=torch.Generator().manual_seed(2027);x=torch.randn(512,6,generator=gen,dtype=torch.float64);v=torch.randn(512,6,generator=gen,dtype=torch.float64);h,valid=horizontal(x,v);assert valid.all()
    _,jv=torch.func.jvp(rotation,(x,),(v,));_,jh=torch.func.jvp(rotation,(x,),(h,));_,jvertical=torch.func.jvp(rotation,(x,),(v-h,))
    error=float(abs(jv-jh).max());null=float(abs(jvertical).max());assert error<1e-10 and null<1e-10
    assert float(abs((h*(v-h)).sum(-1)).max())<1e-10
    return {'random_nondegenerate_rotations':512,'max_physical_jvp_difference':error,'max_vertical_jvp':null,'horizontal_vertical_orthogonal':True,'seed':2027,'dtype':'float64'}

if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
