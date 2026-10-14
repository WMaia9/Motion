"""Equal normalized-L2 motion-feature interventions, independent of evaluator scores.

The budget is ||(y-x)/std||_2 = radius*sqrt(valid_frames) per clip. Radius is
shared across blocks; normalization is reader-specific. Only the null conditions
preserve both rotations and other channels. Other block controls deliberately
hold the remaining features fixed and need not be physically coherent.
"""
import numpy as np

BLOCKS = {'velocity': (193,259), 'contact': (259,263), 'position': (4,67)}
CONDITIONS = ['null_plus','null_minus','null_random','rotation','velocity','contact','position']


def basis(raw):
    d=raw[...,67:193].reshape(*raw.shape[:2],21,6).astype(np.float64)
    a,b=d[...,:3],d[...,3:];u=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-30)
    v=b-(u*b).sum(-1,keepdims=True)*u;v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-30)
    return np.stack([u,v,np.cross(u,v)],axis=-1)


def directions(raw, rep, start, shape):
    return np.stack([np.random.default_rng(20261013+rep*10000+start+i).normal(size=shape) for i in range(len(raw))])


def intervene(raw, lengths, std, condition, radius, rep, start):
    x=raw.astype(np.float64);L=np.asarray(lengths);mask=np.arange(x.shape[1])[None,:]<L[:,None]
    scale=np.asarray(std,dtype=np.float64);assert scale.shape==(263,) and np.all(scale>0)
    target=radius*np.sqrt(L);T=x.shape[1];params={}
    if condition in BLOCKS:
        begin,end=BLOCKS[condition];noise=directions(x,rep,start,(T,end-begin));noise*=mask[...,None]
        norm=np.linalg.norm(noise.reshape(len(x),-1),axis=1);factor=target/norm
        y=x.copy();y[...,begin:end]+=noise*factor[:,None,None]*scale[begin:end]
        params={'scalar':factor}
    else:
        original=x[...,67:193].reshape(len(x),T,21,6);a,b=original[...,:3],original[...,3:]
        if condition.startswith('null'):
            direction=directions(x,rep,start,(T,21,3)) if condition=='null_random' else np.broadcast_to(np.array([1.,-1.,1.])*(1 if condition=='null_plus' else -1),(len(x),T,21,3))
            def changed(s):
                c=s[:,None,None,None]*direction
                return np.concatenate([np.exp(c[...,0:1])*a,np.exp(c[...,1:2])*b+c[...,2:3]*a],-1).reshape(len(x),T,126)
            if condition != 'null_random':
                sign=1 if condition=='null_plus' else -1
                wa=scale[67:193].reshape(21,6)[...,:3];wb=scale[67:193].reshape(21,6)[...,3:]
                def total(z):return (z*mask[...,None]).sum(axis=(1,2))
                AA=total(np.sum((a/wa)**2,-1));BB=total(np.sum((b/wb)**2,-1));CC=total(np.sum((a/wb)**2,-1));BC=total(np.sum(a*b/wb**2,-1))
                def fast_norm(s):
                    ca=np.expm1(sign*s);cb=np.expm1(-sign*s);ch=sign*s
                    return np.sqrt(np.maximum(AA*ca**2+BB*cb**2+CC*ch**2+2*BC*cb*ch,0))
            maximum=2.
        else:
            assert condition=='rotation';axes=directions(x,rep,start,(T,21,3));axes/=np.linalg.norm(axes,axis=-1,keepdims=True)
            K=np.zeros((*axes.shape[:-1],3,3));K[...,0,1]=-axes[...,2];K[...,0,2]=axes[...,1];K[...,1,0]=axes[...,2];K[...,1,2]=-axes[...,0];K[...,2,0]=-axes[...,1];K[...,2,1]=axes[...,0];K2=K@K;R=basis(raw)
            def changed(s):
                theta=s[:,None,None,None,None];Q=np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*K2;rot=R@Q
                return np.concatenate([rot[...,0],rot[...,1]],-1).reshape(len(x),T,126)
            maximum=np.pi/2
            base=np.concatenate([R[...,0],R[...,1]],-1).reshape(len(x),T,126)-x[...,67:193]
            RK=R@K;RK2=R@K2
            first=np.concatenate([RK[...,0],RK[...,1]],-1).reshape(len(x),T,126)
            second=np.concatenate([RK2[...,0],RK2[...,1]],-1).reshape(len(x),T,126)
            fields=[z/scale[67:193]*mask[...,None] for z in [base,first,second]]
            DD,AA,BB=[np.sum(z*z,axis=(1,2)) for z in fields]
            DA=np.sum(fields[0]*fields[1],axis=(1,2));DB=np.sum(fields[0]*fields[2],axis=(1,2));AB=np.sum(fields[1]*fields[2],axis=(1,2))
            def fast_norm(s):
                sn=np.sin(s);cs=1-np.cos(s)
                return np.sqrt(np.maximum(DD+AA*sn**2+BB*cs**2+2*DA*sn+2*DB*cs+2*AB*sn*cs,0))
        def norm(s):
            delta=(changed(s)-x[...,67:193])/scale[67:193];delta*=mask[...,None]
            return np.linalg.norm(delta.reshape(len(x),-1),axis=1)
        search_norm=norm if condition=='null_random' else fast_norm
        lo=np.zeros(len(x));hi=np.full(len(x),maximum);assert np.all(search_norm(hi)>target),'Registered radius cannot be reached in declared domain'
        for _ in range(28):
            mid=(lo+hi)/2;below=search_norm(mid)<target;lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        scalar=(lo+hi)/2;y=x.copy();y[...,67:193]=changed(scalar);params={'scalar':scalar}
    y[~mask]=x[~mask];y=y.astype(raw.dtype)
    # Restoring unaltered blocks by assignment preserves their exact input bits.
    if condition in BLOCKS:begin,end=BLOCKS[condition]
    else:begin,end=67,193
    y[...,:begin]=raw[...,:begin];y[...,end:]=raw[...,end:];y[~mask]=raw[~mask]
    actual=np.linalg.norm((((y.astype(float)-x)/scale)*mask[...,None]).reshape(len(x),-1),axis=1)
    assert np.max(abs(actual-target)/target)<5e-6,(condition,float(np.max(abs(actual-target)/target)))
    assert np.isfinite(y).all()
    return y,params,actual
