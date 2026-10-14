"""Caption-free controls for real rotation-only sign inputs.

Gaussian controls match each clip's actual canonicalization displacement. The
primary Gaussian protects the common preprocessed left-hand stream. Physical
controls rotate original vectors together, preserving their Gram coefficients.
"""
import numpy as np

SEEDS=[20261014,20261015,20261016]
CONDITIONS=['raw','canonical']+[f'gaussian_support_{s}' for s in SEEDS]+['gaussian_all_20261014','physical_1deg','physical_matched']
ACTIVE=np.r_[np.arange(10),np.arange(25,40)]


def canonical(x):
    d=x.reshape(-1,40,6).astype(float);a=d[...,:3]
    u=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    v=d[...,3:]-(u*d[...,3:]).sum(-1,keepdims=True)*u
    v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
    return np.concatenate([u,v],axis=-1).reshape(x.shape).astype('float32')


def transform(x,condition,index,corpus_index):
    source=x.astype(float);C=canonical(x);budget=float(np.linalg.norm(C.astype(float)-source))
    if condition=='raw':return x.copy(),{'budget':budget,'scalar':0.,'feasible':True}
    if condition=='canonical':return C,{'budget':budget,'scalar':1.,'feasible':True}
    d=source.reshape(-1,40,6)
    if condition.startswith('gaussian'):
        seed=int(condition.rsplit('_',1)[1]);noise=np.random.default_rng(seed+corpus_index*1000000+index).normal(size=d.shape)
        if condition.startswith('gaussian_support'):noise[:,10:25]=0.
        scalar=budget/np.linalg.norm(noise) if budget>0 else 0.
        y=(d+scalar*noise).reshape(x.shape).astype('float32')
    else:
        axis=np.random.default_rng(20261014+corpus_index*1000000+index).normal(size=(40,3));axis/=np.linalg.norm(axis,axis=1,keepdims=True)
        # Row vectors use a right multiplication; rotate both rows together.
        K=np.zeros((40,3,3));a,b,c=axis.T
        K[:,0,1]=-c;K[:,0,2]=b;K[:,1,0]=c;K[:,1,2]=-a;K[:,2,0]=-b;K[:,2,1]=a
        rows=d.reshape(len(x),40,2,3)
        perp=rows-(rows*axis[None,:,None,:]).sum(-1,keepdims=True)*axis[None,:,None,:]
        A=float(np.sum(perp[:,ACTIVE]**2))
        if condition=='physical_1deg':scalar=np.pi/180
        else:
            assert condition=='physical_matched'
            if budget**2>4*A+1e-10:return None,{'budget':budget,'scalar':None,'feasible':False,'reason':'canonicalization norm exceeds rigid-vector SO3 range'}
            scalar=2*np.arcsin(min(1.,budget/(2*np.sqrt(A)))) if A>0 else 0.
        Q=np.eye(3)+np.sin(scalar)*K+(1-np.cos(scalar))*(K@K)
        y=(rows@Q[None,:,:,:]).reshape(x.shape).astype('float32')
    if condition.startswith('gaussian_support') or condition.startswith('physical'):y[:,60:150]=x[:,60:150]
    actual=float(np.linalg.norm(y.astype(float)-source))
    if condition!='physical_1deg':assert abs(actual-budget)<=5e-6*max(1.,budget),(condition,index,actual,budget)
    assert np.isfinite(y).all()
    return y,{'budget':budget,'scalar':float(scalar),'realized_L2':actual,'feasible':True}
