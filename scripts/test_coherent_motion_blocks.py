"""Check matched norms, temporal directions, rotations and protected inputs."""
import unittest
import numpy as np
from scripts.coherent_motion_blocks import intervene, CONDITIONS, AXIS
from scripts.matched_motion_blocks import basis, BLOCKS


class CoherentBlocks(unittest.TestCase):
    def test_fixed_directions_and_budget(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(3,12,263)).astype('float32');L=np.array([12,8,5])
        R=basis(x);x[...,67:193]=np.concatenate([R[...,0],R[...,1]],-1).reshape(3,12,126)
        std=rng.uniform(.2,.8,263);mask=np.arange(12)[None,:]<L[:,None]
        for radius in [1.,3.]:
            for condition in CONDITIONS:
                y,s,n=intervene(x,L,std,condition,radius)
                np.testing.assert_allclose(n,radius*np.sqrt(L),rtol=5e-6)
                block=condition.split('_')[0];begin,end=BLOCKS.get(block,(67,193))
                np.testing.assert_array_equal(y[~mask],x[~mask])
                np.testing.assert_array_equal(y[...,:begin],x[...,:begin]);np.testing.assert_array_equal(y[...,end:],x[...,end:])
                if block=='rotation':
                    Q=np.swapaxes(basis(x),-1,-2)@basis(y)
                    for i,l in enumerate(L):
                        np.testing.assert_allclose(Q[i,:l],np.broadcast_to(Q[i,0,0],Q[i,:l].shape),atol=2e-7,rtol=0)
                    self.assertGreater(np.max(abs(Q[mask]-np.eye(3))),.01)
                else:
                    d=y[...,begin:end].astype(float)-x[...,begin:end]
                    for i,l in enumerate(L):np.testing.assert_allclose(d[i,:l],np.broadcast_to(d[i,0],d[i,:l].shape),atol=3e-7,rtol=0)
                z,_,_=intervene(x,L,std,condition,radius);np.testing.assert_array_equal(z,y)


if __name__=='__main__':unittest.main()
