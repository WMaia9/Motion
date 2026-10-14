"""Check norm fairness, geometry and frame/channel protection before scoring."""
import unittest
import numpy as np
from scripts.matched_motion_blocks import intervene,basis,CONDITIONS,BLOCKS

class MatchedBlocks(unittest.TestCase):
    def test_exact_null_geometry_and_equal_norm(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(3,12,263)).astype('float32');L=np.array([12,8,5]);R=basis(x);x[...,67:193]=np.concatenate([R[...,0],R[...,1]],-1).reshape(3,12,126);std=rng.uniform(.2,.8,263)
        for radius in [1.,3.]:
            for condition in CONDITIONS:
                y,p,n=intervene(x,L,std,condition,radius,0,0);np.testing.assert_allclose(n,radius*np.sqrt(L),rtol=5e-6)
                mask=np.arange(12)[None,:]<L[:,None];np.testing.assert_array_equal(y[~mask],x[~mask]);start,end=BLOCKS.get(condition,(67,193));np.testing.assert_array_equal(y[...,:start],x[...,:start]);np.testing.assert_array_equal(y[...,end:],x[...,end:])
                if condition.startswith('null'):np.testing.assert_allclose(basis(y)[mask],basis(x)[mask],atol=2e-7,rtol=0)
                if condition=='rotation':self.assertGreater(np.max(abs(basis(y)[mask]-basis(x)[mask])),.01)
                z,_,_=intervene(x,L,std,condition,radius,0,0);np.testing.assert_array_equal(z,y)

if __name__=='__main__':unittest.main()
