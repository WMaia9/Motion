"""The portable defense must preserve geometry and reject undefined rotations."""
import unittest
import numpy as np
from tools.canonicalize_motion import canonicalize6d,canonicalize_motion

class CanonicalizationContract(unittest.TestCase):
    def test_full_fiber_and_idempotence(self):
        rng=np.random.default_rng(1);d=rng.normal(size=(100,6));c=canonicalize6d(d)
        a,b=d[:,:3],d[:,3:];r=rng.uniform(-4,4,size=(100,3))
        recoded=np.concatenate([np.exp(r[:,:1])*a,np.exp(r[:,1:2])*b+r[:,2:3]*a],-1)
        np.testing.assert_allclose(canonicalize6d(recoded),c,atol=1e-12)
        np.testing.assert_allclose(canonicalize6d(c),c,atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(c[:,:3],axis=-1),1,atol=1e-14)
        np.testing.assert_allclose((c[:,:3]*c[:,3:]).sum(-1),0,atol=1e-14)
    def test_native_schemas_and_padding(self):
        rng=np.random.default_rng(2)
        for width in [263,251]:
            x=rng.normal(size=(2,12,width)).astype(np.float32);L=np.array([8,12]);x[0,8:]=0
            y=canonicalize_motion(x,L);j=(width-11)//12;s=4+3*j;e=s+6*j
            self.assertTrue(np.array_equal(x[...,:s].view('u4'),y[...,:s].view('u4')))
            self.assertTrue(np.array_equal(x[...,e:].view('u4'),y[...,e:].view('u4')))
            self.assertTrue(np.array_equal(x[0,8:].view('u4'),y[0,8:].view('u4')))
            self.assertTrue(np.isfinite(y).all())
    def test_invalid_domain(self):
        for x in [[0,0,0,1,0,0],[1,0,0,2,0,0],[1,0,0,np.nan,1,0]]:
            with self.assertRaises(ValueError):canonicalize6d(np.array(x,dtype=float))
    def test_finite_inputs_cannot_produce_invalid_rotation_on_norm_overflow(self):
        # Finite coordinates alone do not guarantee a finite computed norm.
        # Reject numerical overflow instead of returning a zero basis vector.
        with np.errstate(over='ignore',invalid='ignore'):
            with self.assertRaises(ValueError):
                canonicalize6d(np.array([[1e308,0.,0.,0.,1.,0.]]))

if __name__=='__main__':unittest.main()
