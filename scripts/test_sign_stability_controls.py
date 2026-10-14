"""Check that matched physical controls preserve gauge and change rotations."""
import unittest
import numpy as np
from scripts.sign_stability_controls import transform,canonical,CONDITIONS


class SignStability(unittest.TestCase):
    def test_geometry_norm_and_protected_support(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(31,240)).astype('float32');x=canonical(x)
        d=x.reshape(-1,40,6);d[:,:10,:3]*=1.02;d[:,25:,3:]+=0.025*d[:,25:,:3]
        for condition in CONDITIONS:
            y,p=transform(x,condition,3,0);self.assertTrue(p['feasible'])
            if condition!='physical_1deg' and condition!='raw':self.assertAlmostEqual(np.linalg.norm(y.astype(float)-x),p['budget'],places=5)
            if condition.startswith('gaussian_support') or condition.startswith('physical'):np.testing.assert_array_equal(y[:,60:150],x[:,60:150])
            if condition.startswith('physical'):
                a=x.reshape(-1,40,2,3).astype(float);b=y.reshape(-1,40,2,3).astype(float)
                np.testing.assert_allclose(a@np.swapaxes(a,-1,-2),b@np.swapaxes(b,-1,-2),atol=2e-7,rtol=0)
                self.assertGreater(np.max(abs(canonical(x)-canonical(y))),1e-3)
            z,_=transform(x,condition,3,0);np.testing.assert_array_equal(y,z)


if __name__=='__main__':unittest.main()
