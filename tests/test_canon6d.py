"""Contract tests for canon6d: exact invariance, idempotence and preserved channels."""
import importlib.util
import unittest

import numpy as np

from canon6d import canonicalize, canonicalize6d, recode, rotation_block, rotation_matrix


class TestCanonicalize6D(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.d = self.rng.normal(size=(500, 6))

    def test_recoding_preserves_rotation(self):
        c = self.rng.uniform(-2, 2, size=(500, 3))
        r = recode(self.d, c[:, 0], c[:, 1], c[:, 2])
        np.testing.assert_allclose(rotation_matrix(r), rotation_matrix(self.d), atol=1e-10)

    def test_all_encodings_share_one_canonical_form(self):
        c = self.rng.uniform(-2, 2, size=(500, 3))
        r = recode(self.d, c[:, 0], c[:, 1], c[:, 2])
        np.testing.assert_allclose(canonicalize6d(r), canonicalize6d(self.d), atol=1e-10)

    def test_idempotent_and_orthonormal(self):
        c = canonicalize6d(self.d)
        np.testing.assert_allclose(canonicalize6d(c), c, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(c[:, :3], axis=-1), 1, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(c[:, 3:], axis=-1), 1, atol=1e-14)
        np.testing.assert_allclose((c[:, :3] * c[:, 3:]).sum(-1), 0, atol=1e-14)

    def test_rejects_undefined_rotations(self):
        for bad in ([0, 0, 0, 1, 0, 0], [1, 0, 0, 2, 0, 0], [1, 0, 0, np.nan, 1, 0]):
            with self.assertRaises(ValueError):
                canonicalize6d(np.array(bad, dtype=float))


class TestMotionFeatures(unittest.TestCase):
    def test_rotation_blocks(self):
        self.assertEqual(rotation_block(263), slice(67, 193))
        self.assertEqual(rotation_block(251), slice(64, 184))
        with self.assertRaises(ValueError):
            rotation_block(240)

    def test_only_rotation_channels_change_and_padding_is_kept(self):
        rng = np.random.default_rng(1)
        for width in (263, 251):
            x = rng.normal(size=(3, 20, width)).astype(np.float32)
            lengths = np.array([12, 20, 5])
            y = canonicalize(x, lengths)
            blk = rotation_block(width)
            self.assertTrue(np.array_equal(x[..., :blk.start].view('u4'), y[..., :blk.start].view('u4')))
            self.assertTrue(np.array_equal(x[..., blk.stop:].view('u4'), y[..., blk.stop:].view('u4')))
            self.assertTrue(np.array_equal(x[0, 12:].view('u4'), y[0, 12:].view('u4')))
            np.testing.assert_allclose(canonicalize(y, lengths), y, atol=1e-6)

    def test_matches_the_implementation_used_in_the_paper(self):
        spec = importlib.util.spec_from_file_location('legacy', 'tools/canonicalize_motion.py')
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        x = np.random.default_rng(2).normal(size=(4, 30, 263)).astype(np.float32)
        lengths = np.array([30, 10, 1, 25])
        self.assertTrue(np.array_equal(canonicalize(x, lengths), legacy.canonicalize_motion(x, lengths)))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class TestTorch(unittest.TestCase):
    def test_torch_matches_numpy_and_is_differentiable(self):
        import torch
        from canon6d.torch import canonicalize as canonicalize_torch
        x = np.random.default_rng(3).normal(size=(2, 8, 263))
        t = torch.tensor(x, requires_grad=True)
        y = canonicalize_torch(t)
        np.testing.assert_allclose(y.detach().numpy(), canonicalize(x), atol=1e-10)
        y.sum().backward()
        self.assertTrue(torch.isfinite(t.grad).all())


if __name__ == '__main__':
    unittest.main()
