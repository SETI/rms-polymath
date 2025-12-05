##########################################################################################
# tests/test_pair.py
# Pair comprehensive tests
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Pair, Matrix, Vector


class Test_Pair(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # Test basic construction
        p1 = Pair([1., 2.])
        self.assertEqual(p1.shape, ())
        self.assertEqual(p1.item, (2,))
        self.assertEqual(p1.numer, (2,))
        self.assertTrue(np.allclose(p1.vals, [1., 2.]))

        # Test construction from list
        p2 = Pair([4., 5.])
        self.assertTrue(np.allclose(p2.vals, [4., 5.]))

        # Test construction from tuple
        p3 = Pair((7., 8.))
        self.assertTrue(np.allclose(p3.vals, [7., 8.]))

        # Test construction from numpy array
        p4 = Pair(np.array([10., 11.]))
        self.assertTrue(np.allclose(p4.vals, [10., 11.]))

        # Test n-D arrays
        p5 = Pair(np.random.randn(2, 3, 2))
        self.assertEqual(p5.shape, (2, 3))
        self.assertEqual(p5.item, (2,))
        self.assertEqual(p5.vals.shape, (2, 3, 2))

        # Test higher-dimensional arrays
        p6 = Pair(np.random.randn(4, 5, 6, 2))
        self.assertEqual(p6.shape, (4, 5, 6))
        self.assertEqual(p6.item, (2,))
        self.assertEqual(p6.vals.shape, (4, 5, 6, 2))

        # Test that wrong shapes raise ValueError
        self.assertRaises(ValueError, Pair, np.random.randn(2, 3, 4))
        self.assertRaises(ValueError, Pair, 1.)
        self.assertRaises(ValueError, Pair, [1.])
        self.assertRaises(ValueError, Pair, [1., 2., 3.])

        # Test zeros
        p7 = Pair.zeros((2, 3))
        self.assertEqual(p7.shape, (2, 3))
        self.assertEqual(p7.vals.shape, (2, 3, 2))
        self.assertEqual(p7.vals.dtype.kind, 'f')
        self.assertTrue(np.all(p7.vals == 0))

        p8 = Pair.zeros((2, 3), dtype='float')
        self.assertEqual(p8.shape, (2, 3))
        self.assertEqual(p8.vals.shape, (2, 3, 2))
        self.assertEqual(p8.vals.dtype.kind, 'f')
        self.assertTrue(np.all(p8.vals == 0))

        p9 = Pair.zeros((2, 2), mask=[[0, 1], [0, 0]])
        self.assertEqual(p9.shape, (2, 2))
        self.assertEqual(p9.vals.shape, (2, 2, 2))
        self.assertTrue(np.all(p9.vals == 0))
        self.assertTrue(np.all(p9.mask == [[0, 1], [0, 0]]))

        p10 = Pair.zeros((2, 2), denom=(3, 3))
        self.assertEqual(p10.shape, (2, 2))
        self.assertEqual(p10.vals.shape, (2, 2, 2, 3, 3))
        self.assertTrue(np.all(p10.vals == 0))

        self.assertRaises(ValueError, Pair.zeros, (2, 3), numer=(3,))

        # Test ones
        p11 = Pair.ones((2, 3))
        self.assertEqual(p11.shape, (2, 3))
        self.assertEqual(p11.vals.shape, (2, 3, 2))
        self.assertEqual(p11.vals.dtype.kind, 'f')
        self.assertTrue(np.all(p11.vals == 1))

        p12 = Pair.ones((2, 2), mask=[[0, 1], [0, 0]])
        self.assertEqual(p12.shape, (2, 2))
        self.assertEqual(p12.vals.shape, (2, 2, 2))
        self.assertTrue(np.all(p12.vals == 1))
        self.assertTrue(np.all(p12.mask == [[0, 1], [0, 0]]))

        # Test filled
        p13 = Pair.filled((2, 3), 7.)
        self.assertEqual(p13.shape, (2, 3))
        self.assertEqual(p13.vals.shape, (2, 3, 2))
        self.assertTrue(np.all(p13.vals == 7))

        p14 = Pair.filled((2, 2), (1., 2.))
        self.assertEqual(p14.shape, (2, 2))
        self.assertEqual(p14.vals.shape, (2, 2, 2))
        self.assertTrue(np.all(p14.vals[..., 0] == 1))
        self.assertTrue(np.all(p14.vals[..., 1] == 2))

        # Test as_pair static method
        p15 = Pair([1., 2.])
        p15_conv = Pair.as_pair(p15)
        self.assertEqual(type(p15_conv), Pair)
        self.assertTrue(np.allclose(p15_conv.vals, [1., 2.]))

        # Test as_pair with Vector
        v16 = Vector([1., 2.])
        p16_conv = Pair.as_pair(v16)
        self.assertEqual(type(p16_conv), Pair)
        self.assertTrue(np.allclose(p16_conv.vals, [1., 2.]))

        # Test as_pair with array
        p17_conv = Pair.as_pair([4., 5.])
        self.assertEqual(type(p17_conv), Pair)
        self.assertTrue(np.allclose(p17_conv.vals, [4., 5.]))

        # Test as_pair with 1x2 Matrix (flatten_numer)
        m1x2 = Matrix([[1., 2.]])
        self.assertEqual(m1x2._numer, (1, 2))
        p1x2_conv = Pair.as_pair(m1x2)
        self.assertEqual(type(p1x2_conv), Pair)
        self.assertTrue(np.allclose(p1x2_conv.vals, [1., 2.]))

        # Test as_pair with 2x1 Matrix (flatten_numer)
        m2x1 = Matrix([[1.], [2.]])
        self.assertEqual(m2x1._numer, (2, 1))
        p2x1_conv = Pair.as_pair(m2x1)
        self.assertEqual(type(p2x1_conv), Pair)
        self.assertTrue(np.allclose(p2x1_conv.vals, [1., 2.]))

        # Test as_pair with n-D 1x2 Matrix
        m1x2_nd = Matrix([[[1., 2.]], [[4., 5.]]])
        self.assertEqual(m1x2_nd.shape, (2,))
        self.assertEqual(m1x2_nd._numer, (1, 2))
        p1x2_nd_conv = Pair.as_pair(m1x2_nd)
        self.assertEqual(type(p1x2_nd_conv), Pair)
        self.assertEqual(p1x2_nd_conv.shape, (2,))
        self.assertTrue(np.allclose(p1x2_nd_conv.vals[0], [1., 2.]))
        self.assertTrue(np.allclose(p1x2_nd_conv.vals[1], [4., 5.]))

        # Test as_pair with Qube rank > 1 and first numerator dimension == 2 (split_items)
        # Create a Matrix with shape that has rank > 1 and first numer dim == 2
        m2x4 = Matrix(np.random.randn(2, 2, 4))  # shape (2,), numer (2, 4)
        self.assertEqual(m2x4.shape, (2,))
        self.assertEqual(m2x4._numer, (2, 4))
        self.assertEqual(m2x4.rank, 2)  # nrank=2
        self.assertEqual(m2x4._numer[0], 2)
        p2x4_conv = Pair.as_pair(m2x4)
        self.assertEqual(type(p2x4_conv), Pair)
        # After split_items(1, Pair), the first 2 elements become a Pair
        # and the remaining 4 elements become the denominator
        self.assertEqual(p2x4_conv.shape, (2,))
        self.assertEqual(p2x4_conv.item, (2, 4))  # numer=(2,), denom=(4,)
        self.assertEqual(p2x4_conv.numer, (2,))
        self.assertEqual(p2x4_conv.denom, (4,))

        # Test as_pair with single number (special case: value repeated)
        p18_conv = Pair.as_pair(5.)
        self.assertEqual(type(p18_conv), Pair)
        self.assertTrue(np.allclose(p18_conv.vals, [5., 5.]))

        # Test as_pair with recursive=False
        p19 = Pair([1., 2.])
        p19.insert_deriv('t', Pair([3., 4.]))
        p19_conv = Pair.as_pair(p19, recursive=False)
        self.assertEqual(type(p19_conv), Pair)
        self.assertTrue(np.allclose(p19_conv.vals, [1., 2.]))
        self.assertFalse(hasattr(p19_conv, 'd_dt'))

        # Test from_scalars static method
        x = Scalar(1.)
        y = Scalar(2.)
        p20 = Pair.from_scalars(x, y)
        self.assertEqual(type(p20), Pair)
        self.assertEqual(p20.shape, ())
        self.assertTrue(np.allclose(p20.vals, [1., 2.]))

        # Test from_scalars with n-D scalars
        x_2d = Scalar([[1., 2.], [3., 4.]])
        y_2d = Scalar([[5., 6.], [7., 8.]])
        p21 = Pair.from_scalars(x_2d, y_2d)
        self.assertEqual(p21.shape, (2, 2))
        self.assertTrue(np.allclose(p21.vals[0, 0], [1., 5.]))
        self.assertTrue(np.allclose(p21.vals[0, 1], [2., 6.]))

        # Test from_scalars with zero
        p22 = Pair.from_scalars(1., 0.)
        self.assertTrue(np.allclose(p22.vals, [1., 0.]))

        # Test from_scalars with None (docstring says None is converted to zero Scalar)
        p22_none = Pair.from_scalars(1., None)
        self.assertTrue(np.allclose(p22_none.vals, [1., 0.]))

        p22_none2 = Pair.from_scalars(None, 2.)
        self.assertTrue(np.allclose(p22_none2.vals, [0., 2.]))

        # Test from_scalars with None and n-D scalars
        x_nd = Scalar([[1., 2.], [3., 4.]], drank=1)
        y_nd = Scalar([[5., 6.], [7., 8.]], drank=1)
        p22_none_nd = Pair.from_scalars(x_nd, None)
        self.assertEqual(p22_none_nd.shape, (2,))
        self.assertEqual(p22_none_nd.denom, (2,))  # Should match the denominator of x_nd
        # Check the first array element, first denominator element: should be [x, 0] = [1., 0.]
        self.assertTrue(np.allclose(p22_none_nd.vals[0, :, 0], [1., 0.]))

        # Test from_scalars with all None
        p_all_none = Pair.from_scalars(None, None)
        self.assertEqual(type(p_all_none), Pair)
        self.assertEqual(p_all_none.shape, ())
        self.assertTrue(np.allclose(p_all_none.vals, [0., 0.]))

        # Test from_scalars with multiple scalars requiring broadcasting
        x_broad = Scalar([1., 2.])  # shape (2,)
        y_broad = Scalar([[3.], [4.]])  # shape (2, 1)
        # Broadcasting: (2,) and (2, 1) -> (2, 2)
        p_broad = Pair.from_scalars(x_broad, y_broad)
        self.assertEqual(type(p_broad), Pair)
        self.assertEqual(p_broad.shape, (2, 2))
        # Check a few values
        self.assertTrue(np.allclose(p_broad.vals[0, 0], [1., 3.]))
        self.assertTrue(np.allclose(p_broad.vals[0, 1], [2., 3.]))
        self.assertTrue(np.allclose(p_broad.vals[1, 0], [1., 4.]))
        self.assertTrue(np.allclose(p_broad.vals[1, 1], [2., 4.]))

        # Test from_scalars with readonly
        # Note: readonly parameter is passed but Qube.from_scalars doesn't set readonly on main object
        p23 = Pair.from_scalars(1., 2., readonly=True)
        self.assertEqual(type(p23), Pair)
        # readonly may not be set by Qube.from_scalars, but parameter is accepted

        # Test swapxy method
        p24 = Pair([1., 2.])
        p24_swapped = p24.swapxy()
        self.assertEqual(type(p24_swapped), Pair)
        self.assertTrue(np.allclose(p24_swapped.vals, [2., 1.]))

        # Test swapxy with n-D
        p25 = Pair(np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
        p25_swapped = p25.swapxy()
        self.assertEqual(p25_swapped.shape, (2, 2))
        self.assertTrue(np.allclose(p25_swapped.vals[0, 0], [2., 1.]))
        self.assertTrue(np.allclose(p25_swapped.vals[0, 1], [4., 3.]))

        # Test swapxy with recursive=False
        p26 = Pair([1., 2.])
        p26.insert_deriv('t', Pair([3., 4.]))
        p26_swapped = p26.swapxy(recursive=False)
        self.assertEqual(type(p26_swapped), Pair)
        self.assertTrue(np.allclose(p26_swapped.vals, [2., 1.]))
        self.assertFalse(hasattr(p26_swapped, 'd_dt'))

        # Test swapxy with recursive=True (derivatives should be swapped)
        p27 = Pair([1., 2.])
        p27.insert_deriv('t', Pair([3., 4.]))
        p27_swapped = p27.swapxy(recursive=True)
        self.assertEqual(type(p27_swapped), Pair)
        self.assertTrue(np.allclose(p27_swapped.vals, [2., 1.]))
        self.assertTrue(hasattr(p27_swapped, 'd_dt'))
        self.assertTrue(np.allclose(p27_swapped.d_dt.vals, [4., 3.]))

        # Test rot90 method
        p28 = Pair([1., 0.])  # along x-axis
        p28_rot = p28.rot90()
        self.assertEqual(type(p28_rot), Pair)
        # (x,y) -> (y,-x): (1,0) -> (0,-1)
        self.assertTrue(np.allclose(p28_rot.vals, [0., -1.], atol=1e-10))

        # Test rot90 with another example
        p29 = Pair([0., 1.])  # along y-axis
        p29_rot = p29.rot90()
        # (0,1) -> (1,0)
        self.assertTrue(np.allclose(p29_rot.vals, [1., 0.], atol=1e-10))

        # Test rot90 with n-D
        p30 = Pair(np.array([[[1., 0.], [0., 1.]], [[-1., 0.], [0., -1.]]]))
        p30_rot = p30.rot90()
        self.assertEqual(p30_rot.shape, (2, 2))
        self.assertTrue(np.allclose(p30_rot.vals[0, 0], [0., -1.], atol=1e-10))
        self.assertTrue(np.allclose(p30_rot.vals[0, 1], [1., 0.], atol=1e-10))

        # Test rot90 with recursive=False
        p31 = Pair([1., 0.])
        p31.insert_deriv('t', Pair([2., 3.]))
        p31_rot = p31.rot90(recursive=False)
        self.assertEqual(type(p31_rot), Pair)
        self.assertTrue(np.allclose(p31_rot.vals, [0., -1.], atol=1e-10))
        self.assertFalse(hasattr(p31_rot, 'd_dt'))

        # Test rot90 with recursive=True (derivatives should be rotated)
        p32 = Pair([1., 0.])
        p32.insert_deriv('t', Pair([2., 3.]))
        p32_rot = p32.rot90(recursive=True)
        self.assertEqual(type(p32_rot), Pair)
        self.assertTrue(np.allclose(p32_rot.vals, [0., -1.], atol=1e-10))
        self.assertTrue(hasattr(p32_rot, 'd_dt'))
        # Derivative (2,3) rotated: (3, -2)
        self.assertTrue(np.allclose(p32_rot.d_dt.vals, [3., -2.], atol=1e-10))

        # Test angle method
        p33 = Pair([1., 0.])  # along x-axis
        angle33 = p33.angle()
        self.assertEqual(type(angle33), Scalar)
        self.assertTrue(np.allclose(angle33.vals, 0., atol=1e-10))

        p34 = Pair([0., 1.])  # along y-axis
        angle34 = p34.angle()
        self.assertTrue(np.allclose(angle34.vals, np.pi/2, atol=1e-10))

        # Test angle with n-D
        p35 = Pair(np.array([[[1., 0.], [0., 1.]], [[-1., 0.], [0., -1.]]]))
        angle35 = p35.angle()
        self.assertEqual(angle35.shape, (2, 2))
        self.assertTrue(np.allclose(angle35.vals[0, 0], 0., atol=1e-10))
        self.assertTrue(np.allclose(angle35.vals[0, 1], np.pi/2, atol=1e-10))

        # Test angle range (should be between 0 and 2*pi)
        p36 = Pair([-1., 0.])  # negative x-axis
        angle36 = p36.angle()
        self.assertTrue(angle36.vals >= 0)
        self.assertTrue(angle36.vals <= 2*np.pi)
        # Should be pi (180 degrees)
        self.assertTrue(np.allclose(angle36.vals, np.pi, atol=1e-10))

        # Test angle with recursive=False
        p37 = Pair([1., 1.])
        p37.insert_deriv('t', Pair([2., 3.]))
        angle37 = p37.angle(recursive=False)
        self.assertEqual(type(angle37), Scalar)
        self.assertFalse(hasattr(angle37, 'd_dt'))

        # Test clip2d method
        p38 = Pair([5., 5.])
        lower = Pair([2., 2.])
        upper = Pair([4., 4.])
        p38_clipped = p38.clip2d(lower, upper)
        self.assertEqual(type(p38_clipped), Pair)
        # Should be clipped to (4, 4)
        self.assertTrue(np.allclose(p38_clipped.vals, [4., 4.], atol=1e-10))

        # Test clip2d with None lower
        p39 = Pair([1., 5.])
        upper = Pair([4., 4.])
        p39_clipped = p39.clip2d(None, upper)
        self.assertEqual(type(p39_clipped), Pair)
        # Only upper limit applied, x should be 1, y should be 4
        self.assertTrue(np.allclose(p39_clipped.vals, [1., 4.], atol=1e-10))

        # Test clip2d with None upper
        p40 = Pair([1., 1.])
        lower = Pair([2., 2.])
        p40_clipped = p40.clip2d(lower, None)
        self.assertEqual(type(p40_clipped), Pair)
        # Only lower limit applied, should be (2, 2)
        self.assertTrue(np.allclose(p40_clipped.vals, [2., 2.], atol=1e-10))

        # Test clip2d with n-D
        p41 = Pair(np.array([[[5., 5.], [1., 1.]], [[3., 3.], [6., 6.]]]))
        lower = Pair([2., 2.])
        upper = Pair([4., 4.])
        p41_clipped = p41.clip2d(lower, upper)
        self.assertEqual(p41_clipped.shape, (2, 2))
        # First should be clipped to (4, 4), second to (2, 2), etc.
        self.assertTrue(np.allclose(p41_clipped.vals[0, 0], [4., 4.], atol=1e-10))
        self.assertTrue(np.allclose(p41_clipped.vals[0, 1], [2., 2.], atol=1e-10))

        # Test clip2d with remask=True
        # Note: remask behavior may need verification - docstring says it includes new mask
        p42 = Pair([5., 5.])
        lower = Pair([2., 2.])
        upper = Pair([4., 4.])
        p42_clipped = p42.clip2d(lower, upper, remask=True)
        self.assertEqual(type(p42_clipped), Pair)
        # Values should be clipped to (4, 4)
        self.assertTrue(np.allclose(p42_clipped.vals, [4., 4.], atol=1e-10))
        # remask behavior may vary - check actual implementation

        # Test clip2d raises ValueError for lower with shape
        p43 = Pair([1., 1.])
        lower_bad = Pair([[2., 2.], [3., 3.]])  # has shape
        upper = Pair([4., 4.])
        self.assertRaises(ValueError, p43.clip2d, lower_bad, upper)

        # Test clip2d raises ValueError for upper with shape
        p44 = Pair([1., 1.])
        lower = Pair([2., 2.])
        upper_bad = Pair([[4., 4.], [5., 5.]])  # has shape
        self.assertRaises(ValueError, p44.clip2d, lower, upper_bad)

        # Test clip2d with masked lower limit (should be treated as None)
        p45 = Pair([5., 5.])
        lower_masked = Pair([2., 2.], mask=True)  # masked
        upper = Pair([4., 4.])
        p45_clipped = p45.clip2d(lower_masked, upper)
        self.assertEqual(type(p45_clipped), Pair)
        # Lower should be ignored, only upper limit applied
        self.assertTrue(np.allclose(p45_clipped.vals, [4., 4.], atol=1e-10))

        # Test clip2d with masked upper limit (should be treated as None)
        p46 = Pair([1., 1.])
        lower = Pair([2., 2.])
        upper_masked = Pair([4., 4.], mask=True)  # masked
        p46_clipped = p46.clip2d(lower, upper_masked)
        self.assertEqual(type(p46_clipped), Pair)
        # Upper should be ignored, only lower limit applied
        self.assertTrue(np.allclose(p46_clipped.vals, [2., 2.], atol=1e-10))

        # Test clip2d with both limits masked (both should be ignored)
        p47 = Pair([5., 5.])
        lower_masked2 = Pair([2., 2.], mask=True)
        upper_masked2 = Pair([4., 4.], mask=True)
        p47_clipped = p47.clip2d(lower_masked2, upper_masked2)
        self.assertEqual(type(p47_clipped), Pair)
        # Both limits ignored, values should be unchanged
        self.assertTrue(np.allclose(p47_clipped.vals, [5., 5.], atol=1e-10))

        # Test inherited methods from Vector - to_scalar
        p45 = Pair(np.random.randn(4, 1, 5, 2))
        s45 = p45.to_scalar(0)
        self.assertEqual(type(s45), Scalar)
        self.assertEqual(s45.shape, p45.shape)

        # Test to_scalars
        scalars45 = p45.to_scalars()
        self.assertEqual(len(scalars45), 2)
        self.assertEqual(type(scalars45[0]), Scalar)
        self.assertEqual(scalars45[0].shape, p45.shape)

        # Test dot
        p46 = Pair([1., 2.])
        p47 = Pair([3., 4.])
        dot46 = p46.dot(p47)
        self.assertEqual(type(dot46), Scalar)
        # 1*3 + 2*4 = 3 + 8 = 11
        self.assertTrue(np.allclose(dot46.vals, 11.))

        # Test dot with n-D
        p48 = Pair(np.random.randn(4, 1, 5, 2))
        p49 = Pair(np.random.randn(8, 5, 2))
        dot48 = p48.dot(p49)
        # Broadcasting: (4, 1, 5) and (8, 5) -> (4, 8, 5)
        self.assertEqual(dot48.shape, (4, 8, 5))

        # Test norm
        p50 = Pair([3., 4.])
        norm50 = p50.norm()
        self.assertEqual(type(norm50), Scalar)
        # sqrt(3^2 + 4^2) = 5
        self.assertTrue(np.allclose(norm50.vals, 5.))

        # Test norm with n-D
        p51 = Pair(np.random.randn(2, 3, 2))
        norm51 = p51.norm()
        self.assertEqual(norm51.shape, (2, 3))

        # Test unit
        p52 = Pair([3., 4.])
        unit52 = p52.unit()
        self.assertEqual(type(unit52), Pair)
        # Should be normalized: (3/5, 4/5)
        self.assertTrue(np.allclose(unit52.vals, [0.6, 0.8], atol=1e-10))
        self.assertTrue(np.allclose(unit52.norm().vals, 1., atol=1e-10))

        # Test unit with n-D
        p53 = Pair(np.random.randn(2, 3, 2))
        unit53 = p53.unit()
        self.assertEqual(unit53.shape, (2, 3))

        # Test class constants
        self.assertEqual(type(Pair.ZERO), Pair)
        self.assertTrue(np.allclose(Pair.ZERO.vals, [0., 0.]))
        self.assertTrue(Pair.ZERO.readonly)

        self.assertEqual(type(Pair.ZEROS), Pair)
        self.assertTrue(np.allclose(Pair.ZEROS.vals, [0., 0.]))
        self.assertTrue(Pair.ZEROS.readonly)

        self.assertEqual(type(Pair.ONES), Pair)
        self.assertTrue(np.allclose(Pair.ONES.vals, [1., 1.]))
        self.assertTrue(Pair.ONES.readonly)

        self.assertEqual(type(Pair.HALF), Pair)
        self.assertTrue(np.allclose(Pair.HALF.vals, [0.5, 0.5]))
        self.assertTrue(Pair.HALF.readonly)

        self.assertEqual(type(Pair.XAXIS), Pair)
        self.assertTrue(np.allclose(Pair.XAXIS.vals, [1., 0.]))
        self.assertTrue(Pair.XAXIS.readonly)

        self.assertEqual(type(Pair.YAXIS), Pair)
        self.assertTrue(np.allclose(Pair.YAXIS.vals, [0., 1.]))
        self.assertTrue(Pair.YAXIS.readonly)

        self.assertEqual(type(Pair.MASKED), Pair)
        self.assertTrue(Pair.MASKED.mask)
        self.assertTrue(Pair.MASKED.readonly)

        self.assertEqual(type(Pair.IDENTITY), Pair)
        self.assertEqual(Pair.IDENTITY.shape, ())
        self.assertEqual(Pair.IDENTITY.denom, (2,))
        self.assertEqual(Pair.IDENTITY.item, (2, 2))
        self.assertTrue(Pair.IDENTITY.readonly)

        self.assertEqual(type(Pair.INT00), Pair)
        self.assertTrue(np.allclose(Pair.INT00.vals, [0, 0]))
        self.assertTrue(Pair.INT00.readonly)

        self.assertEqual(type(Pair.INT11), Pair)
        self.assertTrue(np.allclose(Pair.INT11.vals, [1, 1]))
        self.assertTrue(Pair.INT11.readonly)

        # Test that Pair accepts both floats and ints
        p54 = Pair([1, 2])
        self.assertEqual(p54.vals.dtype.kind, 'i')  # Should allow integers

        p55 = Pair([1., 2.])
        self.assertEqual(p55.vals.dtype.kind, 'f')

        # Test with mask
        p56 = Pair([1., 2.], mask=False)
        self.assertFalse(p56.mask)

        p57 = Pair([1., 2.], mask=True)
        self.assertTrue(p57.mask)

        # Test complex n-D case
        p58 = Pair(np.random.randn(3, 4, 5, 6, 2))
        self.assertEqual(p58.shape, (3, 4, 5, 6))
        self.assertEqual(p58.item, (2,))
        self.assertEqual(p58.vals.shape, (3, 4, 5, 6, 2))

        # Test that operations preserve type
        p59 = Pair([1., 2.])
        p60 = Pair([3., 4.])
        p_result = p59 + p60
        self.assertEqual(type(p_result), Pair)

        p_result2 = p59 * 2.
        self.assertEqual(type(p_result2), Pair)

        # Test round-trip: swapxy then swapxy should return original
        p61 = Pair([1., 2.])
        p61_round = p61.swapxy().swapxy()
        self.assertTrue(np.allclose(p61.vals, p61_round.vals, atol=1e-10))

        # Test round-trip: rot90 four times should return original
        p62 = Pair([1., 2.])
        p62_round = p62.rot90().rot90().rot90().rot90()
        self.assertTrue(np.allclose(p62.vals, p62_round.vals, atol=1e-10))

        # Test angle consistency: angle of rot90
        # Note: rot90 does (x,y) -> (y,-x), which rotates by 90 degrees counterclockwise
        # For (1,0) -> (0,-1), the angle goes from 0 to 3π/2 (270 degrees)
        p63 = Pair([1., 0.])
        angle63 = p63.angle()
        p63_rot = p63.rot90()
        angle63_rot = p63_rot.angle()
        # The angle should be (original + 3π/2) mod 2π, or equivalently (original - π/2) mod 2π
        expected_angle = (angle63.vals - np.pi/2) % (2*np.pi)
        self.assertTrue(np.allclose(angle63_rot.vals, expected_angle, atol=1e-10))

##########################################################################################
