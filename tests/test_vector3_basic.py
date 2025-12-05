##########################################################################################
# tests/test_vector3_basic.py
# Vector3 basic construction, factory methods, static methods, and class constants
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Vector3, Matrix, Vector


class Test_Vector3_Basic(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # Test basic construction
        v1 = Vector3([1., 2., 3.])
        self.assertEqual(v1.shape, ())
        self.assertEqual(v1.item, (3,))
        self.assertEqual(v1.numer, (3,))
        self.assertTrue(np.allclose(v1.vals, [1., 2., 3.]))

        # Test construction from list
        v2 = Vector3([4., 5., 6.])
        self.assertTrue(np.allclose(v2.vals, [4., 5., 6.]))

        # Test construction from tuple
        v3 = Vector3((7., 8., 9.))
        self.assertTrue(np.allclose(v3.vals, [7., 8., 9.]))

        # Test construction from numpy array
        v4 = Vector3(np.array([10., 11., 12.]))
        self.assertTrue(np.allclose(v4.vals, [10., 11., 12.]))

        # Test that wrong shapes raise ValueError
        self.assertRaises(ValueError, Vector3, np.random.randn(3, 4, 5))
        self.assertRaises(ValueError, Vector3, 1.)
        self.assertRaises(ValueError, Vector3, [1., 2.])
        self.assertRaises(ValueError, Vector3, [1., 2., 3., 4.])

        # Test automatic coercion of booleans
        v_bool = Vector3([True, True, False])
        self.assertTrue(np.allclose(v_bool.vals, [1., 1., 0.]))

        # Test zeros
        v7 = Vector3.zeros((2, 3))
        self.assertEqual(v7.shape, (2, 3))
        self.assertEqual(v7.vals.shape, (2, 3, 3))
        self.assertEqual(v7.vals.dtype.kind, 'f')
        self.assertTrue(np.all(v7.vals == 0))

        v8 = Vector3.zeros((2, 3), dtype='float')
        self.assertEqual(v8.shape, (2, 3))
        self.assertEqual(v8.vals.shape, (2, 3, 3))
        self.assertEqual(v8.vals.dtype.kind, 'f')
        self.assertTrue(np.all(v8.vals == 0))

        v9 = Vector3.zeros((2, 2), mask=[[0, 1], [0, 0]])
        self.assertEqual(v9.shape, (2, 2))
        self.assertEqual(v9.vals.shape, (2, 2, 3))
        self.assertTrue(np.all(v9.vals == 0))
        self.assertTrue(np.all(v9.mask == [[0, 1], [0, 0]]))

        v10 = Vector3.zeros((2, 2), denom=(3, 3))
        self.assertEqual(v10.shape, (2, 2))
        self.assertEqual(v10.vals.shape, (2, 2, 3, 3, 3))
        self.assertTrue(np.all(v10.vals == 0))

        self.assertRaises(ValueError, Vector3.zeros, (2, 3), numer=(4,))

        # Test ones
        v11 = Vector3.ones((2, 3))
        self.assertEqual(v11.shape, (2, 3))
        self.assertEqual(v11.vals.shape, (2, 3, 3))
        self.assertEqual(v11.vals.dtype.kind, 'f')
        self.assertTrue(np.all(v11.vals == 1))

        v12 = Vector3.ones((2, 2), mask=[[0, 1], [0, 0]])
        self.assertEqual(v12.shape, (2, 2))
        self.assertEqual(v12.vals.shape, (2, 2, 3))
        self.assertTrue(np.all(v12.vals == 1))
        self.assertTrue(np.all(v12.mask == [[0, 1], [0, 0]]))

        # Test filled
        v13 = Vector3.filled((2, 3), 7.)
        self.assertEqual(v13.shape, (2, 3))
        self.assertEqual(v13.vals.shape, (2, 3, 3))
        self.assertTrue(np.all(v13.vals == 7))

        v14 = Vector3.filled((2, 2), (1., 2., 3.))
        self.assertEqual(v14.shape, (2, 2))
        self.assertEqual(v14.vals.shape, (2, 2, 3))
        self.assertTrue(np.all(v14.vals[..., 0] == 1))
        self.assertTrue(np.all(v14.vals[..., 1] == 2))
        self.assertTrue(np.all(v14.vals[..., 2] == 3))

        # Test as_vector3 static method
        v15 = Vector3([1., 2., 3.])
        v15_conv = Vector3.as_vector3(v15)
        self.assertEqual(type(v15_conv), Vector3)
        self.assertTrue(np.allclose(v15_conv.vals, [1., 2., 3.]))

        # Test as_vector3 with Vector
        v16 = Vector([1., 2., 3.])
        v16_conv = Vector3.as_vector3(v16)
        self.assertEqual(type(v16_conv), Vector3)
        self.assertTrue(np.allclose(v16_conv.vals, [1., 2., 3.]))

        # Test as_vector3 with array
        v17_conv = Vector3.as_vector3([4., 5., 6.])
        self.assertEqual(type(v17_conv), Vector3)
        self.assertTrue(np.allclose(v17_conv.vals, [4., 5., 6.]))

        # Test as_vector3 with 1x3 Matrix
        m1x3 = Matrix([[1., 2., 3.]])
        self.assertEqual(m1x3._numer, (1, 3))
        v1x3_conv = Vector3.as_vector3(m1x3)
        self.assertEqual(type(v1x3_conv), Vector3)
        self.assertTrue(np.allclose(v1x3_conv.vals, [1., 2., 3.]))

        # Test as_vector3 with 3x1 Matrix
        m3x1 = Matrix([[1.], [2.], [3.]])
        self.assertEqual(m3x1._numer, (3, 1))
        v3x1_conv = Vector3.as_vector3(m3x1)
        self.assertEqual(type(v3x1_conv), Vector3)
        self.assertTrue(np.allclose(v3x1_conv.vals, [1., 2., 3.]))

        # Test as_vector3 with n-D 1x3 Matrix
        m1x3_nd = Matrix([[[1., 2., 3.]], [[4., 5., 6.]]])
        self.assertEqual(m1x3_nd.shape, (2,))
        self.assertEqual(m1x3_nd._numer, (1, 3))
        v1x3_nd_conv = Vector3.as_vector3(m1x3_nd)
        self.assertEqual(type(v1x3_nd_conv), Vector3)
        self.assertEqual(v1x3_nd_conv.shape, (2,))
        self.assertTrue(np.allclose(v1x3_nd_conv.vals[0], [1., 2., 3.]))
        self.assertTrue(np.allclose(v1x3_nd_conv.vals[1], [4., 5., 6.]))

        # Test as_vector3 with Qube rank > 1 and first numerator dimension == 3
        # Create a Vector with shape that has rank > 1 and first numer dim == 3
        # This would be a Vector with drank > 0, where the first numer dim is 3
        # Actually, let's create a Matrix with shape (3, N) where N > 1
        # But wait, for line 53, we need arg.rank > 1 and arg._numer[0] == 3
        # rank = nrank + drank, so we need nrank + drank > 1 and _numer[0] == 3
        # For a Matrix with _numer = (3, 4), we have nrank=2, so rank=2 > 1, and _numer[0] == 3
        m3x4 = Matrix(np.random.randn(2, 3, 4))  # shape (2,), numer (3, 4)
        self.assertEqual(m3x4.shape, (2,))
        self.assertEqual(m3x4._numer, (3, 4))
        self.assertEqual(m3x4.rank, 2)  # nrank=2
        self.assertEqual(m3x4._numer[0], 3)
        v3x4_conv = Vector3.as_vector3(m3x4)
        self.assertEqual(type(v3x4_conv), Vector3)
        # After split_items(1, Vector3), the first 3 elements become a Vector3
        # and the remaining 4 elements become the denominator
        self.assertEqual(v3x4_conv.shape, (2,))
        self.assertEqual(v3x4_conv.item, (3, 4))  # numer=(3,), denom=(4,)
        self.assertEqual(v3x4_conv.numer, (3,))
        self.assertEqual(v3x4_conv.denom, (4,))

        # Test from_scalars static method
        x = Scalar(1.)
        y = Scalar(2.)
        z = Scalar(3.)
        v18 = Vector3.from_scalars(x, y, z)
        self.assertEqual(type(v18), Vector3)
        self.assertEqual(v18.shape, ())
        self.assertTrue(np.allclose(v18.vals, [1., 2., 3.]))

        # Test from_scalars with n-D scalars
        x_2d = Scalar([[1., 2.], [3., 4.]])
        y_2d = Scalar([[5., 6.], [7., 8.]])
        z_2d = Scalar([[9., 10.], [11., 12.]])
        v19 = Vector3.from_scalars(x_2d, y_2d, z_2d)
        self.assertEqual(v19.shape, (2, 2))
        self.assertTrue(np.allclose(v19.vals[0, 0], [1., 5., 9.]))
        self.assertTrue(np.allclose(v19.vals[0, 1], [2., 6., 10.]))

        # Test from_scalars with zero
        v20 = Vector3.from_scalars(1., 0., 3.)
        self.assertTrue(np.allclose(v20.vals, [1., 0., 3.]))

        # Test from_scalars with None (docstring says None is converted to zero Scalar)
        v20_none = Vector3.from_scalars(1., None, 3.)
        self.assertTrue(np.allclose(v20_none.vals, [1., 0., 3.]))

        # Test from_scalars with None and n-D scalars
        x_nd = Scalar([[1., 2.], [3., 4.]], drank=1)
        y_nd = Scalar([[5., 6.], [7., 8.]], drank=1)
        v20_none_nd = Vector3.from_scalars(x_nd, None, y_nd)
        self.assertEqual(v20_none_nd.shape, (2,))
        self.assertEqual(v20_none_nd.denom, (2,))  # Should match the denominator of x_nd and y_nd
        # Check the first array element, first denominator element: should be [x, 0, y] = [1., 0., 5.]
        self.assertTrue(np.allclose(v20_none_nd.vals[0, :, 0], [1., 0., 5.]))

        # Test from_scalars with all None (lines 97-99: all three are None)
        v_all_none = Vector3.from_scalars(None, None, None)
        self.assertEqual(type(v_all_none), Vector3)
        self.assertEqual(v_all_none.shape, ())
        self.assertTrue(np.allclose(v_all_none.vals, [0., 0., 0.]))

        # Test from_scalars with x=None
        v_x_none = Vector3.from_scalars(None, 2., 3.)
        self.assertEqual(type(v_x_none), Vector3)
        self.assertEqual(v_x_none.shape, ())
        self.assertTrue(np.allclose(v_x_none.vals, [0., 2., 3.]))

        # Test from_scalars with z=None
        v_z_none = Vector3.from_scalars(1., 2., None)
        self.assertEqual(type(v_z_none), Vector3)
        self.assertEqual(v_z_none.shape, ())
        self.assertTrue(np.allclose(v_z_none.vals, [1., 2., 0.]))

        # Test from_scalars with exactly 1 non-None arg (skips if block at line 108, goes directly to 110)
        # This tests the case where len(scalars) = 1, so the if len(scalars) > 1: block is skipped
        v_one_arg = Vector3.from_scalars(None, 2., None)
        self.assertEqual(type(v_one_arg), Vector3)
        self.assertEqual(v_one_arg.shape, ())
        self.assertTrue(np.allclose(v_one_arg.vals, [0., 2., 0.]))

        # Test from_scalars with multiple scalars requiring broadcasting (lines 108-110)
        # Create scalars with different shapes that need broadcasting
        x_broad = Scalar([1., 2.])  # shape (2,)
        y_broad = Scalar([[3.], [4.]])  # shape (2, 1)
        z_broad = Scalar(5.)  # shape ()
        # Broadcasting: (2,) and (2, 1) and () -> (2, 2)
        v_broad = Vector3.from_scalars(x_broad, y_broad, z_broad)
        self.assertEqual(type(v_broad), Vector3)
        self.assertEqual(v_broad.shape, (2, 2))
        # Check a few values
        self.assertTrue(np.allclose(v_broad.vals[0, 0], [1., 3., 5.]))
        self.assertTrue(np.allclose(v_broad.vals[0, 1], [2., 3., 5.]))
        self.assertTrue(np.allclose(v_broad.vals[1, 0], [1., 4., 5.]))
        self.assertTrue(np.allclose(v_broad.vals[1, 1], [2., 4., 5.]))

        # Test from_scalars with broadcasting and None (lines 108-110, 117)
        # x is None, y and z need broadcasting - this ensures len(scalars) = 2, triggering line 108
        y_broad2 = Scalar([3., 4.])  # shape (2,)
        z_broad2 = Scalar([[5.], [6.]])  # shape (2, 1)
        v_broad_none = Vector3.from_scalars(None, y_broad2, z_broad2)
        self.assertEqual(type(v_broad_none), Vector3)
        self.assertEqual(v_broad_none.shape, (2, 2))
        # Check that x component is zero everywhere
        self.assertTrue(np.allclose(v_broad_none.vals[:, :, 0], 0.))
        # Check a few values for y and z components
        self.assertTrue(np.allclose(v_broad_none.vals[0, 0], [0., 3., 5.]))
        self.assertTrue(np.allclose(v_broad_none.vals[0, 1], [0., 4., 5.]))

        # Test from_scalars with exactly 2 non-None args that need broadcasting (lines 108-110)
        # This explicitly tests the case where len(scalars) = 2, ensuring the if block is entered
        # Case 1: x=None, y and z have different shapes requiring broadcast
        y_broad3 = Scalar([1., 2.])  # shape (2,)
        z_broad3 = Scalar([[3.], [4.]])  # shape (2, 1) - different shape requires broadcast
        v_broad2 = Vector3.from_scalars(None, y_broad3, z_broad3)
        self.assertEqual(type(v_broad2), Vector3)
        self.assertEqual(v_broad2.shape, (2, 2))  # Broadcast result: (2,) and (2,1) -> (2,2)
        # Verify the broadcast worked correctly
        self.assertTrue(np.allclose(v_broad2.vals[0, 0], [0., 1., 3.]))
        self.assertTrue(np.allclose(v_broad2.vals[0, 1], [0., 2., 3.]))
        self.assertTrue(np.allclose(v_broad2.vals[1, 0], [0., 1., 4.]))
        self.assertTrue(np.allclose(v_broad2.vals[1, 1], [0., 2., 4.]))

        # Case 2: y=None, x and z have different shapes requiring broadcast (lines 108-110)
        x_broad4 = Scalar([1., 2.])  # shape (2,)
        z_broad4 = Scalar([[3.], [4.]])  # shape (2, 1)
        v_broad3 = Vector3.from_scalars(x_broad4, None, z_broad4)
        self.assertEqual(type(v_broad3), Vector3)
        self.assertEqual(v_broad3.shape, (2, 2))
        # Verify the broadcast worked correctly
        self.assertTrue(np.allclose(v_broad3.vals[0, 0], [1., 0., 3.]))
        self.assertTrue(np.allclose(v_broad3.vals[0, 1], [2., 0., 3.]))
        self.assertTrue(np.allclose(v_broad3.vals[1, 0], [1., 0., 4.]))
        self.assertTrue(np.allclose(v_broad3.vals[1, 1], [2., 0., 4.]))

        # Case 3: All three non-None, but with different shapes requiring broadcast (lines 108-110)
        # This ensures len(scalars) = 3, which is > 1, so should enter the if block
        x_broad5 = Scalar([1., 2.])  # shape (2,)
        y_broad5 = Scalar([[3.], [4.]])  # shape (2, 1)
        z_broad5 = Scalar(5.)  # shape ()
        v_broad4 = Vector3.from_scalars(x_broad5, y_broad5, z_broad5)
        self.assertEqual(type(v_broad4), Vector3)
        self.assertEqual(v_broad4.shape, (2, 2))  # Broadcast: (2,), (2,1), () -> (2,2)
        # Verify the broadcast worked correctly
        self.assertTrue(np.allclose(v_broad4.vals[0, 0], [1., 3., 5.]))
        self.assertTrue(np.allclose(v_broad4.vals[0, 1], [2., 3., 5.]))
        self.assertTrue(np.allclose(v_broad4.vals[1, 0], [1., 4., 5.]))
        self.assertTrue(np.allclose(v_broad4.vals[1, 1], [2., 4., 5.]))

        # Test from_ra_dec_length static method
        ra = Scalar(0.)  # along x-axis
        dec = Scalar(0.)  # in equatorial plane
        length = Scalar(1.)
        v21 = Vector3.from_ra_dec_length(ra, dec, length)
        self.assertEqual(type(v21), Vector3)
        # Should be unit vector along x-axis: (1, 0, 0)
        self.assertTrue(np.allclose(v21.vals, [1., 0., 0.], atol=1e-10))

        # Test from_ra_dec_length with default length
        v22 = Vector3.from_ra_dec_length(ra, dec)
        self.assertTrue(np.allclose(v22.vals, [1., 0., 0.], atol=1e-10))

        # Test from_cylindrical static method
        radius = Scalar(1.)
        longitude = Scalar(0.)  # along x-axis
        z_coord = Scalar(0.)
        v26 = Vector3.from_cylindrical(radius, longitude, z_coord)
        self.assertEqual(type(v26), Vector3)
        # Should be (1, 0, 0)
        self.assertTrue(np.allclose(v26.vals, [1., 0., 0.], atol=1e-10))

        # Test from_cylindrical with default z
        v27 = Vector3.from_cylindrical(radius, longitude)
        self.assertTrue(np.allclose(v27.vals, [1., 0., 0.], atol=1e-10))

        # Test class constants
        self.assertEqual(type(Vector3.ZERO), Vector3)
        self.assertTrue(np.allclose(Vector3.ZERO.vals, [0., 0., 0.]))
        self.assertTrue(Vector3.ZERO.readonly)

        self.assertEqual(type(Vector3.ONES), Vector3)
        self.assertTrue(np.allclose(Vector3.ONES.vals, [1., 1., 1.]))
        self.assertTrue(Vector3.ONES.readonly)

        self.assertEqual(type(Vector3.XAXIS), Vector3)
        self.assertTrue(np.allclose(Vector3.XAXIS.vals, [1., 0., 0.]))
        self.assertTrue(Vector3.XAXIS.readonly)

        self.assertEqual(type(Vector3.YAXIS), Vector3)
        self.assertTrue(np.allclose(Vector3.YAXIS.vals, [0., 1., 0.]))
        self.assertTrue(Vector3.YAXIS.readonly)

        self.assertEqual(type(Vector3.ZAXIS), Vector3)
        self.assertTrue(np.allclose(Vector3.ZAXIS.vals, [0., 0., 1.]))
        self.assertTrue(Vector3.ZAXIS.readonly)

        self.assertEqual(type(Vector3.MASKED), Vector3)
        self.assertTrue(Vector3.MASKED.mask)
        self.assertTrue(Vector3.MASKED.readonly)

        self.assertEqual(type(Vector3.AXES), tuple)
        self.assertEqual(len(Vector3.AXES), 3)
        self.assertEqual(Vector3.AXES[0], Vector3.XAXIS)
        self.assertEqual(Vector3.AXES[1], Vector3.YAXIS)
        self.assertEqual(Vector3.AXES[2], Vector3.ZAXIS)

        # Test that Vector3 only accepts floats (not ints)
        # Integers should be coerced to float
        v84 = Vector3([1, 2, 3])
        self.assertEqual(v84.vals.dtype.kind, 'f')

        # Test with mask
        v85 = Vector3([1., 2., 3.], mask=False)
        self.assertFalse(v85.mask)

        v86 = Vector3([1., 2., 3.], mask=True)
        self.assertTrue(v86.mask)

##########################################################################################
