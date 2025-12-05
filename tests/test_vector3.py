##########################################################################################
# tests/test_vector3.py
# Vector3 comprehensive tests
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Vector3, Matrix, Vector


class Test_Vector3(unittest.TestCase):

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

        # Test n-D arrays
        v5 = Vector3(np.random.randn(2, 3, 3))
        self.assertEqual(v5.shape, (2, 3))
        self.assertEqual(v5.item, (3,))
        self.assertEqual(v5.vals.shape, (2, 3, 3))

        # Test higher-dimensional arrays
        v6 = Vector3(np.random.randn(4, 5, 6, 3))
        self.assertEqual(v6.shape, (4, 5, 6))
        self.assertEqual(v6.item, (3,))
        self.assertEqual(v6.vals.shape, (4, 5, 6, 3))

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

        # Test as_vector3 with 1x3 Matrix (line 49: flatten_numer)
        m1x3 = Matrix([[1., 2., 3.]])
        self.assertEqual(m1x3._numer, (1, 3))
        v1x3_conv = Vector3.as_vector3(m1x3)
        self.assertEqual(type(v1x3_conv), Vector3)
        self.assertTrue(np.allclose(v1x3_conv.vals, [1., 2., 3.]))

        # Test as_vector3 with 3x1 Matrix (line 49: flatten_numer)
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

        # Test as_vector3 with Qube rank > 1 and first numerator dimension == 3 (line 53: split_items)
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

        # Test from_scalars with x=None (line 117: x is None)
        v_x_none = Vector3.from_scalars(None, 2., 3.)
        self.assertEqual(type(v_x_none), Vector3)
        self.assertEqual(v_x_none.shape, ())
        self.assertTrue(np.allclose(v_x_none.vals, [0., 2., 3.]))

        # Test from_scalars with z=None (line 121: z is None)
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

        # Test from_ra_dec_length with n-D inputs
        ra_2d = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]])
        dec_2d = Scalar([[0., 0.], [0., 0.]])
        v23 = Vector3.from_ra_dec_length(ra_2d, dec_2d, 2.)
        self.assertEqual(v23.shape, (2, 2))
        # First should be along x, second along y, etc.
        self.assertTrue(np.allclose(v23.vals[0, 0], [2., 0., 0.], atol=1e-10))

        # Test to_ra_dec_length method
        v24 = Vector3([1., 0., 0.])
        ra24, dec24, length24 = v24.to_ra_dec_length()
        self.assertEqual(type(ra24), Scalar)
        self.assertEqual(type(dec24), Scalar)
        self.assertEqual(type(length24), Scalar)
        self.assertTrue(np.allclose(ra24.vals, 0., atol=1e-10))
        self.assertTrue(np.allclose(dec24.vals, 0., atol=1e-10))
        self.assertTrue(np.allclose(length24.vals, 1., atol=1e-10))

        # Test to_ra_dec_length with n-D
        v25 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        ra25, dec25, length25 = v25.to_ra_dec_length()
        self.assertEqual(ra25.shape, (2, 2))
        self.assertEqual(dec25.shape, (2, 2))
        self.assertEqual(length25.shape, (2, 2))

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

        # Test from_cylindrical with n-D inputs
        radius_2d = Scalar([[1., 2.], [3., 4.]])
        longitude_2d = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]])
        v28 = Vector3.from_cylindrical(radius_2d, longitude_2d, 0.)
        self.assertEqual(v28.shape, (2, 2))

        # Test to_cylindrical method
        v29 = Vector3([1., 0., 0.])
        radius29, longitude29, z29 = v29.to_cylindrical()
        self.assertEqual(type(radius29), Scalar)
        self.assertEqual(type(longitude29), Scalar)
        self.assertEqual(type(z29), Scalar)
        self.assertTrue(np.allclose(radius29.vals, 1., atol=1e-10))
        self.assertTrue(np.allclose(longitude29.vals, 0., atol=1e-10))
        self.assertTrue(np.allclose(z29.vals, 0., atol=1e-10))

        # Test to_cylindrical with n-D
        v30 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        radius30, longitude30, z30 = v30.to_cylindrical()
        self.assertEqual(radius30.shape, (2, 2))
        self.assertEqual(longitude30.shape, (2, 2))
        self.assertEqual(z30.shape, (2, 2))

        # Test longitude method
        v31 = Vector3([1., 0., 0.])
        lon31 = v31.longitude()
        self.assertEqual(type(lon31), Scalar)
        self.assertTrue(np.allclose(lon31.vals, 0., atol=1e-10))

        v32 = Vector3([0., 1., 0.])
        lon32 = v32.longitude()
        self.assertTrue(np.allclose(lon32.vals, np.pi/2, atol=1e-10))

        # Test longitude with n-D
        v33 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[-1., 0., 0.], [0., -1., 0.]]]))
        lon33 = v33.longitude()
        self.assertEqual(lon33.shape, (2, 2))

        # Test latitude method
        v34 = Vector3([1., 0., 0.])
        lat34 = v34.latitude()
        self.assertEqual(type(lat34), Scalar)
        self.assertTrue(np.allclose(lat34.vals, 0., atol=1e-10))

        v35 = Vector3([0., 0., 1.])
        lat35 = v35.latitude()
        self.assertTrue(np.allclose(lat35.vals, np.pi/2, atol=1e-10))

        # Test latitude with n-D
        v36 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        lat36 = v36.latitude()
        self.assertEqual(lat36.shape, (2, 2))

        # Test spin method
        v37 = Vector3([1., 0., 0.])
        pole = Vector3([0., 0., 1.])  # z-axis
        angle = Scalar(np.pi/2)
        v37_spun = v37.spin(pole, angle)
        self.assertEqual(type(v37_spun), Vector3)
        # Rotating (1,0,0) about z-axis by pi/2 should give (0,1,0)
        self.assertTrue(np.allclose(v37_spun.vals, [0., 1., 0.], atol=1e-10))

        # Test spin with angle=None (uses pole magnitude)
        v38 = Vector3([1., 0., 0.])
        pole38 = Vector3([0., 0., np.pi/2])  # magnitude is pi/2
        v38_spun = v38.spin(pole38)
        self.assertEqual(type(v38_spun), Vector3)

        # Test spin with n-D
        v39 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        pole39 = Vector3([0., 0., 1.])
        angle39 = Scalar(np.pi/2)
        v39_spun = v39.spin(pole39, angle39)
        self.assertEqual(v39_spun.shape, (2, 2))

        # Test offset_angles method
        v40 = Vector3([1., 0., 0.])
        v41 = Vector3([0., 1., 0.])
        lon_off, lat_off = v40.offset_angles(v41)
        self.assertEqual(type(lon_off), Scalar)
        self.assertEqual(type(lat_off), Scalar)
        # Should have some angular offset
        self.assertTrue(np.isfinite(lon_off.vals))
        self.assertTrue(np.isfinite(lat_off.vals))

        # Test offset_angles with n-D
        v42 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        v43 = Vector3([1., 0., 0.])
        lon_off2, lat_off2 = v42.offset_angles(v43)
        self.assertEqual(lon_off2.shape, (2, 2))
        self.assertEqual(lat_off2.shape, (2, 2))

        # Test inherited methods from Vector - to_scalar
        v44 = Vector3(np.random.randn(4, 1, 5, 3))
        s44 = v44.to_scalar(0)
        self.assertEqual(type(s44), Scalar)
        self.assertEqual(s44.shape, v44.shape)

        # Test to_scalars
        scalars44 = v44.to_scalars()
        self.assertEqual(len(scalars44), 3)
        self.assertEqual(type(scalars44[0]), Scalar)
        self.assertEqual(scalars44[0].shape, v44.shape)

        # Test as_column
        v45 = Vector3([1., 2., 3.])
        m45 = v45.as_column()
        self.assertEqual(type(m45), Matrix)
        self.assertEqual(m45.numer, (3, 1))
        self.assertTrue(np.allclose(m45.vals[..., 0], [1., 2., 3.]))

        # Test as_row
        v46 = Vector3([1., 2., 3.])
        m46 = v46.as_row()
        self.assertEqual(type(m46), Matrix)
        self.assertEqual(m46.numer, (1, 3))
        self.assertTrue(np.allclose(m46.vals[0, :], [1., 2., 3.]))

        # Test as_diagonal
        v47 = Vector3([1., 2., 3.])
        m47 = v47.as_diagonal()
        self.assertEqual(type(m47), Matrix)
        self.assertEqual(m47.numer, (3, 3))
        self.assertTrue(np.allclose(m47.vals[0, 0], 1.))
        self.assertTrue(np.allclose(m47.vals[1, 1], 2.))
        self.assertTrue(np.allclose(m47.vals[2, 2], 3.))

        # Test dot
        v48 = Vector3([1., 2., 3.])
        v49 = Vector3([4., 5., 6.])
        dot48 = v48.dot(v49)
        self.assertEqual(type(dot48), Scalar)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        self.assertTrue(np.allclose(dot48.vals, 32.))

        # Test dot with n-D
        v50 = Vector3(np.random.randn(4, 1, 5, 3))
        v51 = Vector3(np.random.randn(8, 5, 3))
        dot50 = v50.dot(v51)
        # Broadcasting: (4, 1, 5) and (8, 5) -> (4, 8, 5)
        self.assertEqual(dot50.shape, (4, 8, 5))

        # Test norm
        v52 = Vector3([3., 4., 0.])
        norm52 = v52.norm()
        self.assertEqual(type(norm52), Scalar)
        # sqrt(3^2 + 4^2 + 0^2) = 5
        self.assertTrue(np.allclose(norm52.vals, 5.))

        # Test norm with n-D
        v53 = Vector3(np.random.randn(2, 3, 3))
        norm53 = v53.norm()
        self.assertEqual(norm53.shape, (2, 3))

        # Test unit
        v54 = Vector3([3., 4., 0.])
        unit54 = v54.unit()
        self.assertEqual(type(unit54), Vector3)
        # Should be normalized: (3/5, 4/5, 0)
        self.assertTrue(np.allclose(unit54.vals, [0.6, 0.8, 0.], atol=1e-10))
        self.assertTrue(np.allclose(unit54.norm().vals, 1., atol=1e-10))

        # Test unit with n-D
        v55 = Vector3(np.random.randn(2, 3, 3))
        unit55 = v55.unit()
        self.assertEqual(unit55.shape, (2, 3))

        # Test cross
        v56 = Vector3([1., 0., 0.])
        v57 = Vector3([0., 1., 0.])
        cross56 = v56.cross(v57)
        self.assertEqual(type(cross56), Vector3)
        # Should be (0, 0, 1)
        self.assertTrue(np.allclose(cross56.vals, [0., 0., 1.], atol=1e-10))

        # Test cross with n-D
        v58 = Vector3(np.random.randn(4, 1, 5, 3))
        v59 = Vector3(np.random.randn(8, 5, 3))
        cross58 = v58.cross(v59)
        # Broadcasting: (4, 1, 5) and (8, 5) -> (4, 8, 5)
        self.assertEqual(cross58.shape, (4, 8, 5))

        # Test ucross
        v60 = Vector3([1., 0., 0.])
        v61 = Vector3([0., 1., 0.])
        ucross60 = v60.ucross(v61)
        self.assertEqual(type(ucross60), Vector3)
        # Should be unit vector (0, 0, 1)
        self.assertTrue(np.allclose(ucross60.vals, [0., 0., 1.], atol=1e-10))
        self.assertTrue(np.allclose(ucross60.norm().vals, 1., atol=1e-10))

        # Test outer
        v62 = Vector3([1., 2., 3.])
        v63 = Vector3([4., 5., 6.])
        outer62 = v62.outer(v63)
        self.assertEqual(type(outer62), Matrix)
        # Outer product should be 3x3 matrix
        self.assertEqual(outer62.numer, (3, 3))

        # Test perp
        v64 = Vector3([1., 1., 0.])
        v65 = Vector3([1., 0., 0.])
        perp64 = v64.perp(v65)
        self.assertEqual(type(perp64), Vector3)
        # Component of (1,1,0) perpendicular to (1,0,0) should be (0,1,0)
        self.assertTrue(np.allclose(perp64.vals, [0., 1., 0.], atol=1e-10))

        # Test proj
        v66 = Vector3([1., 1., 0.])
        v67 = Vector3([1., 0., 0.])
        proj66 = v66.proj(v67)
        self.assertEqual(type(proj66), Vector3)
        # Projection of (1,1,0) onto (1,0,0) should be (1,0,0)
        self.assertTrue(np.allclose(proj66.vals, [1., 0., 0.], atol=1e-10))

        # Test sep
        v68 = Vector3([1., 0., 0.])
        v69 = Vector3([0., 1., 0.])
        sep68 = v68.sep(v69)
        self.assertEqual(type(sep68), Scalar)
        # Separation angle between (1,0,0) and (0,1,0) should be pi/2
        self.assertTrue(np.allclose(sep68.vals, np.pi/2, atol=1e-10))

        # Test sep with n-D
        v70 = Vector3(np.random.randn(2, 3, 3))
        v71 = Vector3(np.random.randn(2, 3, 3))
        sep70 = v70.sep(v71)
        self.assertEqual(sep70.shape, (2, 3))

        # Test cross_product_as_matrix
        v72 = Vector3([1., 2., 3.])
        m72 = v72.cross_product_as_matrix()
        self.assertEqual(type(m72), Matrix)
        self.assertEqual(m72.numer, (3, 3))
        # Test that matrix * vector equals cross product
        v73 = Vector3([4., 5., 6.])
        cross72 = v72.cross(v73)
        m72_v73 = m72 * v73
        self.assertTrue(np.allclose(m72_v73.vals, cross72.vals, atol=1e-10))

        # Test cross_product_as_matrix with n-D
        v74 = Vector3(np.random.randn(2, 3, 3))
        m74 = v74.cross_product_as_matrix()
        self.assertEqual(m74.shape, (2, 3))
        self.assertEqual(m74.numer, (3, 3))

        # Test element_mul
        v75 = Vector3([1., 2., 3.])
        v76 = Vector3([4., 5., 6.])
        elem_mul75 = v75.element_mul(v76)
        self.assertEqual(type(elem_mul75), Vector3)
        # Should be (4, 10, 18)
        self.assertTrue(np.allclose(elem_mul75.vals, [4., 10., 18.]))

        # Test element_mul with n-D
        v77 = Vector3(np.random.randn(2, 3, 3))
        v78 = Vector3(np.random.randn(2, 3, 3))
        elem_mul77 = v77.element_mul(v78)
        self.assertEqual(elem_mul77.shape, (2, 3))

        # Test element_div
        v79 = Vector3([4., 10., 18.])
        v80 = Vector3([4., 5., 6.])
        elem_div79 = v79.element_div(v80)
        self.assertEqual(type(elem_div79), Vector3)
        # Should be (1, 2, 3)
        self.assertTrue(np.allclose(elem_div79.vals, [1., 2., 3.], atol=1e-10))

        # Test element_div with n-D
        v81 = Vector3(np.random.randn(2, 3, 3))
        v82 = Vector3(np.random.randn(2, 3, 3))
        elem_div81 = v81.element_div(v82)
        self.assertEqual(elem_div81.shape, (2, 3))

        # Test __abs__ (norm)
        v83 = Vector3([3., 4., 0.])
        abs83 = abs(v83)
        self.assertEqual(type(abs83), Scalar)
        self.assertTrue(np.allclose(abs83.vals, 5.))

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

        # Test complex n-D case
        v87 = Vector3(np.random.randn(3, 4, 5, 6, 3))
        self.assertEqual(v87.shape, (3, 4, 5, 6))
        self.assertEqual(v87.item, (3,))
        self.assertEqual(v87.vals.shape, (3, 4, 5, 6, 3))

        # Test that operations preserve type
        v88 = Vector3([1., 2., 3.])
        v89 = Vector3([4., 5., 6.])
        v_result = v88 + v89
        self.assertEqual(type(v_result), Vector3)

        v_result2 = v88 * 2.
        self.assertEqual(type(v_result2), Vector3)

        # Test round-trip conversions
        v90 = Vector3([1., 2., 3.])
        ra90, dec90, length90 = v90.to_ra_dec_length()
        v90_recon = Vector3.from_ra_dec_length(ra90, dec90, length90)
        self.assertTrue(np.allclose(v90.vals, v90_recon.vals, atol=1e-10))

        v91 = Vector3([1., 2., 3.])
        radius91, longitude91, z91 = v91.to_cylindrical()
        v91_recon = Vector3.from_cylindrical(radius91, longitude91, z91)
        self.assertTrue(np.allclose(v91.vals, v91_recon.vals, atol=1e-10))

        # Test n-D round-trip
        v92 = Vector3(np.random.randn(2, 3, 3))
        ra92, dec92, length92 = v92.to_ra_dec_length()
        v92_recon = Vector3.from_ra_dec_length(ra92, dec92, length92)
        self.assertEqual(v92_recon.shape, (2, 3))
        self.assertTrue(np.allclose(v92.vals, v92_recon.vals, atol=1e-10))

##########################################################################################
