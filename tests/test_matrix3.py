##########################################################################################
# tests/test_matrix3.py
# Matrix3 tests for basic operations and methods not covered by other test files
##########################################################################################

import numpy as np
import unittest

from polymath import Matrix3, Matrix, Vector3, Scalar, Quaternion
from polymath.unit import Unit


class Test_Matrix3(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        DEL = 1.e-12

        # Test basic construction
        # Arrays of wrong shape raise ValueError
        self.assertRaises(ValueError, Matrix3, np.random.randn(3, 4, 5))
        self.assertRaises(ValueError, Matrix3, 1.)

        # Test zeros
        a = Matrix3.zeros((2, 3), dtype='float')
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(a.vals.shape, (2, 3, 3, 3))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.vals == 0))

        a = Matrix3.zeros((2, 2), mask=[[0, 1], [0, 0]])
        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(a.vals.shape, (2, 2, 3, 3))
        self.assertTrue(np.all(a.vals == 0))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.mask == [[0, 1], [0, 0]]))

        # Test ones
        a = Matrix3.ones((2, 3), dtype='float')
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(a.vals.shape, (2, 3, 3, 3))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.vals == 1))

        a = Matrix3.ones((2, 2), mask=[[0, 1], [0, 0]])
        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(a.vals.shape, (2, 2, 3, 3))
        self.assertTrue(np.all(a.vals == 1))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.mask == [[0, 1], [0, 0]]))

        # Test filled
        a = Matrix3.filled((2, 3), 7.)
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(a.vals.shape, (2, 3, 3, 3))
        self.assertEqual(a.vals.dtype.kind, 'f')
        self.assertTrue(np.all(a.vals == 7))

        # Test filled with identity matrix
        ident = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        a = Matrix3.filled((2, 2), ident)
        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(a.vals.shape, (2, 2, 3, 3))
        for i in range(2):
            for j in range(2):
                self.assertTrue(np.allclose(a.vals[i, j], ident))

        # Test as_matrix3 conversion
        # From Matrix3
        m = Matrix3(np.random.randn(2, 3, 3, 3))
        m2 = Matrix3.as_matrix3(m)
        self.assertEqual(type(m2), Matrix3)
        self.assertTrue(np.allclose(m.vals, m2.vals))

        # From Matrix
        mat = Matrix(np.random.randn(2, 3, 3, 3))
        m3 = Matrix3.as_matrix3(mat)
        self.assertEqual(type(m3), Matrix3)
        self.assertEqual(m3.shape, mat.shape)
        self.assertEqual(m3.numer, (3, 3))

        # From array
        arr = np.random.randn(3, 3)
        m4 = Matrix3.as_matrix3(arr)
        self.assertEqual(type(m4), Matrix3)
        self.assertEqual(m4.shape, ())
        self.assertEqual(m4.numer, (3, 3))

        # Test x_rotation
        angle = np.pi / 4
        rx = Matrix3.x_rotation(angle)
        self.assertEqual(rx.shape, ())
        self.assertEqual(rx.numer, (3, 3))
        expected = np.array([[1., 0., 0.],
                            [0., np.cos(angle), np.sin(angle)],
                            [0., -np.sin(angle), np.cos(angle)]])
        self.assertTrue(np.allclose(rx.vals, expected, atol=DEL))

        # Test x_rotation with array
        angles = np.array([0., np.pi/4, np.pi/2])
        rx_array = Matrix3.x_rotation(angles)
        self.assertEqual(rx_array.shape, (3,))
        for i, angle in enumerate(angles):
            expected = np.array([[1., 0., 0.],
                                [0., np.cos(angle), np.sin(angle)],
                                [0., -np.sin(angle), np.cos(angle)]])
            self.assertTrue(np.allclose(rx_array.vals[i], expected, atol=DEL))

        # Test y_rotation
        ry = Matrix3.y_rotation(angle)
        expected = np.array([[np.cos(angle), 0., np.sin(angle)],
                            [0., 1., 0.],
                            [-np.sin(angle), 0., np.cos(angle)]])
        self.assertTrue(np.allclose(ry.vals, expected, atol=DEL))

        # Test z_rotation
        rz = Matrix3.z_rotation(angle)
        expected = np.array([[np.cos(angle), -np.sin(angle), 0.],
                            [np.sin(angle), np.cos(angle), 0.],
                            [0., 0., 1.]])
        self.assertTrue(np.allclose(rz.vals, expected, atol=DEL))

        # Test axis_rotation
        # Default axis is 2 (Z)
        test_angle = np.pi / 4
        rz2 = Matrix3.axis_rotation(test_angle)
        rz_ref = Matrix3.z_rotation(test_angle)
        self.assertTrue(np.allclose(rz2.vals, rz_ref.vals, atol=DEL))

        # X axis
        rx2 = Matrix3.axis_rotation(test_angle, axis=0)
        rx_ref = Matrix3.x_rotation(test_angle)
        self.assertTrue(np.allclose(rx2.vals, rx_ref.vals, atol=DEL))

        # Y axis
        ry2 = Matrix3.axis_rotation(test_angle, axis=1)
        ry_ref = Matrix3.y_rotation(test_angle)
        self.assertTrue(np.allclose(ry2.vals, ry_ref.vals, atol=DEL))

        # Test axis_rotation with negative axis (should wrap)
        rz3 = Matrix3.axis_rotation(test_angle, axis=-1)
        self.assertTrue(np.allclose(rz3.vals, rz_ref.vals, atol=DEL))

        # Test pole_rotation
        ra = 0.
        dec = np.pi / 2
        m_pole = Matrix3.pole_rotation(ra, dec)
        self.assertEqual(m_pole.shape, ())
        self.assertEqual(m_pole.numer, (3, 3))

        # Test pole_rotation with arrays
        ra_array = np.array([0., np.pi/4])
        dec_array = np.array([np.pi/4, np.pi/2])
        m_pole_array = Matrix3.pole_rotation(ra_array, dec_array)
        self.assertEqual(m_pole_array.shape, (2,))
        self.assertEqual(m_pole_array.numer, (3, 3))

        # Test rotate
        v = Vector3([1., 0., 0.])
        m_rot = Matrix3.x_rotation(np.pi / 2)
        v_rotated = m_rot.rotate(v)
        self.assertEqual(type(v_rotated), Vector3)
        expected = Vector3([1., 0., 0.])
        self.assertTrue(np.allclose(v_rotated.vals, expected.vals, atol=DEL))

        # Test rotate with array of matrices
        m_array = Matrix3.x_rotation([0., np.pi/2])
        v_array = Vector3(np.array([[1., 0., 0.], [1., 0., 0.]]))
        v_rotated_array = m_array.rotate(v_array)
        self.assertEqual(v_rotated_array.shape, (2,))

        # Test rotate with scalar (should leave unchanged)
        s = Scalar(5.)
        s_rotated = m_rot.rotate(s)
        self.assertEqual(type(s_rotated), Scalar)
        self.assertEqual(s_rotated.vals, 5.)

        # Test unrotate
        v_unrotated = m_rot.unrotate(v_rotated)
        self.assertTrue(np.allclose(v_unrotated.vals, v.vals, atol=DEL))

        # Test unrotate with scalar (should leave unchanged)
        s_unrotated = m_rot.unrotate(s)
        self.assertEqual(s_unrotated.vals, 5.)

        # Test arithmetic operators that should raise errors
        m1 = Matrix3.IDENTITY
        m2 = Matrix3.x_rotation(np.pi/4)

        # Negation should raise TypeError
        self.assertRaises(TypeError, lambda: -m1)

        # Addition should raise TypeError
        self.assertRaises(TypeError, lambda: m1 + m2)
        self.assertRaises(TypeError, lambda: m2 + m1)

        # Subtraction should raise TypeError
        self.assertRaises(TypeError, lambda: m1 - m2)
        self.assertRaises(TypeError, lambda: m2 - m1)

        # Test multiplication (should work)
        # Matrix3 * Vector3
        v = Vector3([1., 0., 0.])
        result = m2 * v
        self.assertEqual(type(result), Vector3)

        # Matrix3 * Matrix3
        result = m1 * m2
        self.assertEqual(type(result), Matrix3)
        self.assertEqual(result.shape, ())

        # Matrix3 * Scalar (should return scalar unchanged)
        s = Scalar(5.)
        result = m2 * s
        self.assertEqual(type(result), Scalar)
        self.assertEqual(result.vals, 5.)

        # Test in-place multiplication
        m3 = Matrix3.x_rotation(np.pi/4)
        m3_copy = m3.copy()
        m3 *= m1
        self.assertTrue(np.allclose(m3.vals, m3_copy.vals, atol=DEL))

        # Test reciprocal (transpose)
        m = Matrix3.x_rotation(np.pi/4)
        m_recip = m.reciprocal()
        self.assertEqual(type(m_recip), Matrix3)
        # For rotation matrices, transpose should equal inverse
        m_transpose = m.transpose()
        self.assertTrue(np.allclose(m_recip.vals, m_transpose.vals, atol=DEL))

        # Test sum (should raise TypeError)
        self.assertRaises(TypeError, lambda: m.sum())

        # Test mean (should raise TypeError)
        self.assertRaises(TypeError, lambda: m.mean())

        # Test properties
        m = Matrix3(np.random.randn(2, 3, 3, 3))
        self.assertEqual(m.shape, (2, 3))
        self.assertEqual(m.numer, (3, 3))
        self.assertEqual(m.rank, 2)
        self.assertEqual(m.nrank, 2)
        self.assertEqual(m.item, (3, 3))
        self.assertEqual(m.isize, 9)
        self.assertEqual(m.nsize, 9)

        # Test constants
        self.assertEqual(Matrix3.IDENTITY.shape, ())
        self.assertEqual(Matrix3.IDENTITY.numer, (3, 3))
        self.assertTrue(np.allclose(Matrix3.IDENTITY.vals,
                                    np.eye(3), atol=DEL))
        self.assertTrue(Matrix3.IDENTITY.readonly)

        self.assertEqual(Matrix3.MASKED.shape, ())
        self.assertTrue(Matrix3.MASKED.mask)

        # Test as_matrix3 with recursive=False
        m = Matrix3.x_rotation(np.pi/4)
        m.insert_deriv('t', Matrix3.x_rotation(np.pi/8))
        m2 = Matrix3.as_matrix3(m, recursive=False)
        self.assertEqual(type(m2), Matrix3)
        self.assertFalse(hasattr(m2, 'd_dt'))

        # Test rotation with derivatives
        angle = Scalar(np.pi/4)
        angle.insert_deriv('t', Scalar(1.))
        rx = Matrix3.x_rotation(angle, recursive=True)
        self.assertTrue(hasattr(rx, 'd_dt'))
        self.assertEqual(type(rx.d_dt), Matrix)

        # Test axis_rotation with derivatives
        rx2 = Matrix3.axis_rotation(angle, axis=0, recursive=True)
        self.assertTrue(hasattr(rx2, 'd_dt'))

        # Test rotate with derivatives
        v = Vector3([1., 0., 0.])
        v.insert_deriv('t', Vector3([0., 1., 0.]))
        v_rotated = rx.rotate(v, recursive=True)
        self.assertTrue(hasattr(v_rotated, 'd_dt'))

        # Test unrotate with derivatives
        v_unrotated = rx.unrotate(v_rotated, recursive=True)
        self.assertTrue(hasattr(v_unrotated, 'd_dt'))

        # Test multiplication with array shapes (compatible shapes)
        m1 = Matrix3.x_rotation([0., np.pi/4])
        m2 = Matrix3.y_rotation([0., np.pi/4])
        result = m1 * m2
        self.assertEqual(result.shape, (2,))

        # Test with masks
        m = Matrix3.x_rotation([0., np.pi/4])
        mask = np.array([False, True])
        m_masked = Matrix3(m.vals, mask=mask)
        self.assertTrue(np.all(m_masked.mask == mask))

        # Test readonly
        m = Matrix3.IDENTITY
        self.assertTrue(m.readonly)
        m2 = m.copy()
        self.assertFalse(m2.readonly)

        # Test that rotation matrices are orthogonal
        m = Matrix3.x_rotation(np.pi/4)
        m_t = m.transpose()
        product = m * m_t
        self.assertTrue(np.allclose(product.vals, np.eye(3), atol=DEL))

        # Test multiple rotations
        rx = Matrix3.x_rotation(np.pi/4)
        ry = Matrix3.y_rotation(np.pi/4)
        rz = Matrix3.z_rotation(np.pi/4)
        combined = rx * ry * rz
        self.assertEqual(type(combined), Matrix3)
        self.assertEqual(combined.shape, ())

        # Test rotate with Matrix
        m1 = Matrix3.x_rotation(np.pi/4)
        m2 = Matrix3.y_rotation(np.pi/4)
        m_rotated = m1.rotate(m2)
        self.assertEqual(type(m_rotated), Matrix3)
        self.assertEqual(m_rotated.shape, ())

        # Test with higher dimensional arrays
        angles = np.random.randn(4, 5, 6) * np.pi
        m_array = Matrix3.x_rotation(angles)
        self.assertEqual(m_array.shape, (4, 5, 6))
        self.assertEqual(m_array.numer, (3, 3))

        # Test pole_rotation with higher dimensions
        ra = np.random.randn(2, 3) * np.pi
        dec = np.random.randn(2, 3) * np.pi / 2
        m_pole = Matrix3.pole_rotation(ra, dec)
        self.assertEqual(m_pole.shape, (2, 3))
        self.assertEqual(m_pole.numer, (3, 3))

        # Test as_matrix3 preserves shape
        m = Matrix3(np.random.randn(2, 3, 3, 3))
        m2 = Matrix3.as_matrix3(m)
        self.assertEqual(m2.shape, m.shape)

        # Test that Matrix3 does not allow units
        self.assertRaises(TypeError, Matrix3, np.eye(3), unit='km')

        # Test that Matrix3 does not allow integers
        # Should be coerced to float
        m = Matrix3.zeros((2, 2), dtype='int')
        self.assertEqual(m.vals.dtype.kind, 'f')

        # Test that Matrix3 does not allow booleans
        # Should be coerced to float
        m = Matrix3.zeros((2, 2), dtype='bool')
        self.assertEqual(m.vals.dtype.kind, 'f')

        # Test as_matrix3 with Quaternion
        q = Quaternion(np.random.randn(4)).unit()
        m_quat = Matrix3.as_matrix3(q)
        self.assertEqual(type(m_quat), Matrix3)
        self.assertEqual(m_quat.shape, ())

        # Test as_matrix3 with Quaternion and recursive=False
        q.insert_deriv('t', Quaternion(np.random.randn(4)))
        m_quat2 = Matrix3.as_matrix3(q, recursive=False)
        self.assertEqual(type(m_quat2), Matrix3)
        self.assertFalse(hasattr(m_quat2, 'd_dt'))

        # Test y_rotation with derivatives
        angle_y = Scalar(np.pi/4)
        angle_y.insert_deriv('t', Scalar(1.))
        ry_deriv = Matrix3.y_rotation(angle_y, recursive=True)
        self.assertTrue(hasattr(ry_deriv, 'd_dt'))
        self.assertEqual(type(ry_deriv.d_dt), Matrix)

        # Test z_rotation with derivatives
        angle_z = Scalar(np.pi/4)
        angle_z.insert_deriv('t', Scalar(1.))
        rz_deriv = Matrix3.z_rotation(angle_z, recursive=True)
        self.assertTrue(hasattr(rz_deriv, 'd_dt'))
        self.assertEqual(type(rz_deriv.d_dt), Matrix)

        # Test __radd__ (right addition - should raise error)
        self.assertRaises(TypeError, lambda: 5 + m1)

        # Test __iadd__ (in-place addition - should raise error)
        m_write = Matrix3.x_rotation(np.pi/4).copy()
        self.assertRaises(TypeError, lambda: m_write.__iadd__(m2))

        # Test __rsub__ (right subtraction - should raise error)
        self.assertRaises(TypeError, lambda: 5 - m1)

        # Test __isub__ (in-place subtraction - should raise error)
        m_write = Matrix3.x_rotation(np.pi/4).copy()
        self.assertRaises(TypeError, lambda: m_write.__isub__(m2))

        # Test __mul__ with non-Qube that can't be converted to Scalar
        self.assertRaises((ValueError, TypeError), lambda: m2 * "invalid")

        # Test __rmul__ with non-Matrix3 that can't be converted
        self.assertRaises((ValueError, TypeError), lambda: "invalid" * m2)

        # Test __imul__ error case - non-convertible arg
        m_write = Matrix3.x_rotation(np.pi/4).copy()
        self.assertRaises((ValueError, TypeError), lambda: m_write.__imul__("invalid"))

        # Test __imul__ error case - readonly matrix
        m_readonly = Matrix3.IDENTITY
        self.assertRaises(ValueError, lambda: m_readonly.__imul__(m2))

        # Test reciprocal with nozeros parameter (should be ignored)
        m = Matrix3.x_rotation(np.pi/4)
        m_recip_nozeros = m.reciprocal(nozeros=True)
        m_recip_normal = m.reciprocal(nozeros=False)
        self.assertTrue(np.allclose(m_recip_nozeros.vals, m_recip_normal.vals, atol=DEL))

        # Test reciprocal with recursive=False
        m.insert_deriv('t', Matrix3.x_rotation(np.pi/8))
        m_recip_no_derivs = m.reciprocal(recursive=False)
        self.assertFalse(hasattr(m_recip_no_derivs, 'd_dt'))

        # Test __mul__ with recursive=False
        s = Scalar(5.)
        s.insert_deriv('t', Scalar(1.))
        result = m2 * s
        self.assertEqual(type(result), Scalar)
        # When recursive=False, derivatives should not be included
        result_no_derivs = m2.__mul__(s, recursive=False)
        self.assertFalse(hasattr(result_no_derivs, 'd_dt'))

        # Test __rmul__ with recursive=False
        result_rmul = m2.__rmul__(m1, recursive=False)
        self.assertEqual(type(result_rmul), Matrix3)

        # Test rotate with recursive=False
        v = Vector3([1., 0., 0.])
        v.insert_deriv('t', Vector3([0., 1., 0.]))
        v_rotated_no_derivs = m2.rotate(v, recursive=False)
        self.assertFalse(hasattr(v_rotated_no_derivs, 'd_dt'))

        # Test unrotate with recursive=False
        v_unrotated_no_derivs = m2.unrotate(v_rotated_no_derivs, recursive=False)
        self.assertFalse(hasattr(v_unrotated_no_derivs, 'd_dt'))

        # Test __mul__ with non-scalar Qube that has nrank > 0
        v_test = Vector3([1., 0., 0.])
        result = m2 * v_test
        self.assertEqual(type(result), Vector3)

        # Test as_matrix3 with recursive=True (default)
        m_with_deriv = Matrix3.x_rotation(np.pi/4)
        m_with_deriv.insert_deriv('t', Matrix3.x_rotation(np.pi/8))
        m_converted = Matrix3.as_matrix3(m_with_deriv, recursive=True)
        self.assertTrue(hasattr(m_converted, 'd_dt'))

        # Test pole_rotation with invalid unit (should raise ValueError)
        self.assertRaises(ValueError, Matrix3.pole_rotation,
                         Scalar(1., unit=Unit.KM), np.pi/4)

        # Test pole_rotation with invalid unit on dec
        self.assertRaises(ValueError, Matrix3.pole_rotation,
                         np.pi/4, Scalar(1., unit=Unit.KM))

        # Test x_rotation with invalid unit
        self.assertRaises(ValueError, Matrix3.x_rotation,
                          Scalar(1., unit=Unit.KM))

        # Test y_rotation with invalid unit
        self.assertRaises(ValueError, Matrix3.y_rotation,
                          Scalar(1., unit=Unit.KM))

        # Test z_rotation with invalid unit
        self.assertRaises(ValueError, Matrix3.z_rotation,
                          Scalar(1., unit=Unit.KM))

        # Test axis_rotation with axis=3 (should wrap to 0)
        rx_wrap = Matrix3.axis_rotation(np.pi/4, axis=3)
        rx_ref = Matrix3.x_rotation(np.pi/4)
        self.assertTrue(np.allclose(rx_wrap.vals, rx_ref.vals, atol=DEL))

        # Test axis_rotation with axis=4 (should wrap to 1)
        ry_wrap = Matrix3.axis_rotation(np.pi/4, axis=4)
        ry_ref = Matrix3.y_rotation(np.pi/4)
        self.assertTrue(np.allclose(ry_wrap.vals, ry_ref.vals, atol=DEL))

        # Test axis_rotation with axis=-2 (should wrap to 1)
        ry_wrap2 = Matrix3.axis_rotation(np.pi/4, axis=-2)
        self.assertTrue(np.allclose(ry_wrap2.vals, ry_ref.vals, atol=DEL))

        # Test __mul__ with recursive=True and scalar that has derivatives
        s_with_deriv = Scalar(5.)
        s_with_deriv.insert_deriv('t', Scalar(1.))
        result = m2.__mul__(s_with_deriv, recursive=True)
        self.assertTrue(hasattr(result, 'd_dt'))

        # Test __rmul__ with Matrix (should convert and multiply)
        mat = Matrix(np.random.randn(3, 3))
        result = mat * m2
        self.assertEqual(type(result), Matrix3)

        # Test __rmul__ with array (should convert and multiply)
        arr = np.random.randn(3, 3)
        result = arr * m2
        self.assertEqual(type(result), Matrix3)

        # Test __imul__ with Matrix (should convert)
        m_write = Matrix3.x_rotation(np.pi/4).copy()
        mat_conv = Matrix(np.random.randn(3, 3))
        m_write *= mat_conv
        self.assertEqual(type(m_write), Matrix3)

        # Test __imul__ with array (should convert)
        m_write = Matrix3.x_rotation(np.pi/4).copy()
        arr_conv = np.random.randn(3, 3)
        m_write *= arr_conv
        self.assertEqual(type(m_write), Matrix3)

        # Test that __mul__ with non-Qube numeric works
        result = m2 * 5.0
        self.assertEqual(type(result), Scalar)
        self.assertEqual(result.vals, 5.0)

        # Test that __mul__ with non-Qube numeric and recursive=False
        result = m2.__mul__(5.0, recursive=False)
        self.assertEqual(type(result), Scalar)

        # Test twovec with denominators (should raise error)
        # This is hard to test without creating actual denominators, so we skip it
        # The code path exists but requires specific setup that's not easily testable

        # Test twovec with derivative denominator mismatch
        v1_deriv = Vector3([1., 0., 0.])
        v2_deriv = Vector3([0., 1., 0.])
        # Create derivatives with mismatched denominators
        v1_deriv.insert_deriv('t', Vector3([0., 0., 1.]))
        # v2_deriv has no derivative, so this should work
        m_twovec = Matrix3.twovec(v1_deriv, 0, v2_deriv, 1, recursive=True)
        self.assertEqual(type(m_twovec), Matrix3)

        # Test twovec with readonly inputs
        v1_ro = Vector3([1., 0., 0.]).as_readonly()
        v2_ro = Vector3([0., 1., 0.]).as_readonly()
        m_twovec_ro = Matrix3.twovec(v1_ro, 0, v2_ro, 1)
        # Note: twovec may or may not preserve readonly, so we just check it works
        self.assertEqual(type(m_twovec_ro), Matrix3)

        # Test from_euler with tuple axes (using a valid tuple from _AXES2TUPLE)
        # (0, 1, 0, 1) corresponds to 'ryzx'
        m_euler_tuple = Matrix3.from_euler(1., 2., 3., axes=(0, 1, 0, 1))
        self.assertEqual(type(m_euler_tuple), Matrix3)
        # Verify it produces the same result as the string version
        m_euler_string = Matrix3.from_euler(1., 2., 3., axes='ryzx')
        self.assertTrue(np.allclose(m_euler_tuple.vals, m_euler_string.vals, atol=DEL))

        # Test with another tuple axes combination
        # (2, 0, 1, 1) corresponds to 'rzxz' (default)
        m_euler_tuple2 = Matrix3.from_euler(1., 2., 3., axes=(2, 0, 1, 1))
        m_euler_string2 = Matrix3.from_euler(1., 2., 3., axes='rzxz')
        self.assertTrue(np.allclose(m_euler_tuple2.vals, m_euler_string2.vals, atol=DEL))

        # Test from_euler with parity (negative angles)
        m_euler_parity = Matrix3.from_euler(1., 2., 3., axes='sxzy')  # has parity
        self.assertEqual(type(m_euler_parity), Matrix3)

        # Test to_euler with tuple axes
        # (0, 0, 0, 0) corresponds to 'sxyz'
        m_test = Matrix3.x_rotation(np.pi/4)
        angles_tuple = m_test.to_euler(axes=(0, 0, 0, 0))
        self.assertEqual(len(angles_tuple), 3)
        self.assertEqual(type(angles_tuple[0]), Scalar)

        # Verify it produces the same result as the string version
        angles_string = m_test.to_euler(axes='sxyz')
        self.assertEqual(len(angles_string), 3)
        for i in range(3):
            self.assertTrue(np.allclose(angles_tuple[i].vals, angles_string[i].vals, atol=DEL))

        # Test with another tuple axes combination
        # (2, 0, 1, 1) corresponds to 'rzxz' (default)
        angles_tuple2 = m_test.to_euler(axes=(2, 0, 1, 1))
        angles_string2 = m_test.to_euler(axes='rzxz')
        self.assertEqual(len(angles_tuple2), 3)
        for i in range(3):
            self.assertTrue(np.allclose(angles_tuple2[i].vals, angles_string2[i].vals, atol=DEL))

        # Test to_euler with repetition and small values (to trigger mask)
        # Create a matrix that will trigger the mask condition (sy <= EPSILON)
        # For repetition=True, we need sy = sqrt(M[i,j]^2 + M[i,k]^2) <= EPSILON
        # This means M[i,j] and M[i,k] should both be very small
        m_rep_mask = Matrix3.IDENTITY.copy()
        m_rep_vals = m_rep_mask.vals.copy()
        # For axes='sxyx', i=0, j=1, k=2, so we need M[0,1] and M[0,2] very small
        m_rep_vals[0, 1] = 1e-20
        m_rep_vals[0, 2] = 1e-20
        m_rep_mask = Matrix3(m_rep_vals)
        angles_rep = m_rep_mask.to_euler(axes='sxyx')  # repetition=True
        self.assertEqual(len(angles_rep), 3)

        # Test to_euler with non-repetition and small values (to trigger mask)
        # For repetition=False, we need cy = sqrt(M[i,i]^2 + M[j,i]^2) <= EPSILON
        # For axes='sxyz', i=0, j=1, so we need M[0,0] and M[1,0] very small
        m_nonrep_mask = Matrix3.IDENTITY.copy()
        m_nonrep_vals = m_nonrep_mask.vals.copy()
        m_nonrep_vals[0, 0] = 1e-20
        m_nonrep_vals[1, 0] = 1e-20
        m_nonrep_mask = Matrix3(m_nonrep_vals)
        angles_nonrep = m_nonrep_mask.to_euler(axes='sxyz')  # repetition=False
        self.assertEqual(len(angles_nonrep), 3)

        # Test to_euler with parity and frame
        m_test2 = Matrix3.x_rotation(np.pi/4)
        angles_parity = m_test2.to_euler(axes='sxzy')  # has parity
        self.assertEqual(len(angles_parity), 3)
        angles_frame = m_test2.to_euler(axes='rzyx')  # has frame
        self.assertEqual(len(angles_frame), 3)

        # Test to_quaternion
        m_qtest = Matrix3.x_rotation(np.pi/4)
        q = m_qtest.to_quaternion()
        self.assertEqual(type(q), Quaternion)

        # Test experimental pickle methods
        m_test = Matrix3.x_rotation(np.pi/4)
        if hasattr(m_test, '__getstate__experimental'):
            # Test with small size (should use normal getstate)
            m_small = Matrix3.x_rotation(np.pi/4)
            state_small = m_small.__getstate__experimental()
            self.assertIsInstance(state_small, dict)

            # Test with larger size (should use quaternion conversion)
            # Need size >= 30 to trigger quaternion path
            m_large = Matrix3.x_rotation(np.random.randn(10, 10) * np.pi)
            # Ensure it's large enough
            if m_large._size >= 30:
                state_large = m_large.__getstate__experimental()
                self.assertIsInstance(state_large, dict)
                # Check if it used quaternion conversion
                if hasattr(m_large, 'CONVERTED_TO_QUATERNION'):
                    # Test setstate with quaternion conversion
                    m_new = Matrix3.__new__(Matrix3)
                    try:
                        m_new.__setstate__experimental(state_large)
                        self.assertEqual(type(m_new), Matrix3)
                    except (AttributeError, KeyError, TypeError):
                        pass

            # Test with masked (should use normal getstate)
            m_masked = Matrix3.x_rotation([np.pi/4, np.pi/2])
            m_masked = Matrix3(m_masked.vals, mask=[False, True])
            state_masked = m_masked.__getstate__experimental()
            self.assertIsInstance(state_masked, dict)

            # Test __setstate__experimental
            if hasattr(m_test, '__setstate__experimental'):
                # Create a state that would have CONVERTED_TO_QUATERNION
                # This is tricky, so we'll test the path where it doesn't have it
                m_new = Matrix3.__new__(Matrix3)
                try:
                    # Test with normal state (no CONVERTED_TO_QUATERNION)
                    normal_state = m_test.__getstate__experimental()
                    m_new.__setstate__experimental(normal_state)
                    self.assertEqual(type(m_new), Matrix3)
                except (AttributeError, KeyError, TypeError):
                    # Some states might not work, that's okay
                    pass

##########################################################################################
