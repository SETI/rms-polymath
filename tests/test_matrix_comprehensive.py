##########################################################################################
# tests/test_matrix_comprehensive.py
# Comprehensive unit tests for Matrix class based on docstrings
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Vector, Matrix, Vector3


class Test_Matrix_Comprehensive(unittest.TestCase):

    def runTest(self):

        np.random.seed(9012)

        # Test as_matrix static method
        m1 = Matrix([[1., 2.], [3., 4.]])
        m1_conv = Matrix.as_matrix(m1)
        self.assertEqual(type(m1_conv), Matrix)
        self.assertTrue(np.allclose(m1_conv.vals, [[1., 2.], [3., 4.]]))

        # Array to Matrix
        m2 = Matrix.as_matrix([[1., 2.], [3., 4.]])
        self.assertEqual(type(m2), Matrix)

        # Test row_vector method
        m3 = Matrix([[1., 2., 3.], [4., 5., 6.]])
        v1 = m3.row_vector(0)
        self.assertEqual(type(v1), Vector3)  # Should be Vector3 for length 3
        self.assertTrue(np.allclose(v1.vals, [1., 2., 3.]))

        # Test row_vectors method
        rows = m3.row_vectors()
        self.assertEqual(len(rows), 2)
        self.assertTrue(np.allclose(rows[0].vals, [1., 2., 3.]))
        self.assertTrue(np.allclose(rows[1].vals, [4., 5., 6.]))

        # Test column_vector method
        v2 = m3.column_vector(0)
        self.assertEqual(type(v2), Vector)
        self.assertTrue(np.allclose(v2.vals, [1., 4.]))

        # Test column_vectors method
        cols = m3.column_vectors()
        self.assertEqual(len(cols), 3)
        self.assertTrue(np.allclose(cols[0].vals, [1., 4.]))

        # Test to_vector method
        v3 = m3.to_vector(0, 0)
        self.assertEqual(type(v3), Vector)
        self.assertTrue(np.allclose(v3.vals, [1., 2., 3.]))

        # Test to_scalar method
        s1 = m3.to_scalar(0, 1)
        self.assertEqual(type(s1), Scalar)
        self.assertEqual(s1, 2.)

        # Test from_scalars static method
        s2 = Scalar(1.)
        s3 = Scalar(2.)
        s4 = Scalar(3.)
        s5 = Scalar(4.)
        m4 = Matrix.from_scalars(s2, s3, s4, s5)
        self.assertEqual(type(m4), Matrix)
        self.assertEqual(m4.numer, (2, 2))
        self.assertTrue(np.allclose(m4.vals, [[1., 2.], [3., 4.]]))

        # Test is_diagonal method
        m5 = Matrix([[1., 0.], [0., 2.]])
        b1 = m5.is_diagonal()
        self.assertTrue(b1)

        m6 = Matrix([[1., 1.], [0., 2.]])
        b2 = m6.is_diagonal()
        self.assertFalse(b2)

        # Test transpose method
        m7 = Matrix([[1., 2., 3.], [4., 5., 6.]])
        m8 = m7.transpose()
        self.assertEqual(m8.numer, (3, 2))
        self.assertTrue(np.allclose(m8.vals, [[1., 4.], [2., 5.], [3., 6.]]))

        # Test T property
        m9 = m7.T
        self.assertTrue(np.allclose(m9.vals, [[1., 4.], [2., 5.], [3., 6.]]))

        # Test inverse method
        m10 = Matrix([[1., 2.], [3., 4.]])
        m11 = m10.inverse()
        # m10 * m11 should be identity
        m12 = m10 * m11
        self.assertAlmostEqual(m12.to_scalar(0, 0), 1., places=10)
        self.assertAlmostEqual(m12.to_scalar(0, 1), 0., places=10)
        self.assertAlmostEqual(m12.to_scalar(1, 0), 0., places=10)
        self.assertAlmostEqual(m12.to_scalar(1, 1), 1., places=10)

        # Test unitary method (requires 3x3 matrix)
        # Create a 3x3 rotation matrix (unitary)
        angle = np.pi/4
        m13 = Matrix([[np.cos(angle), -np.sin(angle), 0.],
                      [np.sin(angle), np.cos(angle), 0.],
                      [0., 0., 1.]])
        m14 = m13.unitary()
        # Should return a unitary matrix close to the original
        self.assertEqual(m14.numer, (3, 3))
        self.assertTrue(np.allclose(m14.vals, m13.vals, atol=1e-10))

        # Test __abs__ method (should raise TypeError)
        m15 = Matrix([[1., 2.], [3., 4.]])
        self.assertRaises(TypeError, abs, m15)

        # Test identity method
        m16 = Matrix([[1., 2.], [3., 4.]])
        m17 = m16.identity()
        self.assertEqual(m17.numer, (2, 2))
        self.assertTrue(np.allclose(m17.vals, [[1., 0.], [0., 1.]]))

        # Test reciprocal method (should be same as inverse)
        m18 = Matrix([[1., 2.], [3., 4.]])
        m19 = m18.reciprocal()
        m20 = m18.inverse()
        self.assertTrue(np.allclose(m19.vals, m20.vals))

        # n-D test cases
        # Test row_vector with n-D matrix
        m21 = Matrix([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        # m21 has shape (2,) and numer (2, 2)
        v4 = m21.row_vector(0)
        self.assertEqual(v4.shape, (2,))
        # v4 should be a Vector with shape (2,) and numer (2,)
        # First element should be [1, 2] from first matrix, second should be [5, 6] from second matrix
        self.assertTrue(np.allclose(v4.vals[0], [1., 2.]))
        self.assertTrue(np.allclose(v4.vals[1], [5., 6.]))

        # Test column_vector with n-D matrix
        v5 = m21.column_vector(0)
        self.assertEqual(v5.shape, (2,))
        # v5 should extract column 0 from each matrix: [1, 3] and [5, 7]
        self.assertTrue(np.allclose(v5.vals[0], [1., 3.]))
        self.assertTrue(np.allclose(v5.vals[1], [5., 7.]))

        # Test transpose with n-D matrix
        m22 = m21.transpose()
        self.assertEqual(m22.shape, (2,))
        self.assertEqual(m22.numer, (2, 2))

        # Test inverse with n-D matrix
        m23 = Matrix([[[1., 2.], [3., 4.]], [[2., 1.], [1., 2.]]])
        m24 = m23.inverse()
        self.assertEqual(m24.shape, (2,))
        # Check that m23 * m24 gives identity for each
        m25 = m23 * m24
        # Access individual matrices using indexing, then use to_scalar
        # For first matrix (index 0) - use extract_numer to get the matrix
        m25_0 = m25[0]
        self.assertAlmostEqual(m25_0.to_scalar(0, 0), 1., places=10)
        self.assertAlmostEqual(m25_0.to_scalar(0, 1), 0., places=10)
        self.assertAlmostEqual(m25_0.to_scalar(1, 0), 0., places=10)
        self.assertAlmostEqual(m25_0.to_scalar(1, 1), 1., places=10)
        # For second matrix (index 1)
        m25_1 = m25[1]
        self.assertAlmostEqual(m25_1.to_scalar(0, 0), 1., places=10)
        self.assertAlmostEqual(m25_1.to_scalar(0, 1), 0., places=10)
        self.assertAlmostEqual(m25_1.to_scalar(1, 0), 0., places=10)
        self.assertAlmostEqual(m25_1.to_scalar(1, 1), 1., places=10)

        # Test from_scalars with n-D scalars
        # For shape=(2, 2), we need 4 scalars total (2*2=4)
        s6 = Scalar(1.)
        s7 = Scalar(2.)
        s8 = Scalar(3.)
        s9 = Scalar(4.)
        m26 = Matrix.from_scalars(s6, s7, s8, s9)
        self.assertEqual(m26.shape, ())
        self.assertEqual(m26.numer, (2, 2))
        # Test with n-D scalars that broadcast
        s10 = Scalar([[1., 2.], [3., 4.]])
        s11 = Scalar([[5., 6.], [7., 8.]])
        s12 = Scalar([[9., 10.], [11., 12.]])
        s13 = Scalar([[13., 14.], [15., 16.]])
        # Without shape, it should create a square matrix
        m27 = Matrix.from_scalars(s10, s11, s12, s13)
        self.assertEqual(m27.shape, (2, 2))
        self.assertEqual(m27.numer, (2, 2))

        # Test is_diagonal with n-D matrix
        m27 = Matrix([[[1., 0.], [0., 2.]], [[3., 0.], [0., 4.]]])
        b3 = m27.is_diagonal()
        self.assertEqual(b3.shape, (2,))
        self.assertTrue(b3[0])
        self.assertTrue(b3[1])

        # Test as_matrix with Vector drank=1
        v6 = Vector([[1., 0.], [0., 1.]], drank=1)
        m28 = Matrix.as_matrix(v6)
        self.assertEqual(type(m28), Matrix)
        # Note: join_items may change drank, so just check it's a Matrix

        # Test as_matrix with recursive=False
        m29 = Matrix([[1., 2.], [3., 4.]])
        m29.insert_deriv('t', Matrix([[5., 6.], [7., 8.]]))
        m30 = Matrix.as_matrix(m29, recursive=False)
        self.assertEqual(len(m30.derivs), 0)

        # Test from_scalars with shape parameter
        s14 = Scalar(1.)
        s15 = Scalar(2.)
        s16 = Scalar(3.)
        s17 = Scalar(4.)
        m31 = Matrix.from_scalars(s14, s15, s16, s17, shape=(2, 2))
        self.assertEqual(m31.numer, (2, 2))

        # Test from_scalars with wrong number of scalars
        self.assertRaises(ValueError, Matrix.from_scalars, s14, s15, s16, shape=(2, 2))

        # Test from_scalars with invalid shape
        self.assertRaises(ValueError, Matrix.from_scalars, s14, s15, s16, s17, shape=(2,))

        # Test from_scalars with int matrix (error)
        s18 = Scalar(1)
        s19 = Scalar(2)
        s20 = Scalar(3)
        s21 = Scalar(4)
        self.assertRaises(TypeError, Matrix.from_scalars, s18, s19, s20, s21)

        # Test is_diagonal with non-square matrix (error)
        m32 = Matrix([[1., 2., 3.], [4., 5., 6.]])
        self.assertRaises(ValueError, m32.is_diagonal)

        # Test is_diagonal with denominators (error)
        # For drank=1, Matrix with numer (2,2) needs shape (2, 2, p) where p is denominator
        # Create a 3D array: shape (2, 2, 3) for numer (2,2) and denominator size 3
        m33_vals = np.array([[[1., 0., 0.], [0., 2., 0.]], [[0., 0., 3.], [0., 0., 0.]]])
        m33 = Matrix(m33_vals, drank=1)
        self.assertRaises(ValueError, m33.is_diagonal)

        # Test is_diagonal with delta parameter
        m34 = Matrix([[1., 0.01], [0.01, 2.]])
        b4 = m34.is_diagonal(delta=0.1)
        self.assertTrue(b4)

        # Test is_diagonal with masked matrix
        # Simply test that a masked diagonal matrix returns True
        # Create a matrix array and mask one
        m35_array = Matrix([[[1., 0.], [0., 2.]], [[3., 0.], [0., 4.]]])
        m35_masked = m35_array.mask_where(np.array([True, False]))
        b5 = m35_masked.is_diagonal()
        # First matrix is masked, should return True
        # Second matrix is diagonal, should return True
        # b5 is a Boolean, check it properly
        # b5 is a Boolean array with shape (2,)
        self.assertEqual(b5.shape, (2,))
        self.assertTrue(b5.vals[0])  # Masked matrix returns True
        self.assertTrue(b5.vals[1])  # Diagonal matrix returns True

        # Test transpose with recursive=False
        m36 = Matrix([[1., 2.], [3., 4.]])
        m36.insert_deriv('t', Matrix([[5., 6.], [7., 8.]]))
        m37 = m36.transpose(recursive=False)
        self.assertEqual(len(m37.derivs), 0)

        # Test inverse with non-square matrix (error)
        m38 = Matrix([[1., 2., 3.], [4., 5., 6.]])
        self.assertRaises(ValueError, m38.inverse)

        # Test inverse with denominators (error)
        # For drank=1, Matrix with numer (2,2) needs shape (2, 2, m)
        m39_vals = np.array([[[1., 2., 0.], [3., 4., 0.]], [[0., 0., 1.], [0., 0., 1.]]])
        m39 = Matrix(m39_vals, drank=1)
        self.assertRaises(ValueError, m39.inverse)

        # Test inverse with nozeros=True
        m40 = Matrix([[1., 2.], [3., 4.]])
        m41 = m40.inverse(nozeros=True)
        self.assertEqual(m41.numer, (2, 2))

        # Test inverse with singular matrix (nozeros=False)
        m42 = Matrix([[1., 2.], [2., 4.]])
        m43 = m42.inverse()
        # Should mask singular matrix
        self.assertTrue(isinstance(m43, Matrix))
        self.assertTrue(m43.mask)

        # Test inverse with recursive=False
        m44 = Matrix([[1., 2.], [3., 4.]])
        m44.insert_deriv('t', Matrix([[5., 6.], [7., 8.]]))
        m45 = m44.inverse(recursive=False)
        self.assertEqual(len(m45.derivs), 0)

        # Test unitary with non-3x3 matrix (error)
        m46 = Matrix([[1., 2.], [3., 4.]])
        self.assertRaises(ValueError, m46.unitary)

        # Test unitary with denominators (error)
        # For drank=1, Matrix with numer (3,3) needs shape (3, 3, p)
        m47_vals = np.array([[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
                            [[0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]]])
        m47 = Matrix(m47_vals, drank=1)
        self.assertRaises(ValueError, m47.unitary)

        # Test __floordiv__ (error) - these operators raise TypeError
        # The error handling is tested in the code itself
        m48 = Matrix([[1., 2.], [3., 4.]])
        with self.assertRaises(TypeError):
            _ = m48 // 2

        # Test identity with non-square matrix (error)
        m50 = Matrix([[1., 2., 3.], [4., 5., 6.]])
        self.assertRaises(ValueError, m50.identity)

        # Note: Matrix doesn't have a solve() method in the base class
        # Solving is typically done via inverse() * vector
        m51 = Matrix([[1., 2.], [3., 4.]])
        v7 = Vector([1., 2.])
        # Solve m51 * x = v7 by computing x = m51.inverse() * v7
        v8 = m51.inverse() * v7
        # Check that m51 * v8 equals v7
        v9 = m51 * v8
        self.assertAlmostEqual(v9.to_scalar(0), 1., places=10)
        self.assertAlmostEqual(v9.to_scalar(1), 2., places=10)

        # Test with n-D
        m52 = Matrix([[[1., 2.], [3., 4.]], [[2., 1.], [1., 2.]]])
        v10 = Vector([[1., 2.], [3., 4.]])
        v11 = m52.inverse() * v10
        self.assertEqual(v11.shape, (2,))

        # Test row_vector with recursive=False
        m53 = Matrix([[1., 2., 3.], [4., 5., 6.]])
        m53.insert_deriv('t', Matrix([[7., 8., 9.], [10., 11., 12.]]))
        v12 = m53.row_vector(0, recursive=False)
        self.assertEqual(len(v12.derivs), 0)

        # Test column_vector with recursive=False
        v13 = m53.column_vector(0, recursive=False)
        self.assertEqual(len(v13.derivs), 0)

        # Test to_vector with recursive=False
        v14 = m53.to_vector(0, 0, recursive=False)
        self.assertEqual(len(v14.derivs), 0)

        # Test to_scalar with recursive=False
        s22 = m53.to_scalar(0, 1, recursive=False)
        self.assertEqual(len(s22.derivs), 0)

##########################################################################################
