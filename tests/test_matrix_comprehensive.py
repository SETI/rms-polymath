##########################################################################################
# tests/test_matrix_comprehensive.py
# Comprehensive unit tests for Matrix class based on docstrings
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Vector, Matrix, Vector3


def test_matrix_comprehensive_test_as_matrix_static_method() -> None:
    """Test as_matrix static method."""

    np.random.seed(9012)

    m1 = Matrix([[1., 2.], [3., 4.]])
    m1_conv = Matrix.as_matrix(m1)
    assert type(m1_conv) == Matrix
    assert np.allclose(m1_conv.vals, [[1., 2.], [3., 4.]])

    m2 = Matrix.as_matrix([[1., 2.], [3., 4.]])
    assert type(m2) == Matrix

    m3 = Matrix([[1., 2., 3.], [4., 5., 6.]])
    v1 = m3.row_vector(0)
    assert type(v1) == Vector3  # Should be Vector3 for length 3
    assert np.allclose(v1.vals, [1., 2., 3.])

    rows = m3.row_vectors()
    assert len(rows) == 2
    assert np.allclose(rows[0].vals, [1., 2., 3.])
    assert np.allclose(rows[1].vals, [4., 5., 6.])

    v2 = m3.column_vector(0)
    assert type(v2) == Vector
    assert np.allclose(v2.vals, [1., 4.])

    cols = m3.column_vectors()
    assert len(cols) == 3
    assert np.allclose(cols[0].vals, [1., 4.])

    v3 = m3.to_vector(0, 0)
    assert type(v3) == Vector
    assert np.allclose(v3.vals, [1., 2., 3.])

    s1 = m3.to_scalar(0, 1)
    assert type(s1) == Scalar
    assert s1 == 2.

    s2 = Scalar(1.)
    s3 = Scalar(2.)
    s4 = Scalar(3.)
    s5 = Scalar(4.)
    m4 = Matrix.from_scalars(s2, s3, s4, s5)
    assert type(m4) == Matrix
    assert m4.numer == (2, 2)
    assert np.allclose(m4.vals, [[1., 2.], [3., 4.]])

    m5 = Matrix([[1., 0.], [0., 2.]])
    b1 = m5.is_diagonal()
    assert b1
    m6 = Matrix([[1., 1.], [0., 2.]])
    b2 = m6.is_diagonal()
    assert not b2

    m7 = Matrix([[1., 2., 3.], [4., 5., 6.]])
    m8 = m7.transpose()
    assert m8.numer == (3, 2)
    assert np.allclose(m8.vals, [[1., 4.], [2., 5.], [3., 6.]])

    m9 = m7.T
    assert np.allclose(m9.vals, [[1., 4.], [2., 5.], [3., 6.]])

    m10 = Matrix([[1., 2.], [3., 4.]])
    m11 = m10.inverse()

    m12 = m10 * m11
    assert m12.to_scalar(0, 0) == 1. or abs(m12.to_scalar(0, 0) - 1.) <= 1e-10
    assert m12.to_scalar(0, 1) == 0. or abs(m12.to_scalar(0, 1) - 0.) <= 1e-10
    assert m12.to_scalar(1, 0) == 0. or abs(m12.to_scalar(1, 0) - 0.) <= 1e-10
    assert m12.to_scalar(1, 1) == 1. or abs(m12.to_scalar(1, 1) - 1.) <= 1e-10

    angle = np.pi/4
    m13 = Matrix([[np.cos(angle), -np.sin(angle), 0.],
                  [np.sin(angle), np.cos(angle), 0.],
                  [0., 0., 1.]])
    m14 = m13.unitary()

    assert m14.numer == (3, 3)
    assert np.allclose(m14.vals, m13.vals, atol=1e-10)

    m15 = Matrix([[1., 2.], [3., 4.]])
    with pytest.raises(TypeError):
        abs(m15)

    m16 = Matrix([[1., 2.], [3., 4.]])
    m17 = m16.identity()
    assert m17.numer == (2, 2)
    assert np.allclose(m17.vals, [[1., 0.], [0., 1.]])

    m18 = Matrix([[1., 2.], [3., 4.]])
    m19 = m18.reciprocal()
    m20 = m18.inverse()
    assert np.allclose(m19.vals, m20.vals)

    m21 = Matrix([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])

    v4 = m21.row_vector(0)
    assert v4.shape == (2,)

    assert np.allclose(v4.vals[0], [1., 2.])
    assert np.allclose(v4.vals[1], [5., 6.])

    v5 = m21.column_vector(0)
    assert v5.shape == (2,)

    assert np.allclose(v5.vals[0], [1., 3.])
    assert np.allclose(v5.vals[1], [5., 7.])

    m22 = m21.transpose()
    assert m22.shape == (2,)
    assert m22.numer == (2, 2)

    m23 = Matrix([[[1., 2.], [3., 4.]], [[2., 1.], [1., 2.]]])
    m24 = m23.inverse()
    assert m24.shape == (2,)

    m25 = m23 * m24

    m25_0 = m25[0]
    assert m25_0.to_scalar(0, 0) == 1. or abs(m25_0.to_scalar(0, 0) - 1.) <= 1e-10
    assert m25_0.to_scalar(0, 1) == 0. or abs(m25_0.to_scalar(0, 1) - 0.) <= 1e-10
    assert m25_0.to_scalar(1, 0) == 0. or abs(m25_0.to_scalar(1, 0) - 0.) <= 1e-10
    assert m25_0.to_scalar(1, 1) == 1. or abs(m25_0.to_scalar(1, 1) - 1.) <= 1e-10

    m25_1 = m25[1]
    assert m25_1.to_scalar(0, 0) == 1. or abs(m25_1.to_scalar(0, 0) - 1.) <= 1e-10
    assert m25_1.to_scalar(0, 1) == 0. or abs(m25_1.to_scalar(0, 1) - 0.) <= 1e-10
    assert m25_1.to_scalar(1, 0) == 0. or abs(m25_1.to_scalar(1, 0) - 0.) <= 1e-10
    assert m25_1.to_scalar(1, 1) == 1. or abs(m25_1.to_scalar(1, 1) - 1.) <= 1e-10

    s6 = Scalar(1.)
    s7 = Scalar(2.)
    s8 = Scalar(3.)
    s9 = Scalar(4.)
    m26 = Matrix.from_scalars(s6, s7, s8, s9)
    assert m26.shape == ()
    assert m26.numer == (2, 2)

    s10 = Scalar([[1., 2.], [3., 4.]])
    s11 = Scalar([[5., 6.], [7., 8.]])
    s12 = Scalar([[9., 10.], [11., 12.]])
    s13 = Scalar([[13., 14.], [15., 16.]])

    m27 = Matrix.from_scalars(s10, s11, s12, s13)
    assert m27.shape == (2, 2)
    assert m27.numer == (2, 2)

    m27 = Matrix([[[1., 0.], [0., 2.]], [[3., 0.], [0., 4.]]])
    b3 = m27.is_diagonal()
    assert b3.shape == (2,)
    assert b3[0]
    assert b3[1]

    v6 = Vector([[1., 0.], [0., 1.]], drank=1)
    m28 = Matrix.as_matrix(v6)
    assert type(m28) == Matrix
    # Note: join_items may change drank, so just check it's a Matrix

    m29 = Matrix([[1., 2.], [3., 4.]])
    m29.insert_deriv('t', Matrix([[5., 6.], [7., 8.]]))
    m30 = Matrix.as_matrix(m29, recursive=False)
    assert len(m30.derivs) == 0

    s14 = Scalar(1.)
    s15 = Scalar(2.)
    s16 = Scalar(3.)
    s17 = Scalar(4.)
    m31 = Matrix.from_scalars(s14, s15, s16, s17, shape=(2, 2))
    assert m31.numer == (2, 2)

    with pytest.raises(ValueError):
        Matrix.from_scalars(s14, s15, s16, shape=(2, 2))

    with pytest.raises(ValueError):
        Matrix.from_scalars(s14, s15, s16, s17, shape=(2,))

    s18 = Scalar(1)
    s19 = Scalar(2)
    s20 = Scalar(3)
    s21 = Scalar(4)
    with pytest.raises(TypeError):
        Matrix.from_scalars(s18, s19, s20, s21)

    m32 = Matrix([[1., 2., 3.], [4., 5., 6.]])
    with pytest.raises(ValueError):
        m32.is_diagonal()

    m33_vals = np.array([[[1., 0., 0.], [0., 2., 0.]], [[0., 0., 3.], [0., 0., 0.]]])
    m33 = Matrix(m33_vals, drank=1)
    with pytest.raises(ValueError):
        m33.is_diagonal()

    m34 = Matrix([[1., 0.01], [0.01, 2.]])
    b4 = m34.is_diagonal(delta=0.1)
    assert b4

    m35_array = Matrix([[[1., 0.], [0., 2.]], [[3., 0.], [0., 4.]]])
    m35_masked = m35_array.mask_where(np.array([True, False]))
    b5 = m35_masked.is_diagonal()

    assert b5.shape == (2,)
    assert b5.vals[0]  # Masked matrix returns True
    assert b5.vals[1]  # Diagonal matrix returns True

    m36 = Matrix([[1., 2.], [3., 4.]])
    m36.insert_deriv('t', Matrix([[5., 6.], [7., 8.]]))
    m37 = m36.transpose(recursive=False)
    assert len(m37.derivs) == 0

    m38 = Matrix([[1., 2., 3.], [4., 5., 6.]])
    with pytest.raises(ValueError):
        m38.inverse()

    m39_vals = np.array([[[1., 2., 0.], [3., 4., 0.]], [[0., 0., 1.], [0., 0., 1.]]])
    m39 = Matrix(m39_vals, drank=1)
    with pytest.raises(ValueError):
        m39.inverse()

    m40 = Matrix([[1., 2.], [3., 4.]])
    m41 = m40.inverse(nozeros=True)
    assert m41.numer == (2, 2)

    m42 = Matrix([[1., 2.], [2., 4.]])
    m43 = m42.inverse()

    assert isinstance(m43, Matrix)
    assert m43.mask

    m44 = Matrix([[1., 2.], [3., 4.]])
    m44.insert_deriv('t', Matrix([[5., 6.], [7., 8.]]))
    m45 = m44.inverse(recursive=False)
    assert len(m45.derivs) == 0

    m46 = Matrix([[1., 2.], [3., 4.]])
    with pytest.raises(ValueError):
        m46.unitary()

    m47_vals = np.array([[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
                        [[0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]]])
    m47 = Matrix(m47_vals, drank=1)
    with pytest.raises(ValueError):
        m47.unitary()

    m48 = Matrix([[1., 2.], [3., 4.]])
    with pytest.raises(TypeError):
        _ = m48 // 2

    m50 = Matrix([[1., 2., 3.], [4., 5., 6.]])
    with pytest.raises(ValueError):
        m50.identity()


def test_matrix_comprehensive_note_matrix_doesn_t_have_a_solve_method_in_the_base_class_so() -> None:
    """Note: Matrix doesn't have a solve() method in the base class # Solving is typically done via inverse() * vector."""

    np.random.seed(9012)

    m51 = Matrix([[1., 2.], [3., 4.]])
    v7 = Vector([1., 2.])

    v8 = m51.inverse() * v7

    v9 = m51 * v8
    assert v9.to_scalar(0) == 1. or abs(v9.to_scalar(0) - 1.) <= 1e-10
    assert v9.to_scalar(1) == 2. or abs(v9.to_scalar(1) - 2.) <= 1e-10


def test_matrix_comprehensive_test_with_n_d() -> None:
    """Test with n-D."""

    np.random.seed(9012)

    m52 = Matrix([[[1., 2.], [3., 4.]], [[2., 1.], [1., 2.]]])
    v10 = Vector([[1., 2.], [3., 4.]])
    v11 = m52.inverse() * v10
    assert v11.shape == (2,)


def test_matrix_comprehensive_test_row_vector_with_recursive_false() -> None:
    """Test row_vector with recursive=False."""

    np.random.seed(9012)

    m53 = Matrix([[1., 2., 3.], [4., 5., 6.]])
    m53.insert_deriv('t', Matrix([[7., 8., 9.], [10., 11., 12.]]))
    v12 = m53.row_vector(0, recursive=False)
    assert len(v12.derivs) == 0

    v13 = m53.column_vector(0, recursive=False)
    assert len(v13.derivs) == 0

    v14 = m53.to_vector(0, 0, recursive=False)
    assert len(v14.derivs) == 0

    s22 = m53.to_scalar(0, 1, recursive=False)
    assert len(s22.derivs) == 0


##########################################################################################
