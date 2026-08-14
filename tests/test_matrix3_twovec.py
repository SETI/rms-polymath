##########################################################################################
# tests/test_matrix3_twovec.py
##########################################################################################

import numpy as np

from polymath import Matrix, Matrix3, Vector3


def test_matrix3_twovec_these_all_regenerate_the_identity_matrix() -> None:
    """These all regenerate the Identity matrix."""

    np.random.seed(7877)
    DEL = 1.e-12

    mat = Matrix3.twovec(Vector3.XAXIS, 0, Vector3.YAXIS, 1)
    assert (Matrix.IDENTITY3 - mat.vals).rms() < DEL
    mat = Matrix3.twovec(Vector3.YAXIS, 1, Vector3.ZAXIS, 2)
    assert (Matrix.IDENTITY3 - mat.vals).rms() < DEL
    mat = Matrix3.twovec(Vector3.YAXIS, 1, Vector3.YAXIS + Vector3.ZAXIS, 2)
    assert (Matrix.IDENTITY3 - mat.vals).rms() < DEL
    mat = Matrix3.twovec(Vector3.YAXIS, 1, Vector3.YAXIS + Vector3.ZAXIS, 2)
    assert (Matrix.IDENTITY3 - mat.vals).rms() < DEL
    mat = Matrix3.twovec(Vector3.YAXIS, 1, Vector3.ZAXIS - 99*Vector3.YAXIS, 2)
    assert (Matrix.IDENTITY3 - mat.vals).rms() < DEL

    N = 100
    a = Vector3(np.random.randn(N,3)).unit()
    mat = Matrix3.twovec(a, 2, Vector3.XAXIS, 0)
    for i in range(N):

        # The new Y-axis is perpendicular to X
        assert mat.values[i,1,0] == 0. or abs(mat.values[i,1,0] - 0.) <= DEL

        # The new Z-axis coincides with the line of sight
        assert mat.values[i,2,0] == a.values[i,0] or abs(mat.values[i,2,0] - a.values[i,0]) <= DEL
        assert mat.values[i,2,1] == a.values[i,1] or abs(mat.values[i,2,1] - a.values[i,1]) <= DEL
        assert mat.values[i,2,2] == a.values[i,2] or abs(mat.values[i,2,2] - a.values[i,2]) <= DEL

    a = Vector3(np.random.randn(N,3), mask=np.random.randn(N) < -0.5)
    b = Vector3(np.random.randn(N,3), mask=np.random.randn(N) < -0.5)
    mat = Matrix3.twovec(a, 2, b, 1)
    assert np.all(mat.mask == (a.mask | b.mask))


def test_matrix3_twovec_with_derivatives() -> None:
    """With derivatives."""

    np.random.seed(7877)
    DEL = 1.e-12

    DEL = 1.e-12
    N = 100
    a = Vector3(np.random.randn(N,3), mask=(np.random.rand(N) < 0.01))
    da_dt = Vector3(np.random.randn(N,3))
    a.insert_deriv('t', da_dt)
    b = Vector3(np.random.randn(N,3), mask=(np.random.rand(N) < 0.1))
    db_dt = Vector3(np.random.randn(N,3))
    b.insert_deriv('t', db_dt)
    mat = Matrix3.twovec(a, 1, b, 0)
    mat_x_a = mat * a
    mat_x_b = mat * b
    assert np.max(np.abs(mat_x_a.vals[:,0])) < DEL
    assert np.max(np.abs(mat_x_a.vals[:,2])) < DEL
    assert np.max(mat_x_b.vals[:,2]) < DEL
    assert np.min(mat_x_b.vals[:,0]) > 0.   # positive half-plane!
    assert np.all(mat.mask == (a.mask | b.mask))
    EPS = 1.e-8
    mat1 = Matrix3.twovec(a.wod + EPS/2 * da_dt, 1, b.wod + EPS/2 * db_dt, 0)
    mat0 = Matrix3.twovec(a.wod - EPS/2 * da_dt, 1, b.wod - EPS/2 * db_dt, 0)
    dmat_dt = (Matrix(mat1) - Matrix(mat0)) / EPS
    diffs = (dmat_dt.vals - mat.d_dt.vals)[~mat.mask]
    assert np.max(np.abs(diffs)) < 1.e-6


def test_matrix3_twovec_with_derivatives_denoms() -> None:
    """With derivatives, denoms."""

    np.random.seed(7877)

    N = 100
    a = Vector3(np.random.randn(N,3))
    da_dt = Vector3(np.random.randn(N,3,2,3), drank=2)
    a.insert_deriv('t', da_dt)
    b = Vector3(np.random.randn(N,3))
    db_dt = Vector3(np.random.randn(N,3,2,3), drank=2)
    b.insert_deriv('t', db_dt)
    mat = Matrix3.twovec(a, 1, b, 0)
    EPS = 1.e-8
    for i in range(2):
        for j in range(3):
            mat1 = Matrix3.twovec(a.wod + EPS/2 * da_dt.vals[...,i,j], 1,
                                  b.wod + EPS/2 * db_dt.vals[...,i,j], 0)
            mat0 = Matrix3.twovec(a.wod - EPS/2 * da_dt.vals[...,i,j], 1,
                                  b.wod - EPS/2 * db_dt.vals[...,i,j], 0)
            dmat_dt = (Matrix(mat1) - Matrix(mat0)) / EPS

            assert np.max(np.abs(dmat_dt.vals - mat.d_dt.vals[...,i,j])) < 2.e-6


##########################################################################################
