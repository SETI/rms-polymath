##########################################################################################
# tests/test_quaternion_matrix3.py
##########################################################################################

import numpy as np

import pytest

from polymath import Quaternion, Matrix, Matrix3


def test_quaternion_matrix3_from_identity() -> None:
    """The identity matrix converts to the identity quaternion."""

    q = Quaternion.from_matrix3(Matrix3.IDENTITY)
    assert q.values[0] == 1.
    assert q.values[1] == 0.
    assert q.values[2] == 0.
    assert q.values[3] == 0.
    assert not q.mask


@pytest.mark.parametrize('angle', [1.e-2, 1.e-4, 1.e-6, 1.e-8, 0.])
def test_quaternion_matrix3_from_near_identity(angle: float) -> None:
    """Rotations near the identity convert without loss of precision."""

    mat = Matrix3.from_euler(angle, 0., 0., 'rzxz')
    q = Quaternion.from_matrix3(mat)

    assert q.values[0] == pytest.approx(np.cos(0.5 * angle), abs=1.e-15)
    assert q.values[3] == pytest.approx(np.sin(0.5 * angle), abs=1.e-15)
    assert np.abs(q.to_matrix3().values - mat.values).max() <= 1.e-15


def test_quaternion_matrix3_one_quaternion() -> None:
    """One quaternion."""

    np.random.seed(2496)

    # Quaternion to Matrix3 and back

    a = Quaternion(np.random.rand(4)).unit()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)
    DEL = 1.e-14
    for j in range(4):
        assert a.values[j] == b.values[j] or abs(a.values[j] - b.values[j]) <= DEL
    assert not a.readonly
    assert not b.readonly
    a = a.as_readonly()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)
    assert not b.readonly


def test_quaternion_matrix3_n_quaternions() -> None:
    """N Quaternions."""

    np.random.seed(2496)

    # Quaternion to Matrix3 and back

    N = 100
    a = Quaternion(np.random.rand(N,4)).unit()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)
    DEL = 1.e-14
    for i in range(N):
        for j in range(4):
            assert a.values[i,j] == b.values[i,j] or abs(a.values[i,j] - b.values[i,j]) <= DEL
    assert not a.readonly
    assert not b.readonly
    a = a.as_readonly()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)
    assert not b.readonly

    # Quaternion to Euler angles and back


def test_quaternion_matrix3_n_quaternions_without_unit() -> None:
    """N Quaternions, without unit()."""

    np.random.seed(2496)

    # Quaternion to Matrix3 and back

    N = 100
    a = Quaternion(np.random.rand(N,4))
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)
    aa = a.unit()
    DEL = 1.e-14
    for i in range(N):
        for j in range(4):
            assert aa.values[i,j] == b.values[i,j] or abs(aa.values[i,j] - b.values[i,j]) <= DEL
    assert not aa.readonly
    assert not b.readonly


def test_quaternion_matrix3_n_quaternions_with_unit() -> None:
    """N Quaternions, with unit()."""

    np.random.seed(2496)

    # Quaternion to Matrix3 and back

    N = 100
    a = Quaternion(np.random.rand(N,4)).unit()
    mat = a.to_matrix3()
    b = Quaternion.from_matrix3(mat)
    aa = a.unit()
    DEL = 5.e-14
    for i in range(N):
        for j in range(4):
            assert a.values[i,j] == b.values[i,j] or abs(a.values[i,j] - b.values[i,j]) <= DEL
    assert not aa.readonly
    assert not b.readonly


def _euler_path(angles: np.ndarray, velocity: np.ndarray, t: float) -> Matrix3:
    """A Matrix3 at one point along a straight path through Euler angle space."""

    return Matrix3.from_euler(*(angles + t * velocity), 'rzxz')


@pytest.mark.parametrize('angles', [(0., 0., 0.),           # trace branch
                                    (1.e-7, 0., 0.),        # trace branch
                                    (0.3, 0.2, 0.1),        # trace branch
                                    (1.0, 2.0, 3.0),        # diagonal branch
                                    (0., np.pi, 0.),        # diagonal branch, x largest
                                    (np.pi, 0., 0.)])       # diagonal branch, z largest
def test_quaternion_matrix3_from_matrix3_with_derivatives(
        angles: tuple[float, float, float]) -> None:
    """Matrix3 to Quaternion derivatives match finite differences in every branch."""

    np.random.seed(2496)

    angles_ = np.array(angles)
    velocity = np.random.randn(3)
    EPS = 1.e-6

    mat = _euler_path(angles_, velocity, 0.)
    dmat_dt = (_euler_path(angles_, velocity, EPS).values
               - _euler_path(angles_, velocity, -EPS).values) / (2. * EPS)
    mat.insert_deriv('t', Matrix(dmat_dt))

    q = Quaternion.from_matrix3(mat)

    dq_dt = ((Quaternion.from_matrix3(_euler_path(angles_, velocity, EPS)).values
              - Quaternion.from_matrix3(_euler_path(angles_, velocity, -EPS)).values)
             / (2. * EPS))

    DEL = 1.e-8
    for j in range(4):
        assert q.d_dt.values[j] == pytest.approx(dq_dt[j], abs=DEL)


def test_quaternion_matrix3_from_matrix3_derivative_round_trip() -> None:
    """A derivative survives the round trip Quaternion to Matrix3 and back."""

    np.random.seed(2496)

    N = 20
    a = Quaternion(np.random.randn(N,4)).unit()

    # from_matrix3() returns the quaternion whose largest component is positive
    signs = np.sign(a.values[np.arange(N), np.argmax(np.abs(a.values), axis=-1)])
    a = Quaternion(a.values * signs[:,np.newaxis])

    # The derivative of a rotation is orthogonal to the quaternion; a parallel
    # component leaves the matrix unchanged and so cannot be recovered
    da_dt = np.random.randn(N,4)
    da_dt -= np.sum(da_dt * a.values, axis=-1)[:,np.newaxis] * a.values
    a.insert_deriv('t', Quaternion(da_dt))

    b = Quaternion.from_matrix3(a.to_matrix3(recursive=True))

    DEL = 1.e-13
    for i in range(N):
        for j in range(4):
            assert b.values[i,j] == pytest.approx(a.values[i,j], abs=DEL)
            assert b.d_dt.values[i,j] == pytest.approx(da_dt[i,j], abs=DEL)


def test_quaternion_matrix3_from_matrix3_with_denominator() -> None:
    """A Matrix3 derivative with a denominator yields one Quaternion column each."""

    np.random.seed(2496)

    N = 7
    angles = np.random.randn(N,3)
    mat = Matrix3.from_euler(angles[:,0], angles[:,1], angles[:,2], 'rzxz')

    dmat_du = np.random.randn(N,3,3,2)
    mat.insert_deriv('u', Matrix(dmat_du, drank=1))
    q = Quaternion.from_matrix3(mat)

    assert q.d_du.denom == (2,)

    DEL = 1.e-14
    for c in range(2):
        column = mat.wod
        column.insert_deriv('u', Matrix(dmat_du[...,c]))
        expected = Quaternion.from_matrix3(column).d_du

        for i in range(N):
            for j in range(4):
                assert q.d_du.values[i,j,c] == pytest.approx(expected.values[i,j],
                                                             abs=DEL)


def test_quaternion_matrix3_from_matrix3_derivative_masking() -> None:
    """A masked Matrix3 yields a masked Quaternion derivative."""

    np.random.seed(2496)

    angles = np.random.randn(5,3)
    mat = Matrix3.from_euler(angles[:,0], angles[:,1], angles[:,2], 'rzxz')
    mat = mat.mask_where(np.array([False, True, False, False, True]))
    mat.insert_deriv('t', Matrix(np.random.randn(5,3,3)))

    q = Quaternion.from_matrix3(mat)

    assert not q.d_dt.mask[0]
    assert q.d_dt.mask[1]
    assert not q.d_dt.mask[2]
    assert not q.d_dt.mask[3]
    assert q.d_dt.mask[4]


def test_quaternion_matrix3_quaternion_to_matrix3_with_derivatives() -> None:
    """Quaternion to Matrix3, with derivatives."""

    np.random.seed(2496)

    # Quaternion to Matrix3 and back

    N = 100
    x = Quaternion(np.random.rand(N,4))
    x.insert_deriv('t', Quaternion(np.random.rand(N,4)))
    y = x.to_matrix3(recursive=True)
    EPS = 1.e-6
    y1 = Matrix.as_matrix((x + (EPS,0,0,0)).to_matrix3(recursive=False))
    y0 = Matrix.as_matrix((x - (EPS,0,0,0)).to_matrix3(recursive=False))
    dy_dx0 = 0.5 * (y1 - y0) / EPS
    y1 = Matrix.as_matrix((x + (0,EPS,0,0)).to_matrix3(recursive=False))
    y0 = Matrix.as_matrix((x - (0,EPS,0,0)).to_matrix3(recursive=False))
    dy_dx1 = 0.5 * (y1 - y0) / EPS
    y1 = Matrix.as_matrix((x + (0,0,EPS,0)).to_matrix3(recursive=False))
    y0 = Matrix.as_matrix((x - (0,0,EPS,0)).to_matrix3(recursive=False))
    dy_dx2 = 0.5 * (y1 - y0) / EPS
    y1 = Matrix.as_matrix((x + (0,0,0,EPS)).to_matrix3(recursive=False))
    y0 = Matrix.as_matrix((x - (0,0,0,EPS)).to_matrix3(recursive=False))
    dy_dx3 = 0.5 * (y1 - y0) / EPS
    dy_dt = (dy_dx0 * x.d_dt.values[...,0] +
             dy_dx1 * x.d_dt.values[...,1] +
             dy_dx2 * x.d_dt.values[...,2] +
             dy_dx3 * x.d_dt.values[...,3])
    DEL = 1.e-5
    for i in range(N):
        for j in range(3):
            for k in range(3):
                assert dy_dt.values[i,j,k] == y.d_dt.values[i,j,k] or abs(dy_dt.values[i,j,k] - y.d_dt.values[i,j,k]) <= DEL


##########################################################################################
