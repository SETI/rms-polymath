##########################################################################################
# tests/test_vector_cross_3x3.py
##########################################################################################

import numpy as np
import pytest

from polymath import Vector, Unit


def test_vector_cross_3x3_test_units() -> None:
    """Test units."""

    np.random.seed(9797)
    omega = Vector(np.random.randn(30,3))
    omega_as_matrix = omega.cross_product_as_matrix()
    vec = Vector(np.random.randn(20,30,3))
    cross1 = omega_as_matrix * vec
    cross2 = omega.cross(vec)
    assert np.all(np.abs(cross1.values - cross2.values) < 1.e-15)
    dots = omega.dot(cross1)
    assert np.all(np.abs(dots.values) < 1.e-14)

    omega = Vector(np.random.randn(3), unit=Unit.KM)
    omega_as_matrix = omega.cross_product_as_matrix()
    vec = Vector(np.random.randn(3), unit=Unit.SECONDS**(-1))
    cross1 = omega_as_matrix * vec
    cross2 = omega.cross(vec)
    assert cross1.unit_ == Unit.KM/Unit.SECONDS
    assert cross2.unit_ == Unit.KM/Unit.SECONDS


def test_vector_cross_3x3_derivatives_denom() -> None:
    """Derivatives, denom = ()."""

    np.random.seed(9797)
    omega = Vector(np.random.randn(30,3))
    omega_as_matrix = omega.cross_product_as_matrix()
    vec = Vector(np.random.randn(20,30,3))
    cross1 = omega_as_matrix * vec
    cross2 = omega.cross(vec)
    assert np.all(np.abs(cross1.values - cross2.values) < 1.e-15)
    dots = omega.dot(cross1)
    assert np.all(np.abs(dots.values) < 1.e-14)

    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))
    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))
    z = y.cross(x)
    assert 'f' in x.derivs
    assert hasattr(x, 'd_df')
    assert 'g' not in x.derivs
    assert not hasattr(x, 'd_dg')
    assert 'h' in x.derivs
    assert hasattr(x, 'd_dh')
    assert 'f' not in y.derivs
    assert not hasattr(y, 'd_df')
    assert 'g' in y.derivs
    assert hasattr(y, 'd_dg')
    assert 'h' in y.derivs
    assert hasattr(y, 'd_dh')
    assert 'f' in z.derivs
    assert hasattr(z, 'd_df')
    assert 'g' in z.derivs
    assert hasattr(z, 'd_dg')
    assert 'h' in z.derivs
    assert hasattr(z, 'd_dh')
    EPS = 1.e-6
    z1 = y.cross(x + (EPS,0,0))
    z0 = y.cross(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.cross(x + (0,EPS,0))
    z0 = y.cross(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.cross(x + (0,0,EPS))
    z0 = y.cross(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0,0)).cross(x)
    z0 = (y - (EPS,0,0)).cross(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS,0)).cross(x)
    z0 = (y - (0,EPS,0)).cross(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,0,EPS)).cross(x)
    z0 = (y - (0,0,EPS)).cross(x)
    dz_dy2 = 0.5 * (z1 - z0) / EPS
    dz_df = (dz_dx0 * x.d_df.values[:,0] +
             dz_dx1 * x.d_df.values[:,1] +
             dz_dx2 * x.d_df.values[:,2])
    dz_dg = (dz_dy0 * y.d_dg.values[:,0] +
             dz_dy1 * y.d_dg.values[:,1] +
             dz_dy2 * y.d_dg.values[:,2])
    dz_dh = (dz_dx0 * x.d_dh.values[:,0] + dz_dy0 * y.d_dh.values[:,0] +
             dz_dx1 * x.d_dh.values[:,1] + dz_dy1 * y.d_dh.values[:,1] +
             dz_dx2 * x.d_dh.values[:,2] + dz_dy2 * y.d_dh.values[:,2])
    for i in range(N):
        for k in range(3):
            assert z.d_df.values[i,k] == dz_df.values[i,k] or abs(z.d_df.values[i,k] - dz_df.values[i,k]) <= EPS
            assert z.d_dg.values[i,k] == dz_dg.values[i,k] or abs(z.d_dg.values[i,k] - dz_dg.values[i,k]) <= EPS
            assert z.d_dh.values[i,k] == dz_dh.values[i,k] or abs(z.d_dh.values[i,k] - dz_dh.values[i,k]) <= EPS

    z = y.cross_product_as_matrix() * x
    for i in range(N):
        for k in range(3):
            assert z.d_df.values[i,k] == dz_df.values[i,k] or abs(z.d_df.values[i,k] - dz_df.values[i,k]) <= EPS
            assert z.d_dg.values[i,k] == dz_dg.values[i,k] or abs(z.d_dg.values[i,k] - dz_dg.values[i,k]) <= EPS
            assert z.d_dh.values[i,k] == dz_dh.values[i,k] or abs(z.d_dh.values[i,k] - dz_dh.values[i,k]) <= EPS


def test_vector_cross_3x3_derivatives_denom_2() -> None:
    """Derivatives, denom = (2,)."""

    np.random.seed(9797)
    omega = Vector(np.random.randn(30,3))
    omega_as_matrix = omega.cross_product_as_matrix()
    vec = Vector(np.random.randn(20,30,3))
    cross1 = omega_as_matrix * vec
    cross2 = omega.cross(vec)
    assert np.all(np.abs(cross1.values - cross2.values) < 1.e-15)
    dots = omega.dot(cross1)
    assert np.all(np.abs(dots.values) < 1.e-14)

    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))
    x.insert_deriv('f', Vector(np.random.randn(N,3,2), drank=1))
    x.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('g', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))
    z = y.cross(x)
    assert 'f' in x.derivs
    assert hasattr(x, 'd_df')
    assert 'g' not in x.derivs
    assert not hasattr(x, 'd_dg')
    assert 'h' in x.derivs
    assert hasattr(x, 'd_dh')
    assert 'f' not in y.derivs
    assert not hasattr(y, 'd_df')
    assert 'g' in y.derivs
    assert hasattr(y, 'd_dg')
    assert 'h' in y.derivs
    assert hasattr(y, 'd_dh')
    assert 'f' in z.derivs
    assert hasattr(z, 'd_df')
    assert 'g' in z.derivs
    assert hasattr(z, 'd_dg')
    assert 'h' in z.derivs
    assert hasattr(z, 'd_dh')
    EPS = 1.e-6
    z1 = y.cross(x + (EPS,0,0))
    z0 = y.cross(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.cross(x + (0,EPS,0))
    z0 = y.cross(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.cross(x + (0,0,EPS))
    z0 = y.cross(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0,0)).cross(x)
    z0 = (y - (EPS,0,0)).cross(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS,0)).cross(x)
    z0 = (y - (0,EPS,0)).cross(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,0,EPS)).cross(x)
    z0 = (y - (0,0,EPS)).cross(x)
    dz_dy2 = 0.5 * (z1 - z0) / EPS
    dz_df0 = (dz_dx0 * x.d_df.values[:,0,0] +
              dz_dx1 * x.d_df.values[:,1,0] +
              dz_dx2 * x.d_df.values[:,2,0])
    dz_df1 = (dz_dx0 * x.d_df.values[:,0,1] +
              dz_dx1 * x.d_df.values[:,1,1] +
              dz_dx2 * x.d_df.values[:,2,1])
    dz_dg0 = (dz_dy0 * y.d_dg.values[:,0,0] +
              dz_dy1 * y.d_dg.values[:,1,0] +
              dz_dy2 * y.d_dg.values[:,2,0])
    dz_dg1 = (dz_dy0 * y.d_dg.values[:,0,1] +
              dz_dy1 * y.d_dg.values[:,1,1] +
              dz_dy2 * y.d_dg.values[:,2,1])
    dz_dh0 = (dz_dx0 * x.d_dh.values[:,0,0] + dz_dy0 * y.d_dh.values[:,0,0] +
              dz_dx1 * x.d_dh.values[:,1,0] + dz_dy1 * y.d_dh.values[:,1,0] +
              dz_dx2 * x.d_dh.values[:,2,0] + dz_dy2 * y.d_dh.values[:,2,0])
    dz_dh1 = (dz_dx0 * x.d_dh.values[:,0,1] + dz_dy0 * y.d_dh.values[:,0,1] +
              dz_dx1 * x.d_dh.values[:,1,1] + dz_dy1 * y.d_dh.values[:,1,1] +
              dz_dx2 * x.d_dh.values[:,2,1] + dz_dy2 * y.d_dh.values[:,2,1])
    for i in range(N):
        for k in range(3):
            assert z.d_df.values[i,k,0] == dz_df0.values[i,k] or abs(z.d_df.values[i,k,0] - dz_df0.values[i,k]) <= EPS
            assert z.d_dg.values[i,k,0] == dz_dg0.values[i,k] or abs(z.d_dg.values[i,k,0] - dz_dg0.values[i,k]) <= EPS
            assert z.d_dh.values[i,k,0] == dz_dh0.values[i,k] or abs(z.d_dh.values[i,k,0] - dz_dh0.values[i,k]) <= EPS

            assert z.d_df.values[i,k,1] == dz_df1.values[i,k] or abs(z.d_df.values[i,k,1] - dz_df1.values[i,k]) <= EPS
            assert z.d_dg.values[i,k,1] == dz_dg1.values[i,k] or abs(z.d_dg.values[i,k,1] - dz_dg1.values[i,k]) <= EPS
            assert z.d_dh.values[i,k,1] == dz_dh1.values[i,k] or abs(z.d_dh.values[i,k,1] - dz_dh1.values[i,k]) <= EPS

    assert y.cross(x, recursive=False).derivs == {}
    assert hasattr(x, 'd_df')
    assert hasattr(x, 'd_dh')
    assert hasattr(y, 'd_dg')
    assert hasattr(y, 'd_dh')
    assert not hasattr(y.cross(x, recursive=False), 'd_df')
    assert not hasattr(y.cross(x, recursive=False), 'd_dg')
    assert not hasattr(y.cross(x, recursive=False), 'd_dh')


def test_vector_cross_3x3_read_only_status_should_be_preserved() -> None:
    """Read-only status should be preserved."""

    np.random.seed(9797)
    omega = Vector(np.random.randn(30,3))
    omega_as_matrix = omega.cross_product_as_matrix()
    vec = Vector(np.random.randn(20,30,3))
    cross1 = omega_as_matrix * vec
    cross2 = omega.cross(vec)
    assert np.all(np.abs(cross1.values - cross2.values) < 1.e-15)
    dots = omega.dot(cross1)
    assert np.all(np.abs(dots.values) < 1.e-14)

    N = 10
    y = Vector(np.random.randn(N,3))
    x = Vector(np.random.randn(N,3))
    assert not x.readonly
    assert not y.readonly
    assert not y.cross(x).readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not y.as_readonly().cross(x.as_readonly()).readonly
    assert not y.as_readonly().cross(x).readonly
    assert not y.cross(x.as_readonly()).readonly


def test_vector_cross_product_as_matrix_reproduces_the_cross_product() -> None:
    """The matrix multiplied by another vector equals the cross product."""

    rng = np.random.default_rng(31)
    v = Vector(rng.normal(size=(2, 3, 3)))
    w = Vector(rng.normal(size=(2, 3, 3)))

    assert np.allclose((v.cross_product_as_matrix() * w).values, v.cross(w).values)


def test_vector_cross_product_as_matrix_supports_a_denominator() -> None:
    """A Vector carrying a denominator keeps it, with the matrix axes ahead of it."""

    v = Vector(np.arange(6.).reshape(3, 2), drank=1)
    result = v.cross_product_as_matrix()

    assert result.numer == (3, 3)
    assert result.denom == (2,)

    # Each denominator column is the cross-product matrix of that column of the input
    for j in range(2):
        column = Vector(v.values[:, j])
        assert np.all(result.values[..., j] == column.cross_product_as_matrix().values)


def test_vector_cross_product_as_matrix_carries_a_jacobian_derivative() -> None:
    """A derivative with a denominator passes through instead of raising."""

    v = Vector([1., 2., 3.])
    v.insert_deriv('xy', Vector(np.arange(6.).reshape(3, 2), drank=1))
    result = v.cross_product_as_matrix()

    assert result.derivs['xy'].numer == (3, 3)
    assert result.derivs['xy'].denom == (2,)
    assert np.all(result.derivs['xy'].values
                  == v.derivs['xy'].cross_product_as_matrix().values)


def test_vector_cross_product_as_matrix_requires_three_components() -> None:
    """A Vector of any length other than three is rejected."""

    with pytest.raises(ValueError, match='requires item shape'):
        Vector([1., 2.]).cross_product_as_matrix()


##########################################################################################
