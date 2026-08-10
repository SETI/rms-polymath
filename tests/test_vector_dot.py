##########################################################################################
# tests/test_vector_dot.py
##########################################################################################

import numpy as np
import pytest

from polymath import Unit, Vector


def test_vector_dot_test_units() -> None:
    """Test units."""

    np.random.seed(5795)
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,4))
    with pytest.raises(ValueError):
        a.dot(b)
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,5))
    assert a.dot(b) == np.sum(a.values * b.values, axis=-1)

    omega = Vector(np.random.randn(3), unit=Unit.KM)
    omega_as_matrix = omega.cross_product_as_matrix()
    vec = Vector(np.random.randn(3), unit=Unit.SECONDS**(-1))
    cross1 = omega_as_matrix * vec
    cross2 = omega.dot(vec)
    assert cross1.unit_ == Unit.KM/Unit.SECONDS
    assert cross2.unit_ == Unit.KM/Unit.SECONDS


def test_vector_dot_derivatives() -> None:
    """Derivatives."""

    np.random.seed(5795)
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,4))
    with pytest.raises(ValueError):
        a.dot(b)
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,5))
    assert a.dot(b) == np.sum(a.values * b.values, axis=-1)

    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))
    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))
    z = y.dot(x)
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
    z1 = y.dot(x + (EPS,0,0))
    z0 = y.dot(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.dot(x + (0,EPS,0))
    z0 = y.dot(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.dot(x + (0,0,EPS))
    z0 = y.dot(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0,0)).dot(x)
    z0 = (y - (EPS,0,0)).dot(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS,0)).dot(x)
    z0 = (y - (0,EPS,0)).dot(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,0,EPS)).dot(x)
    z0 = (y - (0,0,EPS)).dot(x)
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
        assert z.d_df.values[i] == dz_df.values[i] or abs(z.d_df.values[i] - dz_df.values[i]) <= EPS
        assert z.d_dg.values[i] == dz_dg.values[i] or abs(z.d_dg.values[i] - dz_dg.values[i]) <= EPS
        assert z.d_dh.values[i] == dz_dh.values[i] or abs(z.d_dh.values[i] - dz_dh.values[i]) <= EPS

    assert y.dot(x, recursive=False).derivs == {}
    assert hasattr(x, 'd_df')
    assert hasattr(x, 'd_dh')
    assert hasattr(y, 'd_dg')
    assert hasattr(y, 'd_dh')
    assert not hasattr(y.dot(x, recursive=False), 'd_df')
    assert not hasattr(y.dot(x, recursive=False), 'd_dg')
    assert not hasattr(y.dot(x, recursive=False), 'd_dh')


def test_vector_dot_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(5795)
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,4))
    with pytest.raises(ValueError):
        a.dot(b)
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,5))
    assert a.dot(b) == np.sum(a.values * b.values, axis=-1)

    N = 10
    y = Vector(np.random.randn(N,7))
    x = Vector(np.random.randn(N,7))
    assert not x.readonly
    assert not y.readonly
    assert not y.dot(x).readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not y.as_readonly().dot(x.as_readonly()).readonly
    assert not y.as_readonly().dot(x).readonly
    assert not y.dot(x.as_readonly()).readonly


##########################################################################################
