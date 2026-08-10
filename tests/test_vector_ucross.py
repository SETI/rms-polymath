##########################################################################################
# tests/test_vector_ucross.py
##########################################################################################

import numpy as np

from polymath import Unit, Vector


def test_vector_ucross_single_values() -> None:
    """Single values."""

    np.random.seed(2418)

    x = Vector((1.,0.,0.))
    y = Vector((0.,1.,0.))
    z = Vector((0.,0.,1.))
    assert x.ucross(y) == z
    assert y.ucross(z) == x
    assert z.ucross(x) == y
    assert not x.ucross(y).mask
    assert x.ucross(x).mask
    assert (3*x).ucross(4*y) == z
    assert (-3*y).ucross(7*z) == -x


def test_vector_ucross_array_values() -> None:
    """Array values."""

    np.random.seed(2418)

    N = 100
    x = Vector(np.random.randn(N*3).reshape(N,3))
    y = Vector(np.random.randn(N*3).reshape(N,3))
    z = x.ucross(y)
    for i in range(N):
        assert x.dot(z)[i] == 0. or abs(x.dot(z)[i] - 0.) <= 1.e-12
        assert y.dot(z)[i] == 0. or abs(y.dot(z)[i] - 0.) <= 1.e-12
        assert z.dot(z)[i] == 1. or abs(z.dot(z)[i] - 1.) <= 1.e-12


def test_vector_ucross_units_are_stripped() -> None:
    """Units are stripped."""

    np.random.seed(2418)

    N = 10
    x = Vector(np.random.randn(N*3).reshape(N,3), unit=Unit.KM)
    y = Vector(np.random.randn(N*3).reshape(N,3), unit=Unit.SEC)
    z = x.ucross(y)
    assert z.unit_ == Unit.UNITLESS
    N = 10
    x = Vector(np.random.randn(N*3).reshape(N,3))
    y = Vector(np.random.randn(N*3).reshape(N,3))
    z = x.ucross(y)
    assert (z.unit_ is None)


def test_vector_ucross_derivatives_denom() -> None:
    """Derivatives, denom = ()."""

    np.random.seed(2418)

    N = 6
    x = Vector(np.random.randn(N*3).reshape(N,3))
    y = Vector(np.random.randn(N*3).reshape(N,3))
    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))
    z = y.ucross(x)
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
    z1 = y.ucross(x + (EPS,0,0))
    z0 = y.ucross(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.ucross(x + (0,EPS,0))
    z0 = y.ucross(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.ucross(x + (0,0,EPS))
    z0 = y.ucross(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0,0)).ucross(x)
    z0 = (y - (EPS,0,0)).ucross(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS,0)).ucross(x)
    z0 = (y - (0,EPS,0)).ucross(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,0,EPS)).ucross(x)
    z0 = (y - (0,0,EPS)).ucross(x)
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


def test_vector_ucross_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(2418)

    N = 10
    y = Vector(np.random.randn(N*3).reshape(N,3))
    x = Vector(np.random.randn(N*3).reshape(N,3))
    assert not x.readonly
    assert not y.readonly
    assert not y.ucross(x).readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not y.as_readonly().ucross(x.as_readonly()).readonly
    assert not y.as_readonly().ucross(x).readonly
    assert not y.ucross(x.as_readonly()).readonly


##########################################################################################
