##########################################################################################
# tests/test_vector_outer.py
##########################################################################################

import numpy as np

from polymath import Unit, Vector


def test_vector_outer_test_units() -> None:
    """Test units."""

    np.random.seed(9008)
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,2))
    assert a.outer(b).shape == (3,10)
    assert a.outer(b).numer == (5,2)
    assert a.outer(b).denom == ()
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,5))
    assert a.outer(b) == (a.values.reshape((1,10,5,1)) *
                                 b.values.reshape((3,10,1,5)))

    a = Vector(np.random.randn(3), unit=Unit.KM)
    b = Vector(np.random.randn(3), unit=Unit.SECONDS**(-1))
    assert a.outer(b).unit_ == Unit.KM/Unit.SECONDS
    assert b.outer(a).unit_ == Unit.KM/Unit.SECONDS


def test_vector_outer_derivatives() -> None:
    """Derivatives."""

    np.random.seed(9008)
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,2))
    assert a.outer(b).shape == (3,10)
    assert a.outer(b).numer == (5,2)
    assert a.outer(b).denom == ()
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,5))
    assert a.outer(b) == (a.values.reshape((1,10,5,1)) *
                                 b.values.reshape((3,10,1,5)))

    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))
    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))
    z = y.outer(x)
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
    z1 = y.outer(x + (EPS,0,0))
    z0 = y.outer(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.outer(x + (0,EPS,0))
    z0 = y.outer(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.outer(x + (0,0,EPS))
    z0 = y.outer(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0,0)).outer(x)
    z0 = (y - (EPS,0,0)).outer(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS,0)).outer(x)
    z0 = (y - (0,EPS,0)).outer(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,0,EPS)).outer(x)
    z0 = (y - (0,0,EPS)).outer(x)
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
        for j in range(3):
            for k in range(3):
                assert z.d_df.values[i,j,k] == dz_df.values[i,j,k] or abs(z.d_df.values[i,j,k] - dz_df.values[i,j,k]) <= EPS
                assert z.d_dg.values[i,j,k] == dz_dg.values[i,j,k] or abs(z.d_dg.values[i,j,k] - dz_dg.values[i,j,k]) <= EPS
                assert z.d_dh.values[i,j,k] == dz_dh.values[i,j,k] or abs(z.d_dh.values[i,j,k] - dz_dh.values[i,j,k]) <= EPS

    assert y.outer(x, recursive=False).derivs == {}
    assert hasattr(x, 'd_df')
    assert hasattr(x, 'd_dh')
    assert hasattr(y, 'd_dg')
    assert hasattr(y, 'd_dh')
    assert not hasattr(y.outer(x, recursive=False), 'd_df')
    assert not hasattr(y.outer(x, recursive=False), 'd_dg')
    assert not hasattr(y.outer(x, recursive=False), 'd_dh')


def test_vector_outer_read_only_status_should_be_preserved() -> None:
    """Read-only status should be preserved."""

    np.random.seed(9008)
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,2))
    assert a.outer(b).shape == (3,10)
    assert a.outer(b).numer == (5,2)
    assert a.outer(b).denom == ()
    a = Vector(np.random.randn(10,5))
    b = Vector(np.random.randn(3,10,5))
    assert a.outer(b) == (a.values.reshape((1,10,5,1)) *
                                 b.values.reshape((3,10,1,5)))

    N = 10
    y = Vector(np.random.randn(N,7))
    x = Vector(np.random.randn(N,7))
    assert not x.readonly
    assert not y.readonly
    assert not y.outer(x).readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not y.as_readonly().outer(x.as_readonly()).readonly
    assert not y.as_readonly().outer(x).readonly
    assert not y.outer(x.as_readonly()).readonly


##########################################################################################
