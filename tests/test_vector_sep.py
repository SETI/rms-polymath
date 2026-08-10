##########################################################################################
# tests/test_vector_sep.py
##########################################################################################

import numpy as np

from polymath import Unit, Vector


def test_vector_sep_single_values() -> None:
    """Single values."""

    np.random.seed(8393)

    DEL = 1.e-12
    a = Vector((2,0,0))
    assert a.sep(Vector((0,1,0))) == 0.50 * np.pi or abs(a.sep(Vector((0,1,0))) - 0.50 * np.pi) <= DEL
    assert a.sep(Vector((1,0,1))) == 0.25 * np.pi or abs(a.sep(Vector((1,0,1))) - 0.25 * np.pi) <= DEL
    assert a.sep(Vector((-1,0,1))) == 0.75 * np.pi or abs(a.sep(Vector((-1,0,1))) - 0.75 * np.pi) <= DEL
    assert a.sep(Vector((-1,0,0))) == 1.00 * np.pi or abs(a.sep(Vector((-1,0,0))) - 1.00 * np.pi) <= DEL


def test_vector_sep_multiple_values() -> None:
    """Multiple values."""

    np.random.seed(8393)

    N = 100
    a = Vector(np.random.randn(N,3))
    b = Vector(np.random.randn(N,3))
    sep = a.sep(b)
    sep1 = a.unit().dot(b.unit()).arccos()
    for i in range(N):
        assert sep[i] == sep1[i] or abs(sep[i] - sep1[i]) <= 1.e-10
    sep2 = a.unit().cross(b.unit()).norm().arcsin()
    mask = (a.dot(b) < 0.)
    sep2[mask] = np.pi - sep2[mask]
    for i in range(N):
        assert sep[i] == sep2[i] or abs(sep[i] - sep2[i]) <= 2.e-10


def test_vector_sep_test_units() -> None:
    """Test units."""

    np.random.seed(8393)

    N = 10
    a = Vector(np.random.randn(N,3), unit=Unit.KM)
    b = Vector(np.random.randn(N,3), unit=Unit.KM)
    assert (a.sep(b).mask is False)
    assert (a.sep(b).unit_ is None)
    a = Vector(np.random.randn(N,3), unit=Unit.KM)
    b = Vector(np.random.randn(N,3), unit=Unit.CM)
    assert (a.sep(b).mask is False)
    assert (a.sep(b).unit_ is None)
    a = Vector(np.random.randn(N,3), unit=Unit.KM)
    b = Vector(np.random.randn(N,3), unit=Unit.S)
    assert (a.sep(b).mask is False)
    assert (a.sep(b).unit_ is None)


def test_vector_sep_derivatives() -> None:
    """Derivatives."""

    np.random.seed(8393)

    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))
    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))
    z = y.sep(x)
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
    z1 = y.sep(x + (EPS,0,0))
    z0 = y.sep(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.sep(x + (0,EPS,0))
    z0 = y.sep(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.sep(x + (0,0,EPS))
    z0 = y.sep(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0,0)).sep(x)
    z0 = (y - (EPS,0,0)).sep(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS,0)).sep(x)
    z0 = (y - (0,EPS,0)).sep(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,0,EPS)).sep(x)
    z0 = (y - (0,0,EPS)).sep(x)
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

    assert y.sep(x, recursive=False).derivs == {}
    assert hasattr(x, 'd_df')
    assert hasattr(x, 'd_dh')
    assert hasattr(y, 'd_dg')
    assert hasattr(y, 'd_dh')
    assert not hasattr(y.sep(x, recursive=False), 'd_df')
    assert not hasattr(y.sep(x, recursive=False), 'd_dg')
    assert not hasattr(y.sep(x, recursive=False), 'd_dh')


def test_vector_sep_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(8393)

    N = 10
    y = Vector(np.random.randn(N,7))
    x = Vector(np.random.randn(N,7))
    assert not x.readonly
    assert not y.readonly
    assert not y.sep(x).readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not y.as_readonly().sep(x.as_readonly()).readonly
    assert not y.as_readonly().sep(x).readonly
    assert not y.sep(x.as_readonly()).readonly


##########################################################################################
