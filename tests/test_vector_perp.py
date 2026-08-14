##########################################################################################
# tests/test_vector_perp.py
##########################################################################################

import numpy as np

from polymath import Unit, Vector


def test_vector_perp_single_values() -> None:
    """Single values."""

    np.random.seed(2435)

    assert Vector((2,3,0)).perp((0,7,0)) == (2,0,0)
    assert Vector((2,3,0)).perp((-1,0,0)) == (0,3,0)
    assert Vector((2,3,0),True).perp((-1,0,0)).mask
    assert Vector((2,3,0)).perp((0,0,0)).mask
    assert Vector((0,0,0)).perp((1,1,1)).norm() == 0.


def test_vector_perp_arrays_and_masks() -> None:
    """Arrays and masks."""

    np.random.seed(2435)

    N = 100
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))
    z = y.perp(x)
    for i in range(N):
        assert z[i].cross(x[i]).norm() == z[i].norm() * x[i].norm() or abs(z[i].cross(x[i]).norm() - z[i].norm() * x[i].norm()) <= 1.e-14
        assert z[i].dot(x[i]) == 0. or abs(z[i].dot(x[i]) - 0.) <= 1.e-14
    N = 100
    x = Vector(np.random.randn(N,3), np.random.randn(N) < -0.5)
    y = Vector(np.random.randn(N,3), np.random.randn(N) < -0.5)
    z = y.perp(x)
    zero_mask = (np.random.randn(N) < -0.5) # Insert some zero-valued vectors
    x[zero_mask] = Vector.ZERO3
    z = y.perp(x)
    assert np.all(z.mask == (x.mask | y.mask | zero_mask))

    xx = x[~z.mask]
    zz = z[~z.mask]
    for i in range(len(zz)):
        assert zz[i].cross(xx[i]).norm() == zz[i].norm() * xx[i].norm() or abs(zz[i].cross(xx[i]).norm() - zz[i].norm() * xx[i].norm()) <= 1.e-14
        assert zz[i].dot(xx[i]) == 0. or abs(zz[i].dot(xx[i]) - 0.) <= 1.e-14


def test_vector_perp_test_units() -> None:
    """Test units."""

    np.random.seed(2435)

    N = 100
    x = Vector(np.random.randn(N,3), unit=Unit.KM)
    y = Vector(np.random.randn(N,3), unit=Unit.SECONDS**(-1))
    z = y.perp(x)
    assert z.unit_ == Unit.SECONDS**(-1)


def test_vector_perp_derivatives_denom() -> None:
    """Derivatives, denom = ()."""

    np.random.seed(2435)

    N = 100
    x = Vector(np.random.randn(N*3).reshape((N,3)))
    y = Vector(np.random.randn(N*3).reshape((N,3)))
    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))
    z = y.perp(x)
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
    z1 = y.perp(x + (EPS,0,0))
    z0 = y.perp(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.perp(x + (0,EPS,0))
    z0 = y.perp(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.perp(x + (0,0,EPS))
    z0 = y.perp(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0,0)).perp(x)
    z0 = (y - (EPS,0,0)).perp(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS,0)).perp(x)
    z0 = (y - (0,EPS,0)).perp(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,0,EPS)).perp(x)
    z0 = (y - (0,0,EPS)).perp(x)
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
    DEL = 1.e-5
    for i in range(N):
        for k in range(3):
            assert z.d_df.values[i,k] == dz_df.values[i,k] or abs(z.d_df.values[i,k] - dz_df.values[i,k]) <= DEL
            assert z.d_dg.values[i,k] == dz_dg.values[i,k] or abs(z.d_dg.values[i,k] - dz_dg.values[i,k]) <= DEL
            assert z.d_dh.values[i,k] == dz_dh.values[i,k] or abs(z.d_dh.values[i,k] - dz_dh.values[i,k]) <= DEL


def test_vector_perp_derivatives_denom_2() -> None:
    """Derivatives, denom = (2,)."""

    np.random.seed(2435)

    N = 100
    x = Vector(np.random.randn(N*3).reshape(N,3))
    y = Vector(np.random.randn(N*3).reshape(N,3))
    x.insert_deriv('f', Vector(np.random.randn(N,3,2), drank=1))
    x.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('g', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))
    z = y.perp(x)
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
    z1 = y.perp(x.wod + (EPS,0,0))
    z0 = y.perp(x.wod - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.perp(x.wod + (0,EPS,0))
    z0 = y.perp(x.wod - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.perp(x.wod + (0,0,EPS))
    z0 = y.perp(x.wod - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    z1 = (y.wod + (EPS,0,0)).perp(x.wod)
    z0 = (y.wod - (EPS,0,0)).perp(x.wod)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y.wod + (0,EPS,0)).perp(x.wod)
    z0 = (y.wod - (0,EPS,0)).perp(x.wod)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y.wod + (0,0,EPS)).perp(x.wod)
    z0 = (y.wod - (0,0,EPS)).perp(x.wod)
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
    DEL = 1.e-5
    for i in range(N):
        for k in range(3):
            assert z.d_df.values[i,k,0] == dz_df0.values[i,k] or abs(z.d_df.values[i,k,0] - dz_df0.values[i,k]) <= DEL
            assert z.d_dg.values[i,k,0] == dz_dg0.values[i,k] or abs(z.d_dg.values[i,k,0] - dz_dg0.values[i,k]) <= DEL
            assert z.d_dh.values[i,k,0] == dz_dh0.values[i,k] or abs(z.d_dh.values[i,k,0] - dz_dh0.values[i,k]) <= DEL

            assert z.d_df.values[i,k,1] == dz_df1.values[i,k] or abs(z.d_df.values[i,k,1] - dz_df1.values[i,k]) <= DEL
            assert z.d_dg.values[i,k,1] == dz_dg1.values[i,k] or abs(z.d_dg.values[i,k,1] - dz_dg1.values[i,k]) <= DEL
            assert z.d_dh.values[i,k,1] == dz_dh1.values[i,k] or abs(z.d_dh.values[i,k,1] - dz_dh1.values[i,k]) <= DEL

    assert y.perp(x, recursive=False).derivs == {}
    assert hasattr(x, 'd_df')
    assert hasattr(x, 'd_dh')
    assert hasattr(y, 'd_dg')
    assert hasattr(y, 'd_dh')
    assert not hasattr(y.perp(x, recursive=False), 'd_df')
    assert not hasattr(y.perp(x, recursive=False), 'd_dg')
    assert not hasattr(y.perp(x, recursive=False), 'd_dh')


def test_vector_perp_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(2435)

    N = 10
    y = Vector(np.random.randn(N*3).reshape(N,3))
    x = Vector(np.random.randn(N*3).reshape(N,3))
    assert not x.readonly
    assert not y.readonly
    assert not y.perp(x).readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not y.as_readonly().perp(x.as_readonly()).readonly
    assert not y.as_readonly().perp(x).readonly
    assert not y.perp(x.as_readonly()).readonly


##########################################################################################
