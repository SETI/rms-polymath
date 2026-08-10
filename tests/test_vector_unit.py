##########################################################################################
# tests/test_vector_unit.py
##########################################################################################

import numpy as np

from polymath import Vector


def test_vector_unit_single_values() -> None:
    """Single values."""

    np.random.seed(3455)

    x = Vector((1.,2.,4.,8.))
    u = x.unit()
    assert np.sum(u.values**2) == 1. or abs(np.sum(u.values**2) - 1.) <= 1.e-15
    assert x.dot(u) == x.norm() or abs(x.dot(u) - x.norm()) <= 1.e-15
    x = Vector((1.,2.,4.,8.), mask=True)
    u = x.unit()
    assert (u.mask is True)
    x = Vector((0.,0.,0.,0.,0.), mask=False)
    u = x.unit()
    assert (u.mask is True)


def test_vector_unit_arrays_and_masks() -> None:
    """Arrays and masks."""

    np.random.seed(3455)

    x = Vector(np.zeros((30,7)))
    u = x.unit()
    assert np.all(u.mask)
    x = Vector(np.random.randn(30,7))
    u = x.unit()
    assert not np.any(u.mask)
    N = 100
    x = Vector(np.random.randn(N,7),
               mask=(np.random.randn(N) < -0.3))    # Mask out a fraction
    u = x.unit()
    assert np.all(u.mask == x.mask)
    utest = u[~u.mask]
    for i in range(len(utest)):
        assert utest[i].norm() == 1. or abs(utest[i].norm() - 1.) <= 1.e-15
    zeros = (np.random.randn(N) < 0.3)
    x.values[zeros] = 0.
    u = x.unit()
    for i in range(N):
        if zeros[i]:
            assert u[i].mask
        else:
            assert u[i].mask == x[i].mask
            if not u[i].mask:
                assert u[i].norm() == 1. or abs(u[i].norm() - 1.) <= 1.e-15


def test_vector_unit_derivatives_denom() -> None:
    """Derivatives, denom = ()."""

    np.random.seed(3455)

    N = 100
    x = Vector(np.random.randn(N,3))
    x.insert_deriv('t', Vector(np.random.randn(N,3)))
    x.insert_deriv('v', Vector(np.random.randn(N,3,3), drank=1,
                               mask=(np.random.randn(N) < -0.4)))
    assert 't' in x.derivs
    assert hasattr(x, 'd_dt')
    assert 'v' in x.derivs
    assert hasattr(x, 'd_dv')
    y = x.unit(recursive=False)
    assert 't' not in y.derivs
    assert not hasattr(y, 'd_dt')
    assert 'v' not in y.derivs
    assert not hasattr(y, 'd_dv')
    y = x.unit()
    assert 't' in y.derivs
    assert hasattr(y, 'd_dt')
    assert 'v' in y.derivs
    assert hasattr(y, 'd_dv')
    EPS = 1.e-6
    y1 = (x + (EPS,0,0)).unit()
    y0 = (x - (EPS,0,0)).unit()
    dy_dx0 = 0.5 * (y1 - y0) / EPS
    y1 = (x + (0,EPS,0)).unit()
    y0 = (x - (0,EPS,0)).unit()
    dy_dx1 = 0.5 * (y1 - y0) / EPS
    y1 = (x + (0,0,EPS)).unit()
    y0 = (x - (0,0,EPS)).unit()
    dy_dx2 = 0.5 * (y1 - y0) / EPS
    dy_dt = (dy_dx0 * x.d_dt.values[:,0] +
             dy_dx1 * x.d_dt.values[:,1] +
             dy_dx2 * x.d_dt.values[:,2])
    dy_dv0 = (dy_dx0 * x.d_dv.values[:,0,0] +
              dy_dx1 * x.d_dv.values[:,1,0] +
              dy_dx2 * x.d_dv.values[:,2,0])
    dy_dv1 = (dy_dx0 * x.d_dv.values[:,0,1] +
              dy_dx1 * x.d_dv.values[:,1,1] +
              dy_dx2 * x.d_dv.values[:,2,1])
    dy_dv2 = (dy_dx0 * x.d_dv.values[:,0,2] +
              dy_dx1 * x.d_dv.values[:,1,2] +
              dy_dx2 * x.d_dv.values[:,2,2])
    for i in range(N):
        for k in range(3):
            assert y.d_dt.values[i,k] == dy_dt.values[i,k] or abs(y.d_dt.values[i,k] - dy_dt.values[i,k]) <= EPS
            assert y.d_dv.values[i,k,0] == dy_dv0.values[i,k] or abs(y.d_dv.values[i,k,0] - dy_dv0.values[i,k]) <= EPS
            assert y.d_dv.values[i,k,1] == dy_dv1.values[i,k] or abs(y.d_dv.values[i,k,1] - dy_dv1.values[i,k]) <= EPS
            assert y.d_dv.values[i,k,2] == dy_dv2.values[i,k] or abs(y.d_dv.values[i,k,2] - dy_dv2.values[i,k]) <= EPS


def test_vector_unit_read_only_status_should_be_preserved() -> None:
    """Read-only status should be preserved."""

    np.random.seed(3455)

    N = 10
    Vector(np.random.randn(N,3))
    x = Vector(np.random.randn(N,3))
    assert not x.readonly
    assert not x.unit().readonly
    assert not x.as_readonly().unit().readonly


##########################################################################################
