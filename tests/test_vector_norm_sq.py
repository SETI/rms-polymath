##########################################################################################
# tests/test_vector_norm_sq.py
##########################################################################################

import numpy as np

from polymath import Vector


def test_vector_norm_sq_single_values() -> None:
    """Single values."""

    np.random.seed(8448)

    x = Vector((-1.,))
    assert x.norm_sq() == 1. or abs(x.norm_sq() - 1.) <= 5e-8
    x = Vector((1.,-2.,4.))
    assert x.norm_sq() == 1+4+16 or abs(x.norm_sq() - 1+4+16) <= 1.e-15
    x = Vector((1.,2.,4.,8.), mask=True)
    assert (x.norm_sq().mask is True)


def test_vector_norm_sq_arrays_and_masks() -> None:
    """Arrays and masks."""

    np.random.seed(8448)

    x = Vector(np.random.randn(3,7))
    n = x.norm_sq()
    assert not np.any(n.mask)
    N = 100
    x = Vector(np.random.randn(N,7),
               mask=(np.random.randn(N) < -0.3))    # Mask out a fraction
    n = x.norm_sq()

    nn = n[~n.mask]
    xx = x[~n.mask]
    for i in range(len(nn)):
        assert nn[i] == np.sum(xx.values[i]**2) or abs(nn[i] - np.sum(xx.values[i]**2)) <= 1.e-14
        assert nn[i].mask == xx[i].mask


def test_vector_norm_sq_derivatives_denom() -> None:
    """Derivatives, denom = ()."""

    np.random.seed(8448)

    N = 100
    x = Vector(np.random.randn(N,3))
    x.insert_deriv('t', Vector(np.random.randn(N,3)))
    x.insert_deriv('v', Vector(np.random.randn(N,3,3), drank=1,
                               mask = (np.random.randn(N) < -0.4)))
    assert 't' in x.derivs
    assert hasattr(x, 'd_dt')
    assert 'v' in x.derivs
    assert hasattr(x, 'd_dv')
    y = x.norm_sq(recursive=False)
    assert 't' not in y.derivs
    assert not hasattr(y, 'd_dt')
    assert 'v' not in y.derivs
    assert not hasattr(y, 'd_dv')
    y = x.norm_sq()
    assert 't' in y.derivs
    assert hasattr(y, 'd_dt')
    assert 'v' in y.derivs
    assert hasattr(y, 'd_dv')
    EPS = 1.e-6
    y1 = (x + (EPS,0,0)).norm_sq()
    y0 = (x - (EPS,0,0)).norm_sq()
    dy_dx0 = 0.5 * (y1 - y0) / EPS
    y1 = (x + (0,EPS,0)).norm_sq()
    y0 = (x - (0,EPS,0)).norm_sq()
    dy_dx1 = 0.5 * (y1 - y0) / EPS
    y1 = (x + (0,0,EPS)).norm_sq()
    y0 = (x - (0,0,EPS)).norm_sq()
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
        assert y.d_dt.values[i] == dy_dt.values[i] or abs(y.d_dt.values[i] - dy_dt.values[i]) <= EPS
        assert y.d_dv.values[i,0] == dy_dv0.values[i] or abs(y.d_dv.values[i,0] - dy_dv0.values[i]) <= EPS
        assert y.d_dv.values[i,1] == dy_dv1.values[i] or abs(y.d_dv.values[i,1] - dy_dv1.values[i]) <= EPS
        assert y.d_dv.values[i,2] == dy_dv2.values[i] or abs(y.d_dv.values[i,2] - dy_dv2.values[i]) <= EPS


def test_vector_norm_sq_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(8448)

    N = 10
    Vector(np.random.randn(N,3))
    x = Vector(np.random.randn(N,3))
    assert not x.readonly
    assert not x.norm_sq().readonly
    assert not x.as_readonly().norm_sq().readonly


##########################################################################################
