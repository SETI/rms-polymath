##########################################################################################
# tests/test_pair_swapxy.py
##########################################################################################

import numpy as np

from polymath import Pair, Unit


def test_pair_swapxy_single_values() -> None:
    """Single values."""

    np.random.seed(1871)

    a = Pair((1,2))
    b = a.swapxy()
    assert b == (2,1)
    assert (a.mask is b.mask)


def test_pair_swapxy_arrays_denoms() -> None:
    """Arrays & denoms."""

    np.random.seed(1871)

    N = 10
    a = Pair(np.arange(N*6).reshape(N,2,3), drank=1)
    b = a.swapxy()
    aparts = a.to_scalars()
    bparts = b.to_scalars()
    assert aparts[0] == bparts[1]
    assert aparts[1] == bparts[0]

    a = Pair(np.random.randn(N,2,3), drank=1,
             mask = (np.random.randn(N) < -0.4))
    b = a.swapxy()
    assert np.all(a.mask == b.mask)


def test_pair_swapxy_unit() -> None:
    """Unit."""

    np.random.seed(1871)

    N = 10
    a = Pair(np.arange(N*6).reshape(N,2,3), drank=1, unit=Unit.DEG)
    b = a.swapxy()
    assert b.units == a.units


def test_pair_swapxy_derivatives_denom() -> None:
    """Derivatives, denom = ()."""

    np.random.seed(1871)

    N = 100
    a = Pair(np.random.randn(N,2))
    a.insert_deriv('t', Pair(np.random.randn(N,2)))
    a.insert_deriv('v', Pair(np.random.randn(N,2,3), drank=1,
                             mask = (np.random.randn(N) < -0.4)))
    assert 't' in a.derivs
    assert hasattr(a, 'd_dt')
    assert 'v' in a.derivs
    assert hasattr(a, 'd_dv')
    b = a.swapxy(recursive=False)
    assert 't' not in b.derivs
    assert not hasattr(b, 'd_dt')
    assert 'v' not in b.derivs
    assert not hasattr(b, 'd_dv')
    b = a.swapxy()
    assert 't' in b.derivs
    assert hasattr(b, 'd_dt')
    assert 'v' in b.derivs
    assert hasattr(b, 'd_dv')
    EPS = 1.e-6
    b1 = (a + (EPS,0)).swapxy()
    b0 = (a - (EPS,0)).swapxy()
    db_da0 = 0.5 * (b1 - b0) / EPS
    b1 = (a + (0,EPS)).swapxy()
    b0 = (a - (0,EPS)).swapxy()
    db_da1 = 0.5 * (b1 - b0) / EPS
    db_dt = (db_da0 * a.d_dt.values[:,0] +
             db_da1 * a.d_dt.values[:,1])
    db_dv0 = (db_da0 * a.d_dv.values[:,0,0] +
              db_da1 * a.d_dv.values[:,1,0])
    db_dv1 = (db_da0 * a.d_dv.values[:,0,1] +
              db_da1 * a.d_dv.values[:,1,1])
    db_dv2 = (db_da0 * a.d_dv.values[:,0,2] +
              db_da1 * a.d_dv.values[:,1,2])
    DEL = 1.e-5
    for i in range(N):
        for k in range(2):
            assert b.d_dt.values[i,k] == db_dt.values[i,k] or abs(b.d_dt.values[i,k] - db_dt.values[i,k]) <= DEL
            assert b.d_dv.values[i,k,0] == db_dv0.values[i,k] or abs(b.d_dv.values[i,k,0] - db_dv0.values[i,k]) <= DEL
            assert b.d_dv.values[i,k,1] == db_dv1.values[i,k] or abs(b.d_dv.values[i,k,1] - db_dv1.values[i,k]) <= DEL
            assert b.d_dv.values[i,k,2] == db_dv2.values[i,k] or abs(b.d_dv.values[i,k,2] - db_dv2.values[i,k]) <= DEL
    da_dt_parts = a.d_dt.to_scalars()
    db_dt_parts = b.d_dt.to_scalars()
    assert da_dt_parts[0] == db_dt_parts[1]
    assert da_dt_parts[1] == db_dt_parts[0]
    da_dv_parts = a.d_dv.to_scalars()
    db_dv_parts = b.d_dv.to_scalars()
    assert da_dv_parts[0] == db_dv_parts[1]
    assert da_dv_parts[1] == db_dv_parts[0]


def test_pair_swapxy_read_only_status_should_be_preserved() -> None:
    """Read-only status should be preserved."""

    np.random.seed(1871)

    N = 10
    a = Pair(np.random.randn(N,2))
    Pair(np.random.randn(N,2))
    assert not a.readonly
    assert not a.swapxy().readonly
    assert a.as_readonly().swapxy().readonly


##########################################################################################
