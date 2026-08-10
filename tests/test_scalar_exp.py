##########################################################################################
# tests/test_scalar_exp.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_exp_individual_values() -> None:
    """Individual values."""

    np.random.seed(1641)

    assert Scalar(1.25).exp() == np.exp(1.25)
    assert type(Scalar(1.25).exp()) == Scalar
    assert Scalar(1).exp() == np.exp(1.)
    assert Scalar(0).exp() == 1.

    assert Scalar((-1,0,1)).exp() == np.exp((-1,0,1))
    assert type(Scalar((-1,0,1)).exp()) == Scalar

    N = 1000
    values = np.random.randn(N) * 10.
    angles = Scalar(values)
    funcvals = angles.exp()
    for i in range(N):
        assert funcvals[i] == np.exp(values[i])
    for i in range(N-1):
        assert funcvals[i:i+2] == np.exp(values[i:i+2])

    values = np.random.randn(10) * 10.
    random = Scalar(values, unit=Unit.KM)
    with pytest.raises(ValueError):
        Scalar.exp(random)
    values = np.random.randn(10) * 10.
    random = Scalar(values, unit=Unit.SECONDS)
    with pytest.raises(ValueError):
        Scalar.exp(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    with pytest.raises(ValueError):
        Scalar.exp(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.UNITLESS)
    assert random.exp() == np.exp(values)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    with pytest.raises(ValueError):
        Scalar.exp(random)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.exp()
    assert np.all(y.mask[x.mask])
    assert not np.any(y.mask[~x.mask])

    N = 100
    x = Scalar(np.random.randn(N) * 10.)
    x.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    x.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)), drank=1))
    assert 't' in x.derivs
    assert 'vec' in x.derivs
    assert hasattr(x, 'd_dt')
    assert hasattr(x, 'd_dvec')
    assert 't' in x.exp().derivs
    assert 'vec' in x.exp().derivs
    assert hasattr(x.exp(), 'd_dt')
    assert hasattr(x.exp(), 'd_dvec')
    EPS = 1.e-6
    y1 = (x + EPS).exp()
    y0 = (x - EPS).exp()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.exp().d_dt
    dy_dvec = x.exp().d_dvec
    for i in range(N):
        assert dy_dx[i] * x.d_dt[i] == dy_dt[i] or abs(dy_dx[i] * x.d_dt[i] - dy_dt[i]) <= max(1,abs(dy_dt[i])) * EPS

        for k in range(3):
            assert dy_dx[i] * x.d_dvec[i].values[k] == dy_dvec[i].values[k] or abs(dy_dx[i] * x.d_dvec[i].values[k] - dy_dvec[i].values[k]) <= max(1,abs(dy_dvec[i].values[k]))*EPS

    assert x.exp(recursive=False).derivs == {}
    assert hasattr(x, 'd_dt')
    assert hasattr(x, 'd_dvec')
    assert not hasattr(x.exp(recursive=False), 'd_dt')
    assert not hasattr(x.exp(recursive=False), 'd_dvec')

    N = 10
    x = Scalar(np.random.randn(N) * 10.)
    assert not x.readonly
    assert not x.exp().readonly
    assert x.as_readonly().readonly
    assert not x.as_readonly().exp().readonly

    N = 1000
    x = Scalar(np.random.randn(N) * 700.)
    with pytest.raises(ValueError):
        x.log(check=False)
    assert (x.exp(check=True).max() < np.inf)
    assert (x.exp(check=True).max() > 1.e200)
    assert type(x.exp(check=True).mask) == np.ndarray
    assert (np.sum(x.exp(check=True).mask) > 0)


##########################################################################################
