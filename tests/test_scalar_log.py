##########################################################################################
# tests/test_scalar_log.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_log_individual_values() -> None:
    """Individual values."""

    np.random.seed(8622)

    assert Scalar(0.3).log() == np.log(0.3)
    assert type(Scalar(0.3).log()) == Scalar
    assert Scalar(1.).log() == np.log(1.)
    assert Scalar(1).log() == 0.

    assert Scalar((1,2,3)).log() == np.log((1,2,3))
    assert type(Scalar((1,2,3)).log()) == Scalar

    N = 1000
    x = Scalar(np.random.randn(N))
    y = x.log()
    for i in range(N):
        if x.values[i] > 0.:
            assert y[i] == np.log(x.values[i])
            assert not y.mask[i]
        else:
            assert y.mask[i]
    for i in range(N-1):
        if np.all(x.values[i:i+2] >= 0):
            assert y[i:i+2] == np.log(x.values[i:i+2])

    values = np.abs(np.random.randn(10))
    random = Scalar(values, unit=Unit.KM)
    assert random.log() == Scalar(np.log(values))
    values = np.abs(np.random.randn(10))
    random = Scalar(values, unit=Unit.SECONDS)
    assert random.log() == Scalar(np.log(values))
    values = np.abs(np.random.randn(10))
    random = Scalar(values, unit=Unit.DEG)
    assert random.log() == Scalar(np.log(values))
    values = np.abs(np.random.randn(10))
    random = Scalar(values, unit=Unit.UNITLESS)
    assert random.log() == Scalar(np.log(values))
    x = Scalar(4., unit=Unit.UNITLESS)
    assert not x.log().mask
    x = Scalar(-4., unit=Unit.UNITLESS)
    assert x.log().mask

    random = Scalar(values, unit=Unit.DEG)
    assert (random.log()._unit is None)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.log()
    assert np.all(y.mask[x.mask])

    N = 100
    x = Scalar(np.random.randn(N))
    x.insert_deriv('t', Scalar(np.random.randn(N)))
    assert 't' in x.derivs
    assert hasattr(x, 'd_dt')
    assert 't' in x.log().derivs
    assert hasattr(x.log(), 'd_dt')
    EPS = 1.e-6
    y1 = (x + EPS).log()
    y0 = (x - EPS).log()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.log().d_dt
    DEL = 1.e-5
    for i in range(N):
        assert dy_dx[i] * x.d_dt[i] == dy_dt[i] or abs(dy_dx[i] * x.d_dt[i] - dy_dt[i]) <= DEL * abs(dy_dt[i])

    assert x.log(recursive=False).derivs == {}
    assert hasattr(x, 'd_dt')
    assert not hasattr(x.log(recursive=False), 'd_dt')

    N = 10
    x = Scalar(np.random.randn(N))
    assert not x.readonly
    assert not x.log().readonly
    assert x.as_readonly().readonly
    assert not x.as_readonly().log().readonly

    N = 1000
    x = Scalar(np.random.randn(N))
    with pytest.raises(ValueError):
        x.log(check=False)
    x = Scalar(np.random.randn(N).clip(1.e-99,1.e99))
    assert x.log() == np.log(x.values)


##########################################################################################
