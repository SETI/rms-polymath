##########################################################################################
# tests/test_scalar_sqrt.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_sqrt_individual_values() -> None:
    """Individual values."""

    np.random.seed(9396)

    assert Scalar(0.3).sqrt() == np.sqrt(0.3)
    assert type(Scalar(0.3).sqrt()) == Scalar
    assert Scalar(4.).sqrt() == np.sqrt(4.)
    assert Scalar(4).sqrt() == 2.

    assert Scalar((1,2,3)).sqrt() == np.sqrt((1,2,3))
    assert type(Scalar((1,2,3)).sqrt()) == Scalar

    N = 1000
    x = Scalar(np.random.randn(N))
    y = x.sqrt()
    for i in range(N):
        if x.values[i] >= 0.:
            assert y[i] == np.sqrt(x.values[i])
            assert not y.mask[i]
        else:
            assert y.mask[i]
    for i in range(N-1):
        if np.all(x.values[i:i+2] >= 0):
            assert y[i:i+2] == np.sqrt(x.values[i:i+2])

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    with pytest.raises(ValueError):
        Scalar.sqrt(random)
    random = Scalar((4.,9.,16.), unit=Unit.KM**2)
    assert random.sqrt() == (2,3,4)
    assert random.sqrt() == Scalar((2,3,4), unit=Unit.KM)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.SECONDS)
    with pytest.raises(ValueError):
        Scalar.sqrt(random)
    random = Scalar(values, unit=Unit.DEG)
    with pytest.raises(ValueError):
        Scalar.sqrt(random)
    random = Scalar(values, unit=Unit.RAD)
    with pytest.raises(ValueError):
        Scalar.sqrt(random)
    x = Scalar(4., unit=Unit.UNITLESS)
    assert not x.sqrt().mask
    x = Scalar(-4., unit=Unit.UNITLESS)
    assert x.sqrt().mask

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.sqrt()
    assert np.all(y.mask[x.mask])

    N = 100
    x = Scalar(np.random.randn(N))
    x.insert_deriv('t', Scalar(np.random.randn(N)))
    assert 't' in x.derivs
    assert hasattr(x, 'd_dt')
    assert 't' in x.sqrt().derivs
    assert hasattr(x.sqrt(), 'd_dt')
    EPS = 1.e-6
    y1 = (x + EPS).sqrt()
    y0 = (x - EPS).sqrt()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.sqrt().d_dt
    DEL = 1.e-5
    for i in range(N):
        assert dy_dx[i] * x.d_dt[i] == dy_dt[i] or abs(dy_dx[i] * x.d_dt[i] - dy_dt[i]) <= abs(dy_dt[i]) * DEL

    N = 10
    x = Scalar(np.random.randn(N))
    assert not x.readonly
    assert not x.sqrt().readonly
    assert x.as_readonly().readonly
    assert not x.as_readonly().sqrt().readonly

    N = 1000
    x = Scalar(np.random.randn(N))
    with pytest.raises(ValueError):
        x.sqrt(check=False)
    x = Scalar(np.random.randn(N).clip(0,1.e308))
    assert x.sqrt() == np.sqrt(x.values)


##########################################################################################
