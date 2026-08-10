##########################################################################################
# tests/test_scalar_arctan.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_arctan_individual_values() -> None:
    """Individual values."""

    np.random.seed(6021)

    assert Scalar(-0.3).arctan() == np.arctan(-0.3)
    assert type(Scalar(-0.3).arctan()) == Scalar
    assert Scalar(0.).arctan() == np.arctan(0.)
    assert Scalar(0).arctan() == 0.

    assert Scalar((-0.1,0.,0.1)).arctan() == np.arctan((-0.1,0.,0.1))
    assert type(Scalar((-0.1,0.,0.1)).arctan()) == Scalar

    N = 1000
    x = Scalar(np.random.randn(N))
    y = x.arctan()
    for i in range(N):
        assert y[i] == np.arctan(x.values[i])
    for i in range(N-1):
        if np.all(np.abs(x.values[i:i+2]) <= 1):
            assert y[i:i+2] == np.arctan(x.values[i:i+2])

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    with pytest.raises(ValueError):
        Scalar.arctan(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.SECONDS)
    with pytest.raises(ValueError):
        Scalar.arctan(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    with pytest.raises(ValueError):
        Scalar.arctan(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.RAD)
    with pytest.raises(ValueError):
        Scalar.arctan(random)
    x = Scalar(3.25, unit=Unit.UNITLESS)
    assert not x.arctan().mask


def test_scalar_arctan_units_should_be_removed() -> None:
    """Units should be removed."""

    np.random.seed(6021)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.UNITLESS)
    assert (random.arctan().units is None)


def test_scalar_arctan_masks() -> None:
    """Masks."""

    np.random.seed(6021)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.arctan()
    assert np.all(y.mask[x.mask])
    assert not np.any(y.mask[~x.mask])


def test_scalar_arctan_derivatives() -> None:
    """Derivatives."""

    np.random.seed(6021)

    N = 100
    x = Scalar(np.random.randn(N))
    x.insert_deriv('t', Scalar(np.random.randn(N)))
    assert 't' in x.derivs
    assert hasattr(x, 'd_dt')
    assert 't' in x.arctan().derivs
    assert hasattr(x.arctan(), 'd_dt')
    EPS = 1.e-6
    y1 = (x + EPS).arctan()
    y0 = (x - EPS).arctan()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.arctan().d_dt
    for i in range(N):
        assert dy_dx[i] * x.d_dt[i] == dy_dt[i] or abs(dy_dx[i] * x.d_dt[i] - dy_dt[i]) <= EPS

    assert x.arctan(recursive=False).derivs == {}
    assert hasattr(x, 'd_dt')
    assert not hasattr(x.arctan(recursive=False), 'd_dt')


def test_scalar_arctan_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(6021)

    N = 10
    x = Scalar(np.random.randn(N))
    assert not x.readonly
    assert not x.arctan().readonly
    assert x.as_readonly().readonly
    assert not x.as_readonly().arctan().readonly


##########################################################################################
