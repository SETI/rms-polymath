##########################################################################################
# tests/test_scalar_cos.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_cos_individual_values() -> None:
    """Individual values."""

    np.random.seed(8948)

    assert Scalar(1.25).cos() == np.cos(1.25)
    assert type(Scalar(1.25).cos()) == Scalar
    assert Scalar(1).cos() == np.cos(1.)
    assert Scalar(0).cos() == 1.

    assert Scalar((-1,0,1)).cos() == np.cos((-1,0,1))
    assert type(Scalar((-1,0,1)).cos()) == Scalar

    N = 1000
    values = np.random.randn(N) * 10.
    angles = Scalar(values)
    funcvals = angles.cos()
    for i in range(N):
        assert funcvals[i] == np.cos(values[i])
    for i in range(N-1):
        assert funcvals[i:i+2] == np.cos(values[i:i+2])

    values = np.random.randn(10) * 10.
    random = Scalar(values, unit=Unit.KM)
    with pytest.raises(ValueError):
        Scalar.cos(random)
    values = np.random.randn(10) * 10.
    random = Scalar(values, unit=Unit.SECONDS)
    with pytest.raises(ValueError):
        Scalar.cos(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert random.cos() == random.cos()        # unit should be OK
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.RAD)
    assert random.cos() == random.cos()        # unit should be OK
    angle = Scalar(3.25, unit=Unit.UNITLESS)
    assert angle.cos() == np.cos(angle.values) # unit should be OK


def test_scalar_cos_units_should_be_removed() -> None:
    """Units should be removed."""

    np.random.seed(8948)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert (random.cos()._unit is None)


def test_scalar_cos_masks() -> None:
    """Masks."""

    np.random.seed(8948)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.cos()
    assert np.all(y.mask[x.mask])
    assert not np.any(y.mask[~x.mask])


def test_scalar_cos_derivatives() -> None:
    """Derivatives."""

    np.random.seed(8948)

    N = 100
    x = Scalar(np.random.randn(N) * 10.)
    x.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    x.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)), drank=1))
    assert 't' in x.derivs
    assert 'vec' in x.derivs
    assert hasattr(x, 'd_dt')
    assert hasattr(x, 'd_dvec')
    assert 't' in x.cos().derivs
    assert 'vec' in x.cos().derivs
    assert hasattr(x.cos(), 'd_dt')
    assert hasattr(x.cos(), 'd_dvec')
    EPS = 1.e-6
    y1 = (x + EPS).cos()
    y0 = (x - EPS).cos()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.cos().d_dt
    dy_dvec = x.cos().d_dvec
    for i in range(N):
        assert dy_dx[i] * x.d_dt[i] == dy_dt[i] or abs(dy_dx[i] * x.d_dt[i] - dy_dt[i]) <= 1.e-5

        for k in range(3):
            assert dy_dx[i] * x.d_dvec[i].values[k] == dy_dvec[i].values[k] or abs(dy_dx[i] * x.d_dvec[i].values[k] - dy_dvec[i].values[k]) <= 1.e-5

    assert x.cos(recursive=False).derivs == {}
    assert hasattr(x, 'd_dt')
    assert hasattr(x, 'd_dvec')
    assert not hasattr(x.cos(recursive=False), 'd_dt')
    assert not hasattr(x.cos(recursive=False), 'd_dvec')


def test_scalar_cos_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(8948)

    N = 10
    x = Scalar(np.random.randn(N) * 10.)
    assert not x.readonly
    assert not x.cos().readonly
    assert x.as_readonly().readonly
    assert not x.as_readonly().cos().readonly


##########################################################################################
