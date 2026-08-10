##########################################################################################
# tests/test_scalar_tan.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_tan_individual_values() -> None:
    """Individual values."""

    np.random.seed(9359)

    assert Scalar(1.25).tan() == np.tan(1.25)
    assert type(Scalar(1.25).tan()) == Scalar
    assert Scalar(1).tan() == np.tan(1.)
    assert Scalar(0).tan() == 0.

    assert Scalar((-1,0,1)).tan() == np.tan((-1,0,1))
    assert type(Scalar((-1,0,1)).tan()) == Scalar

    N = 1000
    values = np.random.randn(N) * 10.
    angles = Scalar(values)
    for i in range(N):
        assert angles.tan()[i] == np.tan(values[i])
    for i in range(N-1):
        assert angles.tan()[i:i+2] == np.tan(values[i:i+2])

    values = np.random.randn(10) * 10.
    random = Scalar(values, unit=Unit.KM)
    with pytest.raises(ValueError):
        Scalar.tan(random)
    values = np.random.randn(10) * 10.
    random = Scalar(values, unit=Unit.SECONDS)
    with pytest.raises(ValueError):
        Scalar.tan(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert random.tan() == random.tan()        # unit should be OK
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.RAD)
    assert random.tan() == random.tan()        # unit should be OK
    angle = Scalar(3.25, unit=Unit.UNITLESS)
    assert angle.tan() == np.tan(angle.values) # unit should be OK


def test_scalar_tan_units_should_be_removed() -> None:
    """Units should be removed."""

    np.random.seed(9359)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert (random.tan().unit_ is None)


def test_scalar_tan_masks() -> None:
    """Masks."""

    np.random.seed(9359)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.tan()
    assert np.all(y.mask[x.mask])
    assert not np.any(y.mask[~x.mask])


def test_scalar_tan_derivatives() -> None:
    """Derivatives."""

    np.random.seed(9359)

    N = 100
    x = Scalar(np.random.randn(N) * 10.)
    x.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    x.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)), drank=1))
    assert 't' in x.derivs
    assert 'vec' in x.derivs
    assert hasattr(x, 'd_dt')
    assert hasattr(x, 'd_dvec')
    assert 't' in x.tan().derivs
    assert 'vec' in x.tan().derivs
    assert hasattr(x.tan(), 'd_dt')
    assert hasattr(x.tan(), 'd_dvec')
    EPS = 1.e-6
    y1 = (x + EPS).tan()
    y0 = (x - EPS).tan()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.tan().d_dt
    dy_dvec = x.tan().d_dvec
    DEL = 5.e-5
    for i in range(N):
        assert dy_dx[i] * x.d_dt[i] == dy_dt[i] or abs(dy_dx[i] * x.d_dt[i] - dy_dt[i]) <= DEL * abs(dy_dt[i])

        for k in range(3):
            assert dy_dx[i] * x.d_dvec[i].values[k] == dy_dvec[i].values[k] or abs(dy_dx[i] * x.d_dvec[i].values[k] - dy_dvec[i].values[k]) <= DEL * abs(dy_dvec[i].values[k])


def test_scalar_tan_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(9359)

    N = 10
    x = Scalar(np.random.randn(N) * 10.)
    assert not x.readonly
    assert not x.tan().readonly
    assert x.as_readonly().readonly
    assert not x.as_readonly().tan().readonly


##########################################################################################
