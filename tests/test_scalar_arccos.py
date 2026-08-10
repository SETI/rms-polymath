##########################################################################################
# tests/test_scalar_arccos.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_arccos_individual_values() -> None:
    """Individual values."""

    np.random.seed(8994)

    assert Scalar(-0.3).arccos() == np.arccos(-0.3)
    assert type(Scalar(-0.3).arccos()) == Scalar
    assert Scalar(0.).arccos() == np.arccos(0.)
    assert Scalar(1).arccos() == 0.
    assert Scalar( 1.).arccos() == 0. or abs(Scalar( 1.).arccos() - 0.) <= 1.e-15
    assert Scalar(-1.).arccos() == np.pi or abs(Scalar(-1.).arccos() - np.pi) <= 1.e-15
    assert Scalar( 0.).arccos() == np.pi/2. or abs(Scalar( 0.).arccos() - np.pi/2.) <= 1.e-15

    assert Scalar((-0.1,0.,0.1)).arccos() == np.arccos((-0.1,0.,0.1))
    assert type(Scalar((-0.1,0.,0.1)).arccos()) == Scalar

    N = 1000
    x = Scalar(np.random.randn(N))
    y = x.arccos()
    for i in range(N):
        if abs(x.values[i]) <= 1.:
            assert y[i] == np.arccos(x.values[i])
            assert not y.mask[i]
        else:
            assert y.mask[i]
    for i in range(N-1):
        if np.all(np.abs(x.values[i:i+2]) <= 1):
            assert y[i:i+2] == np.arccos(x.values[i:i+2])

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    with pytest.raises(ValueError):
        Scalar.arccos(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.SECONDS)
    with pytest.raises(ValueError):
        Scalar.arccos(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    with pytest.raises(ValueError):
        Scalar.arccos(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.RAD)
    with pytest.raises(ValueError):
        Scalar.arccos(random)
    x = Scalar(3.25, unit=Unit.UNITLESS)
    assert x.arccos().mask
    x = Scalar(3.25, unit=Unit.UNITLESS)
    with pytest.raises(ValueError):
        x.arccos(recursive=True, check=False)
    x = Scalar(0.25, unit=Unit.UNITLESS)
    assert not x.arccos().mask
    assert x.arccos() == np.arccos(x.values)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.UNITLESS)
    assert (random.arccos().unit_ is None)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.arccos()
    assert np.all(y.mask[x.mask])

    N = 100
    x = Scalar(np.random.randn(N))
    x.insert_deriv('t', Scalar(np.random.randn(N)))
    assert 't' in x.derivs
    assert hasattr(x, 'd_dt')
    assert 't' in x.arccos().derivs
    assert hasattr(x.arccos(), 'd_dt')
    EPS = 1.e-6
    y1 = (x + EPS).arccos()
    y0 = (x - EPS).arccos()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.arccos().d_dt
    DEL = 5.e-6
    for i in range(N):
        if not dy_dt[i].mask and abs(dy_dt[i]) < 10:  # big errors near end points
            assert dy_dx[i] * x.d_dt[i] == dy_dt[i] or abs(dy_dx[i] * x.d_dt[i] - dy_dt[i]) <= DEL

    assert x.arccos(recursive=False).derivs == {}
    assert hasattr(x, 'd_dt')
    assert not hasattr(x.arccos(recursive=False), 'd_dt')

    N = 10
    x = Scalar(np.random.randn(N))
    assert not x.readonly
    assert not x.arccos().readonly
    assert x.as_readonly().readonly
    assert not x.as_readonly().arccos().readonly

    N = 1000
    x = Scalar(np.random.randn(N))
    with pytest.raises(ValueError):
        x.arccos(check=False)
    x = Scalar(np.random.randn(N).clip(-1,1))
    assert x.arccos() == np.arccos(x.values)


##########################################################################################
