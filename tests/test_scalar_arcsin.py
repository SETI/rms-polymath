##########################################################################################
# tests/test_scalar_arcsin.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_arcsin_individual_values() -> None:
    """Individual values."""

    np.random.seed(7221)

    assert Scalar(-0.3).arcsin() == np.arcsin(-0.3)
    assert type(Scalar(-0.3).arcsin()) == Scalar
    assert Scalar(0.).arcsin() == np.arcsin(0.)
    assert Scalar(0).arcsin() == 0.
    assert Scalar( 1.).arcsin() == np.pi/2. or abs(Scalar( 1.).arcsin() - np.pi/2.) <= 1.e-15
    assert Scalar(-1.).arcsin() == -np.pi/2. or abs(Scalar(-1.).arcsin() - -np.pi/2.) <= 1.e-15
    assert Scalar(0).arcsin() == 0.

    assert Scalar((-0.1,0.,0.1)).arcsin() == np.arcsin((-0.1,0.,0.1))
    assert type(Scalar((-0.1,0.,0.1)).arcsin()) == Scalar

    N = 1000
    x = Scalar(np.random.randn(N))
    y = x.arcsin()
    for i in range(N):
        if abs(x.values[i]) <= 1.:
            assert y[i] == np.arcsin(x.values[i])
            assert not y.mask[i]
        else:
            assert y.mask[i]
    for i in range(N-1):
        if np.all(np.abs(x.values[i:i+2]) <= 1):
            assert y[i:i+2] == np.arcsin(x.values[i:i+2])

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    with pytest.raises(ValueError):
        Scalar.arcsin(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.SECONDS)
    with pytest.raises(ValueError):
        Scalar.arcsin(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    with pytest.raises(ValueError):
        Scalar.arcsin(random)
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.RAD)
    with pytest.raises(ValueError):
        Scalar.arcsin(random)
    x = Scalar(3.25, unit=Unit.UNITLESS)
    assert x.arcsin().mask
    x = Scalar(3.25, unit=Unit.UNITLESS)
    with pytest.raises(ValueError):
        x.arcsin(recursive=True, check=False)
    x = Scalar(0.25, unit=Unit.UNITLESS)
    assert not x.arcsin().mask
    assert x.arcsin() == np.arcsin(x.values)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.UNITLESS)
    assert (random.arcsin().unit_ is None)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.arcsin()
    assert np.all(y.mask[x.mask])

    N = 100
    x = Scalar(np.random.randn(N))
    x.insert_deriv('t', Scalar(np.random.randn(N)))
    assert 't' in x.derivs
    assert hasattr(x, 'd_dt')
    assert 't' in x.arcsin().derivs
    assert hasattr(x.arcsin(), 'd_dt')
    EPS = 1.e-6
    y1 = (x + EPS).arcsin()
    y0 = (x - EPS).arcsin()
    dy_dx = 0.5 * (y1 - y0) / EPS
    dy_dt = x.arcsin().d_dt
    DEL = 3.e-6
    for i in range(N):
        if not dy_dt[i].mask and abs(dy_dt[i]) < 10:    # big errors near end points
            assert dy_dx[i] * x.d_dt[i] == dy_dt[i] or abs(dy_dx[i] * x.d_dt[i] - dy_dt[i]) <= DEL

    assert x.arcsin(recursive=False).derivs == {}
    assert hasattr(x, 'd_dt')
    assert not hasattr(x.arcsin(recursive=False), 'd_dt')

    N = 10
    x = Scalar(np.random.randn(N))
    assert not x.readonly
    assert not x.arcsin().readonly
    assert x.as_readonly().readonly
    assert not x.as_readonly().arcsin().readonly

    N = 1000
    x = Scalar(np.random.randn(N))
    with pytest.raises(ValueError):
        x.arcsin(check=False)
    x = Scalar(np.random.randn(N).clip(-1,1))
    assert x.arcsin() == np.arcsin(x.values)


##########################################################################################
