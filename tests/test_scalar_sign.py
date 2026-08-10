##########################################################################################
# tests/test_scalar_sign.py
##########################################################################################

import numpy as np

from polymath import Scalar, Unit


def test_scalar_sign_individual_values() -> None:
    """Individual values."""

    np.random.seed(5251)

    assert Scalar(1.25).sign() == 1.
    assert type(Scalar(1.25).sign()) == Scalar
    assert Scalar(1).sign() == np.sign(1.)
    assert Scalar(0).sign() == 0.


def test_scalar_sign_multiple_values() -> None:
    """Multiple values."""

    np.random.seed(5251)

    assert Scalar((-1,0,1)).sign() == np.sign((-1,0,1))
    assert type(Scalar((-1,0,1)).sign()) == Scalar


def test_scalar_sign_arrays() -> None:
    """Arrays."""

    np.random.seed(5251)

    N = 1000
    x = Scalar(np.random.randn(N))
    y = x.sign()
    for i in range(N):
        assert y[i] == np.sign(x.values[i])
    for i in range(N-1):
        assert y[i:i+2] == np.sign(x.values[i:i+2])


def test_scalar_sign_test_valid_unit() -> None:
    """Test valid unit."""

    np.random.seed(5251)

    values = np.random.randn(10)
    x = Scalar(values, unit=Unit.KM)
    assert x.sign() == np.sign(values)
    values = np.random.randn(10)
    x = Scalar(values, unit=Unit.SECONDS)
    assert x.sign() == np.sign(values)
    values = np.random.randn(10)
    x = Scalar(values, unit=Unit.DEG)
    assert x.sign() == np.sign(values)
    values = np.random.randn(10)
    x = Scalar(values, unit=Unit.UNITLESS)
    assert x.sign() == np.sign(values)


def test_scalar_sign_units_should_be_removed() -> None:
    """Units should be removed."""

    np.random.seed(5251)

    values = np.random.randn(10)
    x = Scalar(values, unit=Unit.CM)
    assert (x.sign().unit_ is None)


def test_scalar_sign_masks() -> None:
    """Masks."""

    np.random.seed(5251)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.sign()
    assert np.all(y.mask[x.mask])
    assert not np.any(y.mask[~x.mask])


def test_scalar_sign_derivatives_are_removed() -> None:
    """Derivatives are removed."""

    np.random.seed(5251)

    N = 100
    x = Scalar(np.random.randn(N))
    x.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    x.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)), drank=1))
    assert 't' in x.derivs
    assert 'vec' in x.derivs
    assert hasattr(x, 'd_dt')
    assert hasattr(x, 'd_dvec')
    assert 't' not in x.sign().derivs
    assert 'vec' not in x.sign().derivs
    assert not hasattr(x.sign(), 'd_dt')
    assert not hasattr(x.sign(), 'd_dvec')


def test_scalar_sign_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(5251)

    N = 10
    x = Scalar(np.random.randn(N))
    assert not x.readonly
    assert not x.sign().readonly
    assert x.as_readonly().readonly
    assert not x.as_readonly().sign().readonly


##########################################################################################
