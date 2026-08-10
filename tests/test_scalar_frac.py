##########################################################################################
# tests/test_scalar_frac.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_frac_individual_values() -> None:
    """Individual values."""

    np.random.seed(4984)

    assert Scalar( 1.25).frac() == 0.25
    assert Scalar(-1.25).frac() == 0.75
    assert Scalar( 1).frac() == 0.
    assert Scalar(-1).frac() == 0.

    assert Scalar((1.25, -1.25)).frac() == (0.25, 0.75)
    assert Scalar((1.25, -1.25)).frac().is_float()
    assert Scalar((1, -1)).frac() == (0.,0.)
    assert Scalar((1.2, -1.2)).frac().is_float()

    N = 1000
    values = np.random.randn(N) * 10.
    random = Scalar(values)
    frandom = random.frac()
    for i in range(N):
        assert frandom[i] == values[i] % 1.
    for i in range(N-1):
        assert random[i:i+2].frac() == values[i:i+2] % 1.

    values = np.random.randn(10) * 10.
    random = Scalar(values, unit=Unit.KM)
    with pytest.raises(ValueError):
        Scalar.frac(random)
    random = Scalar(values, unit=Unit.DEG)
    with pytest.raises(ValueError):
        Scalar.frac(random)
    random = Scalar(3.25, unit=Unit.UNITLESS)
    assert random.frac() == 0.25


def test_scalar_frac_masks() -> None:
    """Masks."""

    np.random.seed(4984)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.frac()
    assert np.all(y.mask[x.mask])
    assert not np.any(y.mask[~x.mask])


def test_scalar_frac_derivatives_should_be_preserved() -> None:
    """Derivatives should be preserved."""

    np.random.seed(4984)

    N = 10
    random = Scalar(np.random.randn(N) * 10.)
    random.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    random.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)),
                                       drank=1))
    assert 't' in random.derivs
    assert 'vec' in random.derivs
    assert hasattr(random, 'd_dt')
    assert hasattr(random, 'd_dvec')
    assert random.frac().derivs == random.derivs
    assert 't' in random.frac().derivs
    assert 'vec' in random.frac().derivs
    assert hasattr(random.frac(), 'd_dt')
    assert hasattr(random.frac(), 'd_dvec')
    N = 10
    random = Scalar(np.arange(10))
    random.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    random.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape(N,3),
                                       drank=1))
    assert 't' in random.derivs
    assert 'vec' in random.derivs
    assert hasattr(random, 'd_dt')
    assert hasattr(random, 'd_dvec')
    assert random.frac().derivs == random.derivs
    assert 't' in random.frac().derivs
    assert 'vec' in random.frac().derivs
    assert hasattr(random.frac(), 'd_dt')
    assert hasattr(random.frac(), 'd_dvec')


def test_scalar_frac_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(4984)

    N = 10
    random = Scalar(np.random.randn(N) * 10.)
    assert not random.readonly
    assert not random.frac().readonly
    assert random.as_readonly().readonly
    assert not random.as_readonly().frac().readonly


##########################################################################################
