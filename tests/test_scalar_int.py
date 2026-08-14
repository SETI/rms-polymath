##########################################################################################
# tests/test_scalar_int.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_int_individual_values() -> None:
    """Individual values."""

    np.random.seed(4353)

    assert Scalar( 1.2).int() == 1
    assert Scalar(-1.2).int() == -2
    assert Scalar( 1).int() == 1
    assert Scalar(-1).int() == -1
    assert Scalar(1.2,True).int() == Scalar(0.).masked_single()
    assert Scalar(1,  True).int() == Scalar(0.).masked_single()

    assert Scalar((1.2, -1.2)).int() == (1,-2)
    assert not Scalar((1.2, -1.2)).int().is_float()
    assert Scalar((1, -1)).int() == (1,-1)
    assert not Scalar((1.2, -1.2)).int().is_float()

    N = 1000
    values = np.random.randn(N) * 10.
    random = Scalar(values)
    irandom = random.int()
    for i in range(N):
        assert irandom[i] == int(np.floor(values[i]))
    for i in range(N-1):
        assert random[i:i+2].int() == np.floor(values[i:i+2])

    values = np.random.randn(10) * 10.
    random = Scalar(values, unit=Unit.KM)
    with pytest.raises(ValueError):
        Scalar.int(random)
    random = Scalar(values, unit=Unit.DEG)
    with pytest.raises(ValueError):
        Scalar.int(random)
    random = Scalar(3.14, unit=Unit.UNITLESS)
    assert random.int() == 3


def test_scalar_int_denominators_are_disallowed() -> None:
    """Denominators are disallowed."""

    a = Scalar([[1., 2.], [3., 4.]], drank=1)
    with pytest.raises(ValueError, match='does not support denominators'):
        a.int()


def test_scalar_int_shift_of_a_shapeless_value() -> None:
    """A shapeless value equal to top shifts down by one; a smaller value does not."""

    assert Scalar(3.).int(3, shift=True) == 2
    assert Scalar(3).int(3, shift=True) == 2
    assert Scalar(2.9).int(3, shift=True) == 2
    assert Scalar(0.5).int(3, shift=True) == 0


def test_scalar_int_clip_without_a_top_value() -> None:
    """Without a top value, clip replaces negative values by zero."""

    a = Scalar([-3.5, 2.5]).int(clip=True)
    assert a == (0, 2)
    assert a.mask is False


def test_scalar_int_clip_and_remask_without_a_top_value() -> None:
    """Without a top value, clip and remask together replace and mask negative values."""

    a = Scalar([-3.5, 2.5]).int(clip=True, remask=True)
    assert a.values[0] == 0
    assert a.values[1] == 2
    assert np.all(a.mask == [True, False])


def test_scalar_int_masks() -> None:
    """Masks."""

    np.random.seed(4353)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    y = x.int()
    assert np.all(y.mask[x.mask])
    assert not np.any(y.mask[~x.mask])


def test_scalar_int_derivatives_should_be_stripped() -> None:
    """Derivatives should be stripped."""

    np.random.seed(4353)

    N = 10
    random = Scalar(np.random.randn(N) * 10.)
    random.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    random.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape(N,3),
                                       drank=1))
    assert 't' in random.derivs
    assert 'vec' in random.derivs
    assert hasattr(random, 'd_dt')
    assert hasattr(random, 'd_dvec')
    assert random.int().derivs == {}
    assert 't' not in random.int().derivs
    assert 'vec' not in random.int().derivs
    assert not hasattr(random.int(), 'd_dt')
    assert not hasattr(random.int(), 'd_dvec')
    N = 10
    random = Scalar(np.arange(10))
    random.insert_deriv('t', Scalar(np.random.randn(N) * 10.))
    random.insert_deriv('vec', Scalar(np.random.randn(3*N).reshape((N,3)),
                                       drank=1))
    assert 't' in random.derivs
    assert 'vec' in random.derivs
    assert hasattr(random, 'd_dt')
    assert hasattr(random, 'd_dvec')
    assert random.int().derivs == {}
    assert 't' not in random.int().derivs
    assert 'vec' not in random.int().derivs
    assert not hasattr(random.int(), 'd_dt')
    assert not hasattr(random.int(), 'd_dvec')


def test_scalar_int_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(4353)

    N = 10
    random = Scalar(np.random.randn(N) * 10.)
    assert not random.readonly
    assert not random.int().readonly
    assert random.as_readonly().readonly
    assert not random.as_readonly().int().readonly


def test_scalar_int_but_int_objects_are_returned_as_is() -> None:
    """But int objects are returned as is."""

    np.random.seed(4353)

    a = Scalar(np.arange(10)).as_readonly()
    assert a.readonly
    assert a.int().readonly


##########################################################################################
