##########################################################################################
# tests/test_qube_any.py
##########################################################################################

import numpy as np
import pytest

from polymath import Qube, Scalar, Boolean, Unit


@pytest.fixture(autouse=True)
def _setup_teardown():
    """Replaces the original setUp and tearDown methods."""
    Qube.prefer_builtins(True)
    yield
    Qube.prefer_builtins(False)


def test_qube_any_individual_values() -> None:
    """Individual values."""

    np.random.seed(3337)

    assert Scalar(0.3).any() == True
    assert type(Scalar(0.3).any()) == bool
    assert Scalar(0.).any() == False
    assert type(Scalar(0.).any()) == bool
    assert Scalar(4, mask=True).any() == Boolean.MASKED
    assert type(Scalar(4, mask=True).any()) == Boolean


def test_qube_any_multiple_values() -> None:
    """Multiple values."""

    np.random.seed(3337)

    assert (Scalar((0,0,1)).any() == True)
    assert type(Scalar((0,0,1)).any()) == bool
    assert Scalar((1.,2.,3.), True).any() == Boolean.MASKED
    assert type(Scalar((1.,2.,3.), True).any()) == Boolean


def test_qube_any_arrays() -> None:
    """Arrays."""

    np.random.seed(3337)

    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    assert x.any() == np.any(x.values)


def test_qube_any_test_unit() -> None:
    """Test unit."""

    np.random.seed(3337)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    assert type(random.any()) == bool
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert type(random.any()) == bool
    values = np.random.randn(10)
    random = Scalar(values, mask=True, unit=None)
    assert random.any() == Boolean.MASKED
    assert random.any().units == None
    assert type(random.any()) == Boolean


def test_qube_any_test_derivs() -> None:
    """Test derivs."""

    np.random.seed(3337)

    values = np.random.randn(10)
    d_dt = Scalar(np.random.randn(10))
    random = Scalar(values)
    random.insert_deriv('t', d_dt)
    assert type(random.any()) == bool


def test_qube_any_masks() -> None:
    """Masks."""

    np.random.seed(3337)

    x = Scalar([0,1,2,3])
    assert x.any()
    x = Scalar(x.values, mask=[False,True,True,True])
    assert not x.any()
    x = Scalar(x.values, mask=[True,True,True,True])
    assert x.any() == Boolean.MASKED


def test_qube_any_any_over_axes() -> None:
    """Any() over axes."""

    np.random.seed(3337)

    values = np.zeros(30).reshape(2,3,5) % 16
    values[0,0,0] = 1
    values[1,1,1] = 1
    x = Scalar(values)
    m0 = x.any(axis=0)
    m01 = x.any(axis=(0,1))
    m012 = x.any(axis=(-1,1,0))
    assert m0.shape == (3,5)
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.any(x.values[:,j,k])
    assert m01.shape == (5,)
    for k in range(5):
        assert m01[k] == np.any(x.values[:,:,k])
    assert np.shape(m012) == ()
    assert type(m012) == bool
    assert m012 == True

    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    x = Scalar(values, mask)
    m0 = x.any(axis=0)
    m01 = x.any(axis=(0,1))
    m012 = x.any(axis=(-1,1,0))
    assert m0.shape == (3,5)
    xx = x.values.copy()
    xx[mask] = False
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.any(xx[:,j,k])
    assert m01.shape == (5,)
    assert m01 == [False, True, False, False, False]
    assert m012 == True
    mask[:,0] = True
    x = Scalar(values, mask)
    m0 = x.any(axis=0)
    m01 = x.any(axis=(0,1))
    m012 = x.any(axis=(-1,1,0))
    for j in (1,2):
        for k in range(5):
            assert m0[j,k] == np.any(x.values[:,j,k])
    j = 0
    for k in range(5):
        assert m0[j,k] == Scalar.MASKED
#         self.assertTrue(np.any(m0[j,k].values == np.any(x.values[:,j,k])))
# Changed 3/14. No need to set values where masked
    x = Scalar(values, True)
    m0 = x.any(axis=0)
    m01 = x.any(axis=(0,1))
    m012 = x.any(axis=(-1,1,0))
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == Boolean.MASKED
    for k in range(5):
        assert m01[k] == Boolean.MASKED
    assert m012 == Boolean.MASKED


def test_qube_any_tests_test_qube_tvl_any_py() -> None:
    """tests/test_qube_tvl_any.py."""

    np.random.seed(3337)

    x = Boolean([True, True, True, True])
    assert x.any() == True
    assert x.tvl_any() == True
    x = Boolean([False, False, False, False], [False, False, False, False])
    assert x.any() == False
    assert x.tvl_any() == False
    x = Boolean([False, False, False, True], [False, False, False, False])
    assert x.any() == True
    assert x.tvl_any() == True
    x = Boolean([False, False, False, True], [False, False, False, True])
    assert x.any() == False
    assert x.tvl_any() == Boolean.MASKED
    x = Boolean([True, False, False, True], [False, False, False, True])
    assert x.any() == True
    assert x.tvl_any() == True
    x = Boolean([False, True, True], True)
    assert x.any() == Boolean.MASKED
    assert x.tvl_any() == Boolean.MASKED
    x = Boolean([False, True, True], [True, True, True])
    assert x.any() == Boolean.MASKED
    assert x.tvl_any() == Boolean.MASKED


##########################################################################################
