##########################################################################################
# tests/test_qube_all.py
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


def test_qube_all_individual_values() -> None:
    """Individual values."""

    np.random.seed(7456)

    assert Scalar(0.3).all() == True
    assert type(Scalar(0.3).all()) == bool
    assert Scalar(0.).all() == False
    assert type(Scalar(0.).all()) == bool
    assert Scalar(4, mask=True).all() == Boolean.MASKED
    assert type(Scalar(4, mask=True).all()) == Boolean


def test_qube_all_multiple_values() -> None:
    """Multiple values."""

    np.random.seed(7456)

    assert (Scalar((1,2,3)).all() == True)
    assert type(Scalar((1,2,3)).all()) == bool
    assert (Scalar((0., 1.,2.,3.)).all() == False)
    assert type(Scalar((0., 1.,2.,3.)).all()) == bool
    assert Scalar((1.,2.,3.), True).all() == Boolean.MASKED
    assert type(Scalar((1.,2.,3.), True).all()) == Boolean


def test_qube_all_arrays() -> None:
    """Arrays."""

    np.random.seed(7456)

    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    assert x.all() == np.all(x.values)


def test_qube_all_test_unit() -> None:
    """Test unit."""

    np.random.seed(7456)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    assert type(random.all()) == bool
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert type(random.all()) == bool
    values = np.random.randn(10)
    random = Scalar(values, mask=True, unit=None)
    assert random.all() == Boolean.MASKED
    assert random.all().units == None
    assert type(random.all()) == Boolean


def test_qube_all_test_derivs() -> None:
    """Test derivs."""

    np.random.seed(7456)

    values = np.random.randn(10)
    d_dt = Scalar(np.random.randn(10))
    random = Scalar(values)
    random.insert_deriv('t', d_dt)
    assert type(random.all()) == bool


def test_qube_all_masks() -> None:
    """Masks."""

    np.random.seed(7456)

    x = Scalar([0,1,2,3])
    assert not x.all()
    x = Scalar(x.values, mask=[True,False,False,False])
    assert x.all()
    x = Scalar(x.values, mask=[True,True,True,True])
    assert x.all() == Boolean.MASKED


def test_qube_all_all_over_axes() -> None:
    """All() over axes."""

    np.random.seed(7456)

    x = Scalar(np.arange(30).reshape(2,3,5) % 16)
    m0 = x.all(axis=0)
    m01 = x.all(axis=(0,1))
    m012 = x.all(axis=(-1,1,0))
    assert m0.shape == (3,5)
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.all(x.values[:,j,k])
    assert m01.shape == (5,)
    for k in range(5):
        assert m01[k] == np.all(x.values[:,:,k])
    assert np.shape(m012) == ()
    assert type(m012) == bool
    assert m012 == 0


def test_qube_all_maxes_with_masks() -> None:
    """Maxes with masks."""

    np.random.seed(7456)

    values = np.arange(30).reshape(2,3,5) % 16
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    x = Scalar(values, mask)
    m0 = x.all(axis=0)
    m01 = x.all(axis=(0,1))
    m012 = x.all(axis=(-1,1,0))
    assert m0.shape == (3,5)
    xx = x.values.copy()
    xx[mask] = 1
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.all(xx[:,j,k])
    assert m01.shape == (5,)
    assert m01 == [True, False, True, True, True]
    assert m012 == False
    values = np.arange(30).reshape(2,3,5) % 16
    mask = np.zeros((2,3,5), dtype='bool')
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.all(axis=0)
    for j in (0,2):
        for k in range(5):
            assert m0[j,k] == np.all(x.values[:,j,k])
    j = 1
    for k in range(5):
        assert m0[j,k] == Scalar.MASKED
        assert np.all(m0[j,k].values == np.all(x.values[:,j,k]))
    x = Scalar(values, True)
    m0 = x.all(axis=0)
    m01 = x.all(axis=(0,1))
    m012 = x.all(axis=(-1,1,0))
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == Boolean.MASKED
    for k in range(5):
        assert m01[k] == Boolean.MASKED
    assert m012 == Boolean.MASKED


def test_qube_all_tests_test_qube_tvl_all_py() -> None:
    """tests/test_qube_tvl_all.py."""

    np.random.seed(7456)

    x = Boolean([True, True, True, True])
    assert x.all() == True
    assert x.tvl_all() == True
    x = Boolean([True, True, True, True], [False, False, False, False])
    assert x.all() == True
    assert x.tvl_all() == True
    x = Boolean([True, True, True, True], [False, False, False, True])
    assert x.all() == True
    assert x.tvl_all() == Boolean.MASKED
    x = Boolean([False, True, True], [False, False, False])
    assert x.all() == False
    assert x.tvl_all() == False
    x = Boolean([False, True, True], [False, True, True])
    assert x.all() == False
    assert x.tvl_all() == False
    x = Boolean([False, True, True], [True, True, True])
    assert x.all() == Boolean.MASKED
    assert x.tvl_all() == Boolean.MASKED
    x = Boolean([False, True, True], [True, False, True])
    assert x.all() == True
    assert x.tvl_all() == Boolean.MASKED


##########################################################################################
