##########################################################################################
# tests/test_scalar_median.py
##########################################################################################

import numpy as np
import pytest

from polymath import Qube, Scalar, Unit


@pytest.fixture(autouse=True)
def _setup_teardown():
    """Replaces the original setUp and tearDown methods."""
    Qube.prefer_builtins(True)
    yield
    Qube.prefer_builtins(False)


def test_scalar_median_individual_values() -> None:
    """Individual values."""

    np.random.seed(9781)

    assert Scalar(0.3).median() == 0.3
    assert type(Scalar(0.3).median()) == float
    assert Scalar(4).median() == 4
    assert type(Scalar(4).median()) == float
    assert Scalar(4, mask=True).median().mask
    assert type(Scalar(4, mask=True).median()) == Scalar


def test_scalar_median_multiple_values() -> None:
    """Multiple values."""

    np.random.seed(9781)

    assert (Scalar((1,2,3)).median() == 2)
    assert type(Scalar((1,2,3)).median()) == float
    assert (Scalar((1,2,3,4)).median() == 2.5)
    assert type(Scalar((1,2,3,4)).median()) == float
    assert (Scalar((1.,2.,3.)).median() == 2.)
    assert type(Scalar((1.,2,3)).median()) == float


def test_scalar_median_arrays() -> None:
    """Arrays."""

    np.random.seed(9781)

    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    assert x.median() == np.median(x.values)


def test_scalar_median_test_unit() -> None:
    """Test unit."""

    np.random.seed(9781)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    assert random.median().unit_ == Unit.KM
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert random.median().unit_ == Unit.DEG
    values = np.random.randn(10)
    random = Scalar(values, unit=None)
    assert type(random.median()) == float


def test_scalar_median_masks() -> None:
    """Masks."""

    np.random.seed(9781)

    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    assert x.median() == np.median(x.values[~x.mask])
    masked = Scalar(x, mask=True)
    assert masked.median().mask
    assert type(masked.median())


def test_scalar_median_means_over_axes() -> None:
    """Means over axes."""

    np.random.seed(9781)

    x = Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.median(axis=0)
    m01 = x.median(axis=(0,1))
    m012 = x.median(axis=(-1,1,0))
    assert m0.is_float()
    assert m01.is_float()
    assert isinstance(m012, float)
    assert m0.shape == (3,5)
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.median(x.values[:,j,k])
    assert m01.shape == (5,)
    for k in range(5):
        assert m01[k] == np.median(x.values[:,:,k])
    assert np.shape(m012) == ()
    assert type(m012) == float
    assert m012 == np.sum(np.arange(30))/30.


def test_scalar_median_means_with_masks() -> None:
    """Means with masks."""

    np.random.seed(9781)

    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    x = Scalar(np.arange(30).reshape(2,3,5), mask)
    m0 = x.median(axis=0)
    m01 = x.median(axis=(0,1))
    m012 = x.median(axis=(-1,1,0))
    assert m0.is_float()
    assert m01.is_float()
    assert isinstance(m012, float)
    assert m0.shape == (3,5)
    assert (m0[0,0] == x.values[1,0,0])
    assert m0[1,1] == x.values[0,1,1]
    for j in range(3):
        for k in range(5):
            if (j,k) in [(0,0), (1,1)]:
                continue
            assert m0[j,k] == np.median(x.values[:,j,k])
    assert m01.shape == (5,)
    assert m01[2] == np.median(x.values[:,:,2])
    assert m01[3] == np.median(x.values[:,:,3])
    assert m01[4] == np.median(x.values[:,:,4])
    indices = (np.array([0,0,1,1,1]), np.array([1,2,0,1,2]),
                                      np.array([0,0,0,0,0]))
    assert m01[0] == np.median(x.values[indices])
    indices = (np.array([0,0,0,1,1]), np.array([0,1,2,0,2]),
                                      np.array([1,1,1,1,1]))
    assert m01[1] == np.median(x.values[indices])
    values = np.arange(30).reshape(2,3,5)
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.median(axis=0)
    assert m0[0,0] == x.values[1,0,0]
    for j in (0,2):
        for k in range(5):
            if (j,k) in [(0,0), (1,1)]:
                continue
            assert m0[j,k] == np.median(x.values[:,j,k])
    j = 1
    for k in range(5):
        assert m0[j,k] == Scalar.MASKED
        assert np.all(m0[j,k].values == np.median(x.values[:,j,k]))


##########################################################################################
