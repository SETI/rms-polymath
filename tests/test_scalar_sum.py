##########################################################################################
# tests/test_scalar_mean.py
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


def test_scalar_sum_individual_values() -> None:
    """Individual values."""

    np.random.seed(3918)

    assert Scalar(0.3).sum() == 0.3
    assert type(Scalar(0.3).sum()) == float
    assert Scalar(4).sum() == 4
    assert type(Scalar(4).sum()) == int
    assert Scalar(4, mask=True).sum().mask
    assert type(Scalar(4, mask=True).sum()) == Scalar


def test_scalar_sum_multiple_values() -> None:
    """Multiple values."""

    np.random.seed(3918)

    assert (Scalar((1,2,3)).sum() == 6)
    assert type(Scalar((1,2,3)).sum()) == int
    assert (Scalar((1.,2.,3.)).sum() == 6.)
    assert type(Scalar((1.,2,3)).sum()) == float


def test_scalar_sum_arrays() -> None:
    """Arrays."""

    np.random.seed(3918)

    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    assert x.sum() == np.sum(x.values)


def test_scalar_sum_test_unit() -> None:
    """Test unit."""

    np.random.seed(3918)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    assert random.sum().unit_ == Unit.KM
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert random.sum().unit_ == Unit.DEG
    values = np.random.randn(10)
    random = Scalar(values, unit=None)
    assert type(random.sum()) == float


def test_scalar_sum_masks() -> None:
    """Masks."""

    np.random.seed(3918)

    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    sumval = 0.
    for i in range(N):
        if not x.mask[i]:
            sumval += x.values[i]
    assert (abs((sumval - x.sum()) / sumval) < 1.e-13)
    masked = Scalar(x, mask=True)
    assert masked.sum().mask
    assert type(masked.sum())


def test_scalar_sum_denominators() -> None:
    """Denominators."""

    np.random.seed(3918)

    a = Scalar(np.arange(24.).reshape(4,3,2), drank=1)
    b = a.sum(axis=1)
    assert b.shape == (4,)
    assert b == Scalar([[6,9],[24,27],[42,45],[60,63]], drank=1)


def test_scalar_sum_sums_over_axes() -> None:
    """Sums over axes."""

    np.random.seed(3918)

    x = Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.sum(axis=0)
    m01 = x.sum(axis=(0,1))
    m012 = x.sum(axis=(-1,1,0))
    assert m0.shape == (3,5)
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.sum(x.values[:,j,k])
    assert m01.shape == (5,)
    for k in range(5):
        assert m01[k] == np.sum(x.values[:,:,k])
    assert np.shape(m012) == ()
    assert type(m012) == int
    assert m012 == np.sum(np.arange(30))


def test_scalar_sum_sums_with_masks() -> None:
    """Sums with masks."""

    np.random.seed(3918)

    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    x = Scalar(np.arange(30).reshape(2,3,5), mask)
    m0 = x.sum(axis=0)
    m01 = x.sum(axis=(0,1))
    m012 = x.sum(axis=(-1,1,0))
    assert m0.shape == (3,5)
    assert m0[0,0] == x.values[1,0,0]
    assert m0[1,1] == x.values[0,1,1]
    for j in range(3):
        for k in range(5):
            if (j,k) in [(0,0), (1,1)]:
                continue
            assert m0[j,k] == np.sum(x.values[:,j,k])
    assert m01.shape == (5,)
    assert m01[0] == np.sum(x.values[:,:,0]) - x.values[0,0,0]
    assert m01[1] == np.sum(x.values[:,:,1]) - x.values[1,1,1]
    assert m01[2] == np.sum(x.values[:,:,2])
    assert m01[3] == np.sum(x.values[:,:,3])
    assert m01[4] == np.sum(x.values[:,:,4])
    assert m012 == np.sum(x.values) - x.values[0,0,0] - x.values[1,1,1]
    values = np.arange(30).reshape(2,3,5)
    mask[0,0,0] = True
    mask[1,1,1] = True
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.sum(axis=0)
    assert m0[0,0] == x.values[1,0,0]
    for j in (0,2):
        for k in range(5):
            if (j,k) in [(0,0), (1,1)]:
                continue
            assert m0[j,k] == np.sum(x.values[:,j,k])
    j = 1
    for k in range(5):
        assert m0[j,k] == Scalar.MASKED
        assert np.all(m0[j,k].values == m0.default)


##########################################################################################
