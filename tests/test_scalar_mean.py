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


def test_scalar_mean_individual_values() -> None:
    """Individual values."""

    np.random.seed(2659)

    assert Scalar(0.3).mean() == 0.3
    assert type(Scalar(0.3).mean()) == float
    assert Scalar(4).mean() == 4
    assert type(Scalar(4).mean()) == float
    assert Scalar(4, mask=True).mean().mask
    assert type(Scalar(4, mask=True).mean()) == Scalar


def test_scalar_mean_multiple_values() -> None:
    """Multiple values."""

    np.random.seed(2659)

    assert (Scalar((1,2,3)).mean() == 2)
    assert type(Scalar((1,2,3)).mean()) == float
    assert (Scalar((1,2,3,4)).mean() == 2.5)
    assert type(Scalar((1,2,3,4)).mean()) == float
    assert (Scalar((1.,2.,3.)).mean() == 2.)
    assert type(Scalar((1.,2,3)).mean()) == float


def test_scalar_mean_arrays() -> None:
    """Arrays."""

    np.random.seed(2659)

    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    assert x.mean() == np.mean(x.values)


def test_scalar_mean_test_unit() -> None:
    """Test unit."""

    np.random.seed(2659)

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    assert random.mean().unit_ == Unit.KM
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert random.mean().unit_ == Unit.DEG
    values = np.random.randn(10)
    random = Scalar(values, unit=None)
    assert type(random.mean()) == float


def test_scalar_mean_masks() -> None:
    """Masks."""

    np.random.seed(2659)

    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    meanval = 0.
    count = 0
    for i in range(N):
        if not x.mask[i]:
            count += 1
            meanval += x.values[i]
    meanval /= count
    assert (abs((meanval - x.mean()) / meanval) < 5.e-14)
    masked = Scalar(x, mask=True)
    assert masked.mean().mask
    assert type(masked.mean())


def test_scalar_mean_means_over_axes() -> None:
    """Means over axes."""

    np.random.seed(2659)

    x = Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.mean(axis=0)
    m01 = x.mean(axis=(0,1))
    m012 = x.mean(axis=(-1,1,0))
    assert m0.is_float()
    assert m01.is_float()
    if Qube.prefer_builtins():
        assert isinstance(m012, float)
    else:
        assert m012.is_float()
    assert m0.shape == (3,5)
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.mean(x.values[:,j,k])
    assert m01.shape == (5,)
    for k in range(5):
        assert m01[k] == np.mean(x.values[:,:,k])
    assert np.shape(m012) == ()
    assert type(m012) == float
    assert m012 == np.sum(np.arange(30))/30.


def test_scalar_mean_means_with_masks() -> None:
    """Means with masks."""

    np.random.seed(2659)

    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    x = Scalar(np.arange(30).reshape(2,3,5), mask)
    m0 = x.mean(axis=0)
    m01 = x.mean(axis=(0,1))
    m012 = x.mean(axis=(-1,1,0))
    assert m0.is_float()
    assert m01.is_float()
    if Qube.prefer_builtins():
        assert isinstance(m012, float)
    else:
        assert m012.is_float()
    assert m0.shape == (3,5)
    assert m0[0,0] == x.values[1,0,0]
    assert m0[1,1] == x.values[0,1,1]
    for j in range(3):
        for k in range(5):
            if (j,k) in [(0,0), (1,1)]:
                continue
            assert m0[j,k] == np.mean(x.values[:,j,k])
    assert m01.shape == (5,)
    assert m01[0] == (np.sum(x.values[:,:,0]) - x.values[0,0,0]) / 5.
    assert m01[1] == (np.sum(x.values[:,:,1]) - x.values[1,1,1]) / 5.
    assert m01[2] == np.sum(x.values[:,:,2]) / 6.
    assert m01[3] == np.sum(x.values[:,:,3]) / 6.
    assert m01[4] == np.sum(x.values[:,:,4]) / 6.
    values = np.arange(30).reshape(2,3,5)
    mask = np.zeros((2,3,5), dtype='bool')
    mask[0,0,0] = True
    mask[1,1,1] = True
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.mean(axis=0)
    assert m0[0,0] == x.values[1,0,0]
    for j in (0,2):
        for k in range(5):
            if (j,k) in [(0,0), (1,1)]:
                continue
            assert m0[j,k] == np.mean(x.values[:,j,k])
    j = 1
    for k in range(5):
        assert m0[j,k] == Scalar.MASKED
        assert np.all(m0[j,k].values == m0.default)


##########################################################################################
