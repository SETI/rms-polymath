##########################################################################################
# tests/test_scalar_min.py
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


def test_scalar_min_individual_values() -> None:
    """Individual values."""

    np.random.seed(2956)

    assert Scalar(0.3).min() == 0.3
    assert type(Scalar(0.3).min()) == float
    assert Scalar(4).min() == 4
    assert type(Scalar(4).min()) == int
    assert Scalar(4, mask=True).min().mask
    assert type(Scalar(4, mask=True).min()) == Scalar

    assert (Scalar((1,2,3)).min() == 1)
    assert type(Scalar((1,2,3)).min()) == int
    assert (Scalar((1.,2.,3.)).min() == 1.)
    assert type(Scalar((1.,2,3)).min()) == float

    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    assert x.min() == np.min(x.values)
    argmin = x.argmin()
    assert x.flatten()[argmin] == x.min()

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    assert random.min().unit_ == Unit.KM
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert random.min().unit_ == Unit.DEG
    values = np.random.randn(10)
    random = Scalar(values, unit=None)
    assert type(random.min()) == float

    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    minval = np.inf
    for i in range(N):
        if (not x.mask[i]) and (x.values[i] < minval):
            minval = x.values[i]
    assert minval == x.min()
    argmin = x.argmin()
    assert x[argmin] == x.min()

    x = x.mask_where_eq(minval)
    assert (x.min() > minval)
    argmin = x.argmin()
    assert x.flatten()[argmin] == x.min()
    masked = Scalar(x, mask=True)
    assert masked.min().mask
    assert type(masked.min())
    argmin = x.argmin()
    assert x[argmin] == x.min()

    a = Scalar([1.,2.], drank=1)
    with pytest.raises(ValueError):
        a.min()


def test_scalar_min_mins_over_axes() -> None:
    """Mins over axes."""

    np.random.seed(2956)

    x = Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.min(axis=0)
    m01 = x.min(axis=(0,1))
    m012 = x.min(axis=(-1,1,0))
    assert m0.shape == (3,5)
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.min(x.values[:,j,k])
    assert m01.shape == (5,)
    for k in range(5):
        assert m01[k] == np.min(x.values[:,:,k])
    assert np.shape(m012) == ()
    assert type(m012) == int
    assert m012 == 0
    argmin = x.argmin(axis=0)
    for j in range(3):
        for k in range(5):
            assert x[argmin[j,k],j,k] == m0[j,k]


def test_scalar_min_mins_with_masks() -> None:
    """Mins with masks."""

    np.random.seed(2956)

    values = np.arange(30).reshape(2,3,5)
    mask = (values < 5)
    x = Scalar(values, mask)
    m0 = x.min(axis=0)
    m01 = x.min(axis=(0,1))
    m012 = x.min(axis=(-1,1,0))
    assert m0.shape == (3,5)
    xx = x.values.copy()
    xx[xx < 5] += 100
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.min(xx[:,j,k])
    assert m01.shape == (5,)
    assert m01 == [5,6,7,8,9]
    assert m012 == 5
    argmin = x.argmin(axis=0)
    for j in range(3):
        for k in range(5):
            assert x[argmin[j,k],j,k] == m0[j,k]
    values = np.arange(30).reshape(2,3,5)
    mask = (values < 5)
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.min(axis=0)
    for j in (0,2):
        for k in range(5):
            assert m0[j,k] == np.min(xx[:,j,k])
    j = 1
    for k in range(5):
        assert m0[j,k] == Scalar.MASKED
        assert np.all(m0[j,k].values == np.min(x.values[:,j,k]))


##########################################################################################
