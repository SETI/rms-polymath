##########################################################################################
# tests/test_scalar_max.py
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


def test_scalar_max_individual_values() -> None:
    """Individual values."""

    np.random.seed(7250)

    assert Scalar(0.3).max() == 0.3
    assert type(Scalar(0.3).max()) == float
    assert Scalar(4).max() == 4
    assert type(Scalar(4).max()) == int
    assert Scalar(4, mask=True).max().mask
    assert type(Scalar(4, mask=True).max()) == Scalar

    assert (Scalar((1,2,3)).max() == 3)
    assert type(Scalar((1,2,3)).max()) == int
    assert (Scalar((1,2,3)).argmax() == 2)
    assert (Scalar((1.,2.,3.)).max() == 3.)
    assert type(Scalar((1.,2,3)).max()) == float
    assert (Scalar((1,2,3)).argmax() == 2)

    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    assert x.max() == np.max(x.values)
    argmax = x.argmax()
    assert x.flatten()[argmax] == x.max()

    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.KM)
    assert random.max().unit_ == Unit.KM
    values = np.random.randn(10)
    random = Scalar(values, unit=Unit.DEG)
    assert random.max().unit_ == Unit.DEG
    values = np.random.randn(10)
    random = Scalar(values, unit=None)
    assert type(random.max()) == float

    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    maxval = -np.inf
    for i in range(N):
        if (not x.mask[i]) and (x.values[i] > maxval):
            maxval = x.values[i]
    assert maxval == x.max()
    argmax = x.argmax()
    assert x[argmax] == x.max()

    x = x.mask_where_eq(maxval)
    assert (x.max() < maxval)
    argmax = x.argmax()
    assert x.flatten()[argmax] == x.max()
    masked = Scalar(x, mask=True)
    assert masked.max().mask
    assert type(masked.max())
    argmax = x.argmax()
    assert x[argmax] == x.max()

    a = Scalar([1.,2.], drank=1)
    with pytest.raises(ValueError):
        a.max()


def test_scalar_max_maxes_over_axes() -> None:
    """Maxes over axes."""

    np.random.seed(7250)

    x = -Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.max(axis=0)
    m01 = x.max(axis=(0,1))
    m012 = x.max(axis=(-1,1,0))
    assert m0.shape == (3,5)
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.max(x.values[:,j,k])
    assert m01.shape == (5,)
    for k in range(5):
        assert m01[k] == np.max(x.values[:,:,k])
    assert np.shape(m012) == ()
    assert type(m012) == int
    assert m012 == 0
    argmax = x.argmax(axis=0)
    for j in range(3):
        for k in range(5):
            assert x[argmax[j,k],j,k] == m0[j,k]


def test_scalar_max_maxes_with_masks() -> None:
    """Maxes with masks."""

    np.random.seed(7250)

    values = -np.arange(30).reshape(2,3,5)
    mask = (values > -5)
    x = Scalar(values, mask)
    m0 = x.max(axis=0)
    m01 = x.max(axis=(0,1))
    m012 = x.max(axis=(-1,1,0))
    assert m0.shape == (3,5)
    xx = x.values.copy()
    xx[xx > -5] -= 100
    for j in range(3):
        for k in range(5):
            assert m0[j,k] == np.max(xx[:,j,k])
    assert m01.shape == (5,)
    assert m01 == [-5,-6,-7,-8,-9]
    assert m012 == -5
    argmax = x.argmax(axis=0)
    for j in range(3):
        for k in range(5):
            assert x[argmax[j,k],j,k] == m0[j,k]
    values = -np.arange(30).reshape(2,3,5)
    mask = (values > -5)
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.max(axis=0)
    for j in (0,2):
        for k in range(5):
            assert m0[j,k] == np.max(xx[:,j,k])
    j = 1
    for k in range(5):
        assert m0[j,k] == Scalar.MASKED
        assert np.all(m0[j,k].values == np.max(x.values[:,j,k]))


##########################################################################################
