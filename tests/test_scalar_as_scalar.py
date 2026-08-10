##########################################################################################
# tests/test_scalar_as_scalar.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Vector, Boolean, Unit


def test_scalar_as_scalar_units_case() -> None:
    """Units case."""

    np.random.seed(3560)
    N = 10
    a = Scalar(np.random.randn(N))
    da_dt = Scalar(np.random.randn(N,6), drank=1)
    a.insert_deriv('t', da_dt)
    b = Scalar.as_scalar(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Unit.CM
    b = Scalar.as_scalar(a)
    assert type(b)
    assert b.units == Unit.CM
    assert b.shape == ()
    assert b.numer == ()
    assert b.values == 1.e-5

    a = Vector(np.random.randn(N,3))
    with pytest.raises(ValueError):
        Scalar.as_scalar(a)


def test_scalar_as_scalar_boolean_case() -> None:
    """Boolean case."""

    np.random.seed(3560)
    N = 10
    a = Scalar(np.random.randn(N))
    da_dt = Scalar(np.random.randn(N,6), drank=1)
    a.insert_deriv('t', da_dt)
    b = Scalar.as_scalar(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Boolean(np.random.randn(N) < 0.)
    b = Scalar.as_scalar(a)
    assert type(b)
    assert b.units == None
    assert b.shape == (N,)
    assert b.numer == ()
    assert b == a
    b = Scalar.as_scalar(Boolean(True))
    assert type(b)
    assert b.units == None
    assert b.shape == ()
    assert b.numer == ()
    assert b.values == 1


def test_scalar_as_scalar_other_cases() -> None:
    """Other cases."""

    np.random.seed(3560)
    N = 10
    a = Scalar(np.random.randn(N))
    da_dt = Scalar(np.random.randn(N,6), drank=1)
    a.insert_deriv('t', da_dt)
    b = Scalar.as_scalar(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    b = Scalar.as_scalar(3.14159)
    assert type(b)
    assert (b.units is None)
    assert b.shape == ()
    assert b.numer == ()
    assert b == 3.14159
    a = np.arange(120).reshape((2,4,3,5))
    b = Scalar.as_scalar(a)
    assert type(b)
    assert (b.units is None)
    assert b.shape == (2,4,3,5)
    assert b.numer == ()
    assert b == a


##########################################################################################
