##########################################################################################
# tests/test_pair_as_pair.py
##########################################################################################

import numpy as np

from polymath import Matrix, Pair, Unit


def test_pair_as_pair_matrix_case_2x1() -> None:
    """Matrix case, 2x1."""

    np.random.seed(2046)
    N = 10
    a = Pair(np.random.randn(N,2))
    da_dt = Pair(np.random.randn(N,2))
    a.insert_deriv('t', da_dt)
    b = Pair.as_pair(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Matrix(np.random.randn(N,2,1), unit=Unit.REV)
    da_dt = Matrix(np.random.randn(N,2,1,6), drank=1)
    a.insert_deriv('t', da_dt)
    b = Pair.as_pair(a)
    assert type(b)
    assert a.units == b.units
    assert a.shape == b.shape
    assert a.numer == (2,1)
    assert b.numer == (2,)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == b.shape
    assert b.d_dt.numer == (2,)
    assert b.d_dt.denom == (6,)
    assert np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel())
    b = Pair.as_pair(a, recursive=False)
    assert not hasattr(b, 'd_dt')


def test_pair_as_pair_matrix_case_1x2() -> None:
    """Matrix case, 1x2."""

    np.random.seed(2046)
    N = 10
    a = Pair(np.random.randn(N,2))
    da_dt = Pair(np.random.randn(N,2))
    a.insert_deriv('t', da_dt)
    b = Pair.as_pair(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Matrix(np.random.randn(N,1,2), unit=Unit.REV)
    da_dt = Matrix(np.random.randn(N,1,2,6), drank=1)
    a.insert_deriv('t', da_dt)
    b = Pair.as_pair(a)
    assert type(b)
    assert a.units == b.units
    assert a.shape == b.shape
    assert a.numer == (1,2)
    assert b.numer == (2,)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == b.shape
    assert b.d_dt.numer == (2,)
    assert b.d_dt.denom == (6,)
    assert np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel())
    b = Pair.as_pair(a, recursive=False)
    assert not hasattr(b, 'd_dt')


def test_pair_as_pair_other_cases() -> None:
    """Other cases."""

    np.random.seed(2046)
    N = 10
    a = Pair(np.random.randn(N,2))
    da_dt = Pair(np.random.randn(N,2))
    a.insert_deriv('t', da_dt)
    b = Pair.as_pair(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    b = Pair.as_pair((1,2))
    assert type(b)
    assert (b.units is None)
    assert b.shape == ()
    assert b.numer == (2,)
    assert b == (1,2)
    a = np.arange(120).reshape((5,4,3,2))
    b = Pair.as_pair(a)
    assert type(b)
    assert (b.units is None)
    assert b.shape == (5,4,3)
    assert b.numer == (2,)
    assert b == a


##########################################################################################
