##########################################################################################
# tests/test_vector_as_vector.py
##########################################################################################

import numpy as np

from polymath import Matrix, Pair, Scalar, Unit, Vector


def test_vector_as_vector_matrix_case_nx1() -> None:
    """Matrix case, Nx1."""

    np.random.seed(4469)
    N = 10
    a = Vector(np.random.randn(N,6))
    da_dt = Vector(np.random.randn(N,6))
    a.insert_deriv('t', da_dt)
    b = Vector.as_vector(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Matrix(np.random.randn(N,7,1), unit=Unit.REV)
    da_dt = Matrix(np.random.randn(N,7,1,6), drank=1)
    a.insert_deriv('t', da_dt)
    b = Vector.as_vector(a)
    assert type(b)
    assert a.unit_ == b.unit_
    assert a.shape == b.shape
    assert a.numer == (7,1)
    assert b.numer == (7,)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == b.shape
    assert b.d_dt.numer == (7,)
    assert b.d_dt.denom == (6,)
    assert np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel())
    b = Vector.as_vector(a, recursive=False)
    assert not hasattr(b, 'd_dt')


def test_vector_as_vector_matrix_case_1xn() -> None:
    """Matrix case, 1xN."""

    np.random.seed(4469)
    N = 10
    a = Vector(np.random.randn(N,6))
    da_dt = Vector(np.random.randn(N,6))
    a.insert_deriv('t', da_dt)
    b = Vector.as_vector(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Matrix(np.random.randn(N,1,7), unit=Unit.REV)
    da_dt = Matrix(np.random.randn(N,1,7,6), drank=1)
    a.insert_deriv('t', da_dt)
    b = Vector.as_vector(a)
    assert type(b)
    assert a.unit_ == b.unit_
    assert a.shape == b.shape
    assert a.numer == (1,7)
    assert b.numer == (7,)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == b.shape
    assert b.d_dt.numer == (7,)
    assert b.d_dt.denom == (6,)
    assert np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel())
    b = Vector.as_vector(a, recursive=False)
    assert not hasattr(b, 'd_dt')


def test_vector_as_vector_scalar_case() -> None:
    """Scalar case."""

    np.random.seed(4469)
    N = 10
    a = Vector(np.random.randn(N,6))
    da_dt = Vector(np.random.randn(N,6))
    a.insert_deriv('t', da_dt)
    b = Vector.as_vector(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Scalar(np.random.randn(N), unit=Unit.UNITLESS)
    da_dt = Scalar(np.random.randn(N,6), drank=1)
    a.insert_deriv('t', da_dt)
    b = Vector.as_vector(a)
    assert type(b)
    assert a.unit_ == b.unit_
    assert a.shape == b.shape
    assert a.numer == ()
    assert b.numer == (1,)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == b.shape
    assert b.d_dt.numer == (1,)
    assert np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel())
    b = Vector.as_vector(a, recursive=False)
    assert not hasattr(b, 'd_dt')
    a = Scalar(7.)
    b = Vector.as_vector(a)
    assert b._values == 7.
    assert b._numer == (1,)
    a = Scalar(np.arange(60).reshape(20,3), drank=1)
    b = Vector.as_vector(a)
    assert np.all(b.vals[:,0,:] == a.vals)
    assert b.shape == (20,)
    assert b.item == (1,3)


def test_vector_as_vector_pair_case() -> None:
    """Pair case."""

    np.random.seed(4469)
    N = 10
    a = Vector(np.random.randn(N,6))
    da_dt = Vector(np.random.randn(N,6))
    a.insert_deriv('t', da_dt)
    b = Vector.as_vector(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Pair(np.random.randn(N,2), unit=Unit.DEG)
    da_dt = Pair(np.random.randn(N,2,6), drank=1)
    a.insert_deriv('t', da_dt)
    b = Vector.as_vector(a)
    assert type(b)
    assert a.unit_ == b.unit_
    assert a.shape == b.shape
    assert a.numer == b.numer
    assert np.all(a.values.ravel() == b.values.ravel())
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == b.shape
    assert b.d_dt.numer == a.numer
    assert np.all(a.d_dt.values.ravel() == b.d_dt.values.ravel())
    b = Vector.as_vector(a, recursive=False)
    assert not hasattr(b, 'd_dt')


def test_vector_as_vector_other_cases() -> None:
    """Other cases."""

    np.random.seed(4469)
    N = 10
    a = Vector(np.random.randn(N,6))
    da_dt = Vector(np.random.randn(N,6))
    a.insert_deriv('t', da_dt)
    b = Vector.as_vector(a, recursive=False)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    b = Vector.as_vector((1,2,3))
    assert type(b)
    assert (b.unit_ is None)
    assert b.shape == ()
    assert b.numer == (3,)
    assert b == (1,2,3)
    a = np.arange(120).reshape((2,4,3,5))
    b = Vector.as_vector(a)
    assert type(b)
    assert (b.unit_ is None)
    assert b.shape == (2,4,3)
    assert b.numer == (5,)
    assert b == a


##########################################################################################
