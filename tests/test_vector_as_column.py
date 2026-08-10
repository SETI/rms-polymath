##########################################################################################
# tests/test_vector_as_column.py
##########################################################################################

import numpy as np

from polymath import Matrix, Vector, Unit


def test_vector_as_column_check_units_and_masks() -> None:
    """check units and masks."""

    np.random.seed(1684)
    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.as_column()
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,1)
    assert b.values.shape == (N,1,1)
    assert type(b) == Matrix

    N = 100
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5), unit=Unit.RAD)
    b = a.as_column()
    assert a.units == b.units
    assert np.all(b.values[...,0] == a.values)
    assert np.all(b.mask == a.mask)
    a.values[0,0] = 22.
    assert b.values[0,0,0] == 22.


def test_vector_as_column_check_derivatives() -> None:
    """check derivatives."""

    np.random.seed(1684)
    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.as_column()
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,1)
    assert b.values.shape == (N,1,1)
    assert type(b) == Matrix

    N = 100
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    da_dt = Vector(np.random.randn(N,4))
    da_dv = Vector(np.random.randn(N,4,2), drank=1)
    a.insert_deriv('t', da_dt)
    a.insert_deriv('v', da_dv)
    assert hasattr(a, 'd_dt')
    assert hasattr(a, 'd_dv')
    b = a.as_column(recursive=False)
    assert not hasattr(b, 'd_dt')
    assert not hasattr(b, 'd_dv')
    b = a.as_column(recursive=True)
    assert hasattr(b, 'd_dt')
    assert hasattr(b, 'd_dv')
    assert b.d_dt.shape == a.shape
    assert b.d_dt.numer == (4,1)
    assert b.d_dt.denom == ()
    assert b.d_dv.shape == a.shape
    assert b.d_dv.numer == (4,1)
    assert b.d_dv.denom == (2,)
    assert np.all(a.values == b.values[...,0])
    assert np.all(a.mask == b.mask)
    assert np.all(a.d_dt.values == b.d_dt.values[...,0])
    assert np.all(a.d_dv.values == b.d_dv.values[...,0,:])


def test_vector_as_column_read_only_status() -> None:
    """read-only status."""

    np.random.seed(1684)
    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.as_column()
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,1)
    assert b.values.shape == (N,1,1)
    assert type(b) == Matrix

    N = 10
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    assert not a.readonly
    b = a.as_column()
    assert not b.readonly
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    a = a.as_readonly()
    assert a.readonly
    b = a.as_column()
    assert b.readonly


##########################################################################################
