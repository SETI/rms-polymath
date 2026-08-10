##########################################################################################
# tests/test_vector_scalars.py
##########################################################################################

import numpy as np

from polymath import Scalar, Unit, Vector


def test_vector_scalars_check_units_and_masks() -> None:
    """check units and masks."""

    np.random.seed(4464)
    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.to_scalar(0)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,1)
    assert b.values.shape == (N,)
    assert type(b) == Scalar
    c = a.to_scalars()
    assert np.all(a.values.ravel() == c[0].values.ravel())
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Scalar

    N = 100
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5),
               unit=Unit.RAD)
    c = a.to_scalars()
    assert a.unit_ == c[0].unit_
    b = a.to_scalar(1)
    assert b == c[1]
    assert np.all(b.values == a.values[...,1])
    assert np.all(b.mask == a.mask)
    b[0] = 22.
    assert a[0].values[1] == 22.


def test_vector_scalars_check_derivatives() -> None:
    """check derivatives."""

    np.random.seed(4464)
    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.to_scalar(0)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,1)
    assert b.values.shape == (N,)
    assert type(b) == Scalar
    c = a.to_scalars()
    assert np.all(a.values.ravel() == c[0].values.ravel())
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Scalar

    N = 100
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    da_dt = Vector(np.random.randn(N,4))
    da_dv = Vector(np.random.randn(N,4,2), drank=1)
    a.insert_deriv('t', da_dt)
    a.insert_deriv('v', da_dv)
    assert hasattr(a, 'd_dt')
    assert hasattr(a, 'd_dv')
    b = a.to_scalar(3, recursive=False)
    assert not hasattr(b, 'd_dt')
    assert not hasattr(b, 'd_dv')
    b = a.to_scalar(3, recursive=True)
    assert hasattr(b, 'd_dt')
    assert hasattr(b, 'd_dv')
    assert b.d_dt.shape == a.shape
    assert b.d_dt.numer == ()
    assert b.d_dt.denom == ()
    assert b.d_dv.shape == a.shape
    assert b.d_dv.numer == ()
    assert b.d_dv.denom == (2,)
    assert np.all(a.values[...,3] == b.values)
    assert np.all(a.mask == b.mask)
    assert np.all(a.d_dt.values[...,3] == b.d_dt.values)
    assert np.all(a.d_dv.values[...,3,:] == b.d_dv.values)
    c = a.to_scalars(recursive=False)[3]
    assert not hasattr(c, 'd_dt')
    assert not hasattr(c, 'd_dv')
    c = a.to_scalars(recursive=True)[3]
    assert hasattr(c, 'd_dt')
    assert hasattr(c, 'd_dv')
    assert c.d_dt.shape == a.shape
    assert c.d_dt.numer == ()
    assert c.d_dt.denom == ()
    assert c.d_dv.shape == a.shape
    assert c.d_dv.numer == ()
    assert c.d_dv.denom == (2,)
    assert np.all(a.values[...,3] == c.values)
    assert np.all(a.mask == c.mask)
    assert np.all(a.d_dt.values[...,3] == c.d_dt.values)
    assert np.all(a.d_dv.values[...,3,:] == c.d_dv.values)


def test_vector_scalars_read_only_status() -> None:
    """read-only status."""

    np.random.seed(4464)
    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.to_scalar(0)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,1)
    assert b.values.shape == (N,)
    assert type(b) == Scalar
    c = a.to_scalars()
    assert np.all(a.values.ravel() == c[0].values.ravel())
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Scalar

    N = 10
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    assert not a.readonly
    b = a.to_scalar(3)
    assert not b.readonly
    c = a.to_scalars()[3]
    assert not c.readonly
    a = Vector(np.random.randn(N,4), mask=(np.random.randn(N) < -0.5))
    a.as_readonly()
    assert a.readonly
    b = a.to_scalar(3)
    assert b.readonly     # because of memory overlap
    c = a.to_scalars()[3]
    assert c.readonly     # because of memory overlap


def test_vector_scalars_from_scalars_args() -> None:
    """from_scalars(*args)."""

    np.random.seed(4464)
    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.to_scalar(0)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,1)
    assert b.values.shape == (N,)
    assert type(b) == Scalar
    c = a.to_scalars()
    assert np.all(a.values.ravel() == c[0].values.ravel())
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Scalar

    a = 1.
    b = Scalar((2,3,4), mask=(True,False,False))
    c = np.random.randn(4,3)
    test = Vector.from_scalars(a,b,c)
    assert np.all(test.values[...,0] == 1)
    assert np.all(test.values[...,1] == (2,3,4))
    assert np.all(test.values[...,2] == c)
    assert np.all(test.mask == [True,False,False])
    assert test.readonly == False
    b = b.as_readonly()
    c = Scalar(c).as_readonly()
    test = Vector.from_scalars(a,b,c)
    assert test.readonly == False


def test_vector_scalars_from_scalars_args_with_derivatives() -> None:
    """from_scalars(*args), with derivatives."""

    np.random.seed(4464)
    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.to_scalar(0)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,1)
    assert b.values.shape == (N,)
    assert type(b) == Scalar
    c = a.to_scalars()
    assert np.all(a.values.ravel() == c[0].values.ravel())
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Scalar

    a = 1.
    b = Scalar([2,3,4], mask=(True,False,False))
    c = np.random.randn(4,3)
    b.insert_deriv('t', Scalar([3,4,5], mask=(False,True,False)))
    test = Vector.from_scalars(a,b,c, recursive=True)
    assert np.all(test.values[...,0] == 1)
    assert np.all(test.values[...,1] == (2,3,4))
    assert np.all(test.values[...,2] == c)
    assert np.all(test.mask == [True,False,False])
    assert test.readonly == False
    assert test.d_dt.values.shape == (4,3,3)
    assert np.all(test.d_dt.values[...,0] == 0)
    assert np.all(test.d_dt.values[...,1] == (3,4,5))
    assert np.all(test.d_dt.values[...,2] == 0)


def test_vector_scalars_from_scalars_args_with_derivatives_denominators() -> None:
    """from_scalars(*args), with derivatives, denominators."""

    np.random.seed(4464)
    N = 100
    a = Vector(np.random.randn(N,1))
    b = a.to_scalar(0)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,1)
    assert b.values.shape == (N,)
    assert type(b) == Scalar
    c = a.to_scalars()
    assert np.all(a.values.ravel() == c[0].values.ravel())
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Scalar

    a = 1.
    b = Scalar((2,3,4), mask=(True,False,False))    # shape=(3,), item=()
    db_dt = Scalar(np.arange(100,112).reshape(3,2,2), drank=2,
                   mask=[False,True,False])
    b.insert_deriv('t', db_dt)
    c = Scalar(np.random.randn(4,3), mask=(np.random.rand(4,3) < 0.3))
    # shape=(4,3), item=()

    dc_dt = Scalar(np.random.randn(4,3,2,2), drank=2, mask=c.mask)
    c.insert_deriv('t', dc_dt)
    abc = Vector.from_scalars(a, b, c, recursive=True)  # shape=(4,3), item=(3,)

    assert np.all(abc.values[...,0] == 1)
    assert np.all(abc.values[...,1] == (2,3,4))
    assert np.all(abc.values[...,2] == c.values)
    assert np.all(abc.mask == (c.mask | [True,False,False]))
    assert abc.readonly == False
    assert abc.d_dt.values.shape == (4,3,3,2,2)
    assert np.all(abc.d_dt.values[...,0,:,:] == 0)
    assert (np.all(abc.d_dt.values[...,1,:,:].flatten() ==
                           4*list(range(100,112))))
    assert np.all(abc.d_dt.values[...,2,:,:] == c.d_dt.values)
    assert np.all(abc.d_dt.mask == (db_dt.mask | dc_dt.mask))


##########################################################################################
