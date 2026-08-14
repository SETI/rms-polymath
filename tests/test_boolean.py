##########################################################################################
# test/test_boolean.py
##########################################################################################

import numbers
import numpy as np
import pytest

from polymath import Boolean, Scalar, Unit


def test_boolean_zeros() -> None:
    """zeros."""

    np.random.seed(7768)

    ##################################################################################
    # Constructor
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(a)
    assert a == b
    b = Boolean(a)
    assert a == b
    a = np.array([True,False])
    b = Boolean(a[0])
    assert b
    assert isinstance(b.vals, bool)
    a = np.array(True)      # shapeless array
    b = Boolean(a)
    assert b
    assert b.vals
    assert str(b) == 'Boolean(True)'
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    assert a[~mask] == values[~mask]
    assert np.all(a.as_mask_where_nonzero() == a.values & ~mask)
    assert np.all(a.as_mask_where_zero() == ~a.values & ~mask)
    assert np.all(a.as_mask_where_nonzero_or_masked() == a.values | mask)
    assert np.all(a.as_mask_where_zero_or_masked() == ~a.values | mask)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, False)
    assert a == values
    assert (a == a.as_mask_where_nonzero())
    assert (~a == a.as_mask_where_zero())
    assert (a == a.as_mask_where_nonzero_or_masked())
    assert (~a == a.as_mask_where_zero_or_masked())
    assert Boolean(True, True) == Boolean.MASKED
    assert Boolean(True, False) == True
    assert Boolean(False, False) == False
    assert Boolean(False, True) == Boolean.MASKED
    a = Boolean(N//2 * [True] + N//2 * [False])
    assert a[:N//2] == True
    assert a[N//2:] == False
    a = Scalar(np.random.randn(N).clip(0,100))
    b = Boolean(a)
    assert a[~b] == 0.
    assert (a[b] != 0.).all()
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999))
    b = Boolean(a)
    assert b == (a.data != 0.)
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean(a)
    assert b[a.mask] == Boolean.MASKED
    assert np.all(b[a.data == 0.].as_mask_where_nonzero() == False)

    ##################################################################################
    # Disallowed base class operations
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    da_dt = Boolean(np.random.randn(N))
    with pytest.raises(TypeError):
        a.insert_deriv('t', da_dt)
    with pytest.raises(TypeError):
        Boolean(a.values, unit=Unit.KM)

    ##################################################################################
    # Other constructors
    ##################################################################################

    a = Boolean.zeros((2,3), dtype='int')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'b'
    assert np.all(a.vals == False)
    a = Boolean.zeros((2,3), dtype='float')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'b'
    assert np.all(a.vals == False)
    a = Boolean.zeros((2,3), dtype='bool')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'b'
    assert np.all(a.vals == False)
    a = Boolean.zeros((2,2), mask=[[0,1],[0,0]])
    assert a.shape == (2,2)
    assert np.all(a.vals == False)
    assert np.all(a.mask == [[0,1],[0,0]])
    with pytest.raises(ValueError):
        Boolean.zeros((2,3), numer=(3,))
    with pytest.raises(ValueError):
        Boolean.zeros((2,3), denom=(3,))

    a = Boolean.ones((2,3), dtype='int')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'b'
    assert np.all(a.vals == True)
    a = Boolean.ones((2,3), dtype='float')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'b'
    assert np.all(a.vals == True)
    a = Boolean.ones((2,3), dtype='bool')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'b'
    assert np.all(a.vals == True)
    a = Boolean.ones((2,2), mask=[[0,1],[0,0]])
    assert a.shape == (2,2)
    assert np.all(a.vals == 1)
    assert np.all(a.mask == [[0,1],[0,0]])
    with pytest.raises(ValueError):
        Boolean.ones((2,3), numer=(3,))
    with pytest.raises(ValueError):
        Boolean.ones((2,3), denom=(3,))

    a = Boolean.filled((2,3), 7)
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'b'
    assert np.all(a.vals == True)
    a = Boolean.filled((2,3), 7.)
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'b'
    assert np.all(a.vals == True)
    a = Boolean.filled((2,2), 7, mask=[[0,1],[0,0]])
    assert a.shape == (2,2)
    assert np.all(a.vals == True)
    assert np.all(a.mask == [[0,1],[0,0]])
    with pytest.raises(ValueError):
        Boolean.ones(7, (2,3), numer=(3,))
    with pytest.raises(ValueError):
        Boolean.ones(7, (2,3), denom=(3,))

    ##################################################################################
    # as_boolean
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean.as_boolean(a)
    assert (a is b)
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean.as_boolean(Scalar(a))
    assert a is not b
    assert b[a.mask] == Boolean.MASKED
    assert np.all(b[a.data == 0.].as_mask_where_nonzero() == False)
    a = Boolean.as_boolean(True)
    assert a == True
    assert type(a) == Boolean
    a = Boolean.as_boolean(False)
    assert a == False
    assert type(a) == Boolean
    a = Boolean.as_boolean(2)
    assert a == True
    assert type(a) == Boolean
    a = Boolean.as_boolean(0)
    assert a == False
    assert type(a) == Boolean
    a = Boolean.as_boolean(-2.)
    assert a == True
    assert type(a) == Boolean
    a = Boolean.as_boolean(0.)
    assert a == False
    assert type(a) == Boolean
    arg = np.array([True, False])
    a = Boolean.as_boolean(arg)
    assert a[0]
    assert not a[1]
    b = Boolean.as_boolean(arg[0])      # np.bool_
    assert b.vals is True

    ##################################################################################
    # as_int(), as_numeric(), as_index()
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    c = a.as_int()
    assert c == a
    assert c[a] == 1
    assert c[~a] == 0
    assert type(c) == Scalar
    assert c.values.dtype == np.dtype('int8')
    assert not a.readonly
    assert not c.readonly
    assert a.as_readonly().readonly
    assert not (~a.as_readonly()).readonly
    a = Boolean(True)
    c = a.as_int()
    assert c == a
    assert c == 1
    assert type(c.values) == int
    a = Boolean(False)
    c = a.as_int()
    assert c == a
    assert c == 0
    assert type(c.values) == int
    a = Boolean(False)
    c = a.as_numeric()
    assert c == a
    assert c == 0
    assert type(c.values) == int
    a = Boolean(np.random.randn(N) < 0.)
    k = a.as_index()
    assert k == a
    assert a[k] == 1
    assert a[~k] == 0
    assert type(k) == np.ndarray
    assert k.dtype == np.dtype('bool')
    a = Boolean(np.random.randn(N) < 0., np.random.randn(N) < 0.)
    k = a.as_index()
    assert np.all(k == a.vals & ~a.mask)
    assert np.all(a[k])
    assert not np.any(a[~k])
    assert type(k) == np.ndarray
    assert k.dtype == np.dtype('bool')
    a = Boolean(True)
    k = a.as_index()
    assert k == 1
    assert isinstance(k, numbers.Integral)

    ##################################################################################
    # as_float()
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    c = a.as_float()
    assert c == a
    assert c[a] == 1.
    assert c[~a] == 0.
    assert type(c) == Scalar
    assert c.values.dtype == np.dtype('float')
    assert not a.readonly
    assert not c.readonly
    assert a.as_readonly().readonly
    assert not (~a.as_readonly()).readonly
    a = Boolean(True)
    c = a.as_float()
    assert c == a
    assert c == 1.
    assert type(c.values) == float
    a = Boolean(False)
    c = a.as_float()
    assert c == a
    assert c == 0.
    assert type(c.values) == float

    ##################################################################################
    # sum()
    ##################################################################################
    N = 100
    a = Boolean([0,1,0,1,0])
    assert a.sum() == 2
    assert a.sum(value=False) == 3

    ##################################################################################
    # ~ operator (not), logical_not()
    ##################################################################################
    a = Boolean((False, False, True, True), (False, True, True, False))
    b = ~a
    assert b[0] == True
    assert b[1] == Boolean.MASKED
    assert b[2] == Boolean.MASKED
    assert b[3] == False
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    c = ~a
    assert c == np.logical_not(a.values)
    c = a.logical_not()
    assert c == np.logical_not(a.values)
    assert not a.readonly
    assert not c.readonly
    assert a.as_readonly().readonly
    assert not (~a.as_readonly()).readonly

    ##################################################################################
    # & operator (and)
    #
    # Truth table for three-valued logic
    #           False       Masked      True
    # False     False       False       False
    # Masked    False       Masked      Masked
    # True      False       Masked      True
    ##################################################################################
    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]
    ab = a.tvl_and(b)
    assert ab[0] == False
    assert ab[:,0] == False
    assert ab[3,3] == True
    assert ab[1:,1:3] == Boolean.MASKED
    assert ab[1:3,1:] == Boolean.MASKED
    ab = a & b
    assert ab[0,0] == False
    assert ab[0,3] == False
    assert ab[3,0] == False
    assert ab[3,3] == True
    assert ab[:,1:3] == Boolean.MASKED
    assert ab[1:3,:] == Boolean.MASKED
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(np.random.randn(4,N) < 0.5)
    c = a & b
    assert c == a.values & b.values
    assert (c == a.values & b.values).all()
    assert not a.readonly
    assert not b.readonly
    assert not c.readonly
    assert a.as_readonly().readonly
    assert b.as_readonly().readonly
    assert not (a.as_readonly() & b.as_readonly()).readonly
    assert not (a.as_readonly() & b).readonly
    assert not (a & b.as_readonly()).readonly
    c = a & False
    assert c == False
    assert type(c) == Boolean
    assert c.shape == (N,)
    c = a & True
    assert c == a
    assert type(c) == Boolean
    assert c.shape == (N,)
    c = a & (N * [True])
    assert c == a
    assert type(c) == Boolean
    assert c.shape == (N,)

    ##################################################################################
    # | operator (or)
    #
    # Truth table for three-valued logic
    #               False       Masked(F)   Masked(T)   True
    # False         False       Masked      Masked      True
    # Masked(F)     Masked      Masked      Masked      True
    # Masked(T)     Masked      Masked      Masked      True
    # True          True        True        True        True
    ##################################################################################
    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]
    ab = a.tvl_or(b)
    assert ab[0,0] == False
    assert ab[:,3] == True
    assert ab[3,:] == True
    assert ab[:3,1:3] == Boolean.MASKED
    assert ab[1:3,:3] == Boolean.MASKED
    ab = a | b
    assert ab[0,0] == False
    assert ab[0,3] == True
    assert ab[3,0] == True
    assert ab[3,3] == True
    assert ab[:,1:3] == Boolean.MASKED
    assert ab[1:3,:] == Boolean.MASKED
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(np.random.randn(4,N) < 0.5)
    c = a | b
    assert c == a.values | b.values
    assert not a.readonly
    assert not b.readonly
    assert not c.readonly
    assert a.as_readonly().readonly
    assert b.as_readonly().readonly
    assert not (a.as_readonly() | b.as_readonly()).readonly
    assert not (a.as_readonly() | b).readonly
    assert not (a | b.as_readonly()).readonly
    c = a | False
    assert c == a
    assert type(c) == Boolean
    assert c.shape == (N,)
    c = a | True
    assert c == True
    assert type(c) == Boolean
    assert c.shape == (N,)

    ##################################################################################
    # ^ operator (xor)
    ##################################################################################
    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]
    ab = a ^ b
    assert ab[0,0] == False
    assert ab[3,3] == False
    assert ab[0,3] == True
    assert ab[3,0] == True
    assert ab[:,1:3] == Boolean.MASKED
    assert ab[1:3,:] == Boolean.MASKED
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(np.random.randn(4,N) < 0.5)
    c = a ^ b
    assert c == a.values ^ b.values
    assert (c == (a.values ^ b.values))
    assert not a.readonly
    assert not b.readonly
    assert not c.readonly
    assert a.as_readonly().readonly
    assert b.as_readonly().readonly
    assert not (a.as_readonly() ^ b.as_readonly()).readonly
    assert not (a.as_readonly() ^ b).readonly
    assert not (a ^ b.as_readonly()).readonly
    c = a ^ False
    assert c == a
    assert type(c) == Boolean
    assert c.shape == (N,)
    c = a ^ True
    assert c == ~a
    assert type(c) == Boolean
    assert c.shape == (N,)

    ##################################################################################
    # &= operator
    ##################################################################################
    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]
    ab = Boolean(4*[[False, False, True, True]], 4*[[False, True, True, False]])
    ab &= b
    assert ab[0,0] == False
    assert ab[0,3] == False
    assert ab[3,0] == False
    assert ab[3,3] == True
    assert ab[:,1:3] == Boolean.MASKED
    assert ab[1:3,:] == Boolean.MASKED
    N = 100
    a = Boolean(np.random.randn(4,N) < 0.)
    b = Boolean(np.random.randn(N) < 0.5)
    c = a & b
    a &= b
    assert a == c
    a = Boolean(np.random.randn(4,N) < 0.)
    b = (np.random.randn(N) < 0.5)
    c = a & b
    a &= b
    assert a == c
    a = Boolean(np.random.randn(4,N) < 0.)
    c = a & True
    a &= True
    assert a == c
    a = Boolean(np.random.randn(4,N) < 0.)
    c = a & False
    a &= False
    assert a == c

    ##################################################################################
    # |= operator
    ##################################################################################
    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]
    ab = Boolean(4*[[False, False, True, True]], 4*[[False, True, True, False]])
    ab |= b
    assert ab[0,0] == False
    assert ab[0,3] == True
    assert ab[3,0] == True
    assert ab[3,3] == True
    assert ab[:,1:3] == Boolean.MASKED
    assert ab[1:3,:] == Boolean.MASKED
    N = 100
    a = Boolean(np.random.randn(4,N) < 0.)
    b = Boolean(np.random.randn(N) < 0.5)
    c = a | b
    a |= b
    assert a == c
    a = Boolean(np.random.randn(4,N) < 0.)
    b = (np.random.randn(N) < 0.5)
    c = a | b
    a |= b
    assert a == c
    a = Boolean(np.random.randn(4,N) < 0.)
    c = a | 22.
    a |= 22.
    assert a == c
    a = Boolean(np.random.randn(4,N) < 0.)
    c = a | False
    a |= False
    assert a == c

    ##################################################################################
    # ^= operator
    ##################################################################################
    a = Boolean((False, False, True, True), (False, True, True, False))
    b = a[:,np.newaxis]
    ab = Boolean(4*[[False, False, True, True]], 4*[[False, True, True, False]])
    ab ^= b
    assert ab[0,0] == False
    assert ab[3,3] == False
    assert ab[0,3] == True
    assert ab[3,0] == True
    assert ab[:,1:3] == Boolean.MASKED
    assert ab[1:3,:] == Boolean.MASKED
    N = 100
    a = Boolean(np.random.randn(4,N) < 0.)
    b = Boolean(np.random.randn(N) < 0.5)
    c = a ^ b
    a ^= b
    assert a == c
    a = Boolean(np.random.randn(4,N) < 0.)
    b = (np.random.randn(N) < 0.5)
    c = a ^ b
    a ^= b
    assert a == c
    a = Boolean(np.random.randn(4,N) < 0.)
    c = a ^ 22.
    a ^= True
    assert a == c
    a = Boolean(np.random.randn(4,N) < 0.)
    c = a | 0
    a |= 0.
    assert a == c

    ##################################################################################
    # Other arithmetic
    ##################################################################################
    a = Boolean([True,False])
    assert +a == [1,0]
    assert isinstance(+a, Scalar)
    assert isinstance(+a[0].values, numbers.Integral)
    assert -a == [-1,0]
    assert isinstance(-a, Scalar)
    assert isinstance(-a[0].values, numbers.Integral)
    assert abs(a) == [1,0]
    assert isinstance(abs(a), Scalar)
    assert isinstance(abs(a[0]).values, numbers.Integral)
    with pytest.raises(TypeError):
        a.__iadd__(True)
    with pytest.raises(TypeError):
        a.__isub__(True)
    with pytest.raises(TypeError):
        a.__imul__(True)
    with pytest.raises(TypeError):
        a.__itruediv__(True)
    with pytest.raises(TypeError):
        a.__ifloordiv__(True)
    with pytest.raises(TypeError):
        a.__imod__(True)
    assert a**200 == [1,0]
    assert isinstance(a**2, Scalar)
    assert isinstance((a**2).values[0], numbers.Integral)
    a = Boolean([True, True, False, False], [False, True, False, True])
    assert a**200 == a
    assert isinstance(a**200, Scalar)
    assert (a**200).is_int()
    assert a**200000 == a
    assert (a**200000).is_int()
    assert a**0 == Boolean(np.ones(4), a.mask)
    assert a**(-1) == Boolean([1,1,0,0], [False, True, True, True])
    assert a**(-200000) == a**(-1)
    assert (a**(-200000)).is_int()
    assert a**1. == a
    assert type(a**1.) == Scalar
    assert (a**1.).is_float()
    assert (a**0.).is_float()
    assert a**200000 == a**200000.
    assert a**0 == a**0.
    assert a**(-1) == a**(-1.)
    assert a**(-200000) == a**(-200000.)


def test_boolean_confirm_true_1_in_arithmetic() -> None:
    """Confirm True == 1 in arithmetic."""

    np.random.seed(7768)

    ##################################################################################
    # Constructor
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(a)
    assert a == b
    b = Boolean(a)
    assert a == b
    a = np.array([True,False])
    b = Boolean(a[0])
    assert b
    assert isinstance(b.vals, bool)
    a = np.array(True)      # shapeless array
    b = Boolean(a)
    assert b
    assert b.vals
    assert str(b) == 'Boolean(True)'
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    assert a[~mask] == values[~mask]
    assert np.all(a.as_mask_where_nonzero() == a.values & ~mask)
    assert np.all(a.as_mask_where_zero() == ~a.values & ~mask)
    assert np.all(a.as_mask_where_nonzero_or_masked() == a.values | mask)
    assert np.all(a.as_mask_where_zero_or_masked() == ~a.values | mask)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, False)
    assert a == values
    assert (a == a.as_mask_where_nonzero())
    assert (~a == a.as_mask_where_zero())
    assert (a == a.as_mask_where_nonzero_or_masked())
    assert (~a == a.as_mask_where_zero_or_masked())
    assert Boolean(True, True) == Boolean.MASKED
    assert Boolean(True, False) == True
    assert Boolean(False, False) == False
    assert Boolean(False, True) == Boolean.MASKED
    a = Boolean(N//2 * [True] + N//2 * [False])
    assert a[:N//2] == True
    assert a[N//2:] == False
    a = Scalar(np.random.randn(N).clip(0,100))
    b = Boolean(a)
    assert a[~b] == 0.
    assert (a[b] != 0.).all()
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999))
    b = Boolean(a)
    assert b == (a.data != 0.)
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean(a)
    assert b[a.mask] == Boolean.MASKED
    assert np.all(b[a.data == 0.].as_mask_where_nonzero() == False)

    ##################################################################################
    # Disallowed base class operations
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    da_dt = Boolean(np.random.randn(N))
    with pytest.raises(TypeError):
        a.insert_deriv('t', da_dt)
    with pytest.raises(TypeError):
        Boolean(a.values, unit=Unit.KM)

    ##################################################################################
    # Other constructors
    ##################################################################################

    a = Boolean(True) + 1
    assert a == 2
    assert type(a)
    a = 1 + Boolean(True)
    assert a == 2
    assert type(a)
    a = Boolean(True) - 2
    assert a == -1
    assert type(a)
    a = 3 - Boolean(True)
    assert a == 2
    assert type(a)
    a = Boolean(True) / 2
    assert a == 0.5
    assert type(a)
    a = 2 / Boolean(True)
    assert a == 2
    assert type(a)
    a = Boolean(True) // 1
    assert a == 1
    assert type(a)
    a = 2 // Boolean(True)
    assert a == 2
    assert type(a)
    a = Boolean(True) % 2
    assert a == 1
    assert type(a)
    a = 2 % Boolean(True)
    assert a == 0
    assert type(a)


def test_boolean_confirm_false_0_in_arithmetic() -> None:
    """Confirm False == 0 in arithmetic."""

    np.random.seed(7768)

    ##################################################################################
    # Constructor
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(a)
    assert a == b
    b = Boolean(a)
    assert a == b
    a = np.array([True,False])
    b = Boolean(a[0])
    assert b
    assert isinstance(b.vals, bool)
    a = np.array(True)      # shapeless array
    b = Boolean(a)
    assert b
    assert b.vals
    assert str(b) == 'Boolean(True)'
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    assert a[~mask] == values[~mask]
    assert np.all(a.as_mask_where_nonzero() == a.values & ~mask)
    assert np.all(a.as_mask_where_zero() == ~a.values & ~mask)
    assert np.all(a.as_mask_where_nonzero_or_masked() == a.values | mask)
    assert np.all(a.as_mask_where_zero_or_masked() == ~a.values | mask)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, False)
    assert a == values
    assert (a == a.as_mask_where_nonzero())
    assert (~a == a.as_mask_where_zero())
    assert (a == a.as_mask_where_nonzero_or_masked())
    assert (~a == a.as_mask_where_zero_or_masked())
    assert Boolean(True, True) == Boolean.MASKED
    assert Boolean(True, False) == True
    assert Boolean(False, False) == False
    assert Boolean(False, True) == Boolean.MASKED
    a = Boolean(N//2 * [True] + N//2 * [False])
    assert a[:N//2] == True
    assert a[N//2:] == False
    a = Scalar(np.random.randn(N).clip(0,100))
    b = Boolean(a)
    assert a[~b] == 0.
    assert (a[b] != 0.).all()
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999))
    b = Boolean(a)
    assert b == (a.data != 0.)
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean(a)
    assert b[a.mask] == Boolean.MASKED
    assert np.all(b[a.data == 0.].as_mask_where_nonzero() == False)

    ##################################################################################
    # Disallowed base class operations
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    da_dt = Boolean(np.random.randn(N))
    with pytest.raises(TypeError):
        a.insert_deriv('t', da_dt)
    with pytest.raises(TypeError):
        Boolean(a.values, unit=Unit.KM)

    ##################################################################################
    # Other constructors
    ##################################################################################

    a = Boolean(False) + 1
    assert a == 1
    assert type(a)
    a = 1 + Boolean(False)
    assert a == 1
    assert type(a)
    a = Boolean(False) - 1
    assert a == -1
    assert type(a)
    a = 3 - Boolean(False)
    assert a == 3
    assert type(a)
    a = Boolean(False) / 2
    assert a == 0
    assert type(a)
    a = 2 / Boolean(False)
    assert a.mask
    assert type(a)
    a = Boolean(False) // 1
    assert a == 0
    assert type(a)
    a = 2 // Boolean(False)
    assert a.mask
    assert type(a)
    a = Boolean(False) % 2
    assert a == 0
    assert type(a)
    a = 2 % Boolean(False)
    assert a.mask
    assert type(a)


def test_boolean_test_tuples() -> None:
    """Test tuples."""

    np.random.seed(7768)

    ##################################################################################
    # Constructor
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(a)
    assert a == b
    b = Boolean(a)
    assert a == b
    a = np.array([True,False])
    b = Boolean(a[0])
    assert b
    assert isinstance(b.vals, bool)
    a = np.array(True)      # shapeless array
    b = Boolean(a)
    assert b
    assert b.vals
    assert str(b) == 'Boolean(True)'
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    assert a[~mask] == values[~mask]
    assert np.all(a.as_mask_where_nonzero() == a.values & ~mask)
    assert np.all(a.as_mask_where_zero() == ~a.values & ~mask)
    assert np.all(a.as_mask_where_nonzero_or_masked() == a.values | mask)
    assert np.all(a.as_mask_where_zero_or_masked() == ~a.values | mask)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, False)
    assert a == values
    assert (a == a.as_mask_where_nonzero())
    assert (~a == a.as_mask_where_zero())
    assert (a == a.as_mask_where_nonzero_or_masked())
    assert (~a == a.as_mask_where_zero_or_masked())
    assert Boolean(True, True) == Boolean.MASKED
    assert Boolean(True, False) == True
    assert Boolean(False, False) == False
    assert Boolean(False, True) == Boolean.MASKED
    a = Boolean(N//2 * [True] + N//2 * [False])
    assert a[:N//2] == True
    assert a[N//2:] == False
    a = Scalar(np.random.randn(N).clip(0,100))
    b = Boolean(a)
    assert a[~b] == 0.
    assert (a[b] != 0.).all()
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999))
    b = Boolean(a)
    assert b == (a.data != 0.)
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean(a)
    assert b[a.mask] == Boolean.MASKED
    assert np.all(b[a.data == 0.].as_mask_where_nonzero() == False)

    ##################################################################################
    # Disallowed base class operations
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    da_dt = Boolean(np.random.randn(N))
    with pytest.raises(TypeError):
        a.insert_deriv('t', da_dt)
    with pytest.raises(TypeError):
        Boolean(a.values, unit=Unit.KM)

    ##################################################################################
    # Other constructors
    ##################################################################################

    a = Boolean((True,False)) + 1
    assert Boolean((True,False)) + 1 == (2,1)
    assert 1 + Boolean((True,False)) == (2,1)
    assert Boolean((True,False)) - 1 == (0,-1)
    assert 1 - Boolean((True,False)) == (0,1)
    assert Boolean((True,False)) * 2 == (2,0)
    assert 2 * Boolean((True,False)) == (2,0)
    assert Boolean((True,False)) / 1 == (1,0)
    assert (1 / Boolean((True,False))).mask[0] == False
    assert (1 / Boolean((True,False))).mask[1] == True
    assert Boolean((True,False)) // 1 == (1,0)
    assert (1 // Boolean((True,False))).mask[0] == False
    assert (1 // Boolean((True,False))).mask[1] == True
    assert Boolean((True,False)) % 1 == (0,0)
    assert (1 % Boolean((True,False))).mask[0] == False
    assert (1 % Boolean((True,False))).mask[1] == True

    ##################################################################################
    # More masking
    ##################################################################################
    N = 200
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    mask = a.as_mask_where_nonzero()
    assert a[mask].all()
    assert (a[mask] == True).all()
    mask = a.as_mask_where_zero()
    assert not a[mask].any()
    assert (a[mask] == False).all()
    mask = a.as_mask_where_nonzero_or_masked()
    assert not (a[mask] == False).any()
    mask = a.as_mask_where_zero_or_masked()
    assert not (a[mask] == True).any()

    ##################################################################################
    # Additional coverage tests
    ##################################################################################


def test_boolean_test_identity_method() -> None:
    """Test identity() method."""

    np.random.seed(7768)

    ##################################################################################
    # Constructor
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(a)
    assert a == b
    b = Boolean(a)
    assert a == b
    a = np.array([True,False])
    b = Boolean(a[0])
    assert b
    assert isinstance(b.vals, bool)
    a = np.array(True)      # shapeless array
    b = Boolean(a)
    assert b
    assert b.vals
    assert str(b) == 'Boolean(True)'
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    assert a[~mask] == values[~mask]
    assert np.all(a.as_mask_where_nonzero() == a.values & ~mask)
    assert np.all(a.as_mask_where_zero() == ~a.values & ~mask)
    assert np.all(a.as_mask_where_nonzero_or_masked() == a.values | mask)
    assert np.all(a.as_mask_where_zero_or_masked() == ~a.values | mask)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, False)
    assert a == values
    assert (a == a.as_mask_where_nonzero())
    assert (~a == a.as_mask_where_zero())
    assert (a == a.as_mask_where_nonzero_or_masked())
    assert (~a == a.as_mask_where_zero_or_masked())
    assert Boolean(True, True) == Boolean.MASKED
    assert Boolean(True, False) == True
    assert Boolean(False, False) == False
    assert Boolean(False, True) == Boolean.MASKED
    a = Boolean(N//2 * [True] + N//2 * [False])
    assert a[:N//2] == True
    assert a[N//2:] == False
    a = Scalar(np.random.randn(N).clip(0,100))
    b = Boolean(a)
    assert a[~b] == 0.
    assert (a[b] != 0.).all()
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999))
    b = Boolean(a)
    assert b == (a.data != 0.)
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean(a)
    assert b[a.mask] == Boolean.MASKED
    assert np.all(b[a.data == 0.].as_mask_where_nonzero() == False)

    ##################################################################################
    # Disallowed base class operations
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    da_dt = Boolean(np.random.randn(N))
    with pytest.raises(TypeError):
        a.insert_deriv('t', da_dt)
    with pytest.raises(TypeError):
        Boolean(a.values, unit=Unit.KM)

    ##################################################################################
    # Other constructors
    ##################################################################################

    a = Boolean(True)
    ident = a.identity()
    assert ident == Boolean(True)
    assert ident.readonly


def test_boolean_test_rtruediv_with_non_qube_arg() -> None:
    """Test __rtruediv__ with non-Qube arg."""

    np.random.seed(7768)

    ##################################################################################
    # Constructor
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(a)
    assert a == b
    b = Boolean(a)
    assert a == b
    a = np.array([True,False])
    b = Boolean(a[0])
    assert b
    assert isinstance(b.vals, bool)
    a = np.array(True)      # shapeless array
    b = Boolean(a)
    assert b
    assert b.vals
    assert str(b) == 'Boolean(True)'
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    assert a[~mask] == values[~mask]
    assert np.all(a.as_mask_where_nonzero() == a.values & ~mask)
    assert np.all(a.as_mask_where_zero() == ~a.values & ~mask)
    assert np.all(a.as_mask_where_nonzero_or_masked() == a.values | mask)
    assert np.all(a.as_mask_where_zero_or_masked() == ~a.values | mask)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, False)
    assert a == values
    assert (a == a.as_mask_where_nonzero())
    assert (~a == a.as_mask_where_zero())
    assert (a == a.as_mask_where_nonzero_or_masked())
    assert (~a == a.as_mask_where_zero_or_masked())
    assert Boolean(True, True) == Boolean.MASKED
    assert Boolean(True, False) == True
    assert Boolean(False, False) == False
    assert Boolean(False, True) == Boolean.MASKED
    a = Boolean(N//2 * [True] + N//2 * [False])
    assert a[:N//2] == True
    assert a[N//2:] == False
    a = Scalar(np.random.randn(N).clip(0,100))
    b = Boolean(a)
    assert a[~b] == 0.
    assert (a[b] != 0.).all()
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999))
    b = Boolean(a)
    assert b == (a.data != 0.)
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean(a)
    assert b[a.mask] == Boolean.MASKED
    assert np.all(b[a.data == 0.].as_mask_where_nonzero() == False)

    ##################################################################################
    # Disallowed base class operations
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    da_dt = Boolean(np.random.randn(N))
    with pytest.raises(TypeError):
        a.insert_deriv('t', da_dt)
    with pytest.raises(TypeError):
        Boolean(a.values, unit=Unit.KM)

    ##################################################################################
    # Other constructors
    ##################################################################################

    a = Boolean([True, False])
    result = 2.0 / a
    assert result[0] == 2.0
    assert result[1] == Scalar.MASKED


def test_boolean_test_rtruediv_with_qube_arg() -> None:
    """Test __rtruediv__ with Qube arg."""

    np.random.seed(7768)

    ##################################################################################
    # Constructor
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(a)
    assert a == b
    b = Boolean(a)
    assert a == b
    a = np.array([True,False])
    b = Boolean(a[0])
    assert b
    assert isinstance(b.vals, bool)
    a = np.array(True)      # shapeless array
    b = Boolean(a)
    assert b
    assert b.vals
    assert str(b) == 'Boolean(True)'
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    assert a[~mask] == values[~mask]
    assert np.all(a.as_mask_where_nonzero() == a.values & ~mask)
    assert np.all(a.as_mask_where_zero() == ~a.values & ~mask)
    assert np.all(a.as_mask_where_nonzero_or_masked() == a.values | mask)
    assert np.all(a.as_mask_where_zero_or_masked() == ~a.values | mask)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, False)
    assert a == values
    assert (a == a.as_mask_where_nonzero())
    assert (~a == a.as_mask_where_zero())
    assert (a == a.as_mask_where_nonzero_or_masked())
    assert (~a == a.as_mask_where_zero_or_masked())
    assert Boolean(True, True) == Boolean.MASKED
    assert Boolean(True, False) == True
    assert Boolean(False, False) == False
    assert Boolean(False, True) == Boolean.MASKED
    a = Boolean(N//2 * [True] + N//2 * [False])
    assert a[:N//2] == True
    assert a[N//2:] == False
    a = Scalar(np.random.randn(N).clip(0,100))
    b = Boolean(a)
    assert a[~b] == 0.
    assert (a[b] != 0.).all()
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999))
    b = Boolean(a)
    assert b == (a.data != 0.)
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean(a)
    assert b[a.mask] == Boolean.MASKED
    assert np.all(b[a.data == 0.].as_mask_where_nonzero() == False)

    ##################################################################################
    # Disallowed base class operations
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    da_dt = Boolean(np.random.randn(N))
    with pytest.raises(TypeError):
        a.insert_deriv('t', da_dt)
    with pytest.raises(TypeError):
        Boolean(a.values, unit=Unit.KM)

    ##################################################################################
    # Other constructors
    ##################################################################################

    a = Boolean([True, False])
    result = a / a
    assert result[0] == 1.0
    assert result[1] == Scalar.MASKED

    result = 2 // a
    assert result[0] == 2
    assert result[1] == Scalar.MASKED

    b = Scalar([2, 1])
    result = b // a
    assert result[0] == 2
    assert result[1] == Scalar.MASKED

    result = 2 % a
    assert result[0] == 0
    assert result[1] == Scalar.MASKED

    b = Scalar([2, 1])
    result = b % a
    assert result[0] == 0
    assert result[1] == Scalar.MASKED


def test_boolean_test_le_method() -> None:
    """Test __le__ method."""

    np.random.seed(7768)

    ##################################################################################
    # Constructor
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    b = Boolean(a)
    assert a == b
    b = Boolean(a)
    assert a == b
    a = np.array([True,False])
    b = Boolean(a[0])
    assert b
    assert isinstance(b.vals, bool)
    a = np.array(True)      # shapeless array
    b = Boolean(a)
    assert b
    assert b.vals
    assert str(b) == 'Boolean(True)'
    mask = (np.random.randn(N) < 0.)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, mask)
    assert a[~mask] == values[~mask]
    assert np.all(a.as_mask_where_nonzero() == a.values & ~mask)
    assert np.all(a.as_mask_where_zero() == ~a.values & ~mask)
    assert np.all(a.as_mask_where_nonzero_or_masked() == a.values | mask)
    assert np.all(a.as_mask_where_zero_or_masked() == ~a.values | mask)
    values = (np.random.randn(N) < 0.)
    a = Boolean(values, False)
    assert a == values
    assert (a == a.as_mask_where_nonzero())
    assert (~a == a.as_mask_where_zero())
    assert (a == a.as_mask_where_nonzero_or_masked())
    assert (~a == a.as_mask_where_zero_or_masked())
    assert Boolean(True, True) == Boolean.MASKED
    assert Boolean(True, False) == True
    assert Boolean(False, False) == False
    assert Boolean(False, True) == Boolean.MASKED
    a = Boolean(N//2 * [True] + N//2 * [False])
    assert a[:N//2] == True
    assert a[N//2:] == False
    a = Scalar(np.random.randn(N).clip(0,100))
    b = Boolean(a)
    assert a[~b] == 0.
    assert (a[b] != 0.).all()
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999))
    b = Boolean(a)
    assert b == (a.data != 0.)
    a = np.ma.MaskedArray(np.random.randn(N).clip(0,999),
                          mask=(np.random.randn(N) < 0.))
    b = Boolean(a)
    assert b[a.mask] == Boolean.MASKED
    assert np.all(b[a.data == 0.].as_mask_where_nonzero() == False)

    ##################################################################################
    # Disallowed base class operations
    ##################################################################################
    N = 100
    a = Boolean(np.random.randn(N) < 0.)
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    da_dt = Boolean(np.random.randn(N))
    with pytest.raises(TypeError):
        a.insert_deriv('t', da_dt)
    with pytest.raises(TypeError):
        Boolean(a.values, unit=Unit.KM)

    ##################################################################################
    # Other constructors
    ##################################################################################

    a = Boolean([True, False])
    result = a <= 1
    assert result[0]
    assert result[1]

    result = a < 1
    assert not result[0]
    assert result[1]

    result = a >= 0
    assert result[0]
    assert result[1]

    result = a > 0
    assert result[0]
    assert not result[1]


def test_boolean_power_of_a_shapeless_boolean() -> None:
    """A shapeless Boolean raised to an integer power behaves like the array case."""

    assert (Boolean(True) ** 2).values == 1
    assert (Boolean(False) ** 2).values == 0
    assert (Boolean(False) ** 0).values == 1
    assert (Boolean(False) ** -1).mask is True


def test_boolean_in_place_power_is_not_supported() -> None:
    """In-place exponentiation is rejected, as are the other in-place operators."""

    a = Boolean([True, False])
    with pytest.raises(TypeError, match='operation is not supported'):
        a **= 2


##########################################################################################
