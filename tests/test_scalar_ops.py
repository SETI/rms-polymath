##########################################################################################
# test/test_scalar_ops.py
##########################################################################################

import numpy as np
import operator
import pytest

from collections.abc import Callable

from polymath import Boolean, Scalar, Unit, Vector


def test_scalar_ops_unary_plus() -> None:
    """Unary plus."""

    np.random.seed(4420)

    a = Scalar(1)
    b = +a
    assert b == 1
    assert type(b) == Scalar
    assert b.is_int()
    assert not b.is_float()
    a = Scalar(1.)
    b = +a
    assert b == 1
    assert type(b) == Scalar
    assert not b.is_int()
    assert b.is_float()
    a = Scalar((1,2))
    b = +a
    assert b == (1,2)
    assert type(b) == Scalar
    assert b.is_int()
    assert not b.is_float()
    a = Scalar((1.,2.))
    b = +a
    assert b == (1,2)
    assert type(b) == Scalar
    assert not b.is_int()
    assert b.is_float()

    a = Scalar(1, derivs={'t':Scalar(2)})
    b = +a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = +a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert a.readonly
    assert b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly
    with pytest.raises(ValueError):
        a.__iadd__(1)
    with pytest.raises(ValueError):
        b.__iadd__(1)
    a = Scalar((1,2), derivs={'t':Scalar((3,4))})
    b = +a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (3,4)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar((1,2), derivs={'t':Scalar((3,4))}).as_readonly()
    b = +a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (3,4)
    assert a.readonly
    assert b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly
    with pytest.raises(ValueError):
        a.__iadd__(1)
    with pytest.raises(ValueError):
        b.__iadd__(1)

    a = Scalar(1)
    b = -a
    assert b == -1
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar(1.)
    b = -a
    assert b == -1
    assert type(b) == Scalar
    assert b.is_float()
    a = Scalar((1,2))
    b = -a
    assert b == (-1,-2)
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar((1.,2.))
    b = -a
    assert b == (-1,-2)
    assert type(b) == Scalar
    assert b.is_float()

    a = Scalar(1, derivs={'t':Scalar(2)})
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == -2
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == -2
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    with pytest.raises(ValueError):
        a.__isub__(1)

    b -= 1
    a = Scalar((1,2), derivs={'t':Scalar((3,4))})
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (-3,-4)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar((1,2), derivs={'t':Scalar((3,4))}).as_readonly()
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (-3,-4)
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    with pytest.raises(ValueError):
        a.__isub__(1)

    b -= 1

    a = abs(Scalar(1))
    b = abs(a)
    assert b == 1
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar(-1)
    b = abs(a)
    assert b == 1
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar(1.)
    b = abs(a)
    assert b == 1
    assert type(b) == Scalar
    assert b.is_float()
    a = Scalar(-1.)
    b = abs(a)
    assert b == 1
    assert type(b) == Scalar
    assert b.is_float()
    a = Scalar((1,-2))
    b = abs(a)
    assert b == (1,2)
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar((-1.,2.))
    b = abs(a)
    assert b == (1,2)
    assert type(b) == Scalar
    assert b.is_float()

    a = Scalar(1, derivs={'t':Scalar(2)})
    b = abs(a)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert not a.readonly
    assert not b.readonly
    assert not b.d_dt.readonly
    assert b.d_dt == 2
    a = Scalar(-1, derivs={'t':Scalar(2)})
    b = abs(a)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert not a.readonly
    assert not b.readonly
    assert not b.d_dt.readonly
    assert b.d_dt == -2
    a = Scalar((1,-1), derivs={'t':Scalar((2,2))})
    b = abs(a)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert not a.readonly
    assert not b.readonly
    assert not b.d_dt.readonly
    assert b.d_dt == (2,-2)
    a = Scalar(1).as_readonly()
    b = abs(a)
    assert a.readonly
    assert not b.readonly
    a = Scalar((1,-1), derivs={'t':Scalar((2,2))}).as_readonly()
    b = abs(a)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert a.readonly
    assert not b.readonly
    assert not b.d_dt.readonly

    expr = Scalar(1) + 1
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(1.) + 1
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(1) + 1.
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 1 + Scalar(1)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 1. + Scalar(1)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 1 + Scalar(1.)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((1,2,3)) + 1
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 1 + Scalar((1,2,3))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(1) + (1,2,3)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = (1,2,3) + Scalar(1)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(1) + np.array((1,2,3))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = np.array((1,2,3)) + Scalar(1)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar((1,2,3)) + 1.
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 1. + Scalar((1,2,3))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((1.,2.,3.)) + 1
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 1 + Scalar((1.,2.,3.))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(1) + (1.,2.,3.)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = (1.,2.,3.) + Scalar(1)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(1.) + (1,2,3)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = (1,2,3) + Scalar(1.)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()

    a = Scalar(1, derivs={'t':Scalar(2)})
    b = a + (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly       # writeable because it is a scalar
    assert b.d_dt.readonly        # readonly because of broadcast
    a = Scalar(1, derivs={'t':Scalar(2)})
    b = (1,2,3) + a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert b.d_dt.readonly        # because of broadcast
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = a + (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly        # because of broadcast
    assert b.shape == b.d_dt.shape # d_dt must be broadcasted
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = (1,2,3) + a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly        # because of broadcast
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)})
    c = a + b
    assert c.d_dt == 6
    assert not b.readonly
    assert not c.d_dt.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)}).as_readonly()
    c = a + b
    assert c.d_dt == 6
    assert b.readonly
    assert not c.d_dt.readonly

    a = Scalar((1,2))
    a += 1
    assert a == (2,3)
    a += (2,3)
    assert a == (4,6)
    assert a.is_int()
    with pytest.raises(TypeError):
        a.__iadd__(0.5)
    b = Scalar((1,2), mask=(False,True))
    a += b
    assert a[0] == 5
    assert a[0].mask == False
    assert a[1].mask == True
    a = Scalar((1,2))
    b = Scalar((1,2), derivs={'t':Scalar([(1,1),(2,2)], drank=1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a += b
    assert hasattr(a, 'd_dt')
    assert a == (2,4)
    assert a.d_dt == ((1,1),(2,2))
    b = Scalar((1,2), derivs={'t':Scalar((1,2), drank=0)})
    a_copy = a.copy()
    with pytest.raises(ValueError):
        a.__iadd__(b)
    assert a == a_copy
    b = Scalar((1,2), derivs={'t':Scalar(((1,2),(3,4)), drank=1)})
    a += b
    assert a == (3,6)
    assert a.d_dt == ((2,3),(5,6))

    for avals in (1., np.arange(24.).reshape(4,3,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)):
                continue
            a = Scalar(avals, amask)
            for bvals in (1., np.arange(8.).reshape(2,4,1,1)):
                for bmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(bmask)) > len(np.shape(bvals)):
                        continue
                    b = Scalar(bvals, bmask)

                    test = a + b
                    assert np.shape(test.mask) in ((), np.shape(test))
    for avals in ((1.,2.), np.arange(48.).reshape(4,3,2,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)) - 1:
                continue
            a = Scalar(avals, amask, drank=1)
            for bvals in (1., np.arange(16.).reshape(2,4,1,1,2)):
                for bmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(bmask)) > len(np.shape(bvals)) - 1:
                        continue
                    b = Scalar(bvals, bmask, drank=1)

                    test = a + b
                    assert np.shape(test.mask) in ((), np.shape(test))

    expr = Scalar(3) - 1
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(3.) - 1
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(3) - 1.
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 3 - Scalar(1)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 3. - Scalar(1)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 3 - Scalar(1.)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((3,4,5)) - 1
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 1 - Scalar((-1,-2,-3))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(1) - (-1,-2,-3)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = (3,4,5) - Scalar(1)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(1) - np.array((-1,-2,-3))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = np.array((3,4,5)) - Scalar(1)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar((3,4,5)) - 1.
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 1. - Scalar((-1,-2,-3))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((3.,4.,5.)) - 1
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 1 - Scalar((-1.,-2.,-3.))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(1) - (-1.,-2.,-3.)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = (3.,4.,5.) - Scalar(1)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(1.) - (-1,-2,-3)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = (1,2,3) - Scalar(-1.)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()

    a = Scalar(1, derivs={'t':Scalar(2)})
    b = a - (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert b.d_dt.readonly        # because of broadcast
    a = Scalar(1, derivs={'t':Scalar(-2)})
    b = (1,2,3) - a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert b.d_dt.readonly        # because of broadcast
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = a - (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly        # because of broadcast
    assert b.shape == b.d_dt.shape     # d_dt must be broadcasted
    a = Scalar(1, derivs={'t':Scalar(-2)}).as_readonly()
    b = (1,2,3) - a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == 2
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly        # because of broadcast
    a = Scalar(1, derivs={'t':Scalar(10)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)})
    c = a - b
    assert c.d_dt == 6
    assert not b.readonly
    assert not c.d_dt.readonly
    a = Scalar(1, derivs={'t':Scalar(10)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)}).as_readonly()
    c = a - b
    assert c.d_dt == 6
    assert b.readonly
    assert not c.d_dt.readonly

    a = Scalar((3,4))
    a -= 1
    assert a == (2,3)
    a -= (1,2)
    assert a == (1,1)
    assert a.is_int()
    with pytest.raises(TypeError):
        a.__isub__(0.5)
    a = Scalar((3,4))
    b = Scalar((1,2), mask=(False,True))
    a -= b
    assert a[0] == 2
    assert a[0].mask == False
    assert a[1].mask == True
    a = Scalar((2,4))
    b = Scalar((1,2), derivs={'t':Scalar([(1,1),(2,2)], drank=1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a -= b
    assert hasattr(a, 'd_dt')
    assert a == (1,2)
    assert a.d_dt == ((-1,-1),(-2,-2))
    b = Scalar((1,2), derivs={'t':Scalar((1,2), drank=0)})
    a_copy = a.copy()
    with pytest.raises(ValueError):
        a.__isub__(b)
    assert a == a_copy
    b = Scalar((1,2), derivs={'t':Scalar(((1,2),(3,4)), drank=1)})
    a -= b
    assert a == (0,0)
    assert a.d_dt == ((-2,-3),(-5,-6))

    for avals in (1., np.arange(24.).reshape(4,3,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)):
                continue
            a = Scalar(avals, amask)
            for bvals in (1., np.arange(8.).reshape(2,4,1,1)):
                for bmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(bmask)) > len(np.shape(bvals)):
                        continue
                    b = Scalar(bvals, bmask)

                    test = a - b
                    assert np.shape(test.mask) in ((), np.shape(test))
    for avals in ((1.,2.), np.arange(48.).reshape(4,3,2,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)) - 1:
                continue
            a = Scalar(avals, amask, drank=1)
            for bvals in (1., np.arange(16.).reshape(2,4,1,1,2)):
                for bmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(bmask)) > len(np.shape(bvals)) - 1:
                        continue
                    b = Scalar(bvals, bmask, drank=1)

                    test = a - b
                    assert np.shape(test.mask) in ((), np.shape(test))

    expr = Scalar(1) * 2
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(1.) * 2
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(1) * 2.
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 1 * Scalar(2)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 1. * Scalar(2)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 1 * Scalar(2.)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((1,2,3)) * 2
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 2 * Scalar((1,2,3))
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(2) * (1,2,3)
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = (1,2,3) * Scalar(2)
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(2) * np.array((1,2,3))
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = np.array((1,2,3)) * Scalar(2)
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar((1,2,3)) * 2.
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 2. * Scalar((1,2,3))
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((1.,2.,3.)) * 2
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 2 * Scalar((1.,2.,3.))
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(2) * (1.,2.,3.)
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = (1.,2.,3.) * Scalar(2)
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(2.) * (1,2,3)
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = (1,2,3) * Scalar(2.)
    assert expr == (2,4,6)
    assert type(expr) == Scalar
    assert expr.is_float()

    a = Scalar(1, derivs={'t':Scalar(2)})
    b = a * (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (2,4,6)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar(1, derivs={'t':Scalar(2)})
    b = (1,2,3) * a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (2,4,6)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = a * (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (2,4,6)
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = (1,2,3) * a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (2,4,6)
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(3)})
    c = a * b
    assert c.d_dt == 7
    assert not b.readonly
    assert not c.d_dt.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(3)}).as_readonly()
    c = a * b
    assert c.d_dt == 7
    assert b.readonly
    assert not c.d_dt.readonly

    a = Scalar((1,2))
    a *= 2
    assert a == (2,4)
    a *= (1,2)
    assert a == (2,8)
    assert a.is_int()
    a = Scalar((1,2))
    with pytest.raises(TypeError):
        a.__imul__(0.5)
    a = Scalar((3,4))
    b = Scalar((1,2), mask=(False,True))
    a *= b
    assert a[0] == 3
    assert a[0].mask == False
    assert a[1].mask == True
    a = Scalar((1,2))
    b = Scalar((3,2), derivs={'t':Scalar([(1,3),(2,1)], drank=1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a *= b
    assert hasattr(a, 'd_dt')
    assert a == (3,4)
    assert a.d_dt == ((1,3),(4,2))
    b = Scalar((2,1), derivs={'t':Scalar((1,2), drank=0)})
    a_copy = a.copy()
    with pytest.raises(ValueError):
        a.__imul__(b)
    assert a == a_copy
    b = Scalar((2,1), derivs={'t':Scalar(((1,2),(3,4)), drank=1)})
    a *= b
    assert a == (6,4)
    assert a.d_dt == ((5,12),(16,18))
    # ((3*(1,2) + 2*(1,3), (4*(3,4) + 1*(4,2)

    for avals in (1., np.arange(24.).reshape(4,3,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)):
                continue
            a = Scalar(avals, amask)
            for vvals in ([1.,1.,1.], np.arange(24.).reshape(2,4,1,1,3)):
                for vmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(vmask)) > len(np.shape(vvals)) - 1:
                        continue
                    v = Vector(vvals, vmask)

                    test = a * v
                    assert np.shape(test.mask) in ((), np.shape(test))
    for avals in (1., np.arange(24.).reshape(4,3,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)):
                continue
            a = Scalar(avals, amask)
            for vvals in ([1.,1.,1.], np.arange(24.).reshape(2,4,1,1,3)):
                for vmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(vmask)) > len(np.shape(vvals)) - 1:
                        continue
                    v = Scalar(vvals, vmask, drank=1)

                    test = a * v
                    assert np.shape(test.mask) in ((), np.shape(test))

    expr = Scalar(4) / 2
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 4 / Scalar(2)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((2,4,6)) / 2
    assert expr == (1,2,3)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 6 / Scalar((6,3,2))
    assert expr == (1,2,3)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(6) / (6,3,2)
    assert expr == (1,2,3)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = (2,4,6) / Scalar(2)
    assert expr == (1,2,3)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(6) / np.array((6,3,2))
    assert expr == (1,2,3)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = np.array((2,4,6)) / Scalar(2)
    assert expr == (1,2,3)
    assert type(expr) == Scalar
    assert expr.is_float()

    a = Scalar(1, derivs={'t':Scalar(6)})
    b = a / (6,3,2)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (1,2,3)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar(2, derivs={'t':Scalar(2)})
    b = (-2,-4,-6) / a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (1,2,3)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar(2, derivs={'t':Scalar(2)}).as_readonly()
    b = (-2,-4,-6) / a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (1,2,3)
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar(5, derivs={'t':Scalar(6)}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(4)})
    c = a / b
    assert c.d_dt == -2
    assert not b.readonly
    assert not c.d_dt.readonly
    a = Scalar(5, derivs={'t':Scalar(6)}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(4)}).as_readonly()
    c = a / b
    assert c.d_dt == -2
    assert b.readonly
    assert not c.d_dt.readonly

    a = Scalar((4,6))
    with pytest.raises(TypeError):
        a.__itruediv__(2)
    a = a.as_float()
    a /= 2
    assert a == (2,3)
    a /= (2,1)
    assert a == (1,3)
    a = Scalar((1.,2.))
    a /= 0.5
    assert a == (2,4)
    a = Scalar((3.,4.))
    b = Scalar((1,2), mask=(False,True))
    a /= b
    assert a[0] == 3
    assert a[0].mask == False
    assert a[1].mask == True
    a = Scalar((12.,15.))
    b = Scalar((3,5), derivs={'t':Scalar([(18,9),(5,-10)], drank=1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a /= b
    assert hasattr(a, 'd_dt')
    assert a == (4,3)

    assert a.d_dt.values[0,0] == -24 or abs(a.d_dt.values[0,0] - -24) <= 1.e-14
    assert a.d_dt.values[0,1] == -12 or abs(a.d_dt.values[0,1] - -12) <= 1.e-14
    assert a.d_dt.values[1,0] == -3 or abs(a.d_dt.values[1,0] - -3) <= 1.e-14
    assert a.d_dt.values[1,1] == 6 or abs(a.d_dt.values[1,1] - 6) <= 1.e-14
    b = Scalar((2,1), derivs={'t':Scalar((1,2), drank=0)})
    a_copy = a.copy()
    with pytest.raises(ValueError):
        a.__imul__(b)
    assert a == a_copy
    b = Scalar((2,1), derivs={'t':Scalar(((1,1),(1,1)), drank=1)})
    a /= b
    assert a == (2,3)

    assert a.d_dt.values[0,0] == -13 or abs(a.d_dt.values[0,0] - -13) <= 1.e-14
    assert a.d_dt.values[0,1] == -7 or abs(a.d_dt.values[0,1] - -7) <= 1.e-14
    assert a.d_dt.values[1,0] == -6 or abs(a.d_dt.values[1,0] - -6) <= 1.e-14
    assert a.d_dt.values[1,1] == 3 or abs(a.d_dt.values[1,1] - 3) <= 1.e-14
    a /= 2
    assert a == (1,1.5)
    assert a.d_dt.values[0,0] == -13/2. or abs(a.d_dt.values[0,0] - -13/2.) <= 1.e-14
    assert a.d_dt.values[0,1] == -7/2. or abs(a.d_dt.values[0,1] - -7/2.) <= 1.e-14
    assert a.d_dt.values[1,0] == -6/2. or abs(a.d_dt.values[1,0] - -6/2.) <= 1.e-14
    assert a.d_dt.values[1,1] == 3/2. or abs(a.d_dt.values[1,1] - 3/2.) <= 1.e-14
    a /= 0
    assert a.mask

    for avals in (1., np.arange(24.).reshape(4,3,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)):
                continue
            a = Scalar(avals, amask)
            for vvals in ([1.,1.,1.], np.arange(24.).reshape(2,4,1,1,3)):
                for vmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(vmask)) > len(np.shape(vvals)) - 1:
                        continue
                    v = Vector(vvals, vmask)

                    test = v / a
                    assert np.shape(test.mask) in ((), np.shape(test))
    for avals in (1., np.arange(24.).reshape(4,3,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)):
                continue
            a = Scalar(avals, amask)
            for vvals in ([1.,1.,1.], np.arange(24.).reshape(2,4,1,1,3)):
                for vmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(vmask)) > len(np.shape(vvals)) - 1:
                        continue
                    v = Scalar(vvals, vmask, drank=1)

                    test = v / a
                    assert np.shape(test.mask) in ((), np.shape(test))

    expr = Scalar(5) // 2
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(5.) // 2
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(5) // 2.
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 5 // Scalar(2)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 5. // Scalar(2)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 5 // Scalar(2.)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((5,7,9)) // 2
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar((5.,7.,9.)) // 2
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((5,7,9)) // 2.
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 9 // Scalar((4,3,2))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 9. // Scalar((4,3,2))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 9 // Scalar((4.,3.,2.))
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = np.array((5,7,9)) // Scalar(2)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()

    a = Scalar(1, derivs={'t':Scalar(2)})
    b = a // (1,2,3)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')
    assert not a.readonly
    assert not b.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = a // (1,2,3)
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')
    assert a.readonly
    assert not b.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = (1,2,3) // a
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')
    assert a.readonly
    assert not b.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)})
    c = a // b
    assert not b.readonly
    assert not c.readonly
    a = Scalar(1, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)}).as_readonly()
    c = a // b
    assert not c.readonly

    a = Scalar((4,6))
    a //= 2
    assert a == (2,3)
    a //= (2,1)
    assert a == (1,3)
    assert a.is_int()
    a = Scalar((1,2))
    with pytest.raises(TypeError):
        a.__ifloordiv__(0.5)
    a = Scalar((1.,2.))
    a //= 0.5
    assert a == (2,4)
    assert a.is_float()
    a = Scalar((3,4))
    b = Scalar((1,2), mask=(False,True))
    a //= b
    assert a[0] == 3
    assert a[0].mask == False
    assert a[1].mask == True
    a = Scalar((12,15))
    b = Scalar((3,5), derivs={'t':Scalar([(18,9),(5,-10)], drank=1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a //= b
    assert not hasattr(a, 'd_dt')    # no derivatives in floor division
    a = Scalar((12,15))
    a //= 4
    assert a == (3,3)
    a //= 0
    assert a.mask

    for avals in (1., np.arange(24.).reshape(4,3,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)):
                continue
            a = Scalar(avals, amask)
            for bvals in (1., np.arange(8.).reshape(2,4,1,1)):
                for bmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(bmask)) > len(np.shape(bvals)):
                        continue
                    b = Scalar(bvals, bmask)

                    test = a // b
                    assert np.shape(test.mask) in ((), np.shape(test))

    expr = Scalar(5) % 3
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar(5.) % 3
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar(5) % 3.
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 5 % Scalar(3)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 5. % Scalar(3)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 5 % Scalar(3.)
    assert expr == 2
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((7,8,9)) % 5
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = Scalar((7.,8.,9.)) % 5
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = Scalar((7,8,9)) % 5.
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 9 % Scalar((3,4,5))
    assert expr == (0,1,4)
    assert type(expr) == Scalar
    assert expr.is_int()
    expr = 9. % Scalar((3,4,5))
    assert expr == (0,1,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = 9 % Scalar((3.,4.,5.))
    assert expr == (0,1,4)
    assert type(expr) == Scalar
    assert expr.is_float()
    expr = np.array((7,8,9)) % Scalar(5)
    assert expr == (2,3,4)
    assert type(expr) == Scalar
    assert expr.is_int()

    a = Scalar(9, derivs={'t':Scalar(2)})
    b = a % (3,4,5)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert a.d_dt == b.d_dt
    assert not a.readonly
    assert not b.readonly
    a = Scalar(9, derivs={'t':Scalar(2)}).as_readonly()
    b = a % (3,4,5)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert a.d_dt
    assert a.readonly
    assert not b.readonly
    a = Scalar(5, derivs={'t':Scalar(2)}).as_readonly()
    b = (7,8,9) % a
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')
    assert a.readonly
    assert not b.readonly
    a = Scalar(5, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)})
    c = a % b
    assert not b.readonly
    assert not c.readonly
    a = Scalar(5, derivs={'t':Scalar(2)}).as_readonly()
    b = Scalar(3, derivs={'t':Scalar(4)}).as_readonly()
    c = a % b
    assert not c.readonly

    a = Scalar((5,7))
    a %= 3
    assert a == (2,1)
    a %= (2,3)
    assert a == (0,1)
    assert a.is_int()
    a = Scalar((9.,12.))
    a %= 3.5
    assert a == (2,1.5)
    assert a.is_float()
    a = Scalar((9,12))
    with pytest.raises(TypeError):
        a.__imod__(3.5)
    a = Scalar((3,4))
    b = Scalar((4,2), mask=(False,True))
    a %= b
    assert a[0] == 3
    assert a[0].mask == False
    assert a[1].mask == True
    a = Scalar((12,15))
    b = Scalar((3,5), derivs={'t':Scalar([(18,9),(5,-10)], drank=1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a %= b
    assert not hasattr(a, 'd_dt')    # no derivatives in modulus
    a = Scalar((12,15))
    a %= 4
    assert a == (0,3)
    a %= 0
    assert a.mask

    for avals in (1., np.arange(24.).reshape(4,3,2)):
        for amask in (True, False, np.random.randn(4,3,2) < 0.):
            if len(np.shape(amask)) > len(np.shape(avals)):
                continue
            a = Scalar(avals, amask)
            for bvals in (1., np.arange(8.).reshape(2,4,1,1)):
                for bmask in (True, False, np.random.randn(2,4,1,1) < 0.):
                    if len(np.shape(bmask)) > len(np.shape(bvals)):
                        continue
                    b = Scalar(bvals, bmask)

                    test = a % b
                    assert np.shape(test.mask) in ((), np.shape(test))

    a = Scalar(2)
    b = a**1
    assert b == 2
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar(2)
    b = a**2
    assert b == 4
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar(2)
    b = a**3
    assert b == 8
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar(2.)
    b = a**3
    assert b == 8
    assert type(b) == Scalar
    assert b.is_float()
    a = Scalar(2)
    b = a**3.
    assert b == 8
    assert type(b) == Scalar
    assert b.is_float()
    a = Scalar((0,1,2,3,4,5))
    b = a**3
    assert b == (0,1,8,27,64,125)
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar((0.,1.,2.,3.,4.,5.))
    b = a**3
    assert b == (0,1,8,27,64,125)
    assert type(b) == Scalar
    assert b.is_float()
    a = Scalar((-2,-1,0,1,2,3,4,5))
    b = a**3
    assert b == (-8,-1,0,1,8,27,64,125)
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar((-2,-1,0,1,2,3,4,5))
    b = a**3.
    assert b == (-8,-1,0,1,8,27,64,125)
    assert type(b) == Scalar
    assert b.is_float()
    a = Scalar((0,1,4,9,16,25))
    b = a**0.5
    assert b == (0,1,2,3,4,5)
    assert type(b) == Scalar
    assert b.is_float()
    assert np.all(b.mask == False)
    a = Scalar((-4,-1,0,1,4,9,16,25))
    b = a**0.5
    assert b[2:] == (0,1,2,3,4,5)
    assert type(b) == Scalar
    assert np.all(b.mask == 2*[True] + 6*[False])
    a = Scalar((-4,-1,0,1,4,9,16,25))
    b = a**(-0.5)
    assert type(b) == Scalar
    assert np.all(b.mask == 3*[True] + 5*[False])
    a = Scalar((-2,-1,0,1,2,3,4,5))
    b = a**(-1)
    assert np.all(b.mask == 2*[False] + [True] + 5*[False])
    for i in range(len(a)):
        if a[i] != 0:
            assert a[i]*b[i] == 1. or abs(a[i]*b[i] - 1.) <= 1.e-14

    a = Scalar(np.arange(20) + 1, derivs={'t':Scalar(np.ones(20))})
    b = a**0
    assert b == 1
    assert b._values.dtype.kind == 'i'
    assert b.d_dt == 0
    b = a**0.
    assert b == 1
    assert b._values.dtype.kind == 'f'
    assert b.d_dt == 0
    b = a**1
    assert b == a
    assert b._values.dtype.kind == 'i'
    assert b.d_dt == 1
    b = a**1.
    assert b == a
    assert b._values.dtype.kind == 'f'
    assert b.d_dt == 1
    b = a**2
    assert b == a*a
    assert b._values.dtype.kind == 'i'
    assert b.d_dt == 2*a
    b = a**2.
    assert b == a*a
    assert b._values.dtype.kind == 'f'
    assert b.d_dt == 2*a
    b = a**3
    assert b == a*a*a
    assert b._values.dtype.kind == 'i'
    assert b.d_dt == 3*a*a
    b = a**3.
    assert b == a*a*a
    assert b._values.dtype.kind == 'f'
    assert b.d_dt == 3*a*a
    b = a**4
    assert b == a*a*a*a
    assert b._values.dtype.kind == 'i'
    assert b.d_dt == 4*a*a*a
    b = a**4.
    assert b == a*a*a*a
    assert b._values.dtype.kind == 'f'
    assert b.d_dt == 4*a*a*a
    b = a**5
    assert b == a*a*a*a*a
    assert b._values.dtype.kind == 'i'
    assert b.d_dt == 5*a*a*a*a
    b = a**5.
    assert b == a*a*a*a*a
    assert b._values.dtype.kind == 'f'
    assert b.d_dt == 5*a*a*a*a
    b = a**0.5
    assert (abs(b - a.sqrt()).max() < 1.e-14)
    assert (abs(b.d_dt - 0.5/a.sqrt()).max() < 1.e-14)
    b = a**(-1)
    assert (abs(b*a - 1).max() < 1.e-14)
    assert (abs(b.d_dt + b*b).max() < 1.e-14)

    # Read-only status
# This probably is no longer what we intend
#     self.assertFalse(a.readonly)
#     self.assertFalse((a**0).readonly)
#     self.assertFalse((a**1).readonly)
#     self.assertFalse((a**2).readonly)
#     self.assertFalse((a**3).readonly)
#     self.assertFalse((a**0.5).readonly)
#     self.assertFalse((a**(-0.5)).readonly)
#     self.assertFalse((a**(-1)).readonly)
#
#     b = a.as_readonly()
#     self.assertTrue(b.readonly)
#     self.assertFalse((b**0).readonly)
#     self.assertFalse((b**1).readonly)
#     self.assertFalse((b**2).readonly)
#     self.assertFalse((b**3).readonly)
#     self.assertFalse((b**0.5).readonly)
#     self.assertFalse((b**(-0.5)).readonly)
#     self.assertFalse((b**(-1)).readonly)

    a = Scalar(2)
    b = a**(-0,1,2,3,4)
    assert b == (1,2,4,8,16)
    assert type(b) == Scalar
    assert b.is_int()
    a = Scalar([2,4]).reshape((2,1))
    b = a**(-1,0,1,2,3,4)
    assert b == [[0.5,1,2,4,8,16],[0.25,1,4,16,64,256]]
    assert type(b) == Scalar
    assert b.is_float()
    a = Scalar(2, unit=Unit.KM)
    b = a**2
    assert b.unit_ == Unit.KM**2
    with pytest.raises(ValueError):
        a.__pow__((2,3))
    a = Scalar(0)
    assert a**0 == 1
    assert (a**0).is_int()
    a = Scalar(0.)
    assert a**0 == 1
    assert (a**0).is_float()
    a = Scalar(0)
    assert a**0. == 1
    assert (a**0).is_int()
    a = Scalar(0.)
    assert a**0. == 1
    assert (a**0).is_float()
    a = Scalar(0)
    assert a**-1 == Scalar.MASKED
    a = Scalar(0.)
    assert a**-1 == Scalar.MASKED
    a = Scalar(0)
    assert a**-1. == Scalar.MASKED
    a = Scalar(0.)
    assert a**-1. == Scalar.MASKED
    a = Scalar(-1)
    assert a**0.5 == Scalar.MASKED
    a = Scalar(-1.)
    assert a**0.5 == Scalar.MASKED
    a = Scalar([0,1])
    assert a**-1 == Scalar([1,1],[True,False])
    a = Scalar([0.,1.])
    assert a**-1 == Scalar([1,1],[True,False])
    a = Scalar([0,1])
    assert a**-1. == Scalar([1,1],[True,False])
    a = Scalar([0.,1.])
    assert a**-1. == Scalar([1,1],[True,False])
    a = Scalar([0,1,2]).reshape((3,1))
    b = a**(0,1,2)
    assert b.flatten() == (1,0,0,1,1,1,1,2,4)
    da_dt = Scalar((1.,1.,1.))
    a = Scalar([0,1,2], derivs={'t': Scalar(da_dt)}).reshape((3,1))
    b = a**(0,1,2)
    assert b.flatten() == (1,0,0,1,1,1,1,2,4)
    assert b.d_dt[0] == Scalar((1.,1.,0.), (True,False,False))
    assert b.d_dt[1] == (0,1,2)
    assert b.d_dt[2] == (0,1,4)

    a = Scalar((1,-1))
    b = a.reciprocal()
    assert b == (1,-1)
    assert type(b)
    assert b.is_float()       # automatic conversion to float
    a = Scalar((1,-1,0))
    b = a.reciprocal()
    assert b[:2] == (1,-1)
    assert type(b)
    assert not b.mask[0]
    assert not b.mask[1]
    assert b.mask[2]
    a = Scalar((-2,-1,0,1,2), derivs={'t':Scalar((1,1,2,2,2))})
    b = a.reciprocal()
    assert b[:2] == (-0.5,-1)
    assert b[3:] == (1,0.5)
    assert b[2].mask
    assert hasattr(b, 'd_dt')
    DEL = 1.e-13
    assert b.d_dt[0].values == -0.25 or abs(b.d_dt[0].values - -0.25) <= DEL
    assert b.d_dt[1].values == -1 or abs(b.d_dt[1].values - -1) <= DEL
    assert b.d_dt[2].mask
    assert b.d_dt[3].values == -2 or abs(b.d_dt[3].values - -2) <= DEL
    assert b.d_dt[4].values == -0.5 or abs(b.d_dt[4].values - -0.5) <= DEL
    assert not b.readonly
    assert not b.d_dt.readonly
    a = Scalar((-2,-1,0,1,2), derivs={'t':Scalar((1,1,2,2,2))}).as_readonly()
    b = a.reciprocal()
    assert not b.readonly
    assert not b.d_dt.readonly
    a = Scalar((-2,-1,0,1,2), derivs={'t':Scalar((1,1,2,2,2))})
    b = a.reciprocal(recursive=False)
    assert not hasattr(b, 'd_dt')
    a = Scalar((1,-1))
    b = a.reciprocal(nozeros=True)
    assert b == (1,-1)
    a = Scalar((1,-1,0))
    with pytest.raises(ValueError):
        a.reciprocal(nozeros=True)

    # Comparisons

    assert (Scalar(-0.3) <= -0.3)
    assert (Scalar(-0.3) >= -0.3)
    assert not (Scalar(-0.3) < -0.3)
    assert not (Scalar(-0.3) > -0.3)
    assert type(Scalar(-0.3) <= -0.3) == bool
    assert type(Scalar(-0.3) >= -0.3) == bool
    assert type(Scalar(-0.3) < -0.3) == bool
    assert type(Scalar(-0.3) > -0.3) == bool
    assert (Scalar(-0.3) <= -0.2)
    assert (Scalar(-0.3) >= -0.4)
    assert (Scalar(-0.3) <  -0.2)
    assert (Scalar(-0.3) >  -0.4)
    assert not (Scalar(2,True) <  2)
    assert not (Scalar(2,True) <= 2)
    assert not (Scalar(0,True) >  0)
    assert not (Scalar(0,True) >= 0)
    assert not (Scalar(1,True) <  Scalar(2,True))
    assert not (Scalar(1,True) <= Scalar(0,True))
    assert not (Scalar(1,True) >  Scalar(0,True))
    assert not (Scalar(1,True) >= Scalar(2,True))

    assert (Scalar((-0.1,0.,0.1)) <= (-0.1,0.,0.1)).all()
    assert (Scalar((-0.1,0.,0.1)) >= (-0.1,0.,0.1)).all()
    assert not (Scalar((-0.1,0.,0.1)) < (-0.1,0.,0.1)).all()
    assert not (Scalar((-0.1,0.,0.1)) > (-0.1,0.,0.1)).all()
    assert (Scalar((1,2,3)) >= (1,2,3)).all()
    assert type(Scalar((1,2,3)) >= (1,2,3)) == Boolean
    assert type(Scalar((1,2,3)) <= (1,2,3)) == Boolean
    assert type(Scalar((1,2,3)) >  (1,2,3)) == Boolean
    assert type(Scalar((1,2,3)) <  (1,2,3)) == Boolean
    assert (Scalar((1,2,3)) <= (1,2,3)).all()
    assert not (Scalar((1,2,3)) >  (1,2,3)).all()
    assert not (Scalar((1,2,3)) <  (1,2,3)).all()
    assert (Scalar((1,2,3)) >= (0,2,3)).all()
    assert not (Scalar((1,2,3)) >= (2,2,3)).all()
    assert not (Scalar((1,2,3),[False,False,True]) <= (1,2,3)).all()
    assert not (Scalar((1,2,3),3*[True]) >= (1,2,3)).all()
    assert (Scalar((1,2,3),[False,False,True]) <= (1,2,3)) == [True,True,False]
    assert (Scalar((1,2,3),[False,False,True]) >= (1,2,3)) == [True,True,False]
    assert (Scalar((0,1,2),[False,False,True]) < (1,2,3)) == [True,True,False]
    assert (Scalar((1,2,3),[False,False,True]) > (0,1,2)) == [True,True,False]

    N = 100
    x = Scalar(np.random.randn(N))
    y = Scalar(np.random.randn(N))
    for i in range(N):
        if x.values[i] > y.values[i]:
            assert (x[i] > y[i])
            assert (x[i] >= y[i])
            assert not (x[i] < y[i])
            assert not (x[i] <= y[i])
        else:
            assert not (x[i] > y[i])
            assert not (x[i] >= y[i])
            assert (x[i] < y[i])
            assert (x[i] <= y[i])
    for i in range(N-1):
        if np.all(x.values[i:i+2] > y.values[i:i+2]):
            assert (x[i:i+2] > y[i:i+2]).all()
            assert (x[i:i+2] >= y[i:i+2]).all()
            assert not (x[i:i+2] < y[i:i+2]).all()
            assert not (x[i:i+2] <= y[i:i+2]).all()
        elif np.all(x.values[i:i+2] < y.values[i:i+2]):
            assert not (x[i:i+2] > y[i:i+2]).all()
            assert not (x[i:i+2] >= y[i:i+2]).all()
            assert (x[i:i+2] < y[i:i+2]).all()
            assert (x[i:i+2] <= y[i:i+2]).all()
        else:
            assert not (x[i:i+2] > y[i:i+2]).all()
            assert not (x[i:i+2] >= y[i:i+2]).all()
            assert not (x[i:i+2] < y[i:i+2]).all()
            assert not (x[i:i+2] <= y[i:i+2]).all()

    x = Scalar(np.random.randn(10), unit=Unit.KM)
    y = Scalar(np.random.randn(10), unit=Unit.CM)
    assert ((x > y).mask is False)
    assert ((x < y).mask is False)
    assert ((x >= y).mask is False)
    assert ((x <= y).mask is False)
    x = Scalar(np.random.randn(10), unit=Unit.KM)
    y = Scalar(np.random.randn(10), unit=None)
    assert ((x > y).mask is False)
    assert ((x < y).mask is False)
    assert ((x >= y).mask is False)
    assert ((x <= y).mask is False)
    x = Scalar(np.random.randn(10), unit=Unit.KM)
    y = Scalar(np.random.randn(10), unit=Unit.SECONDS)
    with pytest.raises(ValueError):
        x.__le__(y)
    with pytest.raises(ValueError):
        x.__ge__(y)
    with pytest.raises(ValueError):
        x.__lt__(y)
    with pytest.raises(ValueError):
        x.__gt__(y)


def test_scalar_ops_units_should_be_removed() -> None:
    """Units should be removed."""

    np.random.seed(4420)

    x = Scalar(np.random.randn(10), unit=Unit.KM)
    y = Scalar(np.random.randn(10), unit=Unit.CM)
    assert ((x > y).unit_ is None)
    assert ((x < y).unit_ is None)
    assert ((x >= y).unit_ is None)
    assert ((x <= y).unit_ is None)


def test_scalar_ops_masks() -> None:
    """Masks."""

    np.random.seed(4420)

    N = 100
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -0.2))
    y = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -0.2))
    assert ((x > y).mask is False)
    assert ((x < y).mask is False)
    assert ((x >= y).mask is False)
    assert ((x <= y).mask is False)
    for i in range(N):
        if not x.mask[i] and not y.mask[i]:
            if x.values[i] > y.values[i]:
                assert (x[i] > y[i])
                assert (x[i] >= y[i])
                assert not (x[i] < y[i])
                assert not (x[i] <= y[i])
            else:
                assert not (x[i] > y[i])
                assert not (x[i] >= y[i])
                assert (x[i] < y[i])
                assert (x[i] <= y[i])
        elif x.mask[i] and y.mask[i]:
            assert not (x[i] >= y[i])
            assert not (x[i] <= y[i])
            assert not (x[i] > y[i])
            assert not (x[i] < y[i])
        else:
            assert not (x[i] >= y[i])
            assert not (x[i] <= y[i])
            assert not (x[i] > y[i])
            assert not (x[i] < y[i])

    x = Scalar(np.random.randn(N))
    y = Scalar(np.random.randn(N))
    assert not x.readonly
    assert not y.readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not (x <  y).readonly
    assert not (x >  y).readonly
    assert not (x <= y).readonly
    assert not (x >= y).readonly
    assert not (x.as_readonly() <  y).readonly
    assert not (x.as_readonly() >  y).readonly
    assert not (x.as_readonly() <= y).readonly
    assert not (x.as_readonly() >= y).readonly
    assert not (x <  y.as_readonly()).readonly
    assert not (x >  y.as_readonly()).readonly
    assert not (x <= y.as_readonly()).readonly
    assert not (x >= y.as_readonly()).readonly
    assert not (x.as_readonly() <  y.as_readonly()).readonly
    assert not (x.as_readonly() >  y.as_readonly()).readonly
    assert not (x.as_readonly() <= y.as_readonly()).readonly
    assert not (x.as_readonly() >= y.as_readonly()).readonly


def test_scalar_ops_reciprocal_disallows_denominators() -> None:
    """Reciprocal does not support denominators."""

    a = Scalar([[1., 2.], [3., 4.]], drank=1)
    with pytest.raises(ValueError, match='does not support denominators'):
        a.reciprocal()


@pytest.mark.parametrize(('symbol', 'func'),
                         [('<' , operator.lt),
                          ('<=', operator.le),
                          ('>' , operator.gt),
                          ('>=', operator.ge)])
def test_scalar_ops_comparisons_disallow_denominators(
        symbol: str, func: Callable[[Scalar, Scalar], object]) -> None:
    """The ordering comparisons do not support denominators."""

    a = Scalar([[1., 2.], [3., 4.]], drank=1)
    with pytest.raises(ValueError, match=f'"{symbol}" does not support denominators'):
        func(a, a)


def test_scalar_ops_power_zero_without_derivatives() -> None:
    """Raising to the power zero can skip the derivatives."""

    a = Scalar(3., derivs={'t': Scalar(1.)})
    b = a.__pow__(0, recursive=False)
    assert b == 1.
    assert b.derivs == {}


def test_scalar_ops_power_exponent_disallows_denominators() -> None:
    """An exponent with a denominator is rejected."""

    with pytest.raises(ValueError, match='exponent requires scalar items'):
        Scalar(2.) ** Scalar([[1., 2.]], drank=1)


def test_scalar_ops_power_masks_a_complex_result() -> None:
    """A shapeless power whose result is not real is masked."""

    a = Scalar(-1.) ** Scalar(0.5)
    assert a.mask is True


def test_scalar_ops_power_of_an_array_with_units() -> None:
    """An array with units raised to a single power scales the unit by that power."""

    a = Scalar([1., 4., 9.], unit=Unit.KM**2) ** Scalar(0.5)
    assert a == (1., 2., 3.)
    assert a.unit_ == Unit.KM


##########################################################################################
