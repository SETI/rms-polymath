##########################################################################################
# tests/test_vector_ops.py
##########################################################################################

import numpy as np
import pytest

from polymath import Vector, Scalar


def test_vector_ops_unary_plus() -> None:
    """Unary plus."""

    np.random.seed(3762)

    a = Vector((1,2,3))
    b = +a
    assert b == (1,2,3)
    assert type(b) == Vector
    assert b.is_int()
    assert not b.is_float()
    a = Vector((1.,2.,3.))
    b = +a
    assert b == (1,2,3)
    assert type(b) == Vector
    assert not b.is_int()
    assert b.is_float()
    a = Vector((1,2))
    b = +a
    assert b == (1,2)
    assert type(b) == Vector
    assert b.is_int()
    assert not b.is_float()
    a = Vector((1.,2.))
    b = +a
    assert b == (1,2)
    assert type(b) == Vector
    assert not b.is_int()
    assert b.is_float()

    a = Vector((1,2,3), derivs={'t':Vector((1,1,2))})
    b = +a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (1,1,2)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((1,1,2))})
    a.as_readonly()
    b = +a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (1,1,2)
    assert a.readonly
    assert b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly
    with pytest.raises(ValueError):
        a.__iadd__((1,1,1))
    with pytest.raises(ValueError):
        b.__iadd__((1,1,1))
    a = Vector((1,2), derivs={'t':Vector((3,4))})
    b = +a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (3,4)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2), derivs={'t':Vector((3,4))}).as_readonly()
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

    a = Vector((1,2,3))
    b = -a
    assert b == (-1,-2,-3)
    assert type(b) == Vector
    assert b.is_int()
    a = Vector((1.,2.,3.))
    b = -a
    assert b == (-1.,-2.,-3.)
    assert type(b) == Vector
    assert b.is_float()
    a = Vector((1,2))
    b = -a
    assert b == (-1,-2)
    assert type(b) == Vector
    assert b.is_int()
    a = Vector((1.,2.))
    b = -a
    assert b == (-1,-2)
    assert type(b) == Vector
    assert b.is_float()

    a = Vector((1,2,3), derivs={'t':Vector((1,1,2))})
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (-1,-1,-2)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((1,1,2))}).as_readonly()
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (-1,-1,-2)
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    with pytest.raises(ValueError):
        a.__isub__((1,1,1))

    b += (1,1,1)
    a = Vector((1,2), derivs={'t':Vector((3,4))})
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (-3,-4)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2), derivs={'t':Vector((3,4))}).as_readonly()
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (-3,-4)
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    with pytest.raises(ValueError):
        a.__isub__(1)                # read-only
    with pytest.raises(TypeError):
        b.__isub__(1)                 # class is wrong
    with pytest.raises(ValueError):
        a.__isub__(Vector([1,2]))    # read-only

    # abs()

    x = Vector((-1.,))
    assert abs(x) == 1. or abs(abs(x) - 1.) <= 5e-8
    x = Vector((1.,-2.,4.))
    assert abs(x) == (1+4+16)**0.5 or abs(abs(x) - (1+4+16)**0.5) <= 1.e-15
    x = Vector((1.,2.,4.,8.), mask=True)
    assert (abs(x).mask is True)

    x = Vector(np.random.randn(3,7))
    n = abs(x)
    assert not np.any(n.mask)
    N = 100
    x = Vector(np.random.randn(N,7),
               mask=(np.random.randn(N) < -0.3))    # Mask out a fraction
    n = abs(x)

    nn = n[~n.mask]
    xx = x[~n.mask]
    for i in range(len(nn)):
        assert nn[i]**2 == np.sum(xx.values[i]**2) or abs(nn[i]**2 - np.sum(xx.values[i]**2)) <= 1.e-14
        assert nn[i].mask == xx[i].mask

    N = 100
    x = Vector(np.random.randn(N,3))
    x.insert_deriv('t', Vector(np.random.randn(N,3)))
    x.insert_deriv('v', Vector(np.random.randn(N,3,3), drank=1,
                               mask = (np.random.randn(N) < -0.4)))
    assert 't' in x.derivs
    assert hasattr(x, 'd_dt')
    assert 'v' in x.derivs
    assert hasattr(x, 'd_dv')
    y = x.__abs__(recursive=False)
    assert 't' not in y.derivs
    assert not hasattr(y, 'd_dt')
    assert 'v' not in y.derivs
    assert not hasattr(y, 'd_dv')
    y = abs(x)
    assert 't' in y.derivs
    assert hasattr(y, 'd_dt')
    assert 'v' in y.derivs
    assert hasattr(y, 'd_dv')
    EPS = 1.e-6
    y1 = abs(x + (EPS,0,0))
    y0 = abs(x - (EPS,0,0))
    dy_dx0 = 0.5 * (y1 - y0) / EPS
    y1 = abs(x + (0,EPS,0))
    y0 = abs(x - (0,EPS,0))
    dy_dx1 = 0.5 * (y1 - y0) / EPS
    y1 = abs(x + (0,0,EPS))
    y0 = abs(x - (0,0,EPS))
    dy_dx2 = 0.5 * (y1 - y0) / EPS
    dy_dt = (dy_dx0 * x.d_dt.values[:,0] +
             dy_dx1 * x.d_dt.values[:,1] +
             dy_dx2 * x.d_dt.values[:,2])
    dy_dv0 = (dy_dx0 * x.d_dv.values[:,0,0] +
              dy_dx1 * x.d_dv.values[:,1,0] +
              dy_dx2 * x.d_dv.values[:,2,0])
    dy_dv1 = (dy_dx0 * x.d_dv.values[:,0,1] +
              dy_dx1 * x.d_dv.values[:,1,1] +
              dy_dx2 * x.d_dv.values[:,2,1])
    dy_dv2 = (dy_dx0 * x.d_dv.values[:,0,2] +
              dy_dx1 * x.d_dv.values[:,1,2] +
              dy_dx2 * x.d_dv.values[:,2,2])
    for i in range(N):
        assert y.d_dt.values[i] == dy_dt.values[i] or abs(y.d_dt.values[i] - dy_dt.values[i]) <= EPS
        assert y.d_dv.values[i,0] == dy_dv0.values[i] or abs(y.d_dv.values[i,0] - dy_dv0.values[i]) <= EPS
        assert y.d_dv.values[i,1] == dy_dv1.values[i] or abs(y.d_dv.values[i,1] - dy_dv1.values[i]) <= EPS
        assert y.d_dv.values[i,2] == dy_dv2.values[i] or abs(y.d_dv.values[i,2] - dy_dv2.values[i]) <= EPS

    N = 10
    y = Vector(np.random.randn(N,3))
    x = Vector(np.random.randn(N,3))
    assert not x.readonly
    assert not abs(x).readonly
    assert not x.as_readonly().norm().readonly

    a = Vector((1,2,3))
    with pytest.raises(TypeError):
        a.__add__(1)      # rank mismatch
    expr = Vector((1,2,3)) + (1,2,3)
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((1.,2.,3.)) + (1,2,3)
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((1,2,3)) + (1.,2.,3.)
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1,2,3) + Vector((1,2,3))
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = (1.,2.,3.) + Vector((1.,2.,3.))
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1,2,3) + Vector((1.,2.,3.))
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = np.array((1,2,3)) + Vector((1,2,3))
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector([(1,2,3),(2,3,4)]) + (1,2,3)
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector([(1.,2.,3.),(2.,3.,4.)]) + (1,2,3)
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector([(1,2,3),(2,3,4)]) + (1.,2.,3.)
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1,2,3) + Vector([(1,2,3),(2,3,4)])
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_int()
    expr = (1,2,3) + Vector([(1.,2.,3.),(2.,3.,4.)])
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1.,2.,3.) + Vector([(1,2,3),(2,3,4)])
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((1,2,3)) + ([(1,2,3),(2,3,4)])
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((1,2,3)) + ([(1.,2.,3.),(2.,3.,4.)])
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((1.,2.,3.)) + ([(1,2,3),(2,3,4)])
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = ((1,2,3),(2,3,4)) + Vector((1,2,3))
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_int()
    expr = ((1.,2.,3.),(2.,3.,4.)) + Vector((1,2,3))
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = ((1,2,3),(2,3,4)) + Vector((1.,2.,3.))
    assert expr == ((2,4,6),(3,5,7))
    assert type(expr) == Vector
    assert expr.is_float()

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = a + (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (3,2,1)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = (1,2,3) + a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (3,2,1)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a + (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (3,2,1)
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly        # because objects are identical
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a + [(1,2,3),(4,5,6)]
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == ((3,2,1),(3,2,1)) # d_dt must be broadcasted
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly

    a = Vector((1,2))
    a += (1,1)
    assert a == (2,3)
    a += (2,3)
    assert a == (4,6)
    assert a.is_int()
    with pytest.raises(TypeError):
        a.__iadd__((0.5,1.5))
    a = Vector([(1,2),(3,4)])
    b = Vector([(1,2),(3,4)], mask=(False,True))
    a += b
    assert a[0] == (2,4)
    assert a[0].mask == False
    assert a[1].mask == True
    a = Vector([(1,2),(3,4)])
    b = Vector((1,2), derivs={'t':Vector([(1,1),(2,2)], drank=1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a += b
    assert hasattr(a, 'd_dt')
    assert a == [(2,4),(4,6)]
    assert a.d_dt == ((1,1),(2,2))
    b = Vector((1,2), derivs={'t':Vector((1,2), drank=0)})
    a_copy = a.copy()
    with pytest.raises(ValueError):
        a.__iadd__(b)    # shape mismatch in deriv
    assert a == a_copy                     # but object unchanged
    a = Vector((1,2), derivs={'t':Vector(((1,2),(3,4)), drank=1)})
    b = Vector((3,4), derivs={'t':Vector(((4,3),(2,1)), drank=1)})
    a += b
    assert a == (4,6)
    assert a.d_dt == ((5,5),(5,5))

    a = Vector((1,2,3))
    with pytest.raises(TypeError):
        a.__add__(1)  # rank mismatch
    expr = Vector((1,2,3)) - (1,2,3)
    assert expr == (0,0,0)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((1.,2.,3.)) - (1,2,3)
    assert expr == (0,0,0)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((1,2,3)) - (1.,2.,3.)
    assert expr == (0,0,0)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1,2,3) - Vector((1,2,3))
    assert expr == (0,0,0)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = (1.,2.,3.) - Vector((1.,2.,3.))
    assert expr == (0,0,0)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1,2,3) - Vector((1.,2.,3.))
    assert expr == (0,0,0)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = np.array((1,2,3)) - Vector((1,2,3))
    assert expr == (0,0,0)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector([(1,2,3),(2,3,4)]) - (1,2,3)
    assert expr == ((0,0,0),(1,1,1))
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector([(1.,2.,3.),(2.,3.,4.)]) - (1,2,3)
    assert expr == ((0,0,0),(1,1,1))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector([(1,2,3),(2,3,4)]) - (1.,2.,3.)
    assert expr == ((0,0,0),(1,1,1))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1,2,3) - Vector([(1,2,3),(2,3,4)])
    assert expr == ((0,0,0),(-1,-1,-1))
    assert type(expr) == Vector
    assert expr.is_int()
    expr = (1,2,3) - Vector([(1.,2.,3.),(2.,3.,4.)])
    assert expr == ((0,0,0),(-1,-1,-1))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1.,2.,3.) - Vector([(1,2,3),(2,3,4)])
    assert expr == ((0,0,0),(-1,-1,-1))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((1,2,3)) - ([(1,2,3),(2,3,4)])
    assert expr == ((0,0,0),(-1,-1,-1))
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((1,2,3)) - ([(1.,2.,3.),(2.,3.,4.)])
    assert expr == ((0,0,0),(-1,-1,-1))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((1.,2.,3.)) - ([(1,2,3),(2,3,4)])
    assert expr == ((0,0,0),(-1,-1,-1))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = ((1,2,3),(2,3,4)) - Vector((1,2,3))
    assert expr == ((0,0,0),(1,1,1))
    assert type(expr) == Vector
    assert expr.is_int()
    expr = ((1.,2.,3.),(2.,3.,4.)) - Vector((1,2,3))
    assert expr == ((0,0,0),(1,1,1))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = ((1,2,3),(2,3,4)) - Vector((1.,2.,3.))
    assert expr == ((0,0,0),(1,1,1))
    assert type(expr) == Vector
    assert expr.is_float()

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = a - (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (3,2,1)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = (1,2,3) - a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (-3,-2,-1)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a - (1,2,3)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (3,2,1)
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly        # because objects are identical
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a - [(1,2,3),(4,5,6)]
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == ((3,2,1),(3,2,1)) # d_dt must be broadcasted
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly        # because objects are identical

    a = Vector((1,2))
    a -= (1,1)
    assert a == (0,1)
    a -= (-2,-3)
    assert a == (2,4)
    assert a.is_int()
    with pytest.raises(TypeError):
        a.__isub__((0.5,1.5))
    a = Vector([(1,2),(3,4)])
    b = Vector([(1,2),(3,4)], mask=(False,True))
    a -= b
    assert a[0] == (0,0)
    assert a[0].mask == False
    assert a[1].mask == True
    a = Vector([(1,2),(3,4)])
    b = Vector((1,2), derivs={'t':Vector([(1,1),(2,2)], drank=1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a -= b
    assert hasattr(a, 'd_dt')
    assert a == [(0,0),(2,2)]
    assert a.d_dt == ((-1,-1),(-2,-2))
    b = Vector((1,2), derivs={'t':Vector((1,2), drank=0)})
    a_copy = a.copy()
    with pytest.raises(ValueError):
        a.__iadd__(b)    # shape mismatch in deriv
    assert a == a_copy                     # but object unchanged
    a = Vector((1,2), derivs={'t':Vector(((1,2),(3,4)), drank=1)})
    b = Vector((3,4), derivs={'t':Vector(((4,3),(2,1)), drank=1)})
    a -= b
    assert a == (-2,-2)
    assert a.d_dt == ((-3,-1),(1,3))

    expr = Vector((1,2,3)) * 2
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((1.,2.,3.)) * 2
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((1,2,3)) * 2.
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = 2 * Vector((1,2,3))
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = 2 * Vector((1.,2.,3.))
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = 2. * Vector((1,2,3))
    assert expr == (2,4,6)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((1,2,3)) * (1,2)
    assert expr == [(1,2,3),(2,4,6)]
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((1.,2.,3.)) * (1,2)
    assert expr == [(1,2,3),(2,4,6)]
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((1,2,3)) * (1.,2.)
    assert expr == [(1,2,3),(2,4,6)]
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1,2) * Vector((1,2,3))
    assert expr == [(1,2,3),(2,4,6)]
    assert type(expr) == Vector
    assert expr.is_int()
    expr = (1,2) * Vector((1.,2.,3.))
    assert expr == [(1,2,3),(2,4,6)]
    assert type(expr) == Vector
    assert expr.is_float()
    expr = (1.,2.) * Vector((1,2,3))
    assert expr == [(1,2,3),(2,4,6)]
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector([(1,2,3),(2,3,4)]) * (1,2)
    assert expr == ((1,2,3),(4,6,8))
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector([(1.,2.,3.),(2.,3.,4.)]) * (1,2)
    assert expr == ((1,2,3),(4,6,8))
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector([(1,2,3),(2,3,4)]) * (1.,2.)
    assert expr == ((1,2,3),(4,6,8))
    assert type(expr) == Vector
    assert expr.is_float()

    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = a * 2
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (6,4,2)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = 2 * a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (6,4,2)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = a * (1,2)
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(3,2,1),(6,4,2)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))})
    b = (1,2) * a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(3,2,1),(6,4,2)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((1,2,3), derivs={'t':Vector((3,2,1))}).as_readonly()
    b = a * 2
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (6,4,2)
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Scalar((1,3), derivs={'t':Scalar((1,2))})
    b = Vector((1,2), derivs={'t':Vector((3,2))})
    c = b * a

    assert c == [(1,2),(3,6)]

    assert c.d_dt == [(4,4),(11,10)]
    c = a * b
    assert c == [(1,2),(3,6)]
    assert c.d_dt == [(4,4),(11,10)]

    a = Vector((1,2))
    a *= 2
    assert a == (2,4)
    with pytest.raises(TypeError):
        a.__imul__(0.25)
    a = Vector([(1,2),(3,4)])
    b = (2,3)
    a *= b
    assert a == [(2,4),(9,12)]
    a = Vector([(1,2),(3,4)])
    b = Scalar((2,3), mask=(False,True))
    a *= b
    assert a[0] == (2,4)
    assert a[0].mask == False
    assert a[1].mask == True
    a = Vector([(1,2),(3,4)])
    b = Scalar(2, derivs={'t':Scalar(1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a *= b
    assert hasattr(a, 'd_dt')
    assert a == [(2,4),(6,8)]
    assert a.d_dt == ((1,2),(3,4))
    a = Vector((3,4), derivs={'t':Vector((2,1), drank=0)})
    b = Scalar(2, derivs={'t':Scalar(1)})
    a *= b
    assert a == (6,8)
    assert a.d_dt == (7,6)

    expr = Vector((2,4,6)) / 2
    assert expr == (1,2,3)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((2,4,6)) / (1,2)
    assert expr == [(2,4,6),(1,2,3)]
    assert type(expr) == Vector
    assert expr.is_float()

    a = Vector((2,4,6), derivs={'t':Vector((6,4,2))})
    b = a / 2
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == (3,2,1)
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Vector((2,4,6))
    b = Scalar(2, derivs={'t':Scalar(-2)})
    c = a / b
    assert c == (1,2,3)
    assert c.d_dt == (1,2,3)
    assert not c.readonly
    assert not c.d_dt.readonly
    a = Vector((2,4,6), derivs={'t':Vector((4,6,8))})
    b = Scalar(2, derivs={'t':Scalar(-2)})
    c = a / b
    assert c == (1,2,3)
    assert c.d_dt == -a/b/b*b.d_dt + a.d_dt/b
    assert not c.readonly
    assert not c.d_dt.readonly
    a = Vector((2,4,6), derivs={'t':Vector((4,6,8))}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(-2)})
    c = a / b
    assert c == (1,2,3)
    assert c.d_dt == -a/b/b*b.d_dt + a.d_dt/b
    assert not c.readonly
    assert not c.d_dt.readonly
    a = Vector((2,4,6), derivs={'t':Vector((4,6,8))})
    b = Scalar(2, derivs={'t':Scalar(-2)}).as_readonly()
    c = a / b
    assert c == (1,2,3)
    assert c.d_dt == -a/b/b*b.d_dt + a.d_dt/b
    assert not c.readonly
    assert not c.d_dt.readonly
    a = Vector((2,4,6), derivs={'t':Vector((4,6,8))}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(-2)}).as_readonly()
    c = a / b
    assert c == (1,2,3)
    assert c.d_dt == -a/b/b*b.d_dt + a.d_dt/b
    assert not c.readonly
    assert not c.d_dt.readonly

    a = Vector((4,6))
    with pytest.raises(TypeError):
        a.__itruediv__(2)
    with pytest.raises(TypeError):
        a.__itruediv__(0.5)
    a = Vector((4.,6.))
    a /= 2
    assert a == (2,3)
    a = Vector((1.,2.))
    a /= 0.5
    assert a == (2,4)
    assert a.is_float()
    a = Vector([(3.,4.),(4.,6.)])
    b = Scalar((1,2), mask=(False,True))
    a /= b
    assert a[0] == (3,4)
    assert a[0].mask == False
    assert a[1].mask == True
    a = Vector([(3.,4.),(4.,6.)])
    b = Scalar((1,2), mask=(False,False))
    a /= b
    assert a[0] == (3,4)
    assert a[1] == (2,3)
    a = Vector([(3.,4.),(4.,6.)])
    b = Scalar((1,2), mask=(False,True))
    a /= b
    assert a[0] == (3,4)
    assert a[0].mask == False
    assert a[1].mask == True
    a = Vector((9.,-18.))
    b = Scalar(3, derivs={'t':Scalar((1,2), drank=1)})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    da_dt = -(a/b/b).wod * b.d_dt
    a /= b
    assert hasattr(a, 'd_dt')
    assert a == (3,-6)
    DEL = 1.e-13
    assert a.d_dt.values[0,0] == da_dt.values[0,0] or abs(a.d_dt.values[0,0] - da_dt.values[0,0]) <= DEL
    assert a.d_dt.values[0,1] == da_dt.values[0,1] or abs(a.d_dt.values[0,1] - da_dt.values[0,1]) <= DEL
    assert a.d_dt.values[1,0] == da_dt.values[1,0] or abs(a.d_dt.values[1,0] - da_dt.values[1,0]) <= DEL
    assert a.d_dt.values[1,1] == da_dt.values[1,1] or abs(a.d_dt.values[1,1] - da_dt.values[1,1]) <= DEL
    a = Vector((9.,-18.))
    a /= 0
    assert a.mask

    expr = Vector((2,4,7)) // 2
    assert expr == (1,2,3)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((2.,4.,7.)) // 2
    assert expr == (1,2,3)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((2,4,7)) // 2.
    assert expr == (1,2,3)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((2,4,7)) // (2,3)
    assert expr == [(1,2,3),(0,1,2)]
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((2.,4.,7.)) // (2,3)
    assert expr == [(1,2,3),(0,1,2)]
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((2,4,7)) // (2.,3.)
    assert expr == [(1,2,3),(0,1,2)]
    assert type(expr) == Vector
    assert expr.is_float()

    a = Vector((2,4,7), derivs={'t':Vector((6,4,2))})
    b = a // 2
    assert hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')
    assert not a.readonly
    assert not b.readonly
    a = Vector((2,4,7)).as_readonly()
    b = a // 2
    assert a.readonly
    assert not b.readonly
    a = Vector((2,4,7)).as_readonly()
    b = a // Scalar(2)
    assert a.readonly
    assert not b.readonly
    a = Vector((2,4,7)).as_readonly()
    b = a // Scalar(2).as_readonly()
    assert a.readonly
    assert not b.readonly
    a = Vector((2,4,7)).as_readonly()
    b = a // np.array(2)
    assert a.readonly
    assert not b.readonly

    a = Vector((4,7))
    a //= 2
    assert a == (2,3)
    a = Vector((5,8))
    with pytest.raises(TypeError):
        a.__ifloordiv__(3.5)
    a = Vector((5.,8.))
    a //= 3.5
    assert a == (1,2)  # no automatic conversion to float
    assert a.is_float()
    a = Vector([(3,4),(4,7)])
    b = Scalar((1,2), mask=(False,False))
    a //= b
    assert a == [(3,4),(2,3)]
    a = Vector([(3,4),(4,7)])
    b = Scalar((1,2), mask=(False,True))
    a //= b
    assert a[0] == (3,4)
    assert a[0].mask == False
    assert a[1].mask == True
    a = Vector([(3,4),(4,7)])
    b = Scalar((1,0))
    a //= b
    assert a[0] == (3,4)
    assert a[0].mask == False
    assert a[1].mask == True

    expr = Vector((2,4,7)) % 2
    assert expr == (0,0,1)
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((2.,4.,7.)) % 2
    assert expr == (0,0,1)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((2,4,7)) % 2.
    assert expr == (0,0,1)
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((2,4,7)) % (2,3)
    assert expr == [(0,0,1),(2,1,1)]
    assert type(expr) == Vector
    assert expr.is_int()
    expr = Vector((2.,4.,7.)) % (2,3)
    assert expr == [(0,0,1),(2,1,1)]
    assert type(expr) == Vector
    assert expr.is_float()
    expr = Vector((2,4,7)) % (2.,3.)
    assert expr == [(0,0,1),(2,1,1)]
    assert type(expr) == Vector
    assert expr.is_float()

    a = Vector((2,4,7), derivs={'t':Vector((6,4,2))})
    b = a % 2
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == a.d_dt
    assert not a.readonly
    assert not b.readonly
    a = Vector((2,4,7)).as_readonly()
    b = a % 2
    assert a.readonly
    assert not b.readonly
    a = Vector((2,4,7)).as_readonly()
    b = a % Scalar(2)
    assert a.readonly
    assert not b.readonly
    a = Vector((2,4,7)).as_readonly()
    b = a % Scalar(2).as_readonly()
    assert a.readonly
    assert not b.readonly
    a = Vector((2,4,7)).as_readonly()
    b = a % np.array(2)
    assert a.readonly
    assert not b.readonly

    a = Vector((4,7))
    a %= 2
    assert a == (0,1)
    a = Vector((5,8))
    with pytest.raises(TypeError):
        a.__imod__(3.5)
    a = Vector((5.,8.))
    a %= 3.5
    assert a == (1.5,1)
    a = Vector([(3,4),(4,7)])
    b = Scalar((1,2), mask=(False,False))
    a %= b
    assert a == [(0,0),(0,1)]
    a = Vector([(3,4),(4,7)])
    b = Scalar((1,2), mask=(False,True))
    a %= b
    assert a[0] == (0,0)
    assert a[0].mask == False
    assert a[1].mask == True
    a = Vector([(3,4),(4,7)])
    b = Scalar((1,0))
    a %= b
    assert a[0] == (0,0)
    assert a[0].mask == False
    assert a[1].mask == True

    a = Vector((2,4,7))
    with pytest.raises(TypeError):
        a.reciprocal()


##########################################################################################
