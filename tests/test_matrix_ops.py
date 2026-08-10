##########################################################################################
# tests/test_matrix_ops.py
##########################################################################################


import pytest

from polymath import Matrix, Scalar, Vector


def test_matrix_ops_unary_plus() -> None:
    """Unary plus."""

    a = Matrix([(1,2,3),(3,4,5)])
    b = +a
    assert b == [(1,2,3),(3,4,5)]
    assert type(b) == Matrix
    assert b.is_float()       # Matrix is always float
    assert not hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(1,1)])})
    b = +a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(1,0),(1,1)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(1,1)])}).as_readonly()
    b = +a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(1,0),(1,1)]
    assert a.readonly
    assert b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly
    with pytest.raises(ValueError):
        a.__iadd__([(1,0),(1,1)]) # because readonly

    # Unary minus

    a = Matrix([(1,2,3),(3,4,5)])
    b = -a
    assert b == [(-1,-2,-3),(-3,-4,-5)]
    assert type(b) == Matrix
    assert b.is_float()       # Matrix is always float
    assert not hasattr(a, 'd_dt')
    assert not hasattr(b, 'd_dt')

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(1,1)])})
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(-1,-0),(-1,-1)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(1,1)])}).as_readonly()
    b = -a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(-1,-0),(-1,-1)]
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    with pytest.raises(ValueError):
        a.__isub__([(1,0),(1,1)]) # because readonly

    # abs()

    a = Matrix([(1,0,0),(0,0,1),(0,-1,0)])
    with pytest.raises(TypeError):
        a.__abs__()

    # Addition

    a = Matrix([(1,2,3),(3,4,5)])
    b = a + [(1,1,1),(0,0,0)]
    assert b == [(2,3,4),(3,4,5)]
    assert type(b) == Matrix
    assert b.is_float()       # Matrix is always float
    a = Matrix([(1,2,3),(3,4,5)])
    b = [(1,1,1),(0,0,0)] + a
    assert b == [(2,3,4),(3,4,5)]
    assert type(b) == Matrix
    assert b.is_float()
    a = Matrix([(1,2,3),(3,4,5)])
    b = [(1,1),(0,0)]
    with pytest.raises(ValueError):
        a.__add__(b)

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    b = a + [(1,1),(0,0)]
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(1,1),(-1,-1)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    b = [(1,1),(0,0)] + a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(1,1),(-1,-1)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    a = a.as_readonly()
    b = a + [(1,1),(0,0)]
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(1,1),(-1,-1)]
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly    # deriv is a direct copy

    a = Matrix([(1,2),(3,4)])
    a += [(1,1),(0,0)]
    assert a == [(2,3),(3,4)]
    a = Matrix([(1,2),(3,4)])
    b = Matrix([[(1,1),(0,0)],[(0,1),(2,0)]])
    with pytest.raises(ValueError):
        a.__iadd__(b)    # shape mismatch
    a = Matrix([(1,2),(3,4)])
    b = Matrix([(1,1),(0,0)], derivs={'t':Matrix([(1,1),(2,2)])})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a += b
    assert hasattr(a, 'd_dt')
    assert a == [(2,3),(3,4)]
    assert a.d_dt == [(1,1),(2,2)]
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,2),(3,4)])})
    b = Matrix([(1,1),(0,0)], derivs={'t':Matrix([(4,3),(2,1)])})
    a += b
    assert a == [(2,3),(3,4)]
    assert a.d_dt == ((5,5),(5,5))

    # Subtraction

    a = Matrix([(1,2,3),(3,4,5)])
    b = a - [(1,1,1),(0,0,0)]
    assert b == [(0,1,2),(3,4,5)]
    assert type(b) == Matrix
    assert b.is_float()       # Matrix is always float
    a = Matrix([(1,2,3),(3,4,5)])
    b = [(1,1,1),(0,0,0)] - a
    assert b == [(0,-1,-2),(-3,-4,-5)]
    assert type(b) == Matrix
    assert b.is_float()
    a = Matrix([(1,2,3),(3,4,5)])
    b = [(1,1),(0,0)]
    with pytest.raises(ValueError):
        a.__sub__(b)

    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    b = a - [(1,1),(0,0)]
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(1,1),(-1,-1)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    b = [(1,1),(0,0)] - a
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(-1,-1),(1,1)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,1),(-1,-1)])})
    a = a.as_readonly()
    b = a - [(1,1),(0,0)]
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(1,1),(-1,-1)]
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly        # deriv is an exact copy

    a = Matrix([(1,2),(3,4)])
    a -= [(1,1),(0,0)]
    assert a == [(0,1),(3,4)]
    a = Matrix([(1,2),(3,4)])
    b = Matrix([[(1,1),(0,0)],[(0,1),(2,0)]])
    with pytest.raises(ValueError):
        a.__isub__(b)    # shape mismatch
    a = Matrix([(1,2),(3,4)])
    b = Matrix([(1,1),(0,0)], derivs={'t':Matrix([(1,1),(2,2)])})
    assert not hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    a -= b
    assert hasattr(a, 'd_dt')
    assert a == [(0,1),(3,4)]
    assert a.d_dt == [(-1,-1),(-2,-2)]
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,2),(3,4)])})
    b = Matrix([(1,1),(0,0)], derivs={'t':Matrix([(4,3),(2,1)])})
    a -= b
    assert a == [(0,1),(3,4)]
    assert a.d_dt == ((-3,-1),(1,3))

    # Multiplication

    a = Matrix([(1,2,3),(3,4,5)])
    b = a * 2
    assert b == [(2,4,6),(6,8,10)]
    assert type(b) == Matrix
    assert b.is_float()       # Matrix is always float
    a = Matrix([(1,2,3),(3,4,5)])
    b = 2 * a
    assert b == [(2,4,6),(6,8,10)]
    assert type(b) == Matrix
    assert b.is_float()
    a = Matrix([(1,0),(0,1)])
    b = Matrix([(1,2),(3,4)]) * a
    assert b == [(1,2),(3,4)]
    assert type(b) == Matrix
    a = Matrix([(1,0),(0,1)])
    b = a * Matrix([(1,2),(3,4)])
    assert b == [(1,2),(3,4)]
    assert type(b) == Matrix
    a = Matrix([(1,0,-1),(0,2,-1)])
    b = a * Vector((1,2,3))
    assert b == (-2,1)
    assert type(b) == Vector
    a = Matrix([(1,0,-1),(0,2,-1)])
    b = a * Vector([(1,6),(2,5),(3,4)], drank=1)
    assert b == [(-2,2),(1,6)]
    assert type(b) == Vector

    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(3,2,1),(1,1,1)])})
    b = a * 2
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(6,4,2),(2,2,2)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(3,2,1),(1,1,1)])})
    b = Scalar(2, derivs={'t':Scalar(1)})
    c = a * b
    assert c.d_dt == [(7,4,1),(2,4,1)]
    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(3,2,1),(1,1,1)])})
    a = a.as_readonly()
    b = a * 2
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(6,4,2),(2,2,2)]
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Matrix([(1,0,-1),(0,2,-1)])
    b = Vector((1,2,3))
    c = a * b
    assert c == (-2,1)
    assert not c.readonly
    a = Matrix([(1,0,-1),(0,2,-1)]).as_readonly()
    b = Vector((1,2,3))
    c = a * b
    assert c == (-2,1)
    assert not c.readonly
    a = Matrix([(1,0,-1),(0,2,-1)])
    b = Vector((1,2,3)).as_readonly()
    c = a * b
    assert c == (-2,1)
    assert not c.readonly
    a = Matrix([(1,0,-1),(0,2,-1)]).as_readonly()
    b = Vector((1,2,3)).as_readonly()
    c = a * b
    assert c == (-2,1)
    assert not c.readonly

    a = Matrix([(1,2),(3,4)])
    a *= 2
    assert a == [(2,4),(6,8)]
    a = Matrix([(1,2),(3,4)])
    a *= Matrix([(2,0),(0,2)])
    assert a == [(2,4),(6,8)]
    a = Matrix([(1,2),(3,4)])
    a *= Scalar(2, derivs={'t':Scalar(-1)})
    assert a == [(2,4),(6,8)]
    assert a.d_dt == [(-1,-2),(-3,-4)]
    a = Matrix([(1,2),(3,4)])
    a *= Matrix([(2,0),(0,2)], derivs={'t':Matrix([(-1,0),(0,-1)])})
    assert a == [(2,4),(6,8)]
    assert a.d_dt == [(-1,-2),(-3,-4)]
    a = Matrix([(1,2),(3,4)], derivs={'t':Matrix([(1,0),(0,1)])})
    a *= Matrix([(2,0),(0,2)], derivs={'t':Matrix([(-1,0),(0,-1)])})
    assert a == [(2,4),(6,8)]
    assert a.d_dt == [(1,-2),(-3,-2)]

    # Division

    a = Matrix([(2,4,6),(6,8,10)])
    b = a / 2
    assert b == [(1,2,3),(3,4,5)]
    assert type(b) == Matrix
    assert b.is_float()       # Matrix is always float
    a = Matrix([(1,2,3),(3,4,5)])

    with pytest.raises(ValueError):
        Scalar(2).__truediv__(a)
    a = Matrix([(1,0),(0,-1)])
    b = 2 / a                           # 2 * inverse matrix
    assert b == [(2,0),(0,-2)]
    assert type(b) == Matrix
    a = Matrix([(-1,0),(0,-1)])
    b = Matrix([(1,2),(3,4)]) / a
    assert b == [(-1,-2),(-3,-4)]
    assert type(b) == Matrix
    a = Matrix([(1,0),(0,-1)])
    b = Matrix([(1,2),(3,4)]) / a
    assert b == [(1,-2),(3,-4)]
    assert type(b) == Matrix
    a = Matrix([(1,2),(3,4)])
    b = Matrix([(1,0),(0,1)]) / a
    assert b == a.reciprocal()
    assert type(b) == Matrix
    a = Matrix([(1,2),(3,4)])
    b = 1. / a
    assert b == a.reciprocal()
    assert type(b) == Matrix

    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(6,4,2),(2,2,2)])})
    b = a / 2
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(3,2,1),(1,1,1)]
    assert not a.readonly
    assert not b.readonly
    assert not a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Matrix([(1,0,-1),(0,2,-1)], derivs={'t':Matrix([(6,4,2),(2,2,2)])})
    a = a.as_readonly()
    b = a / 2
    assert hasattr(a, 'd_dt')
    assert hasattr(b, 'd_dt')
    assert b.d_dt == [(3,2,1),(1,1,1)]
    assert a.readonly
    assert not b.readonly
    assert a.d_dt.readonly
    assert not b.d_dt.readonly
    a = Matrix([(1,-1),(0,2)], derivs={'t':Matrix([(6,4),(2,2)])})
    b = Scalar(2, derivs={'t':Scalar(1)})
    c = a / b
    assert c.d_dt == -a/b/b*b.d_dt + a.d_dt/b
    a = Matrix([(1,-1),(0,2)], derivs={'t':Matrix([(6,4),(2,2)])}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(1)})
    c = a / b
    assert not c.readonly
    assert not c.d_dt.readonly
    a = Matrix([(1,-1),(0,2)], derivs={'t':Matrix([(6,4),(2,2)])})
    b = Scalar(2, derivs={'t':Scalar(1)}).as_readonly()
    c = a / b
    assert not c.readonly
    assert not c.d_dt.readonly
    a = Matrix([(1,-1),(0,2)], derivs={'t':Matrix([(6,4),(2,2)])}).as_readonly()
    b = Scalar(2, derivs={'t':Scalar(1)}).as_readonly()
    c = a / b
    assert not c.readonly
    assert not c.d_dt.readonly

    a = Matrix([(2,4),(6,8)])
    a /= 2
    assert a == [(1,2),(3,4)]
    a = Matrix([(2,4),(6,8)])
    a /= Matrix([(2,0),(0,2)])
    assert a == [(1,2),(3,4)]
    a = Matrix([(2,4),(6,8)])
    b = Scalar(2, derivs={'t':Scalar(-1)})
    da_dt = -a/b/b*b.d_dt
    a /= b
    assert a == [(1,2),(3,4)]
    assert a.d_dt == da_dt
    a = Matrix([(2,4),(6,8)], derivs={'t':Matrix([(6,4),(2,2)])})
    b = Matrix([(2,0),(0,2)], derivs={'t':Matrix([(-1,0),(0,-1)])})
    da_dt = -a/b.wod/b.wod*b.d_dt + a.d_dt/b.wod
    a /= b
    assert a == [(1,2),(3,4)]
    assert a.d_dt == da_dt

    # Floor division

    with pytest.raises(TypeError):
        Matrix([(2,4),(6,8)]).__floordiv__(1)
    with pytest.raises(TypeError):
        Matrix([(2,4),(6,8)]).__ifloordiv__(1)

    # Modulus

    with pytest.raises(TypeError):
        Matrix([(2,4),(6,8)]).__mod__(1)
    with pytest.raises(TypeError):
        Matrix([(2,4),(6,8)]).__imod__(1)


##########################################################################################
