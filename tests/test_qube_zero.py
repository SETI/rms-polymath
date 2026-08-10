##########################################################################################
# tests/test_qube_zero.py
##########################################################################################

import numpy as np

from polymath import Boolean, Matrix, Matrix3, Pair, Quaternion, Scalar, Vector, Vector3


def test_qube_zero() -> None:
    """Exercise qube zero."""

    a = Scalar((1,2,3))
    assert a.zero() == 0
    assert type(a.zero()) == Scalar
    assert type(a.zero().values) == int
    assert a.zero().shape == ()
    a = Scalar((1.,2.,3.))
    assert a.zero() == 0
    assert type(a.zero()) == Scalar
    assert type(a.zero().values) == float
    assert a.zero().shape == ()
    a = Boolean([True,False])
    assert a.zero() == False
    assert type(a.zero()) == Boolean
    assert type(a.zero().values) == bool
    assert a.zero().shape == ()
    a = Vector([(1,2,3),(4,5,6)])
    assert a.zero() == (0,0,0)
    assert type(a.zero()) == Vector
    assert a.zero().values.dtype == np.dtype('int')
    assert a.zero().shape == ()
    a = Vector([(1.,2.,3.),(4.,5.,6.)])
    assert a.zero() == (0,0,0)
    assert type(a.zero()) == Vector
    assert a.zero().values.dtype == np.dtype('float')
    assert a.zero().shape == ()
    a = Pair([(1,2),(4,5)])
    assert a.zero() == (0,0)
    assert type(a.zero()) == Pair
    assert a.zero().values.dtype == np.dtype('int')
    assert a.zero().shape == ()
    a = Pair([(1.,2.),(4.,5.)])
    assert a.zero() == (0,0)
    assert type(a.zero()) == Pair
    assert a.zero().values.dtype == np.dtype('float')
    assert a.zero().shape == ()
    a = Vector3([(1,2,3),(4,5,6)])
    assert a.zero() == (0,0,0)
    assert type(a.zero()) == Vector3
    assert a.zero().values.dtype == np.dtype('float')  # coerced
    assert a.zero().shape == ()
    a = Vector3([(1.,2.,3.),(4.,5.,6.)])
    assert a.zero() == (0,0,0)
    assert type(a.zero()) == Vector3
    assert a.zero().values.dtype == np.dtype('float')
    assert a.zero().shape == ()
    a = Quaternion([(1,2,3,4),(4,5,6,7)])
    assert a.zero() == (0,0,0,0)
    assert type(a.zero()) == Quaternion
    assert a.zero().values.dtype == np.dtype('float')  # coerced
    assert a.zero().shape == ()
    a = Quaternion([(1.,2.,3.,4.),(4.,5.,6.,7.)])
    assert a.zero() == (0,0,0,0)
    assert type(a.zero()) == Quaternion
    assert a.zero().values.dtype == np.dtype('float')
    assert a.zero().shape == ()
    a = Matrix([(1,2),(4,5)])
    assert a.zero() == [(0,0),(0,0)]
    assert type(a.zero()) == Matrix
    assert a.zero().values.dtype == np.dtype('float')  # coerced
    assert a.zero().shape == ()
    a = Matrix([(1.,2.),(4.,5.)])
    assert a.zero() == [(0,0),(0,0)]
    assert type(a.zero()) == Matrix
    assert a.zero().values.dtype == np.dtype('float')
    assert a.zero().shape == ()
    a = Matrix3([(1,2,3),(4,5,6),(7,8,9)])
    assert a.zero() == [(0,0,0),(0,0,0),(0,0,0)]
    assert type(a.zero()) == Matrix3
    assert a.zero().values.dtype == np.dtype('float')  # coerced
    assert a.zero().shape == ()
    a = Matrix3([(1.,2.,3.),(4.,5.,6.),(7.,8.,9.)])
    assert a.zero() == [(0,0,0),(0,0,0),(0,0,0)]
    assert type(a.zero()) == Matrix3
    assert a.zero().values.dtype == np.dtype('float')
    assert a.zero().shape == ()


##########################################################################################
