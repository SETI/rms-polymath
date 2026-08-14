##########################################################################################
# tests/test_qube._dentity.py
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Matrix, Matrix3, Pair, Quaternion, Scalar, Vector, Vector3


def test_qube_identity() -> None:
    """Exercise qube identity."""

    a = Scalar((1,2,3))
    assert a.identity() == 1
    assert type(a.identity()) == Scalar
    assert type(a.identity().values) == int
    assert a.identity().shape == ()
    a = Scalar((1.,2.,3.))
    assert a.identity() == 1
    assert type(a.identity()) == Scalar
    assert type(a.identity().values) == float
    assert a.identity().shape == ()
    a = Boolean([True,False])
    assert a.identity() == True
    a = Vector([(1,2,3),(4,5,6)])
    with pytest.raises(TypeError):
        a.identity()
    a = Pair([(1,2),(4,5)])
    with pytest.raises(TypeError):
        a.identity()
    a = Vector3([(1,2,3),(4,5,6)])
    with pytest.raises(TypeError):
        a.identity()
    a = Quaternion([(1,2,3,4),(4,5,6,7)])
    assert a.identity() == (1,0,0,0)
    assert type(a.identity()) == Quaternion
    assert a.identity().values.dtype == np.dtype('float')  # coerced
    assert a.identity().shape == ()
    a = Quaternion([(1.,2.,3.,4.),(4.,5.,6.,7.)])
    assert a.identity() == (1,0,0,0)
    assert type(a.identity()) == Quaternion
    assert a.identity().values.dtype == np.dtype('float')
    assert a.identity().shape == ()
    a = Matrix([(1,2),(4,5)])
    assert a.identity() == [(1,0),(0,1)]
    assert type(a.identity()) == Matrix
    assert a.identity().values.dtype == np.dtype('float')  # coerced
    assert a.identity().shape == ()
    a = Matrix([(1,2,3),(4,5,6),(7,8,9)])
    assert a.identity() == [(1,0,0),(0,1,0),(0,0,1)]
    assert type(a.identity()) == Matrix
    assert a.identity().values.dtype == np.dtype('float')  # coerced
    assert a.identity().shape == ()
    a = Matrix3([(1,2,3),(4,5,6),(7,8,9)])
    assert a.identity() == [(1,0,0),(0,1,0),(0,0,1)]
    assert type(a.identity()) == Matrix3
    assert a.identity().values.dtype == np.dtype('float')  # coerced
    assert a.identity().shape == ()


##########################################################################################
