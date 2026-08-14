##########################################################################################
# tests/test_qube_types/py
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Matrix, Matrix3, Pair, Quaternion, Scalar, Vector, Vector3


def test_qube_types() -> None:
    """Exercise qube types."""

    np.random.seed(6172)

    ##################################################################################
    # mvals(self)
    ##################################################################################
    a = Scalar(np.random.randn(4,5), mask=(np.random.rand(4,5) < 0.2))
    mv = a.mvals
    assert np.all(mv.data == a.values)
    assert np.all(mv.mask == a.mask)
    a = Vector(np.random.randn(4,5,3), mask=(np.random.rand(4,5) < 0.2))
    mv = a.mvals
    assert np.all(mv.data == a.values)
    assert np.all(mv.mask[...,0] == a.mask)
    assert np.all(mv.mask[...,1] == a.mask)
    assert np.all(mv.mask[...,2] == a.mask)
    a = Matrix(np.random.randn(4,5,3,3), mask=(np.random.rand(4,5) < 0.2))
    mv = a.mvals
    assert np.all(mv.data == a.values)
    assert np.all(mv.mask == a.mask[...,np.newaxis,np.newaxis])

    ##################################################################################
    # is_numeric(self)
    ##################################################################################
    assert Boolean.TRUE.is_numeric() == False
    assert Scalar.ONE.is_numeric() == True
    assert Vector.XAXIS3.is_numeric() == True
    assert Vector3.XAXIS.is_numeric() == True
    assert Pair.XAXIS.is_numeric() == True
    assert Matrix.IDENTITY2.is_numeric() == True
    assert Matrix3.IDENTITY.is_numeric() == True

    ##################################################################################
    # is_numeric(self)
    ##################################################################################
    assert Boolean.TRUE.is_numeric() == False
    assert Scalar.ONE.is_numeric() == True
    assert Vector.XAXIS3.is_numeric() == True
    assert Vector3.XAXIS.is_numeric() == True
    assert Pair.XAXIS.is_numeric() == True
    assert Matrix.IDENTITY2.is_numeric() == True
    assert Matrix3.IDENTITY.is_numeric() == True

    ##################################################################################
    # as_numeric(self)
    ##################################################################################
    assert Boolean.TRUE.as_numeric() == 1
    assert Boolean.FALSE.as_numeric() == 0
    assert type(Boolean.TRUE.as_numeric()) == Scalar
    assert type(Boolean.FALSE.as_numeric()) == Scalar
    assert Scalar.ONE.as_numeric() == Scalar.ONE
    assert Vector.XAXIS3.as_numeric() == Vector.XAXIS3
    assert Vector3.XAXIS.as_numeric() == Vector3.XAXIS
    assert Pair.XAXIS.as_numeric() == Pair.XAXIS
    assert Matrix.IDENTITY2.as_numeric() == Matrix.IDENTITY2
    assert Matrix3.IDENTITY.as_numeric() == Matrix3.IDENTITY

    ##################################################################################
    # is_float(self)
    # is_int(self)
    ##################################################################################
    assert Boolean((True,False)).is_int() == False
    assert Boolean((True,False)).is_float() == False
    assert Scalar((1,2,3)).is_int() == True
    assert Scalar((1,2,3)).is_float() == False
    assert Scalar((1.,2.,3.)).is_int() == False
    assert Scalar((1.,2.,3.)).is_float() == True
    assert Vector((1,2,3)).is_int() == True
    assert Vector((1,2,3)).is_float() == False
    assert Vector((1.,2.,3.)).is_int() == False
    assert Vector((1.,2.,3.)).is_float() == True
    assert Vector3((1,2,3)).is_int() == False      # coerced to float
    assert Vector3((1,2,3)).is_float() == True
    assert Vector3((1.,2.,3.)).is_int() == False
    assert Vector3((1.,2.,3.)).is_float() == True
    assert Pair((1,2)).is_int() == True
    assert Pair((1,2)).is_float() == False
    assert Pair((1.,2.)).is_int() == False
    assert Pair((1.,2.)).is_float() == True
    assert Quaternion((1,2,3,4)).is_int() == False # coerced to float
    assert Quaternion((1,2,3,4)).is_float() == True
    assert Quaternion((1.,2.,3.,4.)).is_int() == False
    assert Quaternion((1.,2.,3.,4.)).is_float() == True
    assert Matrix([(1,2),(3,4)]).is_int() == False # coerced to float
    assert Matrix([(1,2),(3,4)]).is_float() == True
    assert Matrix([(1.,2.),(3.,4.)]).is_int() == False
    assert Matrix([(1.,2.),(3.,4.)]).is_float() == True

    ##################################################################################
    # as_float(self)
    # as_int(self)
    ##################################################################################
    assert Boolean(True).as_int() == 1
    assert Boolean(False).as_int() == 0
    assert type(Boolean(True).as_int()) == Scalar
    assert type(Boolean(False).as_int()) == Scalar
    assert type(Boolean(True).as_int().values) == int
    assert type(Boolean(False).as_int().values) == int
    assert Boolean(True).as_float() == 1
    assert Boolean(False).as_float() == 0
    assert type(Boolean(True).as_float()) == Scalar
    assert type(Boolean(False).as_float()) == Scalar
    assert type(Boolean(True).as_float().values) == float
    assert type(Boolean(False).as_float().values) == float
    assert Boolean((True,False)).as_int() == (1,0)
    assert type(Boolean((True,False)).as_int()) == Scalar
    assert Boolean((True,False)).as_int().values.dtype == np.dtype('int8')
    assert Boolean((True,False)).as_float() == (1,0)
    assert type(Boolean((True,False)).as_float()) == Scalar
    assert Boolean((True,False)).as_float().values.dtype == np.dtype('float')
    assert type(Scalar(1.).as_int().values) == int
    assert Scalar((1.,2.)).as_int().values.dtype == np.dtype('int64')
    assert Scalar((1.5,-1.5)).as_int() == (1,-2)
    assert type(Scalar(1).as_float().values) == float
    assert Scalar((1,2)).as_float().values.dtype == np.dtype('float')
    assert Vector((1.,2.)).as_int().values.dtype == np.dtype('int64')
    assert Vector((1.5,-1.5)).as_int().values.dtype == np.dtype('int64')
    assert Vector((1,2)).as_float().values.dtype == np.dtype('float')
    assert Pair((1.,2.)).as_int().values.dtype == np.dtype('int64')
    assert Pair((1.5,-1.5)).as_int().values.dtype == np.dtype('int64')
    assert Pair((1,2)).as_float().values.dtype == np.dtype('float')
    with pytest.raises(TypeError):
        Vector3((1.,2.,3.)).as_int()
    with pytest.raises(TypeError):
        Quaternion((1.,2.,3.,4.)).as_int()
    with pytest.raises(TypeError):
        Matrix([(1,0),(0,1)]).as_int()
    with pytest.raises(TypeError):
        Matrix3([(1,0,0),(0,1,0),(0,0,1)]).as_int()

    ##################################################################################
    # masked_single(self)
    ##################################################################################
    a = Scalar((1,2,3))
    assert a.masked_single() == Scalar.MASKED
    assert type(a.masked_single()) == Scalar
    assert a.masked_single().shape == ()
    a = Boolean([True,False])
    assert a.masked_single() == Boolean.MASKED
    assert type(a.masked_single()) == Boolean
    assert a.masked_single().shape == ()
    a = Vector([(1,2,3),(4,5,6)])
    assert a.masked_single() == Vector.MASKED3
    assert type(a.masked_single()) == Vector
    assert a.masked_single().shape == ()
    a = Pair([(1,2),(4,5)])
    assert a.masked_single() == Pair.MASKED
    assert type(a.masked_single()) == Pair
    assert a.masked_single().shape == ()
    a = Vector3([(1,2,3),(4,5,6)])
    assert a.masked_single() == Vector3.MASKED
    assert type(a.masked_single()) == Vector3
    assert a.masked_single().shape == ()
    a = Quaternion([(1,2,3,4),(4,5,6,7)])
    assert a.masked_single() == Quaternion.MASKED
    assert type(a.masked_single()) == Quaternion
    assert a.masked_single().shape == ()
    a = Matrix([(1,2),(4,5)])
    assert a.masked_single() == Matrix.MASKED2
    assert type(a.masked_single()) == Matrix
    assert a.masked_single().shape == ()
    a = Matrix([(1,2,3),(4,5,6),(7,8,9)])
    assert a.masked_single() == Matrix3.MASKED3
    assert type(a.masked_single()) == Matrix
    assert a.masked_single().shape == ()
    a = Matrix3([(1,2,3),(4,5,6),(7,8,9)])
    assert a.masked_single() == Matrix3.MASKED
    assert type(a.masked_single()) == Matrix3
    assert a.masked_single().shape == ()


##########################################################################################
