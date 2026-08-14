##########################################################################################
# tests/test_qube_reshaping.py
##########################################################################################

import numpy as np
import pytest

from polymath import Pair, Qube, Matrix, Scalar, Vector, Vector3


def test_qube_reshaping_reshape_self_shape_recursive_true() -> None:
    """reshape(self, shape, recursive=True)."""

    np.random.seed(2292)

    a = Vector(np.random.randn(3,4,5,2))
    b = a.reshape((3,4,5))
    assert a.shape == (3,4,5)
    assert b.shape == (3,4,5)
    assert a.numer == (2,)
    assert b.numer == (2,)
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Vector
    a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
    b = a.reshape((6,5,4,3,2))
    assert a.shape == (2,3,4,5,6)
    assert b.shape == (6,5,4,3,2)
    assert a.numer == (3,)
    assert b.numer == (3,)
    assert a.denom == (2,)
    assert b.denom == (2,)
    assert type(b) == Vector
    a = Vector(np.random.randn(2,3,4,5,6,3))
    a.insert_deriv('t', Vector(np.random.randn(3,1,5,6,3,2,2), drank=2))
    assert a.shape == (2,3,4,5,6)
    assert a.numer == (3,)
    assert a.denom == ()
    assert a.d_dt.shape == (2,3,4,5,6) # broadcasted!
    assert a.d_dt.numer == (3,)
    assert a.d_dt.denom == (2,2)
    b = a.reshape((6,5,4,3,2), recursive=False)
    assert b.shape == (6,5,4,3,2)
    assert b.numer == (3,)
    assert b.denom == ()
    assert not hasattr(b, 'd_dt')
    assert type(b) == Vector
    b = a.reshape((6,5,4,3,2), recursive=True)
    assert b.shape == (6,5,4,3,2)
    assert b.numer == (3,)
    assert b.denom == ()
    assert b.d_dt.shape == (6,5,4,3,2)
    assert b.d_dt.numer == (3,)
    assert b.d_dt.denom == (2,2)
    assert type(b) == Vector
    a = Vector(np.random.randn(2,3,4,5,6,3))
    assert not a.readonly
    da_dt = Vector(np.random.randn(3,1,5,6,3,2,2), drank=2)
    assert not da_dt.readonly
    a.insert_deriv('t', da_dt)
    assert not a.readonly
    assert da_dt.readonly     # because of broadcast
    assert a.d_dt.readonly
    b = a.reshape((6,5,4,3,2), recursive=True)
    assert not b.readonly
    assert b.d_dt.readonly
    a = Vector(np.random.randn(2,3,4,5,6,3))
    da_dt = Vector(np.random.randn(2,3,4,5,6,3,2,2), drank=2)
    a.insert_deriv('t', da_dt)
    assert not a.readonly
    assert not a.d_dt.readonly
    b = a.reshape((6,5,4,3,2), recursive=True)
    assert not b.readonly
    assert not b.d_dt.readonly
    a.as_readonly()
    assert a.readonly
    assert a.d_dt.readonly
    b = a.reshape((6,5,4,3,2), recursive=True)
    assert b.readonly
    assert b.d_dt.readonly
    a = Vector3(np.random.randn(2,3,4,5,6,3,2), drank=1)
    b = a.reshape((6,5,4,3,2))
    assert type(b) == Vector3

    a = Scalar(np.random.randn(3,4,5), mask=True)
    b = a.reshape((3,4,5))
    assert a.shape == (3,4,5)
    assert b.shape == (3,4,5)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    a = Scalar(np.random.randn(3,4,5), mask=False)
    b = a.reshape((3,4,5))
    assert a.shape == (3,4,5)
    assert b.shape == (3,4,5)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    a = Scalar(np.random.randn(3,4,5), mask=np.random.randn(3,4,5) < 0.)
    b = a.reshape((3,4,5))
    assert a.shape == (3,4,5)
    assert b.shape == (3,4,5)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    assert (abs(a.sum() - b.sum()) < 3.e-15)

    a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
    b = a.flatten()
    assert a.shape == (2,3,4,5,6)
    assert b.shape == (np.prod(a.shape),)
    assert a.numer == (3,)
    assert b.numer == (3,)
    assert a.denom == (2,)
    assert b.denom == (2,)
    assert type(b) == Vector

    a = Vector(np.random.randn(2,3,4,5,6,3))
    a.insert_deriv('t', Vector(np.random.randn(3,1,5,6,3,2,2), drank=2))
    assert a.shape == (2,3,4,5,6)
    assert a.numer == (3,)
    assert a.denom == ()
    assert a.d_dt.shape == (2,3,4,5,6) # broadcasted!
    assert a.d_dt.numer == (3,)
    assert a.d_dt.denom == (2,2)
    assert not a.readonly
    assert a.d_dt.readonly        # because of broadcast
    b = a.reshape((6,5,4,3,2), recursive=False)
    assert b.shape == (6,5,4,3,2)
    assert b.numer == (3,)
    assert b.denom == ()
    assert not hasattr(b, 'd_dt')
    assert type(b) == Vector
    assert not b.readonly
    b = a.reshape((6,5,4,3,2), recursive=True)
    assert b.shape == (6,5,4,3,2)
    assert b.numer == (3,)
    assert b.denom == ()
    assert b.d_dt.shape == (6,5,4,3,2)
    assert b.d_dt.numer == (3,)
    assert b.d_dt.denom == (2,2)
    assert type(b) == Vector
    assert not b.readonly
    assert b.d_dt.readonly    # because of broadcast

    a = a.as_readonly()
    assert a.readonly
    assert a.d_dt.readonly
    b = a.reshape((6,5,4,3,2), recursive=True)
    assert b.readonly
    assert b.d_dt.readonly

    a = Scalar(np.random.randn(3,4,5), mask=True)
    b = a.flatten()
    assert b.shape == (60,)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    a = Scalar(np.random.randn(3,4,5), mask=False)
    b = a.reshape((3,4,5))
    assert a.shape == (3,4,5)
    assert b.shape == (3,4,5)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    a = Scalar(np.random.randn(3,4,5), mask=np.random.randn(3,4,5) < 0.)
    b = a.reshape((3,4,5))
    assert a.shape == (3,4,5)
    assert b.shape == (3,4,5)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    assert (abs(a.sum() - b.sum()) < 3.e-15)

    a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
    b = a.swap_axes(0,1)
    assert a.shape == (2,3,4,5,6)
    assert b.shape == (3,2,4,5,6)
    assert a.numer == (3,)
    assert b.numer == (3,)
    assert a.denom == (2,)
    assert b.denom == (2,)
    assert type(b) == Vector
    assert a[0] == b[:,0]
    assert a[1] == b[:,1]
    a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
    b = a.swap_axes(0,-1)
    assert a.shape == (2,3,4,5,6)
    assert b.shape == (6,3,4,5,2)
    assert a.numer == (3,)
    assert b.numer == (3,)
    assert a.denom == (2,)
    assert b.denom == (2,)
    assert type(b) == Vector
    assert a[0,:,:,:,0] == b[0,:,:,:,0]
    assert a[1,:,:,:,5] == b[5,:,:,:,1]

    a = Vector3(np.random.randn(2,3,4,5,6,3,2), drank=1)
    b = a.swap_axes(0,-1)
    assert type(b) == Vector3

    a = Vector(np.random.randn(2,3,4,5,6,3))
    a.insert_deriv('t', Vector(np.random.randn(3,1,5,6,3,2,2), drank=2))
    assert a.shape == (2,3,4,5,6)
    assert a.numer == (3,)
    assert a.denom == ()
    assert a.d_dt.shape == (2,3,4,5,6) # broadcasted!
    assert a.d_dt.numer == (3,)
    assert a.d_dt.denom == (2,2)
    b = a.swap_axes(0,-1)
    assert a.shape == (2,3,4,5,6)
    assert b.shape == (6,3,4,5,2)
    assert a.numer == (3,)
    assert b.numer == (3,)
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Vector
    assert a[0,:,:,:,0] == b[0,:,:,:,0]
    assert a[1,:,:,:,5] == b[5,:,:,:,1]
    assert a.d_dt.shape == (2,3,4,5,6)
    assert b.d_dt.shape == (6,3,4,5,2)
    assert a.d_dt.numer == (3,)
    assert b.d_dt.numer == (3,)
    assert a.d_dt.denom == (2,2)
    assert b.d_dt.denom == (2,2)
    assert type(b.d_dt) == Vector
    assert a.d_dt[0,:,:,:,0] == b.d_dt[0,:,:,:,0]
    assert a.d_dt[1,:,:,:,5] == b.d_dt[5,:,:,:,1]

    assert not a.readonly
    assert not b.readonly
    assert a.d_dt.readonly        # because of broadcast
    assert b.d_dt.readonly        # because of broadcast
    a = a.as_readonly()
    b = a.swap_axes(0,-1)
    assert a.readonly
    assert b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly

    a = Scalar(np.random.randn(3,4,5), mask=True)
    b = a.swap_axes(0,-1)
    assert b.shape == (5,4,3)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    a = Scalar(np.random.randn(3,4,5), mask=False)
    b = a.swap_axes(0,-1)
    assert a.shape == (3,4,5)
    assert b.shape == (5,4,3)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    a = Scalar(np.random.randn(3,4,5), mask=np.random.randn(3,4,5) < 0.)
    b = a.swap_axes(0,-1)
    assert a.shape == (3,4,5)
    assert b.shape == (5,4,3)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    assert (abs(a.sum() - b.sum()) < 1.e-14)

    a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
    b = a.roll_axis(1)
    assert a.shape == (2,3,4,5,6)
    assert b.shape == (3,2,4,5,6)
    assert a.numer == (3,)
    assert b.numer == (3,)
    assert a.denom == (2,)
    assert b.denom == (2,)
    assert type(b) == Vector
    assert a[0] == b[:,0]
    assert a[1] == b[:,1]
    a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
    b = a.roll_axis(4,1)
    assert a.shape == (2,3,4,5,6)
    assert b.shape == (2,6,3,4,5)
    assert a.numer == (3,)
    assert b.numer == (3,)
    assert a.denom == (2,)
    assert b.denom == (2,)
    assert type(b) == Vector
    assert a[0,:,:,:,0] == b[0,0,:,:,:]
    assert a[0,:,:,:,1] == b[0,1,:,:,:]
    assert a[0,:,:,:,2] == b[0,2,:,:,:]
    assert a[0,:,:,:,3] == b[0,3,:,:,:]
    assert a[0,:,:,:,4] == b[0,4,:,:,:]
    assert a[0,:,:,:,5] == b[0,5,:,:,:]
    assert a[1,:,:,:,0] == b[1,0,:,:,:]
    assert a[1,:,:,:,1] == b[1,1,:,:,:]
    assert a[1,:,:,:,2] == b[1,2,:,:,:]
    assert a[1,:,:,:,3] == b[1,3,:,:,:]
    assert a[1,:,:,:,4] == b[1,4,:,:,:]
    assert a[1,:,:,:,5] == b[1,5,:,:,:]

    a = Vector3(np.random.randn(2,3,4,5,6,3,2), drank=1)
    b = a.roll_axis(3,1)
    assert type(b) == Vector3
    assert b.shape == (2,5,3,4,6)

    a = Vector(np.random.randn(2,3,4,5,6,3))
    a.insert_deriv('t', Vector(np.random.randn(3,1,5,6,3,2,2), drank=2))
    assert a.shape == (2,3,4,5,6)
    assert a.numer == (3,)
    assert a.denom == ()
    assert a.d_dt.shape == (2,3,4,5,6) # broadcasted!
    assert a.d_dt.numer == (3,)
    assert a.d_dt.denom == (2,2)
    b = a.roll_axis(1)
    assert a.shape == (2,3,4,5,6)
    assert b.shape == (3,2,4,5,6)
    assert a.numer == (3,)
    assert b.numer == (3,)
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Vector
    assert a[0,0] == b[0,0]
    assert a[1,0] == b[0,1]
    assert a[0,1] == b[1,0]
    assert a[1,1] == b[1,1]
    assert a[0,2] == b[2,0]
    assert a[1,2] == b[2,1]
    assert a.d_dt.shape == (2,3,4,5,6)
    assert b.d_dt.shape == (3,2,4,5,6)
    assert a.d_dt.numer == (3,)
    assert b.d_dt.numer == (3,)
    assert a.d_dt.denom == (2,2)
    assert b.d_dt.denom == (2,2)
    assert type(b.d_dt) == Vector
    assert a.d_dt[0,0] == b.d_dt[0,0]
    assert a.d_dt[1,0] == b.d_dt[0,1]
    assert a.d_dt[0,1] == b.d_dt[1,0]
    assert a.d_dt[1,1] == b.d_dt[1,1]
    assert a.d_dt[0,2] == b.d_dt[2,0]
    assert a.d_dt[1,2] == b.d_dt[2,1]

    assert not a.readonly
    assert not b.readonly
    assert a.d_dt.readonly        # because of broadcast
    assert b.d_dt.readonly        # because of broadcast
    a = a.as_readonly()
    b = a.roll_axis(0,-1)
    assert a.readonly
    assert b.readonly
    assert a.d_dt.readonly
    assert b.d_dt.readonly

    a = Scalar(np.random.randn(2,4,3))
    a.insert_deriv('t', Scalar(np.random.randn(3,2), drank=1))
    assert a.shape == (2,4,3)
    assert a.numer == ()
    assert a.denom == ()
    assert a.rank == 0
    assert a.d_dt.shape == (2,4,3)    # broadcasted!
    assert a.d_dt.numer == ()
    assert a.d_dt.denom == (2,)
    assert a.d_dt.rank == 1
    b = a.roll_axis(-2,0,recursive=True,rank=4)
    assert b.shape == (4,1,2,3)
    assert b.numer == ()
    assert b.denom == ()
    assert type(b) == Scalar
    assert a[...,0,:] == b[0]
    assert a[...,1,:] == b[1]
    assert a[...,2,:] == b[2]
    assert a[...,3,:] == b[3]
    assert b.d_dt.shape == (4,1,2,3)
    assert b.d_dt.numer == ()
    assert b.d_dt.denom == (2,)
    assert type(b.d_dt) == Scalar
    assert a.d_dt[...,0,:] == b.d_dt[0]
    assert a.d_dt[...,1,:] == b.d_dt[1]
    assert a.d_dt[...,2,:] == b.d_dt[2]
    assert a.d_dt[...,3,:] == b.d_dt[3]

    a = Scalar(np.random.randn(3,4,5), mask=True)
    b = a.roll_axis(-1)
    assert a.shape == (3,4,5)
    assert b.shape == (5,3,4)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    a = Scalar(np.random.randn(3,4,5), mask=False)
    b = a.roll_axis(-1)
    assert a.shape == (3,4,5)
    assert b.shape == (5,3,4)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    a = Scalar(np.random.randn(3,4,5), mask=np.random.randn(3,4,5) < 0.)
    b = a.roll_axis(-1)
    assert a.shape == (3,4,5)
    assert b.shape == (5,3,4)
    assert a.numer == ()
    assert b.numer == ()
    assert a.denom == ()
    assert b.denom == ()
    assert type(b) == Scalar
    assert (abs(a.sum() - b.sum()) < 5.e-15)

    a = Matrix(np.random.randn(3,1,4,3,2), drank=1)
    assert a.shape == (3,1)
    b = a.broadcast_into_shape((4,3,2))
    assert a[:,0] == b[0,:,0]
    assert a[:,0] == b[3,:,1]
    assert a.readonly     # Because of broadcast of b
    assert b.readonly
    a = Matrix(np.random.randn(3,1,4,3,2), drank=1)
    a.insert_deriv('t', Matrix(np.random.randn(3,1,4,3,2,2), drank=2))
    assert not a.readonly
    assert not a.d_dt.readonly
    b = a.broadcast_into_shape((4,3,2), recursive=False)
    assert a.readonly         # because of broadcast of b
    assert b.readonly         # because of broadcast
    assert not hasattr(b, 'd_dt')
    b = a.broadcast_into_shape((4,3,2), recursive=True)
    assert b.readonly         # because of broadcast
    assert b.d_dt.readonly    # because of broadcast
    a = a.as_readonly()
    assert a.readonly
    assert a.d_dt.readonly
    b = a.broadcast_into_shape((4,3,2), recursive=False)
    assert b.readonly
    assert not hasattr(b, 'd_dt')
    b = a.broadcast_into_shape((4,3,2), recursive=True)
    assert b.readonly
    assert b.d_dt.readonly

    a = Scalar(np.random.randn(2,1,4,1,3,1,3, 2,2), drank=2)
    b = Vector(np.random.randn(  7,4,1,3,7,3, 3))
    c = Matrix(np.random.randn(      4,1,1,1, 3,3,5), drank=1)
    assert Qube.broadcasted_shape(b,c) == (7,4,4,3,7,3)
    assert Qube.broadcasted_shape(b,c,item=(2,)) == (7,4,4,3,7,3,2)
    assert Qube.broadcasted_shape(a,b) == (2,7,4,1,3,7,3)
    assert Qube.broadcasted_shape(a,b,None) == (2,7,4,1,3,7,3)
    assert Qube.broadcasted_shape(a,b,()) == (2,7,4,1,3,7,3)
    assert Qube.broadcasted_shape(a,b,item=(2,)) == (2,7,4,1,3,7,3,2)
    assert Qube.broadcasted_shape(a,c) == (2,1,4,4,3,1,3)
    assert Qube.broadcasted_shape(a,b,c) == (2,7,4,4,3,7,3)
    assert Qube.broadcasted_shape(c,(2,2,2)) == (4,2,2,2)
    with pytest.raises(ValueError):
        Qube.broadcasted_shape(c, (5,2,2,2))
    assert Qube.broadcasted_shape(a,b,c,(),None,(3,),item=(2,2)) == (2,7,4,4,3,7,3,2,2)

    a = Scalar(np.random.randn(2,1,1,3, 2,2), drank=2)
    b = Pair(np.random.randn(    3,1,1, 2))
    c = Matrix(np.random.randn(    4,1, 3,3))
    e = np.array(np.random.randn(3,4,3))
    f = None
    b.insert_deriv('t', Pair(np.random.randn(2,2), drank=1))
    assert b.d_dt.shape == (3,1,1)
    assert b.d_dt.readonly
    (aa,bb,cc,ee,ff) = Qube.broadcast(a,b,c,e,f,recursive=False)
    assert aa.shape == (2,3,4,3)
    assert bb.shape == (2,3,4,3)
    assert cc.shape == (2,3,4,3)
    assert ee.shape == (2,3,4,3)
    assert ff == None
    assert aa.readonly
    assert bb.readonly
    assert cc.readonly
    assert not hasattr(bb, 'd_dt')
    (aa,bb,cc,ee,ff) = Qube.broadcast(a,b,c,e,f,recursive=True)
    assert bb.d_dt.shape == (2,3,4,3)
    assert bb.d_dt.readonly

    # Additional coverage tests for missing lines

    a = Vector([[1., 2., 3.]])  # shape (1,), rank 1
    b = a.broadcast_to(())
    assert b.shape == ()
    assert np.allclose(b.values, [1., 2., 3.])

    a = Scalar([5.])  # shape (1,), _values is ndarray

    original_values = a._values
    a._values = float(original_values[0])  # Convert to Python float
    a._is_array = False
    a._is_scalar = True

    b = a.broadcast_to(())
    assert b.shape == ()
    assert b.values == 5.
    assert isinstance(b.values, (float, int))

    a = Scalar([1., 2., 3.])

    if not isinstance(a._mask, np.ndarray):
        a._mask = np.array([False, True, False])
    b = a.broadcast_to(())
    assert b.shape == ()
    assert b.values == 1.  # First element
    assert isinstance(b.mask, bool)

    a = Scalar(np.arange(12).reshape(3, 4))
    b = a.reshape([6, 2])
    assert b.shape == (6, 2)
    c = a.reshape(12)
    assert c.shape == (12,)

    a = Scalar(np.arange(12).reshape(3, 4))
    b = a.swap_axes(0, 0)
    assert a == b
    b = a.swap_axes(1, 1)
    assert a == b

    a = Scalar(np.arange(12).reshape(3, 4))
    with pytest.raises(ValueError) as cm:
        a.roll_axis(0, 0, rank=1)
    assert 'rank 1 is too small for shape' in str(cm.value)

    a = Scalar(np.arange(12).reshape(3, 4))
    b = a.roll_axis(1, 2)
    assert b.shape == (3, 4)
    a = Scalar(np.arange(12).reshape(3, 4))
    b = a.roll_axis(1, 0)
    assert b.shape == (4, 3)

    a = Scalar(np.arange(12).reshape(3, 4))
    with pytest.raises(ValueError) as cm:
        a.move_axis(0, 1, rank=1)
    assert 'rank 1 is too small for shape' in str(cm.value)


def test_qube_reshaping_move_axis_with_scalar_source_destination() -> None:
    """move_axis with scalar source/destination."""

    np.random.seed(2292)

    a = Scalar(np.arange(12).reshape(3, 4))
    b = a.move_axis(0, 1)
    assert b.shape == (4, 3)
    b = a.move_axis(1, 0)
    assert b.shape == (4, 3)


def test_qube_reshaping_move_axis_reshape_when_ndims_rank_when_rank_3_and_object_has() -> None:
    """move_axis reshape when ndims < rank # When rank=3 and object has shape (3, 4), it gets reshaped to (1, 3, 4) # Then moving axis 0 to position 2 results in (3, 4, 1)."""

    np.random.seed(2292)

    a = Scalar(np.arange(12).reshape(3, 4))
    b = a.move_axis(0, 2, rank=3)
    assert b.shape == (3, 4, 1)


def test_qube_reshaping_stack_function_various_paths() -> None:
    """stack function various paths."""

    np.random.seed(2292)

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)
    assert np.allclose(c.values[0], [1., 2., 3.])
    assert np.allclose(c.values[1], [4., 5., 6.])


def test_qube_reshaping_stack_with_none_args() -> None:
    """stack with None args."""

    np.random.seed(2292)

    a = Scalar([1., 2., 3.])
    b = None
    c = Scalar([4., 5., 6.])
    result = Qube.stack(a, b, c)
    assert result.shape == (3, 3)
    assert np.allclose(result.values[0], [1., 2., 3.])
    assert np.allclose(result.values[1], [0., 0., 0.])
    assert np.allclose(result.values[2], [4., 5., 6.])


def test_qube_reshaping_stack_with_derivatives() -> None:
    """stack with derivatives."""

    np.random.seed(2292)

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([10., 20., 30.]))
    b = Scalar([4., 5., 6.])
    b.insert_deriv('t', Scalar([40., 50., 60.]))
    c = Qube.stack(a, b, recursive=True)
    assert hasattr(c, 'd_dt')
    assert c.d_dt.shape == (2, 3)
    assert np.allclose(c.d_dt.values[0], [10., 20., 30.])
    assert np.allclose(c.d_dt.values[1], [40., 50., 60.])


def test_qube_reshaping_stack_with_mixed_types() -> None:
    """stack with mixed types."""

    np.random.seed(2292)

    a = Scalar([1., 2., 3.])
    b = Scalar([4, 5, 6])
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)


def test_qube_reshaping_stack_with_units() -> None:
    """stack with units."""

    np.random.seed(2292)

    from polymath.unit import Unit
    a = Scalar([1., 2., 3.], unit=Unit.KM)
    b = Scalar([4., 5., 6.], unit=Unit.KM)
    c = Qube.stack(a, b)
    assert c._unit == Unit.KM


def test_qube_reshaping_test_move_axis_with_recursive_true_and_derivatives() -> None:
    """Test move_axis with recursive=True and derivatives."""

    np.random.seed(2292)

    a = Scalar(np.arange(12).reshape(3, 4))
    a.insert_deriv('t', Scalar(np.arange(12).reshape(3, 4)))
    b = a.move_axis(0, 1, recursive=True)
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == (4, 3)


def test_qube_reshaping_test_stack_with_float_arg_logic_float_arg_is_none_or_not_qub() -> None:
    """Test stack with float_arg logic (float_arg is None or not qubed) # Case: float_arg is None."""

    np.random.seed(2292)

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)


def test_qube_reshaping_case_float_arg_is_not_none_but_qubed_is_true_arg_was_convert() -> None:
    """Case: float_arg is not None but qubed is True (arg was converted)."""

    np.random.seed(2292)

    a = np.array([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)


def test_qube_reshaping_test_stack_with_int_arg_logic_int_arg_is_none_or_not_qubed_c() -> None:
    """Test stack with int_arg logic (int_arg is None or not qubed) # Case: int_arg is None, float_arg is None."""

    np.random.seed(2292)

    a = Scalar([1, 2, 3])
    b = Scalar([4, 5, 6])
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)


def test_qube_reshaping_case_int_arg_is_not_none_but_qubed_is_true() -> None:
    """Case: int_arg is not None but qubed is True."""

    np.random.seed(2292)

    a = np.array([1, 2, 3])
    b = Scalar([4, 5, 6])
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)


def test_qube_reshaping_test_stack_with_bool_arg_logic_bool_arg_is_none_or_not_qubed() -> None:
    """Test stack with bool_arg logic (bool_arg is None or not qubed) # Case: bool_arg is None, int_arg is None, float_arg is None."""

    np.random.seed(2292)

    from polymath.boolean import Boolean
    a = Boolean([True, False, True])
    b = Boolean([False, True, False])
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)

    a = np.array([True, False, True])
    b = Boolean([False, True, False])
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)

    a = Scalar([1., 2., 3.])
    b = np.array([4., 5., 6.])  # This will be converted (qubed=True)
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)

    a = Scalar([1, 2, 3])
    b = np.array([4, 5, 6])  # This will be converted (qubed=True)
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)

    a = Boolean([True, False, True])
    b = np.array([False, True, False])  # This will be converted (qubed=True)
    c = Qube.stack(a, b)
    assert c.shape == (2, 3)


##########################################################################################
