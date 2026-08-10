##########################################################################################
# tests/test_qube_reshaping.py
##########################################################################################

import numpy as np
import unittest

from polymath import Pair, Qube, Matrix, Scalar, Vector, Vector3


class Test_qube_reshaping(unittest.TestCase):

    def runTest(self):

        np.random.seed(2292)

        # reshape(self, shape, recursive=True)
        a = Vector(np.random.randn(3,4,5,2))
        b = a.reshape((3,4,5))
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (3,4,5))
        self.assertEqual(a.numer, (2,))
        self.assertEqual(b.numer, (2,))
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Vector)

        a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
        b = a.reshape((6,5,4,3,2))
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(b.shape, (6,5,4,3,2))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(a.denom, (2,))
        self.assertEqual(b.denom, (2,))
        self.assertEqual(type(b), Vector)

        a = Vector(np.random.randn(2,3,4,5,6,3))
        a.insert_deriv('t', Vector(np.random.randn(3,1,5,6,3,2,2), drank=2))
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(a.denom, ())
        self.assertEqual(a.d_dt.shape, (2,3,4,5,6)) # broadcasted!
        self.assertEqual(a.d_dt.numer, (3,))
        self.assertEqual(a.d_dt.denom, (2,2))

        b = a.reshape((6,5,4,3,2), recursive=False)
        self.assertEqual(b.shape, (6,5,4,3,2))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(b.denom, ())
        self.assertFalse(hasattr(b, 'd_dt'))
        self.assertEqual(type(b), Vector)

        b = a.reshape((6,5,4,3,2), recursive=True)
        self.assertEqual(b.shape, (6,5,4,3,2))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(b.denom, ())
        self.assertEqual(b.d_dt.shape, (6,5,4,3,2))
        self.assertEqual(b.d_dt.numer, (3,))
        self.assertEqual(b.d_dt.denom, (2,2))
        self.assertEqual(type(b), Vector)

        a = Vector(np.random.randn(2,3,4,5,6,3))
        self.assertFalse(a.readonly)

        da_dt = Vector(np.random.randn(3,1,5,6,3,2,2), drank=2)
        self.assertFalse(da_dt.readonly)

        a.insert_deriv('t', da_dt)
        self.assertFalse(a.readonly)
        self.assertTrue(da_dt.readonly)     # because of broadcast
        self.assertTrue(a.d_dt.readonly)

        b = a.reshape((6,5,4,3,2), recursive=True)
        self.assertFalse(b.readonly)
        self.assertTrue(b.d_dt.readonly)

        a = Vector(np.random.randn(2,3,4,5,6,3))
        da_dt = Vector(np.random.randn(2,3,4,5,6,3,2,2), drank=2)
        a.insert_deriv('t', da_dt)
        self.assertFalse(a.readonly)
        self.assertFalse(a.d_dt.readonly)

        b = a.reshape((6,5,4,3,2), recursive=True)
        self.assertFalse(b.readonly)
        self.assertFalse(b.d_dt.readonly)

        a.as_readonly()
        self.assertTrue(a.readonly)
        self.assertTrue(a.d_dt.readonly)

        b = a.reshape((6,5,4,3,2), recursive=True)
        self.assertTrue(b.readonly)
        self.assertTrue(b.d_dt.readonly)

        a = Vector3(np.random.randn(2,3,4,5,6,3,2), drank=1)
        b = a.reshape((6,5,4,3,2))
        self.assertEqual(type(b), Vector3)

        # With mask
        a = Scalar(np.random.randn(3,4,5), mask=True)
        b = a.reshape((3,4,5))
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (3,4,5))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        a = Scalar(np.random.randn(3,4,5), mask=False)
        b = a.reshape((3,4,5))
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (3,4,5))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        a = Scalar(np.random.randn(3,4,5), mask=np.random.randn(3,4,5) < 0.)
        b = a.reshape((3,4,5))
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (3,4,5))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        self.assertTrue(abs(a.sum() - b.sum()) < 3.e-15)

        # flatten(self, recursive=True)
        a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
        b = a.flatten()
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(b.shape, (np.prod(a.shape),))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(a.denom, (2,))
        self.assertEqual(b.denom, (2,))
        self.assertEqual(type(b), Vector)

        # Derivatives & read-only status
        a = Vector(np.random.randn(2,3,4,5,6,3))
        a.insert_deriv('t', Vector(np.random.randn(3,1,5,6,3,2,2), drank=2))
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(a.denom, ())
        self.assertEqual(a.d_dt.shape, (2,3,4,5,6)) # broadcasted!
        self.assertEqual(a.d_dt.numer, (3,))
        self.assertEqual(a.d_dt.denom, (2,2))
        self.assertFalse(a.readonly)
        self.assertTrue(a.d_dt.readonly)        # because of broadcast

        b = a.reshape((6,5,4,3,2), recursive=False)
        self.assertEqual(b.shape, (6,5,4,3,2))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(b.denom, ())
        self.assertFalse(hasattr(b, 'd_dt'))
        self.assertEqual(type(b), Vector)
        self.assertFalse(b.readonly)

        b = a.reshape((6,5,4,3,2), recursive=True)
        self.assertEqual(b.shape, (6,5,4,3,2))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(b.denom, ())
        self.assertEqual(b.d_dt.shape, (6,5,4,3,2))
        self.assertEqual(b.d_dt.numer, (3,))
        self.assertEqual(b.d_dt.denom, (2,2))
        self.assertEqual(type(b), Vector)
        self.assertFalse(b.readonly)
        self.assertTrue(b.d_dt.readonly)    # because of broadcast

        # Readonly status
        a = a.as_readonly()
        self.assertTrue(a.readonly)
        self.assertTrue(a.d_dt.readonly)

        b = a.reshape((6,5,4,3,2), recursive=True)
        self.assertTrue(b.readonly)
        self.assertTrue(b.d_dt.readonly)

        # With mask
        a = Scalar(np.random.randn(3,4,5), mask=True)
        b = a.flatten()
        self.assertEqual(b.shape, (60,))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        a = Scalar(np.random.randn(3,4,5), mask=False)
        b = a.reshape((3,4,5))
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (3,4,5))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        a = Scalar(np.random.randn(3,4,5), mask=np.random.randn(3,4,5) < 0.)
        b = a.reshape((3,4,5))
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (3,4,5))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        self.assertTrue(abs(a.sum() - b.sum()) < 3.e-15)

        # swap_axes(self, axis1, axis2, recursive=True)
        a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
        b = a.swap_axes(0,1)
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(b.shape, (3,2,4,5,6))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(a.denom, (2,))
        self.assertEqual(b.denom, (2,))
        self.assertEqual(type(b), Vector)

        self.assertEqual(a[0], b[:,0])
        self.assertEqual(a[1], b[:,1])

        a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
        b = a.swap_axes(0,-1)
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(b.shape, (6,3,4,5,2))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(a.denom, (2,))
        self.assertEqual(b.denom, (2,))
        self.assertEqual(type(b), Vector)

        self.assertEqual(a[0,:,:,:,0], b[0,:,:,:,0])
        self.assertEqual(a[1,:,:,:,5], b[5,:,:,:,1])

        # Try a different subclass
        a = Vector3(np.random.randn(2,3,4,5,6,3,2), drank=1)
        b = a.swap_axes(0,-1)
        self.assertEqual(type(b), Vector3)

        # Derivatives
        a = Vector(np.random.randn(2,3,4,5,6,3))
        a.insert_deriv('t', Vector(np.random.randn(3,1,5,6,3,2,2), drank=2))
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(a.denom, ())
        self.assertEqual(a.d_dt.shape, (2,3,4,5,6)) # broadcasted!
        self.assertEqual(a.d_dt.numer, (3,))
        self.assertEqual(a.d_dt.denom, (2,2))

        b = a.swap_axes(0,-1)
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(b.shape, (6,3,4,5,2))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Vector)

        self.assertEqual(a[0,:,:,:,0], b[0,:,:,:,0])
        self.assertEqual(a[1,:,:,:,5], b[5,:,:,:,1])

        self.assertEqual(a.d_dt.shape, (2,3,4,5,6))
        self.assertEqual(b.d_dt.shape, (6,3,4,5,2))
        self.assertEqual(a.d_dt.numer, (3,))
        self.assertEqual(b.d_dt.numer, (3,))
        self.assertEqual(a.d_dt.denom, (2,2))
        self.assertEqual(b.d_dt.denom, (2,2))
        self.assertEqual(type(b.d_dt), Vector)

        self.assertEqual(a.d_dt[0,:,:,:,0], b.d_dt[0,:,:,:,0])
        self.assertEqual(a.d_dt[1,:,:,:,5], b.d_dt[5,:,:,:,1])

        # Read-only status
        self.assertFalse(a.readonly)
        self.assertFalse(b.readonly)
        self.assertTrue(a.d_dt.readonly)        # because of broadcast
        self.assertTrue(b.d_dt.readonly)        # because of broadcast

        a = a.as_readonly()
        b = a.swap_axes(0,-1)
        self.assertTrue(a.readonly)
        self.assertTrue(b.readonly)
        self.assertTrue(a.d_dt.readonly)
        self.assertTrue(b.d_dt.readonly)

        # With mask
        a = Scalar(np.random.randn(3,4,5), mask=True)
        b = a.swap_axes(0,-1)
        self.assertEqual(b.shape, (5,4,3))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        a = Scalar(np.random.randn(3,4,5), mask=False)
        b = a.swap_axes(0,-1)
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (5,4,3))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        a = Scalar(np.random.randn(3,4,5), mask=np.random.randn(3,4,5) < 0.)
        b = a.swap_axes(0,-1)
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (5,4,3))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        self.assertTrue(abs(a.sum() - b.sum()) < 1.e-14)

        # roll_axis(self, axis, start, recursive=True, rank=None)
        a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
        b = a.roll_axis(1)
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(b.shape, (3,2,4,5,6))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(a.denom, (2,))
        self.assertEqual(b.denom, (2,))
        self.assertEqual(type(b), Vector)

        self.assertEqual(a[0], b[:,0])
        self.assertEqual(a[1], b[:,1])

        a = Vector(np.random.randn(2,3,4,5,6,3,2), drank=1)
        b = a.roll_axis(4,1)
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(b.shape, (2,6,3,4,5))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(a.denom, (2,))
        self.assertEqual(b.denom, (2,))
        self.assertEqual(type(b), Vector)

        self.assertEqual(a[0,:,:,:,0], b[0,0,:,:,:])
        self.assertEqual(a[0,:,:,:,1], b[0,1,:,:,:])
        self.assertEqual(a[0,:,:,:,2], b[0,2,:,:,:])
        self.assertEqual(a[0,:,:,:,3], b[0,3,:,:,:])
        self.assertEqual(a[0,:,:,:,4], b[0,4,:,:,:])
        self.assertEqual(a[0,:,:,:,5], b[0,5,:,:,:])

        self.assertEqual(a[1,:,:,:,0], b[1,0,:,:,:])
        self.assertEqual(a[1,:,:,:,1], b[1,1,:,:,:])
        self.assertEqual(a[1,:,:,:,2], b[1,2,:,:,:])
        self.assertEqual(a[1,:,:,:,3], b[1,3,:,:,:])
        self.assertEqual(a[1,:,:,:,4], b[1,4,:,:,:])
        self.assertEqual(a[1,:,:,:,5], b[1,5,:,:,:])

        # Try a different subclass
        a = Vector3(np.random.randn(2,3,4,5,6,3,2), drank=1)
        b = a.roll_axis(3,1)
        self.assertEqual(type(b), Vector3)
        self.assertEqual(b.shape, (2,5,3,4,6))

        # Derivatives
        a = Vector(np.random.randn(2,3,4,5,6,3))
        a.insert_deriv('t', Vector(np.random.randn(3,1,5,6,3,2,2), drank=2))
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(a.denom, ())
        self.assertEqual(a.d_dt.shape, (2,3,4,5,6)) # broadcasted!
        self.assertEqual(a.d_dt.numer, (3,))
        self.assertEqual(a.d_dt.denom, (2,2))

        b = a.roll_axis(1)
        self.assertEqual(a.shape, (2,3,4,5,6))
        self.assertEqual(b.shape, (3,2,4,5,6))
        self.assertEqual(a.numer, (3,))
        self.assertEqual(b.numer, (3,))
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Vector)

        self.assertEqual(a[0,0], b[0,0])
        self.assertEqual(a[1,0], b[0,1])
        self.assertEqual(a[0,1], b[1,0])
        self.assertEqual(a[1,1], b[1,1])
        self.assertEqual(a[0,2], b[2,0])
        self.assertEqual(a[1,2], b[2,1])

        self.assertEqual(a.d_dt.shape, (2,3,4,5,6))
        self.assertEqual(b.d_dt.shape, (3,2,4,5,6))
        self.assertEqual(a.d_dt.numer, (3,))
        self.assertEqual(b.d_dt.numer, (3,))
        self.assertEqual(a.d_dt.denom, (2,2))
        self.assertEqual(b.d_dt.denom, (2,2))
        self.assertEqual(type(b.d_dt), Vector)

        self.assertEqual(a.d_dt[0,0], b.d_dt[0,0])
        self.assertEqual(a.d_dt[1,0], b.d_dt[0,1])
        self.assertEqual(a.d_dt[0,1], b.d_dt[1,0])
        self.assertEqual(a.d_dt[1,1], b.d_dt[1,1])
        self.assertEqual(a.d_dt[0,2], b.d_dt[2,0])
        self.assertEqual(a.d_dt[1,2], b.d_dt[2,1])

        # Read-only status
        self.assertFalse(a.readonly)
        self.assertFalse(b.readonly)
        self.assertTrue(a.d_dt.readonly)        # because of broadcast
        self.assertTrue(b.d_dt.readonly)        # because of broadcast

        a = a.as_readonly()
        b = a.roll_axis(0,-1)
        self.assertTrue(a.readonly)
        self.assertTrue(b.readonly)
        self.assertTrue(a.d_dt.readonly)
        self.assertTrue(b.d_dt.readonly)

        # Rank
        a = Scalar(np.random.randn(2,4,3))
        a.insert_deriv('t', Scalar(np.random.randn(3,2), drank=1))
        self.assertEqual(a.shape, (2,4,3))
        self.assertEqual(a.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(a.rank,  0)
        self.assertEqual(a.d_dt.shape, (2,4,3))    # broadcasted!
        self.assertEqual(a.d_dt.numer, ())
        self.assertEqual(a.d_dt.denom, (2,))
        self.assertEqual(a.d_dt.rank,  1)

        b = a.roll_axis(-2,0,recursive=True,rank=4)
        self.assertEqual(b.shape, (4,1,2,3))
        self.assertEqual(b.numer, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        self.assertEqual(a[...,0,:], b[0])
        self.assertEqual(a[...,1,:], b[1])
        self.assertEqual(a[...,2,:], b[2])
        self.assertEqual(a[...,3,:], b[3])

        self.assertEqual(b.d_dt.shape, (4,1,2,3))
        self.assertEqual(b.d_dt.numer, ())
        self.assertEqual(b.d_dt.denom, (2,))
        self.assertEqual(type(b.d_dt), Scalar)

        self.assertEqual(a.d_dt[...,0,:], b.d_dt[0])
        self.assertEqual(a.d_dt[...,1,:], b.d_dt[1])
        self.assertEqual(a.d_dt[...,2,:], b.d_dt[2])
        self.assertEqual(a.d_dt[...,3,:], b.d_dt[3])

        # With mask
        a = Scalar(np.random.randn(3,4,5), mask=True)
        b = a.roll_axis(-1)
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (5,3,4))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        a = Scalar(np.random.randn(3,4,5), mask=False)
        b = a.roll_axis(-1)
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (5,3,4))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        a = Scalar(np.random.randn(3,4,5), mask=np.random.randn(3,4,5) < 0.)
        b = a.roll_axis(-1)
        self.assertEqual(a.shape, (3,4,5))
        self.assertEqual(b.shape, (5,3,4))
        self.assertEqual(a.numer, ())
        self.assertEqual(b.numer, ())
        self.assertEqual(a.denom, ())
        self.assertEqual(b.denom, ())
        self.assertEqual(type(b), Scalar)

        self.assertTrue(abs(a.sum() - b.sum()) < 5.e-15)

        # broadcast_into_shape(self, shape, recursive=True, sample_array=None)
        a = Matrix(np.random.randn(3,1,4,3,2), drank=1)
        self.assertEqual(a.shape, (3,1))
        b = a.broadcast_into_shape((4,3,2))

        self.assertEqual(a[:,0], b[0,:,0])
        self.assertEqual(a[:,0], b[3,:,1])

        self.assertTrue(a.readonly)     # Because of broadcast of b
        self.assertTrue(b.readonly)

        a = Matrix(np.random.randn(3,1,4,3,2), drank=1)
        a.insert_deriv('t', Matrix(np.random.randn(3,1,4,3,2,2), drank=2))
        self.assertFalse(a.readonly)
        self.assertFalse(a.d_dt.readonly)

        b = a.broadcast_into_shape((4,3,2), recursive=False)
        self.assertTrue(a.readonly)         # because of broadcast of b
        self.assertTrue(b.readonly)         # because of broadcast
        self.assertFalse(hasattr(b, 'd_dt'))

        b = a.broadcast_into_shape((4,3,2), recursive=True)
        self.assertTrue(b.readonly)         # because of broadcast
        self.assertTrue(b.d_dt.readonly)    # because of broadcast

        a = a.as_readonly()
        self.assertTrue(a.readonly)
        self.assertTrue(a.d_dt.readonly)

        b = a.broadcast_into_shape((4,3,2), recursive=False)
        self.assertTrue(b.readonly)
        self.assertFalse(hasattr(b, 'd_dt'))

        b = a.broadcast_into_shape((4,3,2), recursive=True)
        self.assertTrue(b.readonly)
        self.assertTrue(b.d_dt.readonly)

        # broadcasted_shape(*objects, item=())
        a = Scalar(np.random.randn(2,1,4,1,3,1,3, 2,2), drank=2)
        b = Vector(np.random.randn(  7,4,1,3,7,3, 3))
        c = Matrix(np.random.randn(      4,1,1,1, 3,3,5), drank=1)

        self.assertEqual(Qube.broadcasted_shape(b,c), (7,4,4,3,7,3))
        self.assertEqual(Qube.broadcasted_shape(b,c,item=(2,)), (7,4,4,3,7,3,2))

        self.assertEqual(Qube.broadcasted_shape(a,b), (2,7,4,1,3,7,3))
        self.assertEqual(Qube.broadcasted_shape(a,b,None), (2,7,4,1,3,7,3))
        self.assertEqual(Qube.broadcasted_shape(a,b,()), (2,7,4,1,3,7,3))
        self.assertEqual(Qube.broadcasted_shape(a,b,item=(2,)), (2,7,4,1,3,7,3,2))

        self.assertEqual(Qube.broadcasted_shape(a,c), (2,1,4,4,3,1,3))

        self.assertEqual(Qube.broadcasted_shape(a,b,c), (2,7,4,4,3,7,3))

        self.assertEqual(Qube.broadcasted_shape(c,(2,2,2)), (4,2,2,2))

        self.assertRaises(ValueError, Qube.broadcasted_shape, c, (5,2,2,2))

        self.assertEqual(Qube.broadcasted_shape(a,b,c,(),None,(3,),item=(2,2)),
                                                (2,7,4,4,3,7,3,2,2))

        # broadcast(*objects, recursive=True)
        a = Scalar(np.random.randn(2,1,1,3, 2,2), drank=2)
        b = Pair(np.random.randn(    3,1,1, 2))
        c = Matrix(np.random.randn(    4,1, 3,3))
        e = np.array(np.random.randn(3,4,3))
        f = None

        b.insert_deriv('t', Pair(np.random.randn(2,2), drank=1))
        self.assertEqual(b.d_dt.shape, (3,1,1))
        self.assertTrue(b.d_dt.readonly)

        (aa,bb,cc,ee,ff) = Qube.broadcast(a,b,c,e,f,recursive=False)

        self.assertEqual(aa.shape, (2,3,4,3))
        self.assertEqual(bb.shape, (2,3,4,3))
        self.assertEqual(cc.shape, (2,3,4,3))
        self.assertEqual(ee.shape, (2,3,4,3))
        self.assertEqual(ff, None)

        self.assertTrue(aa.readonly)
        self.assertTrue(bb.readonly)
        self.assertTrue(cc.readonly)

        self.assertFalse(hasattr(bb, 'd_dt'))

        (aa,bb,cc,ee,ff) = Qube.broadcast(a,b,c,e,f,recursive=True)
        self.assertEqual(bb.d_dt.shape, (2,3,4,3))
        self.assertTrue(bb.d_dt.readonly)

        # Additional coverage tests for missing lines

        # Test broadcast_to with shape () for rank > 0
        a = Vector([[1., 2., 3.]])  # shape (1,), rank 1
        b = a.broadcast_to(())
        self.assertEqual(b.shape, ())
        self.assertTrue(np.allclose(b.values, [1., 2., 3.]))

        # Test broadcast_to with shape () for rank 0 with non-ndarray values
        # Create a Scalar with shape (1,) but manually set _values to Python float
        a = Scalar([5.])  # shape (1,), _values is ndarray
        # Manually manipulate to create edge case: shape != () but _values is Python scalar
        # This tests the else branch at line 69
        original_values = a._values
        a._values = float(original_values[0])  # Convert to Python float
        a._is_array = False
        a._is_scalar = True
        # Now a has shape (1,) but _values is a Python float
        b = a.broadcast_to(())
        self.assertEqual(b.shape, ())
        self.assertEqual(b.values, 5.)
        self.assertIsInstance(b.values, (float, int))

        # Test broadcast_to with shape () and array mask
        a = Scalar([1., 2., 3.])
        # Ensure mask is an array
        if not isinstance(a._mask, np.ndarray):
            a._mask = np.array([False, True, False])
        b = a.broadcast_to(())
        self.assertEqual(b.shape, ())
        self.assertEqual(b.values, 1.)  # First element
        self.assertIsInstance(b.mask, bool)

        # reshape with non-tuple shape
        a = Scalar(np.arange(12).reshape(3, 4))
        b = a.reshape([6, 2])
        self.assertEqual(b.shape, (6, 2))
        c = a.reshape(12)
        self.assertEqual(c.shape, (12,))

        # swap_axes when a1 == a2
        a = Scalar(np.arange(12).reshape(3, 4))
        b = a.swap_axes(0, 0)
        self.assertEqual(a, b)
        b = a.swap_axes(1, 1)
        self.assertEqual(a, b)

        # roll_axis ValueError for rank too small
        a = Scalar(np.arange(12).reshape(3, 4))
        with self.assertRaises(ValueError) as cm:
            a.roll_axis(0, 0, rank=1)
        self.assertIn('rank 1 is too small for shape', str(cm.exception))

        # roll_axis when start != rank
        a = Scalar(np.arange(12).reshape(3, 4))
        b = a.roll_axis(1, 2)
        self.assertEqual(b.shape, (3, 4))
        a = Scalar(np.arange(12).reshape(3, 4))
        b = a.roll_axis(1, 0)
        self.assertEqual(b.shape, (4, 3))

        # move_axis ValueError for rank too small
        a = Scalar(np.arange(12).reshape(3, 4))
        with self.assertRaises(ValueError) as cm:
            a.move_axis(0, 1, rank=1)
        self.assertIn('rank 1 is too small for shape', str(cm.exception))

        # move_axis with scalar source/destination
        a = Scalar(np.arange(12).reshape(3, 4))
        b = a.move_axis(0, 1)
        self.assertEqual(b.shape, (4, 3))
        b = a.move_axis(1, 0)
        self.assertEqual(b.shape, (4, 3))

        # move_axis reshape when ndims < rank
        # When rank=3 and object has shape (3, 4), it gets reshaped to (1, 3, 4)
        # Then moving axis 0 to position 2 results in (3, 4, 1)
        a = Scalar(np.arange(12).reshape(3, 4))
        b = a.move_axis(0, 2, rank=3)
        self.assertEqual(b.shape, (3, 4, 1))

        # stack function various paths
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))
        self.assertTrue(np.allclose(c.values[0], [1., 2., 3.]))
        self.assertTrue(np.allclose(c.values[1], [4., 5., 6.]))

        # stack with None args
        a = Scalar([1., 2., 3.])
        b = None
        c = Scalar([4., 5., 6.])
        result = Qube.stack(a, b, c)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.allclose(result.values[0], [1., 2., 3.]))
        self.assertTrue(np.allclose(result.values[1], [0., 0., 0.]))
        self.assertTrue(np.allclose(result.values[2], [4., 5., 6.]))

        # stack with derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([10., 20., 30.]))
        b = Scalar([4., 5., 6.])
        b.insert_deriv('t', Scalar([40., 50., 60.]))
        c = Qube.stack(a, b, recursive=True)
        self.assertTrue(hasattr(c, 'd_dt'))
        self.assertEqual(c.d_dt.shape, (2, 3))
        self.assertTrue(np.allclose(c.d_dt.values[0], [10., 20., 30.]))
        self.assertTrue(np.allclose(c.d_dt.values[1], [40., 50., 60.]))

        # stack with mixed types
        a = Scalar([1., 2., 3.])
        b = Scalar([4, 5, 6])
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

        # stack with units
        from polymath.unit import Unit
        a = Scalar([1., 2., 3.], unit=Unit.KM)
        b = Scalar([4., 5., 6.], unit=Unit.KM)
        c = Qube.stack(a, b)
        self.assertEqual(c._unit, Unit.KM)

        # Test move_axis with recursive=True and derivatives
        a = Scalar(np.arange(12).reshape(3, 4))
        a.insert_deriv('t', Scalar(np.arange(12).reshape(3, 4)))
        b = a.move_axis(0, 1, recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertEqual(b.d_dt.shape, (4, 3))

        # Test stack with float_arg logic (float_arg is None or not qubed)
        # Case: float_arg is None
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

        # Case: float_arg is not None but qubed is True (arg was converted)
        a = np.array([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

        # Test stack with int_arg logic (int_arg is None or not qubed)
        # Case: int_arg is None, float_arg is None
        a = Scalar([1, 2, 3])
        b = Scalar([4, 5, 6])
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

        # Case: int_arg is not None but qubed is True
        a = np.array([1, 2, 3])
        b = Scalar([4, 5, 6])
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

        # Test stack with bool_arg logic (bool_arg is None or not qubed)
        # Case: bool_arg is None, int_arg is None, float_arg is None
        from polymath.boolean import Boolean
        a = Boolean([True, False, True])
        b = Boolean([False, True, False])
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

        # Case: bool_arg is not None but qubed is True
        a = np.array([True, False, True])
        b = Boolean([False, True, False])
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

        # Test stack with float_arg is not None and qubed is True
        # This tests the branch where float_arg is not None and qubed is True
        # so the condition "float_arg is None or not qubed" is False
        a = Scalar([1., 2., 3.])
        b = np.array([4., 5., 6.])  # This will be converted (qubed=True)
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

        # Test stack with int_arg is not None and qubed is True
        a = Scalar([1, 2, 3])
        b = np.array([4, 5, 6])  # This will be converted (qubed=True)
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

        # Test stack with bool_arg is not None and qubed is True
        a = Boolean([True, False, True])
        b = np.array([False, True, False])  # This will be converted (qubed=True)
        c = Qube.stack(a, b)
        self.assertEqual(c.shape, (2, 3))

##########################################################################################
