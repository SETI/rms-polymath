##########################################################################################
# tests/test_qube_items.py
#
#   transpose_numer(self, axis1=0, axis2=1, recursive=True)
#   reshape_numer(self, shape, classes=(), recursive=True)
#   flatten_numer(self, classes=(), recursive=True)
#
#   transpose_denom(self, axis1=0, axis2=1)
#   reshape_denom(self, shape)
#   flatten_denom(self)
#
#   join_items(self, classes)
#   swap_items(self, classes)
#   chain(self, arg)
##########################################################################################

import numpy as np

from polymath import Boolean, Matrix, Matrix3, Quaternion, Scalar, Vector


def test_qube_items() -> None:
    """Exercise qube items."""

    np.random.seed(8736)

    ##################################################################################
    # transpose_numer(self, axis1=0, axis2=1, recursive=True)
    ##################################################################################
    a = Matrix(np.random.randn(5,4,3,2), drank=1)
    b = a.transpose_numer(0,1)
    assert b.shape == (5,)
    assert b.numer == (3,4)
    assert b.denom == (2,)
    assert np.all(a.values[:,:,0] == b.values[:,0])
    assert np.all(a.values[:,:,1] == b.values[:,1])
    assert np.all(a.values[:,:,2] == b.values[:,2])
    a.values[1,3,2] = 42.
    assert np.all(b.values[1,2,3] == 42)
    ####
    a = Matrix(np.random.randn(5,4,3))
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.transpose_numer(0,1,recursive=False)
    assert not hasattr(b, 'd_dt')
    assert a.readonly == False
    assert b.readonly == False
    b = a.transpose_numer(0,1,recursive=True)
    assert np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,0])
    assert np.all(a.d_dt.values[:,:,1] == b.d_dt.values[:,1])
    assert np.all(a.d_dt.values[:,:,2] == b.d_dt.values[:,2])
    a.d_dt.values[1,1,2] = 42.
    assert np.all(b.d_dt.values[1,2,1] == 42)
    assert a.readonly == False
    assert b.readonly == False
    assert a.d_dt.readonly == False
    assert b.d_dt.readonly == False
    a = Matrix(np.random.randn(5,4,3))
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)
    a.as_readonly()
    b = a.transpose_numer(0,1,recursive=True)
    assert a.readonly == True
    assert b.readonly == True
    assert a.d_dt.readonly == True
    assert b.d_dt.readonly == True

    ##################################################################################
    # reshape_numer(self, shape, classes=(), recursive=True)
    ##################################################################################
    a = Matrix(np.random.randn(5,4,3,2), drank=1)
    b = a.reshape_numer((6,2))
    assert b.shape == (5,)
    assert b.numer == (6,2)
    assert b.denom == (2,)
    assert np.all(a.values[:,0,0] == b.values[:,0,0])
    assert np.all(a.values[:,0,1] == b.values[:,0,1])
    assert np.all(a.values[:,0,2] == b.values[:,1,0])
    assert np.all(a.values[:,1,0] == b.values[:,1,1])
    assert np.all(a.values[:,1,1] == b.values[:,2,0])
    assert np.all(a.values[:,1,2] == b.values[:,2,1])
    assert np.all(a.values[:,2,0] == b.values[:,3,0])
    assert np.all(a.values[:,2,1] == b.values[:,3,1])
    assert np.all(a.values[:,2,2] == b.values[:,4,0])
    assert np.all(a.values[:,3,0] == b.values[:,4,1])
    assert np.all(a.values[:,3,1] == b.values[:,5,0])
    assert np.all(a.values[:,3,2] == b.values[:,5,1])
    a.values[1,3,2] = 42.
    assert np.all(b.values[1,5,1] == 42)
    a = Matrix(np.random.randn(5,4,3))
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.reshape_numer((6,2),recursive=False)
    assert not hasattr(b, 'd_dt')
    assert a.readonly == False
    assert b.readonly == False
    b = a.reshape_numer((6,2),recursive=True)
    assert np.all(a.d_dt.values[:,0,0] == b.d_dt.values[:,0,0])
    assert np.all(a.d_dt.values[:,0,1] == b.d_dt.values[:,0,1])
    assert np.all(a.d_dt.values[:,0,2] == b.d_dt.values[:,1,0])
    assert np.all(a.d_dt.values[:,1,0] == b.d_dt.values[:,1,1])
    assert np.all(a.d_dt.values[:,1,1] == b.d_dt.values[:,2,0])
    assert np.all(a.d_dt.values[:,1,2] == b.d_dt.values[:,2,1])
    assert np.all(a.d_dt.values[:,2,0] == b.d_dt.values[:,3,0])
    assert np.all(a.d_dt.values[:,2,1] == b.d_dt.values[:,3,1])
    assert np.all(a.d_dt.values[:,2,2] == b.d_dt.values[:,4,0])
    assert np.all(a.d_dt.values[:,3,0] == b.d_dt.values[:,4,1])
    assert np.all(a.d_dt.values[:,3,1] == b.d_dt.values[:,5,0])
    assert np.all(a.d_dt.values[:,3,2] == b.d_dt.values[:,5,1])
    a.d_dt.values[1,3,2] = 42.
    assert np.all(b.d_dt.values[1,5,1] == 42)
    assert a.readonly == False
    assert b.readonly == False
    assert a.d_dt.readonly == False
    assert b.d_dt.readonly == False
    a = Matrix(np.random.randn(5,4,3)).as_readonly()
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.reshape_numer((6,2),recursive=True)
    assert a.readonly == True
    assert b.readonly == True
    assert a.d_dt.readonly == True
    assert b.d_dt.readonly == True
    a.as_readonly()
    assert a.d_dt.readonly == True
    assert b.d_dt.readonly == True

    ##################################################################################
    # flatten_numer(self, classes=(), recursive=True)
    ##################################################################################
    a = Matrix(np.random.randn(5,4,3,2), drank=1)
    b = a.flatten_numer()
    assert b.shape == (5,)
    assert b.numer == (12,)
    assert b.denom == (2,)
    assert np.all(a.values[:,0,0] == b.values[:,0])
    assert np.all(a.values[:,0,1] == b.values[:,1])
    assert np.all(a.values[:,0,2] == b.values[:,2])
    assert np.all(a.values[:,1,0] == b.values[:,3])
    assert np.all(a.values[:,1,1] == b.values[:,4])
    assert np.all(a.values[:,1,2] == b.values[:,5])
    assert np.all(a.values[:,2,0] == b.values[:,6])
    assert np.all(a.values[:,2,1] == b.values[:,7])
    assert np.all(a.values[:,2,2] == b.values[:,8])
    assert np.all(a.values[:,3,0] == b.values[:,9])
    assert np.all(a.values[:,3,1] == b.values[:,10])
    assert np.all(a.values[:,3,2] == b.values[:,11])
    a.values[1,3,2] = 42.
    assert np.all(b.values[1,11] == 42)
    a = Matrix(np.random.randn(5,4,3))
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.flatten_numer(recursive=False)
    assert not hasattr(b, 'd_dt')
    assert a.readonly == False
    assert b.readonly == False
    b = a.flatten_numer(recursive=True)
    assert np.all(a.d_dt.values[:,0,0] == b.d_dt.values[:,0])
    assert np.all(a.d_dt.values[:,0,1] == b.d_dt.values[:,1])
    assert np.all(a.d_dt.values[:,0,2] == b.d_dt.values[:,2])
    assert np.all(a.d_dt.values[:,1,0] == b.d_dt.values[:,3])
    assert np.all(a.d_dt.values[:,1,1] == b.d_dt.values[:,4])
    assert np.all(a.d_dt.values[:,1,2] == b.d_dt.values[:,5])
    assert np.all(a.d_dt.values[:,2,0] == b.d_dt.values[:,6])
    assert np.all(a.d_dt.values[:,2,1] == b.d_dt.values[:,7])
    assert np.all(a.d_dt.values[:,2,2] == b.d_dt.values[:,8])
    assert np.all(a.d_dt.values[:,3,0] == b.d_dt.values[:,9])
    assert np.all(a.d_dt.values[:,3,1] == b.d_dt.values[:,10])
    assert np.all(a.d_dt.values[:,3,2] == b.d_dt.values[:,11])
    a.d_dt.values[1,3,2] = 42.
    assert np.all(b.d_dt.values[1,11] == 42)
    assert a.readonly == False
    assert b.readonly == False
    assert a.d_dt.readonly == False
    assert b.d_dt.readonly == False
    a = Matrix(np.random.randn(5,4,3)).as_readonly()
    da_dt = Matrix(np.random.randn(5,4,3,2), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.flatten_numer(recursive=True)
    assert a.readonly == True
    assert b.readonly == True
    assert a.d_dt.readonly == True
    assert b.d_dt.readonly == True

    ##################################################################################
    # transpose_denom(self, axis1=0, axis2=1)
    ##################################################################################
    a = Vector(np.random.randn(5,4,3,2), drank=2)
    b = a.transpose_denom(0,1)
    assert b.shape == (5,)
    assert b.numer == (4,)
    assert b.denom == (2,3)
    assert np.all(a.values[...,0] == b.values[...,0,:])
    assert np.all(a.values[...,1] == b.values[...,1,:])
    a.values[...,2,1] = 42.
    assert np.all(b.values[...,1,2] == 42)
    assert a.readonly == False
    assert b.readonly == False
    a = Matrix(np.random.randn(5,4,3,2), drank=2).as_readonly()
    b = a.transpose_denom(0,1)
    assert a.readonly == True
    assert b.readonly == True

    ##################################################################################
    # reshape_denom(self, shape)
    ##################################################################################
    a = Vector(np.random.randn(5,4,3,2), drank=2)
    b = a.reshape_denom((2,3))
    assert b.shape == (5,)
    assert b.numer == (4,)
    assert b.denom == (2,3)
    assert np.all(a.values[...,0,0] == b.values[...,0,0])
    assert np.all(a.values[...,0,1] == b.values[...,0,1])
    assert np.all(a.values[...,1,0] == b.values[...,0,2])
    assert np.all(a.values[...,1,1] == b.values[...,1,0])
    assert np.all(a.values[...,2,0] == b.values[...,1,1])
    assert np.all(a.values[...,2,1] == b.values[...,1,2])
    a.values[1,1,2,1] = 42.
    assert np.all(b.values[1,1,1,2] == 42)
    assert a.readonly == False
    assert b.readonly == False
    a = Vector(np.random.randn(5,4,3,2), drank=2).as_readonly()
    b = a.reshape_denom((2,3))
    assert a.readonly == True
    assert b.readonly == True

    ##################################################################################
    # flatten_denom(self)
    ##################################################################################
    a = Vector(np.random.randn(5,4,3,2), drank=2)
    b = a.flatten_denom()
    assert b.shape == (5,)
    assert b.numer == (4,)
    assert b.denom == (6,)
    assert np.all(a.values[...,0,0] == b.values[...,0])
    assert np.all(a.values[...,0,1] == b.values[...,1])
    assert np.all(a.values[...,1,0] == b.values[...,2])
    assert np.all(a.values[...,1,1] == b.values[...,3])
    assert np.all(a.values[...,2,0] == b.values[...,4])
    assert np.all(a.values[...,2,1] == b.values[...,5])
    a.values[1,1,2,1] = 42.
    assert np.all(b.values[1,1,5] == 42)
    a = Matrix(np.random.randn(5,4,3)).as_readonly()
    b = a.flatten_denom()
    assert a.readonly == True
    assert b.readonly == True

    ##################################################################################
    # join_items(self, classes)
    ##################################################################################
    a = Vector(np.random.randn(5,4,3,2), drank=1)
    b = a.join_items(Matrix)
    assert b.shape == (5,4)
    assert b.numer == (3,2)
    assert b.denom == ()
    b = a.join_items((Boolean,Scalar,Matrix3,Quaternion,Matrix))
    assert type(b) == Matrix
    assert a.readonly == False
    assert b.readonly == False
    a = a.as_readonly()
    b = a.join_items(Matrix)
    assert a.readonly == True
    assert b.readonly == True

    ##################################################################################
    # swap_items(self, classes)
    ##################################################################################
    a = Vector(np.random.randn(5,4,3,2), drank=2)
    b = a.swap_items((Boolean,Scalar,Matrix3,Quaternion,Matrix))
    assert type(b) == Matrix
    assert b.shape == a.shape
    assert b.numer == a.denom
    assert b.denom == a.numer
    assert np.all(a.values[:,0] == b.values[...,0])
    assert np.all(a.values[:,1] == b.values[...,1])
    assert np.all(a.values[:,2] == b.values[...,2])
    assert np.all(a.values[:,3] == b.values[...,3])
    assert a.readonly == False
    assert b.readonly == False
    a = a.as_readonly()
    b = a.swap_items(Matrix)
    assert a.readonly == True
    assert b.readonly == True

    ##################################################################################
    # chain(self, arg)
    ##################################################################################
    a = Vector(np.arange(120).reshape((5,4,3,2)), drank=1)
    b = Vector(np.arange(60,180).reshape((5,4,2,3)), drank=1)
    a_values = a.values.reshape(5,4,3,2,1)
    b_values = b.values.reshape(5,4,1,2,3)
    a_chain_b_vals = np.sum(a_values * b_values, axis=-2)
    assert np.all(a.chain(b).values == a_chain_b_vals)
    assert a.chain(b).shape == (5,4)
    assert a.chain(b).numer == (3,)
    assert a.chain(b).denom == (3,)
    a = Vector(np.arange(60).reshape((5,3,4)), drank=1)
    b = Vector(np.arange(120).reshape((5,4,3,2)), drank=2)
    a_values = a.values.reshape(5,3,4,1,1)
    b_values = b.values.reshape(5,1,4,3,2)
    a_chain_b_vals = np.sum(a_values * b_values, axis=2)
    assert np.all(a.chain(b).values == a_chain_b_vals)
    assert a.chain(b).shape == (5,)
    assert a.chain(b).numer == (3,)
    assert a.chain(b).denom == (3,2)
    a = Vector(np.arange(120).reshape((5,4,3,2)), drank=2)
    b = Matrix(np.arange(270).reshape((5,3,2,3,3)), drank=2)
    a_values = a.values.reshape(5,4,6,1,1)
    b_values = b.values.reshape(5,1,6,3,3)
    a_chain_b_vals = np.sum(a_values * b_values, axis=2)
    assert np.all(a.chain(b).values == a_chain_b_vals)
    assert a_chain_b_vals.shape == (5,4,3,3)
    assert a.chain(b).shape == (5,)
    assert a.chain(b).numer == (4,)
    assert a.chain(b).denom == (3,3)


##########################################################################################
