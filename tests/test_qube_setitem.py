##########################################################################################
# tests/test_qube_setitem.py
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Pair, Scalar, Vector, Vector3


def test_qube_setitem() -> None:
    """Exercise qube setitem."""

    np.random.seed(8343)

    ##################################################################################
    # Qube into Qube, no broadcast, unmasked, with integers, ellipses, colons
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1)
    b = Vector(np.random.randn(4,5,6,3,2), drank=1)
    a[0] = b[0]
    assert np.all(a.values[0] == b.values[0])
    assert np.all(a.mask == b.mask)
    a[:,0] = b[:,0]
    assert np.all(a.values[:,0] == b.values[:,0])
    assert np.all(a.mask == b.mask)
    a[...,0] = b[...,0]
    assert np.all(a.values[:,:,0] == b.values[:,:,0])
    assert np.all(a.mask == b.mask)

    ##################################################################################
    # Same as above, with matching masks
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    b = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    a[0] = b[0]
    assert np.all(a.values[0] == b.values[0])
    assert np.all(a.mask[0] == b.mask[0])
    a[:,0] = b[:,0]
    assert np.all(a.values[:,0] == b.values[:,0])
    assert np.all(a.mask[:,0] == b.mask[:,0])
    a[...,0] = b[...,0]
    assert np.all(a.values[:,:,0] == b.values[:,:,0])
    assert np.all(a.mask[:,:,0] == b.mask[:,:,0])
    a[0,...,0] = b[0,...,1]
    assert np.all(a.values[0,:,0] == b.values[0,:,1])
    assert np.all(a.mask[0,:,0] == b.mask[0,:,1])
    a[...,::-1] = b
    assert np.all(a.values == b.values[:,:,::-1])
    assert np.all(a.mask == b.mask[:,:,::-1])
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.values[:,:,0:5:2] == b.values[:,:,2:5])
    assert np.all(a.mask[:,:,0:5:2] == b.mask[:,:,2:5])

    ##################################################################################
    # Same as above, requiring right mask reshaping
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    b = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=True)
    a[0] = b[0]
    assert np.all(a.values[0] == b.values[0])
    assert np.all(a.mask[0] == True)
    a[:,0] = b[:,0]
    assert np.all(a.values[:,0] == b.values[:,0])
    assert np.all(a.mask[:,0] == True)
    a[...,0] = b[...,0]
    assert np.all(a.values[:,:,0] == b.values[:,:,0])
    assert np.all(a.mask[:,:,0] == True)
    a[0,...,0] = b[0,...,1]
    assert np.all(a.values[0,:,0] == b.values[0,:,1])
    assert np.all(a.mask[0,:,0] == True)
    a[...,::-1] = b
    assert np.all(a.values == b.values[:,:,::-1])
    assert np.all(a.mask == True)
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.values[:,:,0:5:2] == b.values[:,:,2:5])
    assert np.all(a.mask[:,:,0:5:2] == True)

    ##################################################################################
    # Same as above, requiring left mask reshaping
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    b = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    a[0] = b[0]
    assert np.all(a.values[0] == b.values[0])
    assert np.all(a.mask[0] == b.mask[0])
    a[:,0] = b[:,0]
    assert np.all(a.values[:,0] == b.values[:,0])
    assert np.all(a.mask[:,0] == b.mask[:,0])
    a[...,0] = b[...,0]
    assert np.all(a.values[:,:,0] == b.values[:,:,0])
    assert np.all(a.mask[:,:,0] == b.mask[:,:,0])
    a[0,...,0] = b[0,...,1]
    assert np.all(a.values[0,:,0] == b.values[0,:,1])
    assert np.all(a.mask[0,:,0] == b.mask[0,:,1])
    a[...,::-1] = b
    assert np.all(a.values == b.values[:,:,::-1])
    assert np.all(a.mask == b.mask[:,:,::-1])
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.values[:,:,0:5:2] == b.values[:,:,2:5])
    assert np.all(a.mask[:,:,0:5:2] == b.mask[:,:,2:5])

    ##################################################################################
    # Same as above, requiring left and right mask reshaping
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    b = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=True)
    assert type(a.mask) == bool
    assert type(b.mask) == bool
    a[0] = b[0]
    assert np.all(a.values[0] == b.values[0])
    assert np.all(a.mask[0] == True)
    assert type(a.mask) == np.ndarray
    assert type(b.mask) == bool
    a[:,0] = b[:,0]
    assert np.all(a.values[:,0] == b.values[:,0])
    assert np.all(a.mask[:,0] == True)
    a[...,0] = b[...,0]
    assert np.all(a.values[:,:,0] == b.values[:,:,0])
    assert np.all(a.mask[:,:,0] == True)
    a[0,...,0] = b[0,...,1]
    assert np.all(a.values[0,:,0] == b.values[0,:,1])
    assert np.all(a.mask[0,:,0] == True)
    a[...,::-1] = b
    assert np.all(a.values == b.values[:,:,::-1])
    assert np.all(a.mask == True)
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.values[:,:,0:5:2] == b.values[:,:,2:5])
    assert np.all(a.mask[:,:,0:5:2] == True)

    ##################################################################################
    # Same as above, requiring right object broadcasting
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    b = Vector(np.random.randn(6,3,2), drank=1, mask=True)
    a[0] = b
    assert np.all(a.values[0] == b.values)
    assert np.all(a.mask[0] == True)
    a[:,0] = b
    assert np.all(a.values[:,0] == b.values)
    assert np.all(a.mask[:,0] == True)
    b = Vector(np.random.randn(5,6,3,2), drank=1, mask=True)
    a[...,0] = b[...,0]
    assert np.all(a.values[:,:,0] == b.values[:,0])
    assert np.all(a.mask[:,:,0] == True)
    a[0,...,0] = b[...,1]
    assert np.all(a.values[0,:,0] == b.values[:,1])
    assert np.all(a.mask[0,:,0] == True)
    a[...,::-1] = b
    assert np.all(a.values[:,:,::-1] == b.values)
    assert np.all(a.mask == True)
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.values[:,:,0:5:2] == b.values[:,2:5])
    assert np.all(a.mask[:,:,0:5:2] == True)

    ##################################################################################
    # Using boolean arrays as masks
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    b = Vector(np.random.randn(4,5,6,3), mask=True)
    mask = np.array([True,False,False,True])
    a[mask] = b[mask]
    assert np.all(a.values[mask] == b.values[mask])
    assert np.all(a.mask[mask] == True)
    assert np.all(a.values[0] == b.values[0])
    assert not np.all(a.values[1] == b.values[1])
    assert not np.all(a.values[2] == b.values[2])
    assert np.all(a.values[3] == b.values[3])
    assert np.all(a.mask[0] == True)
    assert np.all(a.mask[3] == True)
    mask = np.array([True,False,False,True])
    a[mask] = (0,0,1)
    assert np.all(a.values[mask][...,0] == 0)
    assert np.all(a.values[mask][...,1] == 0)
    assert np.all(a.values[mask][...,2] == 1)
    assert np.all(a.mask[mask] == False)
    assert np.all(a.values[0] == (0,0,1))
    assert not np.all(a.values[1] == b.values[1])
    assert not np.all(a.values[2] == b.values[2])
    assert np.all(a.values[3] == (0,0,1))
    assert np.all(a.mask[0] == False)
    assert np.all(a.mask[3] == False)
    mask = np.array([True,False,False,True])
    b = Vector(np.random.randn(2,5,6,3), mask=False)
    a[mask] = b
    assert np.all(a.values[mask] == b.values)
    assert np.all(a.mask[mask] == False)
    assert np.all(a.mask[0] == False)
    assert np.all(a.mask[3] == False)

    ##################################################################################
    # Same as above, using Boolean subclasses
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    b = Vector(np.random.randn(4,5,6,3), mask=True)
    mask = Boolean(np.array([True,False,False,True]))
    a[mask] = b[mask]
    assert np.all(a.values[mask.values] == b.values[mask.values])
    assert np.all(a.mask[mask.values] == True)
    assert np.all(a.values[0] == b.values[0])
    assert not np.all(a.values[1] == b.values[1])
    assert not np.all(a.values[2] == b.values[2])
    assert np.all(a.values[3] == b.values[3])
    assert np.all(a.mask[0] == True)
    assert np.all(a.mask[3] == True)
    mask = Boolean(np.array([True,False,False,True]))
    a[mask] = (0,0,1)
    assert np.all(a.values[mask.values][...,0] == 0)
    assert np.all(a.values[mask.values][...,1] == 0)
    assert np.all(a.values[mask.values][...,2] == 1)
    assert np.all(a.mask[mask.values] == False)
    assert np.all(a.values[0] == (0,0,1))
    assert not np.all(a.values[1] == b.values[1])
    assert not np.all(a.values[2] == b.values[2])
    assert np.all(a.values[3] == (0,0,1))
    assert np.all(a.mask[0] == False)
    assert np.all(a.mask[3] == False)
    mask = Boolean(np.array([True,False,False,True]))
    b = Vector(np.random.randn(2,5,6,3), mask=False)
    a[mask] = b
    assert np.all(a.values[mask.values] == b.values)
    assert np.all(a.mask[mask.values] == False)
    assert np.all(a.mask[0] == False)
    assert np.all(a.mask[3] == False)

    ##################################################################################
    # Using bool True and False
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    b = Vector(np.random.randn(4,5,6,3), mask=True)
    aa = a.copy()
    bb = b.copy()
    b[False] = a[False]
    assert b == bb
    b[False] = 42.
    assert b == bb
    b[True] = a[True]
    assert b == aa
    a = Scalar(1)
    a[False] = 11
    assert a == 1
    a[True] = 11
    assert a == 11
    a[True] = 3.3
    assert a == 3
    a = Boolean(True)
    a[False] = False
    assert a == True
    a[True] = False
    assert a == False
    a = Vector3([1,2,3])
    a[False] = (3,4,5)
    assert a == (1,2,3)
    a[True] = (3,4,5)
    assert a == (3,4,5)
    a = Scalar(np.arange(10))
    a[False] = 1
    assert a == np.arange(10)
    a = Scalar(np.arange(10))
    a[True] = 11
    assert a == 10*[11]

    ##################################################################################
    # Using tuples, Vectors, Pairs
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3), mask=False)
    b = Vector(np.random.randn(3,6,3), mask=True)
    tup = ((0,1,3),(0,1,3))
    assert a[tup].shape == b.shape
    a[tup] = b
    assert np.all(a.mask[0,0] == True)
    assert np.all(a.mask[1,1] == True)
    assert np.all(a.mask[3,3] == True)
    b = Vector(np.random.randn(3,6,3), mask=False)
    tup = ((0,1,3),(0,1,3))
    assert a[tup].shape == b.shape
    a[tup] = b
    assert np.all(a.values[0,0] == b.values[0])
    assert np.all(a.values[1,1] == b.values[1])
    assert np.all(a.values[3,3] == b.values[2])
    assert np.all(a.mask[0,0] == False)
    assert np.all(a.mask[1,1] == False)
    assert np.all(a.mask[3,3] == False)
    b = Vector(np.random.randn(3,6,3), mask=True)
    pair = Pair([(0,0),(1,1),(3,3)])
    a[pair] = b
    assert np.all(a.mask[0,0] == True)
    assert np.all(a.mask[1,1] == True)
    assert np.all(a.mask[3,3] == True)
    assert a[pair] == a[tup]
    b = Vector(np.random.randn(3,6,3), mask=False)
    pair = Pair([(0,0),(1,1),(3,3)])
    a[pair] = b
    assert np.all(a.values[0,0] == b.values[0])
    assert np.all(a.values[1,1] == b.values[1])
    assert np.all(a.values[3,3] == b.values[2])
    assert np.all(a.mask[0,0] == False)
    assert np.all(a.mask[1,1] == False)
    assert np.all(a.mask[3,3] == False)
    assert a[pair] == a[tup]
    b = Vector(np.random.randn(3,3), mask=True)
    tup = [(0,1,3),(0,1,3),(0,0,0)]
    a[tup] = b
    assert np.all(a.mask[0,0,0] == True)
    assert np.all(a.mask[1,1,0] == True)
    assert np.all(a.mask[3,3,0] == True)
    b = Vector(np.random.randn(3,3), mask=False)
    tup = [(0,1,3),(0,1,3),(0,0,0)]
    a[tup] = b
    assert np.all(a.values[0,0,0] == b.values[0])
    assert np.all(a.values[1,1,0] == b.values[1])
    assert np.all(a.values[3,3,0] == b.values[2])
    assert np.all(a.mask[0,0,0] == False)
    assert np.all(a.mask[1,1,0] == False)
    assert np.all(a.mask[3,3,0] == False)
    b = Vector(np.random.randn(3,3), mask=True)
    vector = Vector([(0,0,0),(1,1,0),(3,3,0)])
    a[vector] = b
    assert np.all(a.mask[0,0,0] == True)
    assert np.all(a.mask[1,1,0] == True)
    assert np.all(a.mask[3,3,0] == True)
    b = Vector(np.random.randn(3,3), mask=False)
    vector = Vector([(0,0,0),(1,1,0),(3,3,0)])
    a[vector] = b
    assert np.all(a.values[0,0,0] == b.values[0])
    assert np.all(a.values[1,1,0] == b.values[1])
    assert np.all(a.values[3,3,0] == b.values[2])
    assert np.all(a.mask[0,0,0] == False)
    assert np.all(a.mask[1,1,0] == False)
    assert np.all(a.mask[3,3,0] == False)
    assert a[vector] == a[tup]

    ##################################################################################
    ############################################################################
    # All the same tests as above for objects with derivatives
    ##################################################################################
    ############################################################################

    ##################################################################################
    # Qube into Qube, no broadcast, unmasked, with integers, ellipses, colons
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1))
    a.insert_deriv('v', Vector(np.random.randn(4,5,6,3,2,3), drank=2))
    aa = a.copy()
    b = Vector(np.random.randn(4,5,6,3,2), drank=1)
    a[0] = b[0]     # derivs are missing in b
    assert a.d_dt[0] == Vector.zeros((), numer=(3,), denom=(2,))
    assert a.d_dv[0] == Vector.zeros((), numer=(3,), denom=(2,3))
    assert a.d_dt[1] == aa.d_dt[1]
    assert a.d_dv[1] == aa.d_dv[1]
    b = Vector(np.random.randn(4,5,6,3,2), drank=1)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1))
    b.insert_deriv('v', Vector(np.random.randn(4,5,6,3,2,3), drank=2))
    a[0] = b[0]
    assert np.all(a.d_dt.values[0] == b.d_dt.values[0])
    assert np.all(a.d_dt.mask == b.d_dt.mask)
    assert np.all(a.d_dv.values[0] == b.d_dv.values[0])
    assert np.all(a.d_dv.mask == b.d_dv.mask)
    a[:,0] = b[:,0]
    assert np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0])
    assert np.all(a.d_dt.mask == b.d_dt.mask)
    assert np.all(a.d_dv.values[:,0] == b.d_dv.values[:,0])
    assert np.all(a.d_dv.mask == b.d_dv.mask)
    a[...,0] = b[...,0]
    assert np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0])
    assert np.all(a.d_dt.mask == b.d_dt.mask)
    assert np.all(a.d_dv.values[:,:,0] == b.d_dv.values[:,:,0])
    assert np.all(a.d_dv.mask == b.d_dv.mask)

    ##################################################################################
    # Same as above, with matching masks
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=a.mask))
    b = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,3), drank=1))
    with pytest.raises(ValueError):
        a.__setitem__(0, b[0])
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=b.mask))
    a[0] = b[0]
    assert np.all(a.d_dt.values[0] == b.d_dt.values[0])
    assert np.all(a.d_dt.mask[0] == b.d_dt.mask[0])
    a[:,0] = b[:,0]
    assert np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0])
    assert np.all(a.d_dt.mask[:,0] == b.d_dt.mask[:,0])
    a[...,0] = b[...,0]
    assert np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0])
    assert np.all(a.d_dt.mask[:,:,0] == b.d_dt.mask[:,:,0])
    a[0,...,0] = b[0,...,1]
    assert np.all(a.d_dt.values[0,:,0] == b.d_dt.values[0,:,1])
    assert np.all(a.d_dt.mask[0,:,0] == b.d_dt.mask[0,:,1])
    a[...,::-1] = b
    assert np.all(a.d_dt.values == b.d_dt.values[:,:,::-1])
    assert np.all(a.d_dt.mask == b.d_dt.mask[:,:,::-1])
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,:,2:5])
    assert np.all(a.d_dt.mask[:,:,0:5:2] == b.d_dt.mask[:,:,2:5])

    ##################################################################################
    # Same as above, requiring right mask reshaping
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=a.mask))
    b = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=True)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=True))
    a[0] = b[0]
    assert np.all(a.d_dt.values[0] == b.d_dt.values[0])
    assert np.all(a.d_dt.mask[0] == True)
    a[:,0] = b[:,0]
    assert np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0])
    assert np.all(a.d_dt.mask[:,0] == True)
    a[...,0] = b[...,0]
    assert np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0])
    assert np.all(a.d_dt.mask[:,:,0] == True)
    a[0,...,0] = b[0,...,1]
    assert np.all(a.d_dt.values[0,:,0] == b.d_dt.values[0,:,1])
    assert np.all(a.d_dt.mask[0,:,0] == True)
    a[...,::-1] = b
    assert np.all(a.d_dt.values == b.d_dt.values[:,:,::-1])
    assert np.all(a.d_dt.mask == True)
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,:,2:5])
    assert np.all(a.d_dt.mask[:,:,0:5:2] == True)

    ##################################################################################
    # Same as above, requiring left mask reshaping
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1))
    b = Vector(np.random.randn(4,5,6,3,2), drank=1,
               mask=(np.random.rand(4,5,6) < 0.2))
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=b.mask))
    a[0] = b[0]
    assert np.all(a.d_dt.values[0] == b.d_dt.values[0])
    assert np.all(a.d_dt.mask[0] == b.d_dt.mask[0])
    a[:,0] = b[:,0]
    assert np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0])
    assert np.all(a.d_dt.mask[:,0] == b.d_dt.mask[:,0])
    a[...,0] = b[...,0]
    assert np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0])
    assert np.all(a.d_dt.mask[:,:,0] == b.d_dt.mask[:,:,0])
    a[0,...,0] = b[0,...,1]
    assert np.all(a.d_dt.values[0,:,0] == b.d_dt.values[0,:,1])
    assert np.all(a.d_dt.mask[0,:,0] == b.d_dt.mask[0,:,1])
    a[...,::-1] = b
    assert np.all(a.d_dt.values == b.d_dt.values[:,:,::-1])
    assert np.all(a.d_dt.mask == b.d_dt.mask[:,:,::-1])
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,:,2:5])
    assert np.all(a.d_dt.mask[:,:,0:5:2] == b.d_dt.mask[:,:,2:5])

    ##################################################################################
    # Same as above, requiring left and right mask reshaping
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=False)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1))
    b = Vector(np.random.randn(4,5,6,3,2), drank=1, mask=True)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3,2), drank=1,
                               mask=True))
    a[0] = b[0]
    assert np.all(a.d_dt.values[0] == b.d_dt.values[0])
    assert np.all(a.d_dt.mask[0] == True)
    assert type(a.d_dt.mask) == np.ndarray
    a[:,0] = b[:,0]
    assert np.all(a.d_dt.values[:,0] == b.d_dt.values[:,0])
    assert np.all(a.d_dt.mask[:,0] == True)
    a[...,0] = b[...,0]
    assert np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,:,0])
    assert np.all(a.d_dt.mask[:,:,0] == True)
    a[0,...,0] = b[0,...,1]
    assert np.all(a.d_dt.values[0,:,0] == b.d_dt.values[0,:,1])
    assert np.all(a.d_dt.mask[0,:,0] == True)
    a[...,::-1] = b
    assert np.all(a.d_dt.values == b.d_dt.values[:,:,::-1])
    assert np.all(a.d_dt.mask == True)
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,:,2:5])
    assert np.all(a.d_dt.mask[:,:,0:5:2] == True)

    ##################################################################################
    # Same as above, requiring right object broadcasting
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3), mask=False)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3)))
    b = Vector(np.random.randn(6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(6,3), mask=True))
    a[0] = b
    assert np.all(a.d_dt.values[0] == b.d_dt.values)
    assert np.all(a.d_dt.mask[0] == True)
    a[:,0] = b
    assert np.all(a.d_dt.values[:,0] == b.d_dt.values)
    assert np.all(a.d_dt.mask[:,0] == True)
    b = Vector(np.random.randn(5,6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(5,6,3), mask=True))
    a[...,0] = b[...,0]
    assert np.all(a.d_dt.values[:,:,0] == b.d_dt.values[:,0])
    assert np.all(a.d_dt.mask[:,:,0] == True)
    a[0,...,0] = b[...,1]
    assert np.all(a.d_dt.values[0,:,0] == b.d_dt.values[:,1])
    assert np.all(a.d_dt.mask[0,:,0] == True)
    a[...,::-1] = b
    assert np.all(a.d_dt.values[:,:,::-1] == b.d_dt.values)
    assert np.all(a.d_dt.mask == True)
    a[...,0:5:2] = b[...,2:5]
    assert np.all(a.d_dt.values[:,:,0:5:2] == b.d_dt.values[:,2:5])
    assert np.all(a.d_dt.mask[:,:,0:5:2] == True)

    ##################################################################################
    # Using boolean arrays as masks
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=a.mask))
    b = Vector(np.random.randn(4,5,6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=True))
    mask = np.array([True,False,False,True])
    a[mask] = b[mask]
    assert np.all(a.d_dt.values[mask] == b.d_dt.values[mask])
    assert np.all(a.d_dt.mask[mask] == True)
    assert np.all(a.d_dt.values[0] == b.d_dt.values[0])
    assert not np.all(a.d_dt.values[1] == b.d_dt.values[1])
    assert not np.all(a.d_dt.values[2] == b.d_dt.values[2])
    assert np.all(a.d_dt.values[3] == b.d_dt.values[3])
    assert np.all(a.d_dt.mask[0] == True)
    assert np.all(a.d_dt.mask[3] == True)
    mask = np.array([True,False,False,True])
    b = Vector(np.random.randn(2,5,6,3), mask=False)
    b.insert_deriv('t', Vector(np.random.randn(2,5,6,3), mask=False))
    a[mask] = b
    assert np.all(a.d_dt.values[mask] == b.d_dt.values)
    assert np.all(a.d_dt.mask[mask] == False)
    assert np.all(a.d_dt.mask[0] == False)
    assert np.all(a.d_dt.mask[3] == False)

    ##################################################################################
    # Same as above, using Boolean subclasses
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3), mask=(np.random.rand(4,5,6) < 0.2))
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=a.mask))
    b = Vector(np.random.randn(4,5,6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=True))
    mask = Boolean(np.array([True,False,False,True]))
    a[mask] = b[mask]
    assert (np.all(a.d_dt.values[mask.values] ==
                           b.d_dt.values[mask.values]))
    assert np.all(a.d_dt.mask[mask.values] == True)
    assert np.all(a.d_dt.values[0] == b.d_dt.values[0])
    assert not np.all(a.d_dt.values[1] == b.d_dt.values[1])
    assert not np.all(a.d_dt.values[2] == b.d_dt.values[2])
    assert np.all(a.d_dt.values[3] == b.d_dt.values[3])
    assert np.all(a.d_dt.mask[0] == True)
    assert np.all(a.d_dt.mask[3] == True)
    mask = Boolean(np.array([True,False,False,True]))
    b = Vector(np.random.randn(2,5,6,3), mask=False)
    b.insert_deriv('t', Vector(np.random.randn(2,5,6,3), mask=False))
    a[mask] = b
    assert np.all(a.d_dt.values[mask.values] == b.d_dt.values)
    assert np.all(a.d_dt.mask[mask.values] == False)
    assert np.all(a.d_dt.mask[0] == False)
    assert np.all(a.d_dt.mask[3] == False)

    ##################################################################################
    # Using tuples, Vectors, Pairs
    ##################################################################################
    a = Vector(np.random.randn(4,5,6,3), mask=False)
    a.insert_deriv('t', Vector(np.random.randn(4,5,6,3), mask=False))
    b = Vector(np.random.randn(3,6,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(3,6,3), mask=True))
    tup = [(0,1,3),(0,1,3)]
    a[tup] = b
    assert np.all(a.d_dt.values[0,0] == b.d_dt.values[0])
    assert np.all(a.d_dt.values[1,1] == b.d_dt.values[1])
    assert np.all(a.d_dt.values[3,3] == b.d_dt.values[2])
    assert np.all(a.d_dt.mask[0,0] == True)
    assert np.all(a.d_dt.mask[1,1] == True)
    assert np.all(a.d_dt.mask[3,3] == True)
    pair = Pair([(0,0),(1,1),(3,3)])
    a[pair] = b
    assert np.all(a.d_dt.values[0,0] == b.d_dt.values[0])
    assert np.all(a.d_dt.values[1,1] == b.d_dt.values[1])
    assert np.all(a.d_dt.values[3,3] == b.d_dt.values[2])
    assert np.all(a.d_dt.mask[0,0] == True)
    assert np.all(a.d_dt.mask[1,1] == True)
    assert np.all(a.d_dt.mask[3,3] == True)
    assert a.d_dt[pair] == a.d_dt[tup]
    b = Vector(np.random.randn(3,3), mask=True)
    b.insert_deriv('t', Vector(np.random.randn(3,3), mask=True))
    tup = [(0,1,3),(0,1,3),(0,0,0)]
    a[tup] = b
    assert np.all(a.d_dt.values[0,0,0] == b.d_dt.values[0])
    assert np.all(a.d_dt.values[1,1,0] == b.d_dt.values[1])
    assert np.all(a.d_dt.values[3,3,0] == b.d_dt.values[2])
    assert np.all(a.d_dt.mask[0,0,0] == True)
    assert np.all(a.d_dt.mask[1,1,0] == True)
    assert np.all(a.d_dt.mask[3,3,0] == True)
    vector = Vector([(0,0,0),(1,1,0),(3,3,0)])
    a[vector] = b
    assert np.all(a.d_dt.values[0,0,0] == b.d_dt.values[0])
    assert np.all(a.d_dt.values[1,1,0] == b.d_dt.values[1])
    assert np.all(a.d_dt.values[3,3,0] == b.d_dt.values[2])
    assert np.all(a.d_dt.mask[0,0,0] == True)
    assert np.all(a.d_dt.mask[1,1,0] == True)
    assert np.all(a.d_dt.mask[3,3,0] == True)
    assert a.d_dt[vector] == a.d_dt[tup]

    ##################################################################################
    # Non-consecutive array indices
    ##################################################################################
    a = Scalar(np.random.randn(7,6,5,4))
    aa = a.copy()
    aa[:,np.array([2,0]),:,np.array([1,3])] = 99.
    assert aa[:,2,:,1] == 99.
    assert aa[:,0,:,3] == 99.
    for i in range(6):
        for j in range(4):
            if (i,j) == (2,1):
                continue
            if (i,j) == (0,3):
                continue
            assert (aa[:,i,:,j] != 99.)
            assert (aa[:,i,:,j] == a[:,i,:,j])
    a = Scalar(np.random.randn(7,6,5,4), mask=(np.random.rand(7,6,5,4) < 0.2))
    aa = a.copy()
    aa[:,np.array([2,0]),:,np.array([1,3])] = 99.
    assert aa[:,2,:,1] == 99.
    assert aa[:,0,:,3] == 99.
    for i in range(6):
        for j in range(4):
            if (i,j) == (2,1):
                continue
            if (i,j) == (0,3):
                continue
            assert (aa[:,i,:,j] != 99.)
            assert (aa[:,i,:,j] == a[:,i,:,j])


def test_qube_setitem_non_consecutive_array_indices_with_an_array_mask() -> None:
    """Assign through non-consecutive array indices when this object's mask is an array."""

    a = Scalar(np.zeros((4,5,6,7)), mask=np.zeros((4,5,6,7), dtype='bool'))
    a[:, np.array([0,1]), :, np.array([0,1])] = Scalar(np.ones((4,2,6)))
    assert a.values[0,0,0,0] == 1.
    assert a.values[0,1,0,1] == 1.
    assert a.values[0,2,0,2] == 0.
    assert not np.any(a.mask)


##########################################################################################
