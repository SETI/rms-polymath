##########################################################################################
# tests/test_vector_mean_sum.py
##########################################################################################

import numpy as np

from polymath import Vector


def test_vector_mean_sum_mean() -> None:
    """Mean."""

    np.random.seed(7365)

    assert Vector([1,2,3,4]).mean() == [1,2,3,4]
    vals = np.random.randn(5,4)
    v = Vector(vals)
    assert v.mean() == np.mean(vals, axis=0)
    assert v.mean(axis=-1) == np.mean(vals, axis=0)
    vals = np.random.randn(5,5,4)
    v = Vector(vals)
    assert v.mean() == np.mean(vals, axis=(0,1))
    assert v.mean(axis=0) == np.mean(vals, axis=0)
    assert v.mean(axis=-2) == np.mean(vals, axis=0)
    assert v.mean(axis=1) == np.mean(vals, axis=1)
    assert v.mean(axis=-1) == np.mean(vals, axis=1)
    vals = np.random.randn(3,5,4)
    mask = 3*[[False,False,True,True,True]]
    v = Vector(vals, mask)
    assert v.mean() == np.mean(vals[:,:2], axis=(0,1))
    assert v.mean(axis=1) == np.mean(vals[:,:2], axis=1)
    assert v.mean(axis=-1) == np.mean(vals[:,:2], axis=1)
    assert v.mean(axis=0)[:2] == np.mean(vals[:,:2], axis=0)
    assert np.all(v.mean(axis=0)[2:].mask) == True


def test_vector_mean_sum_mean_with_derivs() -> None:
    """Mean, with derivs."""

    np.random.seed(7365)

    vals = np.random.randn(5,4)
    dv_dt = Vector(np.random.randn(5,4,2,2), drank=2)
    v = Vector(vals, derivs={'t': dv_dt})
    assert v.mean() == np.mean(vals, axis=0)
    assert v.mean(axis=-1) == np.mean(vals, axis=0)
    assert v.mean().d_dt == np.mean(dv_dt.vals, axis=0)
    assert v.mean(axis=-1).d_dt == np.mean(dv_dt.vals, axis=0)
    vals = np.random.randn(5,5,4)
    dv_dt = Vector(np.random.randn(5,5,4,2,2), drank=2)
    v = Vector(vals, derivs={'t': dv_dt})
    assert v.mean() == np.mean(vals, axis=(0,1))
    assert v.mean(axis=0) == np.mean(vals, axis=0)
    assert v.mean(axis=-2) == np.mean(vals, axis=0)
    assert v.mean(axis=1) == np.mean(vals, axis=1)
    assert v.mean(axis=-1) == np.mean(vals, axis=1)
    assert v.mean().d_dt == np.mean(dv_dt.vals, axis=(0,1))
    assert v.mean(axis=0).d_dt == np.mean(dv_dt.vals, axis=0)
    assert v.mean(axis=-2).d_dt == np.mean(dv_dt.vals, axis=0)
    assert v.mean(axis=1).d_dt == np.mean(dv_dt.vals, axis=1)
    assert v.mean(axis=-1).d_dt == np.mean(dv_dt.vals, axis=1)
    vals = np.random.randn(3,5,4)
    mask = 3*[[False,False,True,True,True]]
    dv_dt = Vector(np.random.randn(3,5,4,2,2), drank=2, mask=mask)
    v = Vector(vals, mask, derivs={'t': dv_dt})
    assert v.mean() == np.mean(vals[:,:2], axis=(0,1))
    assert v.mean(axis=1) == np.mean(vals[:,:2], axis=1)
    assert v.mean(axis=-1) == np.mean(vals[:,:2], axis=1)
    assert v.mean(axis=0)[:2] == np.mean(vals[:,:2], axis=0)
    assert np.all(v.mean(axis=0)[2:].mask) == True
    assert v.mean().d_dt == np.mean(dv_dt.vals[:,:2], axis=(0,1))
    assert v.mean(axis=1).d_dt == np.mean(dv_dt.vals[:,:2], axis=1)
    assert v.mean(axis=-1).d_dt == np.mean(dv_dt.vals[:,:2], axis=1)
    assert v.mean(axis=0)[:2].d_dt == np.mean(dv_dt.vals[:,:2], axis=0)
    assert np.all(v.mean(axis=0)[2:].d_dt.mask) == True


def test_vector_mean_sum_sum() -> None:
    """Sum."""

    np.random.seed(7365)

    assert Vector([1,2,3,4]).sum() == [1,2,3,4]
    vals = np.random.randn(5,4)
    v = Vector(vals)
    assert v.sum() == np.sum(vals, axis=0)
    assert v.sum(axis=-1) == np.sum(vals, axis=0)
    vals = np.random.randn(5,5,4)
    v = Vector(vals)
    assert v.sum() == np.sum(vals, axis=(0,1))
    assert v.sum(axis=0) == np.sum(vals, axis=0)
    assert v.sum(axis=-2) == np.sum(vals, axis=0)
    assert v.sum(axis=1) == np.sum(vals, axis=1)
    assert v.sum(axis=-1) == np.sum(vals, axis=1)
    vals = np.random.randn(3,5,4)
    mask = 3*[[False,False,True,True,True]]
    v = Vector(vals, mask)
    assert v.sum() == np.sum(vals[:,:2], axis=(0,1))
    assert v.sum(axis=1) == np.sum(vals[:,:2], axis=1)
    assert v.sum(axis=-1) == np.sum(vals[:,:2], axis=1)
    assert v.sum(axis=0)[:2] == np.sum(vals[:,:2], axis=0)
    assert np.all(v.sum(axis=0)[2:].mask) == True


def test_vector_mean_sum_sum_with_derivs() -> None:
    """Sum, with derivs."""

    np.random.seed(7365)

    vals = np.random.randn(5,4)
    dv_dt = Vector(np.random.randn(5,4,2,2), drank=2)
    v = Vector(vals, derivs={'t': dv_dt})
    assert v.sum() == np.sum(vals, axis=0)
    assert v.sum(axis=-1) == np.sum(vals, axis=0)
    assert v.sum().d_dt == np.sum(dv_dt.vals, axis=0)
    assert v.sum(axis=-1).d_dt == np.sum(dv_dt.vals, axis=0)
    vals = np.random.randn(5,5,4)
    dv_dt = Vector(np.random.randn(5,5,4,2,2), drank=2)
    v = Vector(vals, derivs={'t': dv_dt})
    assert v.sum() == np.sum(vals, axis=(0,1))
    assert v.sum(axis=0) == np.sum(vals, axis=0)
    assert v.sum(axis=-2) == np.sum(vals, axis=0)
    assert v.sum(axis=1) == np.sum(vals, axis=1)
    assert v.sum(axis=-1) == np.sum(vals, axis=1)
    assert v.sum().d_dt == np.sum(dv_dt.vals, axis=(0,1))
    assert v.sum(axis=0).d_dt == np.sum(dv_dt.vals, axis=0)
    assert v.sum(axis=-2).d_dt == np.sum(dv_dt.vals, axis=0)
    assert v.sum(axis=1).d_dt == np.sum(dv_dt.vals, axis=1)
    assert v.sum(axis=-1).d_dt == np.sum(dv_dt.vals, axis=1)
    vals = np.random.randn(3,5,4)
    mask = 3*[[False,False,True,True,True]]
    dv_dt = Vector(np.random.randn(3,5,4,2,2), drank=2, mask=mask)
    v = Vector(vals, mask, derivs={'t': dv_dt})
    assert v.sum() == np.sum(vals[:,:2], axis=(0,1))
    assert v.sum(axis=1) == np.sum(vals[:,:2], axis=1)
    assert v.sum(axis=-1) == np.sum(vals[:,:2], axis=1)
    assert v.sum(axis=0)[:2] == np.sum(vals[:,:2], axis=0)
    assert np.all(v.sum(axis=0)[2:].mask) == True
    assert v.sum().d_dt == np.sum(dv_dt.vals[:,:2], axis=(0,1))
    assert v.sum(axis=1).d_dt == np.sum(dv_dt.vals[:,:2], axis=1)
    assert v.sum(axis=-1).d_dt == np.sum(dv_dt.vals[:,:2], axis=1)
    assert v.sum(axis=0)[:2].d_dt == np.sum(dv_dt.vals[:,:2], axis=0)
    assert np.all(v.sum(axis=0)[2:].d_dt.mask) == True


##########################################################################################
