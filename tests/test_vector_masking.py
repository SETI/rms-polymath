##########################################################################################
# tests/test_vector_masking.py
##########################################################################################

import numpy as np

from polymath import Scalar, Vector


def test_vector_masking() -> None:
    """Exercise vector masking."""

    a = Vector(np.arange(9).reshape(3,3))   # [[0,1,2],[3,4,5],[6,7,8]]
    mask000 = np.array([False, False, False])
    mask100 = np.array([True , False, False])
    mask110 = np.array([True , True , False])
    mask111 = np.array([True , True , True ])
    mask011 = np.array([False, True , True ])
    mask001 = np.array([False, False, True ])
    assert np.all(a.mask_where_component_le(2,2).mask == mask100)
    assert np.all(a.mask_where_component_le(2,3).mask == mask100)
    assert np.all(a.mask_where_component_le(2,4).mask == mask100)
    assert np.all(a.mask_where_component_le(2,5).mask == mask110)
    assert np.all(a.mask_where_component_le(2,6).mask == mask110)
    assert np.all(a.mask_where_component_lt(2,2).mask == mask000)
    assert np.all(a.mask_where_component_lt(2,3).mask == mask100)
    assert np.all(a.mask_where_component_lt(2,4).mask == mask100)
    assert np.all(a.mask_where_component_lt(2,5).mask == mask100)
    assert np.all(a.mask_where_component_lt(2,6).mask == mask110)
    assert np.all(a.mask_where_component_ge(2,2).mask == mask111)
    assert np.all(a.mask_where_component_ge(2,3).mask == mask011)
    assert np.all(a.mask_where_component_ge(2,4).mask == mask011)
    assert np.all(a.mask_where_component_ge(2,5).mask == mask011)
    assert np.all(a.mask_where_component_ge(2,6).mask == mask001)
    assert np.all(a.mask_where_component_ge(2,7).mask == mask001)
    assert np.all(a.mask_where_component_gt(2,1).mask == mask111)
    assert np.all(a.mask_where_component_gt(2,2).mask == mask011)
    assert np.all(a.mask_where_component_gt(2,3).mask == mask011)
    assert np.all(a.mask_where_component_gt(2,4).mask == mask011)
    assert np.all(a.mask_where_component_gt(2,5).mask == mask001)
    assert np.all(a.mask_where_component_gt(2,6).mask == mask001)
    assert np.all(a.mask_where_component_gt(2,7).mask == mask001)
    assert np.all(a.mask_where_component_gt(2,8).mask == mask000)

    ############################################################################################
    # clip_component(), etc.
    ############################################################################################
    assert a.clip_component(2,2,8,False) == [[0,1,2],[3,4,5],[6,7,8]]
    assert a.clip_component(2,2,7,False) == [[0,1,2],[3,4,5],[6,7,7]]
    assert a.clip_component(2,2,6,False) == [[0,1,2],[3,4,5],[6,7,6]]
    assert a.clip_component(2,2,3,False) == [[0,1,2],[3,4,3],[6,7,3]]
    assert a.clip_component(2,2,None,False) == [[0,1,2],[3,4,5],[6,7,8]]
    assert a.clip_component(2,None,3,False) == [[0,1,2],[3,4,3],[6,7,3]]
    assert a.clip_component(2,2,8,True) == [[0,1,2],[3,4,5],[6,7,8]]
    assert np.all(a.clip_component(2,2,7,True).mask == mask001)
    assert np.all(a.clip_component(2,2,6,True).mask == mask001)
    assert np.all(a.clip_component(2,2,3,True).mask == mask011)
    assert np.all(a.clip_component(2,2,None,True).mask == mask000)
    lower = Scalar([4,3,2])
    upper = Scalar([5,4,3],mask=[0,1,0])
    assert a.clip_component(2,lower,upper,False) == [[0,1,4],[3,4,5],[6,7,3]]


def test_vector_clip_component_assigns_the_limit_value() -> None:
    """A shapeless Vector clips against an upper limit given as a plain number."""

    assert list(Vector([5., 0.]).clip_component(0, None, 2.).values) == [2., 0.]
    assert list(Vector([-5., 0.]).clip_component(0, -2., None).values) == [-2., 0.]


##########################################################################################
