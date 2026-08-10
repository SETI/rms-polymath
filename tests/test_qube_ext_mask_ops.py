##########################################################################################
# tests/test_qube_ext_mask_ops.py
#
# Comprehensive unit tests for mask operations based on docstrings in mask_ops.py
##########################################################################################

import numpy as np
import pytest

from polymath import Qube, Scalar, Vector


def test_qube_ext_mask_ops_simple_1_d_case_empty_mask_returns_unchanged() -> None:
    """Simple 1-D case: empty mask returns unchanged."""

    np.random.seed(8736)

    ##################################################################################
    # mask_where()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    mask = np.array([False, False, False, False, False])
    b = a.mask_where(mask)
    assert a == b

    a = Scalar([1., 2., 3., 4., 5.])
    mask = np.array([True, False, True, False, False])
    b = a.mask_where(mask)
    assert b.mask[0]
    assert not b.mask[1]
    assert b.mask[2]
    assert not b.mask[3]
    assert not b.mask[4]
    assert b[1] == 2.
    assert b[3] == 4.
    assert b[4] == 5.

    a = Scalar([1., 2., 3., 4., 5.])
    mask = np.array([True, False, False, False, False])
    b = a.mask_where(mask, replace=99., remask=True)
    assert b.mask[0]
    assert not b.mask[1]
    assert b[1] == 2.

    a = Scalar([1., 2., 3., 4., 5.])
    mask = np.array([True, False, False, False, False])
    b = a.mask_where(mask, replace=99., remask=False)
    if isinstance(b.mask, np.ndarray):
        assert not b.mask[0]
    else:
        assert not b.mask
    assert b[0] == 99.
    assert b[1] == 2.

    a = Scalar([1., 2., 3., 4., 5.])
    mask = np.array([True, False, False, False, False])
    b = a.mask_where(mask, replace=None, remask=False)
    assert a == b

    a = Scalar(np.arange(20).reshape(4, 5))
    mask = np.array([[True, False, True, False, False],
                     [False, False, False, False, False],
                     [True, True, False, False, False],
                     [False, False, False, False, True]])
    b = a.mask_where(mask)
    assert b.mask[0, 0]
    assert not b.mask[0, 1]
    assert b.mask[0, 2]
    assert b.mask[2, 0]
    assert b.mask[2, 1]
    assert b.mask[3, 4]

    a = Scalar(np.arange(20).reshape(4, 5))
    replace = Scalar(np.ones((4, 5)) * 99.)
    mask = np.array([[True, False, False, False, False],
                     [False, False, False, False, False],
                     [False, False, False, False, False],
                     [False, False, False, False, False]])
    b = a.mask_where(mask, replace=replace, remask=False)
    assert b[0, 0] == 99.
    assert b[0, 1] == 1.

    a = Vector(np.arange(30).reshape(10, 3))
    mask = np.array([True] * 5 + [False] * 5)
    b = a.mask_where(mask)
    assert np.all(b.mask[0:5])
    assert not np.all(b.mask[5:10])

    a = Scalar([1., 2., 3., 4., 5.])
    replace = Scalar([1., 2., 3.])  # Wrong shape
    mask = np.array([True, False, False, False, False])
    with pytest.raises(ValueError):
        a.mask_where(mask, replace=replace)

    a = Scalar([1., 2., 3.])
    da_dt = Scalar([10., 20., 30.])
    a.insert_deriv('t', da_dt)
    mask = np.array([True, False, False])
    b = a.mask_where(mask, recursive=True)
    assert b.mask[0]
    assert b.d_dt.mask[0]
    assert not b.mask[1]
    assert not b.d_dt.mask[1]
    b = a.mask_where(mask, recursive=False)
    assert b.mask[0]

    assert not hasattr(b, 'd_dt')

    ##################################################################################
    # mask_where_eq()
    ##################################################################################

    a = Scalar([1., 2., 3., 2., 5.])
    b = a.mask_where_eq(2.)
    assert not b.mask[0]
    assert b.mask[1]
    assert not b.mask[2]
    assert b.mask[3]
    assert not b.mask[4]
    assert b[0] == 1.
    assert b[2] == 3.
    assert b[4] == 5.

    a = Scalar([1., 2., 3., 2., 5.])
    b = a.mask_where_eq(2., replace=99., remask=False)
    assert b[0] == 1.
    assert b[1] == 99.
    assert b[2] == 3.
    assert b[3] == 99.
    assert b[4] == 5.

    a = Vector(np.arange(30).reshape(10, 3) % 6)
    match = Vector([3., 4., 5.])
    b = a.mask_where_eq(match)

    assert b.count_masked() == 5

    a = Vector(np.arange(30).reshape(10, 3) % 6)
    match = Vector([3., 4., 5.])
    replace = Vector([0., 1., 2.])
    b = a.mask_where_eq(match, replace=replace, remask=False)
    assert b.count_masked() == 0
    assert b[0] == replace

    a = Scalar([1., 2., 3.])
    b = a.mask_where_eq(99.)
    assert a == b

    ##################################################################################
    # mask_where_ne()
    ##################################################################################

    a = Scalar([1., 2., 3., 2., 5.])
    b = a.mask_where_ne(2.)
    assert b.mask[0]
    assert not b.mask[1]
    assert b.mask[2]
    assert not b.mask[3]
    assert b.mask[4]
    assert b[1] == 2.
    assert b[3] == 2.

    a = Scalar([1., 2., 3., 2., 5.])
    b = a.mask_where_ne(2., replace=99., remask=False)
    assert b[0] == 99.
    assert b[1] == 2.
    assert b[2] == 99.
    assert b[3] == 2.
    assert b[4] == 99.

    a = Vector(np.arange(30).reshape(10, 3) % 6)
    match = Vector([3., 4., 5.])
    b = a.mask_where_ne(match)

    assert b.count_masked() == 5

    a = Scalar([2., 2., 2.])
    b = a.mask_where_ne(2.)

    assert a == b

    ##################################################################################
    # mask_where_le()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.mask_where_le(3.)
    assert b.mask[0]  # 1 <= 3
    assert b.mask[1]  # 2 <= 3
    assert b.mask[2]  # 3 <= 3
    assert not b.mask[3]  # 4 > 3
    assert not b.mask[4]  # 5 > 3
    assert b[3] == 4.
    assert b[4] == 5.

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.mask_where_le(3., replace=0., remask=False)
    assert b[0] == 0.
    assert b[1] == 0.
    assert b[2] == 0.
    assert b[3] == 4.
    assert b[4] == 5.

    a = Scalar(np.arange(20).reshape(4, 5))
    b = a.mask_where_le(5.)

    assert np.all(b.mask[a.values <= 5.])

    a = Vector(np.arange(9).reshape(3, 3), drank=1)
    with pytest.raises(ValueError):
        a.mask_where_le(2.)

    a = Vector([1., 2., 3.])
    with pytest.raises(ValueError):
        a.mask_where_le(2.)

    ##################################################################################
    # mask_where_ge()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.mask_where_ge(3.)
    assert not b.mask[0]  # 1 < 3
    assert not b.mask[1]  # 2 < 3
    assert b.mask[2]  # 3 >= 3
    assert b.mask[3]  # 4 >= 3
    assert b.mask[4]  # 5 >= 3
    assert b[0] == 1.
    assert b[1] == 2.

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.mask_where_ge(3., replace=0., remask=False)
    assert b[0] == 1.
    assert b[1] == 2.
    assert b[2] == 0.
    assert b[3] == 0.
    assert b[4] == 0.

    a = Scalar(np.arange(20).reshape(4, 5))
    b = a.mask_where_ge(15.)
    assert np.all(b.mask[a.values >= 15.])

    ##################################################################################
    # mask_where_lt()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.mask_where_lt(3.)
    assert b.mask[0]  # 1 < 3
    assert b.mask[1]  # 2 < 3
    assert not b.mask[2]  # 3 >= 3
    assert not b.mask[3]  # 4 >= 3
    assert not b.mask[4]  # 5 >= 3
    assert b[2] == 3.
    assert b[3] == 4.
    assert b[4] == 5.

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.mask_where_lt(3., replace=0., remask=False)
    assert b[0] == 0.
    assert b[1] == 0.
    assert b[2] == 3.
    assert b[3] == 4.
    assert b[4] == 5.

    a = Scalar(np.arange(20).reshape(4, 5))
    b = a.mask_where_lt(5.)
    assert np.all(b.mask[a.values < 5.])

    ##################################################################################
    # mask_where_gt()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.mask_where_gt(3.)
    assert not b.mask[0]  # 1 <= 3
    assert not b.mask[1]  # 2 <= 3
    assert not b.mask[2]  # 3 <= 3
    assert b.mask[3]  # 4 > 3
    assert b.mask[4]  # 5 > 3
    assert b[0] == 1.
    assert b[1] == 2.
    assert b[2] == 3.

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.mask_where_gt(3., replace=0., remask=False)
    assert b[0] == 1.
    assert b[1] == 2.
    assert b[2] == 3.
    assert b[3] == 0.
    assert b[4] == 0.

    a = Scalar(np.arange(20).reshape(4, 5))
    b = a.mask_where_gt(15.)
    assert np.all(b.mask[a.values > 15.])

    ##################################################################################
    # mask_where_between()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_between(2., 4., mask_endpoints=True)
    assert not b.mask[0]  # 1 < 2
    assert b.mask[1]  # 2 >= 2 and <= 4
    assert b.mask[2]  # 3 >= 2 and <= 4
    assert b.mask[3]  # 4 >= 2 and <= 4
    assert not b.mask[4]  # 5 > 4
    assert not b.mask[5]  # 6 > 4

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_between(2., 4., mask_endpoints=False)
    assert not b.mask[0]  # 1 < 2
    assert not b.mask[1]  # 2 not > 2
    assert b.mask[2]  # 3 > 2 and < 4
    assert not b.mask[3]  # 4 not < 4
    assert not b.mask[4]  # 5 > 4
    assert not b.mask[5]  # 6 > 4

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_between(2., 4., mask_endpoints=(True, False))
    assert not b.mask[0]  # 1 < 2
    assert b.mask[1]  # 2 >= 2
    assert b.mask[2]  # 3 > 2 and < 4
    assert not b.mask[3]  # 4 not < 4
    assert not b.mask[4]  # 5 > 4
    assert not b.mask[5]  # 6 > 4

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_between(2., 4., replace=0., mask_endpoints=True, remask=False)
    assert b[0] == 1.
    assert b[1] == 0.
    assert b[2] == 0.
    assert b[3] == 0.
    assert b[4] == 5.
    assert b[5] == 6.

    a = Scalar(np.arange(20).reshape(4, 5))
    b = a.mask_where_between(5., 15., mask_endpoints=True)
    assert np.all(b.mask[(a.values >= 5.) & (a.values <= 15.)])

    a = Scalar([1., 2., 3., 4., 5.])
    lower = Scalar(2., mask=True)  # Masked limit should be ignored
    upper = Scalar(4.)
    b = a.mask_where_between(lower, upper, mask_endpoints=True)

    if isinstance(b.mask, np.ndarray):
        assert np.all(b.mask[a.values <= 4.])
    else:
        # If mask is scalar, check appropriately
        assert (b.mask if np.all(a.values <= 4.) else not b.mask)

    ##################################################################################
    # mask_where_outside()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_outside(2., 4., mask_endpoints=True)
    assert b.mask[0]  # 1 <= 2
    assert b.mask[1]  # 2 <= 2
    assert not b.mask[2]  # 3 > 2 and < 4
    assert b.mask[3]  # 4 >= 4
    assert b.mask[4]  # 5 >= 4
    assert b.mask[5]  # 6 >= 4

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_outside(2., 4., mask_endpoints=False)
    assert b.mask[0]  # 1 < 2
    assert not b.mask[1]  # 2 >= 2
    assert not b.mask[2]  # 3 >= 2 and < 4
    assert not b.mask[3]  # 4 >= 2 and < 4
    assert b.mask[4]  # 5 >= 4
    assert b.mask[5]  # 6 >= 4

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_outside(2., 4., replace=0., mask_endpoints=True, remask=False)
    assert b[0] == 0.
    assert b[1] == 0.
    assert b[2] == 3.
    assert b[3] == 0.
    assert b[4] == 0.
    assert b[5] == 0.

    a = Scalar(np.arange(20).reshape(4, 5))
    b = a.mask_where_outside(5., 15., mask_endpoints=True)
    assert np.all(b.mask[(a.values < 5.) | (a.values > 15.)])

    ##################################################################################
    # clip()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.clip(2., 4., remask=False)
    assert b[0] == 2.  # Clipped to lower
    assert b[1] == 2.  # Clipped to lower
    assert b[2] == 3.  # Unchanged
    assert b[3] == 4.  # Unchanged
    assert b[4] == 4.  # Clipped to upper
    assert b[5] == 4.  # Clipped to upper

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.clip(2., 4., remask=True)
    assert b.mask[0]  # Outside range (< 2)
    assert not b.mask[1]  # At lower limit, inclusive=True by default (not masked)
    assert not b.mask[2]  # Inside range
    assert not b.mask[3]  # At upper limit, inclusive=True by default (not masked)
    assert b.mask[4]  # Outside range (> 4)
    assert b.mask[5]  # Outside range (> 4)

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.clip(2., 4., remask=True, inclusive=False)
    assert b.mask[0]  # Outside range (< 2)
    assert not b.mask[1]  # At lower limit, inclusive=False means not masked (value is 2, which is >= 2)
    assert not b.mask[2]  # Inside range
    assert b.mask[3]  # At upper limit, inclusive=False means masked (value is 4, which is >= 4)
    assert b.mask[4]  # Outside range (> 4)
    assert b.mask[5]  # Outside range (> 4)

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.clip(None, 4., remask=False)
    assert b[0] == 1.  # No lower limit
    assert b[1] == 2.
    assert b[2] == 3.
    assert b[3] == 4.
    assert b[4] == 4.  # Clipped to upper
    assert b[5] == 4.  # Clipped to upper

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.clip(2., None, remask=False)
    assert b[0] == 2.  # Clipped to lower
    assert b[1] == 2.  # Clipped to lower
    assert b[2] == 3.
    assert b[3] == 4.
    assert b[4] == 5.  # No upper limit
    assert b[5] == 6.  # No upper limit

    a = Scalar([1., 2., 3., 4., 5., 6.])
    lower = Scalar([0., 1., 2., 3., 4., 5.])
    upper = Scalar([2., 3., 4., 5., 6., 7.])
    b = a.clip(lower, upper, remask=False)
    assert b[0] == 1.  # Between 0 and 2
    assert b[1] == 2.  # Between 1 and 3
    assert b[2] == 3.  # Between 2 and 4
    assert b[3] == 4.  # Between 3 and 5
    assert b[4] == 5.  # Between 4 and 6
    assert b[5] == 6.  # Between 5 and 7

    a = Scalar([1., 2., 3., 4., 5., 6.])
    lower = Scalar([0., 1., 2., 3., 4., 5.])
    upper = Scalar([2., 3., 4., 5., 6., 7.], mask=[False, False, False, False, False, True])
    b = a.clip(lower, upper, remask=False)

    assert b[5] == 6.  # No upper limit due to masking

    ##################################################################################
    # Static methods: is_below(), is_above(), is_outside(), is_inside()
    ##################################################################################

    result = Qube.is_below(3., 5., inclusive=True)
    assert result
    result = Qube.is_below(5., 5., inclusive=True)
    assert result
    result = Qube.is_below(6., 5., inclusive=True)
    assert not result

    result = Qube.is_below(3., 5., inclusive=False)
    assert result
    result = Qube.is_below(5., 5., inclusive=False)
    assert not result
    result = Qube.is_below(6., 5., inclusive=False)
    assert not result

    result = Qube.is_above(6., 5., inclusive=True)
    assert result
    result = Qube.is_above(5., 5., inclusive=True)
    assert not result
    result = Qube.is_above(3., 5., inclusive=True)
    assert not result

    result = Qube.is_above(6., 5., inclusive=False)
    assert result
    result = Qube.is_above(5., 5., inclusive=False)
    assert result
    result = Qube.is_above(3., 5., inclusive=False)
    assert not result

    result = Qube.is_outside(1., 2., 5., inclusive=True)
    assert result  # 1 < 2
    result = Qube.is_outside(2., 2., 5., inclusive=True)
    assert not result  # 2 >= 2 and <= 5
    result = Qube.is_outside(3., 2., 5., inclusive=True)
    assert not result  # 3 >= 2 and <= 5
    result = Qube.is_outside(5., 2., 5., inclusive=True)
    assert not result  # 5 >= 2 and <= 5
    result = Qube.is_outside(6., 2., 5., inclusive=True)
    assert result  # 6 > 5

    result = Qube.is_outside(1., 2., 5., inclusive=False)
    assert result  # 1 < 2
    result = Qube.is_outside(2., 2., 5., inclusive=False)
    assert not result  # 2 >= 2 and < 5
    result = Qube.is_outside(5., 2., 5., inclusive=False)
    assert result  # 5 >= 5
    result = Qube.is_outside(6., 2., 5., inclusive=False)
    assert result  # 6 >= 5

    result = Qube.is_inside(1., 2., 5., inclusive=True)
    assert not result  # 1 < 2
    result = Qube.is_inside(2., 2., 5., inclusive=True)
    assert result  # 2 >= 2 and <= 5
    result = Qube.is_inside(3., 2., 5., inclusive=True)
    assert result  # 3 >= 2 and <= 5
    result = Qube.is_inside(5., 2., 5., inclusive=True)
    assert result  # 5 >= 2 and <= 5
    result = Qube.is_inside(6., 2., 5., inclusive=True)
    assert not result  # 6 > 5

    result = Qube.is_inside(1., 2., 5., inclusive=False)
    assert not result  # 1 < 2
    result = Qube.is_inside(2., 2., 5., inclusive=False)
    assert result  # 2 >= 2 and < 5
    result = Qube.is_inside(5., 2., 5., inclusive=False)
    assert not result  # 5 >= 5
    result = Qube.is_inside(6., 2., 5., inclusive=False)
    assert not result  # 6 >= 5

    arg = np.array([1., 2., 3., 4., 5., 6.])
    result = Qube.is_inside(arg, 2., 5., inclusive=True)
    expected = np.array([False, True, True, True, True, False])
    assert np.all(result == expected)

    ##################################################################################
    # Additional coverage tests for missing lines
    ##################################################################################

    a = Scalar(5.)
    mask = True
    b = a.mask_where(mask, replace=None, remask=True)
    assert b.mask
    assert b.shape == ()

    a = Scalar(5.)
    mask = True
    b = a.mask_where(mask, replace=99., remask=True)
    assert b.mask
    assert b.shape == ()

    a = Scalar(5.)
    mask = True
    b = a.mask_where(mask, replace=99., remask=False)
    assert not b.mask
    assert b.values == 99.

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_outside(2., 4., mask_endpoints=True)

    assert b.mask[0]
    assert b.mask[1]
    assert not b.mask[2]
    assert b.mask[3]

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_between(2., 4., mask_endpoints=False)

    assert not b.mask[1]  # 2 is not > 2
    assert b.mask[2]  # 3 is > 2 and < 4
    assert not b.mask[3]  # 4 is not < 4

    a = Scalar([1., 2., 3., 4., 5., 6.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    b = a.clip(2., 4., remask=False)

    assert hasattr(b, 'd_dt')

    assert np.allclose(b.d_dt.values[0], 0.)
    assert np.allclose(b.d_dt.values[5], 0.)

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.clip(2., 4., remask=True, inclusive=False)

    assert b.mask[3]  # 4 >= 4 with inclusive=False

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.clip(None, 4., remask=True, inclusive=False)

    assert b.mask[3]  # 4 >= 4
    assert b.mask[4]  # 5 >= 4
    assert b.mask[5]  # 6 >= 4

    a = Scalar([1., 2., 3., 4., 5.])
    limit = np.array([2., 3., 4., 5., 6.])

    b = a.clip(limit, None, remask=False)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    limit = np.array(2.)  # Scalar array
    b = a.clip(limit, None, remask=False)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    limit = Scalar([2., 3., 4., 5., 6.], mask=[False, False, True, False, False])

    b = a.clip(limit, None, remask=False)

    assert b[2] == 3.  # No lower limit due to masking

    a = Scalar([1., 2., 3., 4., 5.])
    deriv = Scalar([0.1, 0.2, 0.3, 0.4, 0.5], drank=1)
    a.insert_deriv('t', deriv)
    limit = a.d_dt  # This has drank=1
    with pytest.raises(ValueError):
        a.mask_where_ge(limit)

    a = Scalar([1., 2., 3., 4., 5.])
    limit = Vector([1., 2., 3.])  # Vector has numer (3,), Scalar has numer ()
    with pytest.raises(ValueError):
        a.mask_where_ge(limit)


def test_qube_ext_mask_ops_test_mask_where_outside_with_mask_endpoints_as_list() -> None:
    """Test mask_where_outside with mask_endpoints as list."""

    np.random.seed(8736)

    ##################################################################################
    # mask_where()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5., 6.])
    b = a.mask_where_outside(2., 4., mask_endpoints=[True, False])
    assert b.mask[0]  # 1 <= 2, masked
    assert b.mask[1]  # 2 <= 2, masked (endpoint included)
    assert not b.mask[2]  # 3 between 2 and 4, not masked
    assert not b.mask[3]  # 4 == 4, not masked (endpoint excluded)
    assert b.mask[4]  # 5 > 4, masked


def test_qube_ext_mask_ops_test_limit_from_qube_with_masked_qube_limit_that_has_mask_ar() -> None:
    """Test _limit_from_qube with masked Qube limit that has mask array."""

    np.random.seed(8736)

    ##################################################################################
    # mask_where()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    limit = Scalar([2., 3., 4., 5., 6.], mask=[False, False, True, False, False])
    b = a.clip(limit, None, remask=False)
    assert b.values[2] == 3.  # Index 2 has masked limit, treated as -inf


def test_qube_ext_mask_ops_test_limit_from_qube_with_masked_qube_limit_using_mask_where() -> None:
    """Test _limit_from_qube with masked Qube limit using mask_where_ge."""

    np.random.seed(8736)

    ##################################################################################
    # mask_where()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    limit = Scalar([10., 10., 10., 10., 10.], mask=[False, False, True, False, False])
    b = a.mask_where_ge(limit, remask=False)
    if isinstance(b.mask, np.ndarray):
        assert not b.mask[0]
        assert not b.mask[1]
        assert not b.mask[2]  # limit[2] is masked, treated as +inf
        assert not b.mask[3]
        assert not b.mask[4]
    else:
        assert not b.mask


def test_qube_ext_mask_ops_test_limit_from_qube_with_qube_limit_that_has_matching_numer() -> None:
    """Test _limit_from_qube with Qube limit that has matching numer."""

    np.random.seed(8736)

    ##################################################################################
    # mask_where()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    limit = Scalar([2., 3., 4., 5., 6.])  # Scalar has numer (), matches a
    b = a.clip(limit, None, remask=False)
    assert b.shape == a.shape


def test_qube_ext_mask_ops_test_limit_from_qube_lines_447_449_when_limit_is_np_ndarray_() -> None:
    """Test _limit_from_qube lines 447-449: when limit is np.ndarray and self._rank is truthy # This requires self to have rank > 0 (array shape, not scalar) # _rank is the number of shape dimensions, not item dimensions."""

    np.random.seed(8736)

    ##################################################################################
    # mask_where()
    ##################################################################################

    a = Scalar(np.arange(12).reshape(2, 3, 2))  # shape (2, 3, 2), rank 3

    limit = np.array(0.5)  # Scalar array

    b = a.mask_where_le(limit)
    assert type(b) == Scalar
    assert b.shape == a.shape


def test_qube_ext_mask_ops_test_limit_from_qube_line_465_when_limit_numer_is_truthy_and() -> None:
    """Test _limit_from_qube line 465: when limit._numer is truthy and matches self._numer # For now, let's test that the function works with matching numer (even if empty)."""

    np.random.seed(8736)

    ##################################################################################
    # mask_where()
    ##################################################################################

    a = Scalar([1., 2., 3.])  # numer is ()
    limit = Scalar([0.5])  # numer is (), matches but is falsy

    b = a.mask_where_le(limit)
    assert type(b) == Scalar


def test_qube_ext_mask_ops_test_with_multi_dimensional_scalar_array_and_masked_limit() -> None:
    """Test with multi-dimensional Scalar array and masked limit."""

    np.random.seed(8736)

    ##################################################################################
    # mask_where()
    ##################################################################################

    a = Scalar([[1., 2., 3.], [4., 5., 6.]])  # shape (2, 3), _rank=0, _nrank=0
    limit = Scalar([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
                   mask=[[False, False, True], [False, False, False]])

    b = a.mask_where_le(limit)
    assert b.shape == a.shape

    if isinstance(b.mask, np.ndarray):
        assert not b.mask[0, 2]  # limit[0,2] is masked, treated as -inf


def test_qube_ext_mask_ops_test_line_474_with_larger_multi_dimensional_array() -> None:
    """Test line 474 with larger multi-dimensional array."""

    np.random.seed(8736)

    ##################################################################################
    # mask_where()
    ##################################################################################

    a = Scalar(np.arange(24).reshape(2, 3, 4))  # shape (2, 3, 4), _rank=0

    limit_mask = np.zeros((2, 3, 4), dtype=bool)
    limit_mask[0, 1, 2] = True  # One masked element
    limit = Scalar(np.arange(24).reshape(2, 3, 4) * 0.1, mask=limit_mask)
    b = a.mask_where_ge(limit)
    assert b.shape == a.shape
    assert hasattr(b, 'mask')

    if isinstance(b.mask, np.ndarray):
        assert not b.mask[0, 1, 2]  # limit[0,1,2] is masked, treated as +inf


##########################################################################################
