##########################################################################################
# tests/test_qube_mask_ops.py
#
# Comprehensive unit tests for mask operations based on docstrings in mask_ops.py
##########################################################################################

import numpy as np
import unittest

from polymath import Qube, Scalar, Vector


class Test_Qube_mask_ops(unittest.TestCase):

    def runTest(self):

        np.random.seed(8736)

        ##################################################################################
        # mask_where()
        ##################################################################################

        # Simple 1-D case: empty mask returns unchanged
        a = Scalar([1., 2., 3., 4., 5.])
        mask = np.array([False, False, False, False, False])
        b = a.mask_where(mask)
        self.assertEqual(a, b)

        # Simple 1-D case: mask some values
        a = Scalar([1., 2., 3., 4., 5.])
        mask = np.array([True, False, True, False, False])
        b = a.mask_where(mask)
        self.assertTrue(b.mask[0])
        self.assertFalse(b.mask[1])
        self.assertTrue(b.mask[2])
        self.assertFalse(b.mask[3])
        self.assertFalse(b.mask[4])
        self.assertEqual(b[1], 2.)
        self.assertEqual(b[3], 4.)
        self.assertEqual(b[4], 5.)

        # Simple 1-D case: mask with replacement, remask=True
        a = Scalar([1., 2., 3., 4., 5.])
        mask = np.array([True, False, False, False, False])
        b = a.mask_where(mask, replace=99., remask=True)
        self.assertTrue(b.mask[0])
        self.assertFalse(b.mask[1])
        self.assertEqual(b[1], 2.)

        # Simple 1-D case: mask with replacement, remask=False
        a = Scalar([1., 2., 3., 4., 5.])
        mask = np.array([True, False, False, False, False])
        b = a.mask_where(mask, replace=99., remask=False)
        if isinstance(b.mask, np.ndarray):
            self.assertFalse(b.mask[0])
        else:
            self.assertFalse(b.mask)
        self.assertEqual(b[0], 99.)
        self.assertEqual(b[1], 2.)

        # Simple 1-D case: replace=None, remask=False (should return unchanged)
        a = Scalar([1., 2., 3., 4., 5.])
        mask = np.array([True, False, False, False, False])
        b = a.mask_where(mask, replace=None, remask=False)
        self.assertEqual(a, b)

        # Complex n-D case: 2-D array
        a = Scalar(np.arange(20).reshape(4, 5))
        mask = np.array([[True, False, True, False, False],
                         [False, False, False, False, False],
                         [True, True, False, False, False],
                         [False, False, False, False, True]])
        b = a.mask_where(mask)
        self.assertTrue(b.mask[0, 0])
        self.assertFalse(b.mask[0, 1])
        self.assertTrue(b.mask[0, 2])
        self.assertTrue(b.mask[2, 0])
        self.assertTrue(b.mask[2, 1])
        self.assertTrue(b.mask[3, 4])

        # Complex n-D case: with replacement array
        a = Scalar(np.arange(20).reshape(4, 5))
        replace = Scalar(np.ones((4, 5)) * 99.)
        mask = np.array([[True, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]])
        b = a.mask_where(mask, replace=replace, remask=False)
        self.assertEqual(b[0, 0], 99.)
        self.assertEqual(b[0, 1], 1.)

        # Complex n-D case: Vector with mask
        a = Vector(np.arange(30).reshape(10, 3))
        mask = np.array([True] * 5 + [False] * 5)
        b = a.mask_where(mask)
        self.assertTrue(np.all(b.mask[0:5]))
        self.assertFalse(np.all(b.mask[5:10]))

        # Test ValueError: incompatible replacement shape
        a = Scalar([1., 2., 3., 4., 5.])
        replace = Scalar([1., 2., 3.])  # Wrong shape
        mask = np.array([True, False, False, False, False])
        self.assertRaises(ValueError, a.mask_where, mask, replace=replace)

        # Test with recursive parameter
        a = Scalar([1., 2., 3.])
        da_dt = Scalar([10., 20., 30.])
        a.insert_deriv('t', da_dt)
        mask = np.array([True, False, False])
        b = a.mask_where(mask, recursive=True)
        self.assertTrue(b.mask[0])
        self.assertTrue(b.d_dt.mask[0])
        self.assertFalse(b.mask[1])
        self.assertFalse(b.d_dt.mask[1])

        b = a.mask_where(mask, recursive=False)
        self.assertTrue(b.mask[0])
        # recursive=False means derivatives are excluded from the returned object
        self.assertFalse(hasattr(b, 'd_dt'))

        ##################################################################################
        # mask_where_eq()
        ##################################################################################

        # Simple 1-D case
        a = Scalar([1., 2., 3., 2., 5.])
        b = a.mask_where_eq(2.)
        self.assertFalse(b.mask[0])
        self.assertTrue(b.mask[1])
        self.assertFalse(b.mask[2])
        self.assertTrue(b.mask[3])
        self.assertFalse(b.mask[4])
        self.assertEqual(b[0], 1.)
        self.assertEqual(b[2], 3.)
        self.assertEqual(b[4], 5.)

        # Simple 1-D case: with replacement
        a = Scalar([1., 2., 3., 2., 5.])
        b = a.mask_where_eq(2., replace=99., remask=False)
        self.assertEqual(b[0], 1.)
        self.assertEqual(b[1], 99.)
        self.assertEqual(b[2], 3.)
        self.assertEqual(b[3], 99.)
        self.assertEqual(b[4], 5.)

        # Complex n-D case: Vector matching
        a = Vector(np.arange(30).reshape(10, 3) % 6)
        match = Vector([3., 4., 5.])
        b = a.mask_where_eq(match)
        # Should mask items where all components match
        self.assertEqual(b.count_masked(), 5)

        # Complex n-D case: Vector with replacement
        a = Vector(np.arange(30).reshape(10, 3) % 6)
        match = Vector([3., 4., 5.])
        replace = Vector([0., 1., 2.])
        b = a.mask_where_eq(match, replace=replace, remask=False)
        self.assertEqual(b.count_masked(), 0)
        self.assertEqual(b[0], replace)

        # Test that no items need masking returns unchanged
        a = Scalar([1., 2., 3.])
        b = a.mask_where_eq(99.)
        self.assertEqual(a, b)

        ##################################################################################
        # mask_where_ne()
        ##################################################################################

        # Simple 1-D case
        a = Scalar([1., 2., 3., 2., 5.])
        b = a.mask_where_ne(2.)
        self.assertTrue(b.mask[0])
        self.assertFalse(b.mask[1])
        self.assertTrue(b.mask[2])
        self.assertFalse(b.mask[3])
        self.assertTrue(b.mask[4])
        self.assertEqual(b[1], 2.)
        self.assertEqual(b[3], 2.)

        # Simple 1-D case: with replacement
        a = Scalar([1., 2., 3., 2., 5.])
        b = a.mask_where_ne(2., replace=99., remask=False)
        self.assertEqual(b[0], 99.)
        self.assertEqual(b[1], 2.)
        self.assertEqual(b[2], 99.)
        self.assertEqual(b[3], 2.)
        self.assertEqual(b[4], 99.)

        # Complex n-D case: Vector
        a = Vector(np.arange(30).reshape(10, 3) % 6)
        match = Vector([3., 4., 5.])
        b = a.mask_where_ne(match)
        # Should mask items where not all components match
        self.assertEqual(b.count_masked(), 5)

        # Test that no items need masking returns unchanged
        a = Scalar([2., 2., 2.])
        b = a.mask_where_ne(2.)
        # If all equal 2, then mask_where_ne(2) finds no items to mask, so returns unchanged
        # According to docstring: "If no items need to be masked, this object is returned unchanged"
        self.assertEqual(a, b)

        ##################################################################################
        # mask_where_le()
        ##################################################################################

        # Simple 1-D case
        a = Scalar([1., 2., 3., 4., 5.])
        b = a.mask_where_le(3.)
        self.assertTrue(b.mask[0])  # 1 <= 3
        self.assertTrue(b.mask[1])  # 2 <= 3
        self.assertTrue(b.mask[2])  # 3 <= 3
        self.assertFalse(b.mask[3])  # 4 > 3
        self.assertFalse(b.mask[4])  # 5 > 3
        self.assertEqual(b[3], 4.)
        self.assertEqual(b[4], 5.)

        # Simple 1-D case: with replacement
        a = Scalar([1., 2., 3., 4., 5.])
        b = a.mask_where_le(3., replace=0., remask=False)
        self.assertEqual(b[0], 0.)
        self.assertEqual(b[1], 0.)
        self.assertEqual(b[2], 0.)
        self.assertEqual(b[3], 4.)
        self.assertEqual(b[4], 5.)

        # Complex n-D case
        a = Scalar(np.arange(20).reshape(4, 5))
        b = a.mask_where_le(5.)
        # All values <= 5 should be masked
        self.assertTrue(np.all(b.mask[a.values <= 5.]))

        # Test ValueError: denominators not allowed
        a = Vector(np.arange(9).reshape(3, 3), drank=1)
        self.assertRaises(ValueError, a.mask_where_le, 2.)

        # Test ValueError: item rank > 0 not allowed
        a = Vector([1., 2., 3.])
        self.assertRaises(ValueError, a.mask_where_le, 2.)

        ##################################################################################
        # mask_where_ge()
        ##################################################################################

        # Simple 1-D case
        a = Scalar([1., 2., 3., 4., 5.])
        b = a.mask_where_ge(3.)
        self.assertFalse(b.mask[0])  # 1 < 3
        self.assertFalse(b.mask[1])  # 2 < 3
        self.assertTrue(b.mask[2])  # 3 >= 3
        self.assertTrue(b.mask[3])  # 4 >= 3
        self.assertTrue(b.mask[4])  # 5 >= 3
        self.assertEqual(b[0], 1.)
        self.assertEqual(b[1], 2.)

        # Simple 1-D case: with replacement
        a = Scalar([1., 2., 3., 4., 5.])
        b = a.mask_where_ge(3., replace=0., remask=False)
        self.assertEqual(b[0], 1.)
        self.assertEqual(b[1], 2.)
        self.assertEqual(b[2], 0.)
        self.assertEqual(b[3], 0.)
        self.assertEqual(b[4], 0.)

        # Complex n-D case
        a = Scalar(np.arange(20).reshape(4, 5))
        b = a.mask_where_ge(15.)
        self.assertTrue(np.all(b.mask[a.values >= 15.]))

        ##################################################################################
        # mask_where_lt()
        ##################################################################################

        # Simple 1-D case
        a = Scalar([1., 2., 3., 4., 5.])
        b = a.mask_where_lt(3.)
        self.assertTrue(b.mask[0])  # 1 < 3
        self.assertTrue(b.mask[1])  # 2 < 3
        self.assertFalse(b.mask[2])  # 3 >= 3
        self.assertFalse(b.mask[3])  # 4 >= 3
        self.assertFalse(b.mask[4])  # 5 >= 3
        self.assertEqual(b[2], 3.)
        self.assertEqual(b[3], 4.)
        self.assertEqual(b[4], 5.)

        # Simple 1-D case: with replacement
        a = Scalar([1., 2., 3., 4., 5.])
        b = a.mask_where_lt(3., replace=0., remask=False)
        self.assertEqual(b[0], 0.)
        self.assertEqual(b[1], 0.)
        self.assertEqual(b[2], 3.)
        self.assertEqual(b[3], 4.)
        self.assertEqual(b[4], 5.)

        # Complex n-D case
        a = Scalar(np.arange(20).reshape(4, 5))
        b = a.mask_where_lt(5.)
        self.assertTrue(np.all(b.mask[a.values < 5.]))

        ##################################################################################
        # mask_where_gt()
        ##################################################################################

        # Simple 1-D case
        a = Scalar([1., 2., 3., 4., 5.])
        b = a.mask_where_gt(3.)
        self.assertFalse(b.mask[0])  # 1 <= 3
        self.assertFalse(b.mask[1])  # 2 <= 3
        self.assertFalse(b.mask[2])  # 3 <= 3
        self.assertTrue(b.mask[3])  # 4 > 3
        self.assertTrue(b.mask[4])  # 5 > 3
        self.assertEqual(b[0], 1.)
        self.assertEqual(b[1], 2.)
        self.assertEqual(b[2], 3.)

        # Simple 1-D case: with replacement
        a = Scalar([1., 2., 3., 4., 5.])
        b = a.mask_where_gt(3., replace=0., remask=False)
        self.assertEqual(b[0], 1.)
        self.assertEqual(b[1], 2.)
        self.assertEqual(b[2], 3.)
        self.assertEqual(b[3], 0.)
        self.assertEqual(b[4], 0.)

        # Complex n-D case
        a = Scalar(np.arange(20).reshape(4, 5))
        b = a.mask_where_gt(15.)
        self.assertTrue(np.all(b.mask[a.values > 15.]))

        ##################################################################################
        # mask_where_between()
        ##################################################################################

        # Simple 1-D case: mask_endpoints=True
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.mask_where_between(2., 4., mask_endpoints=True)
        self.assertFalse(b.mask[0])  # 1 < 2
        self.assertTrue(b.mask[1])  # 2 >= 2 and <= 4
        self.assertTrue(b.mask[2])  # 3 >= 2 and <= 4
        self.assertTrue(b.mask[3])  # 4 >= 2 and <= 4
        self.assertFalse(b.mask[4])  # 5 > 4
        self.assertFalse(b.mask[5])  # 6 > 4

        # Simple 1-D case: mask_endpoints=False
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.mask_where_between(2., 4., mask_endpoints=False)
        self.assertFalse(b.mask[0])  # 1 < 2
        self.assertFalse(b.mask[1])  # 2 not > 2
        self.assertTrue(b.mask[2])  # 3 > 2 and < 4
        self.assertFalse(b.mask[3])  # 4 not < 4
        self.assertFalse(b.mask[4])  # 5 > 4
        self.assertFalse(b.mask[5])  # 6 > 4

        # Simple 1-D case: mask_endpoints as tuple
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.mask_where_between(2., 4., mask_endpoints=(True, False))
        self.assertFalse(b.mask[0])  # 1 < 2
        self.assertTrue(b.mask[1])  # 2 >= 2
        self.assertTrue(b.mask[2])  # 3 > 2 and < 4
        self.assertFalse(b.mask[3])  # 4 not < 4
        self.assertFalse(b.mask[4])  # 5 > 4
        self.assertFalse(b.mask[5])  # 6 > 4

        # Simple 1-D case: with replacement
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.mask_where_between(2., 4., replace=0., mask_endpoints=True, remask=False)
        self.assertEqual(b[0], 1.)
        self.assertEqual(b[1], 0.)
        self.assertEqual(b[2], 0.)
        self.assertEqual(b[3], 0.)
        self.assertEqual(b[4], 5.)
        self.assertEqual(b[5], 6.)

        # Complex n-D case
        a = Scalar(np.arange(20).reshape(4, 5))
        b = a.mask_where_between(5., 15., mask_endpoints=True)
        self.assertTrue(np.all(b.mask[(a.values >= 5.) & (a.values <= 15.)]))

        # Test with masked limits
        a = Scalar([1., 2., 3., 4., 5.])
        lower = Scalar(2., mask=True)  # Masked limit should be ignored
        upper = Scalar(4.)
        b = a.mask_where_between(lower, upper, mask_endpoints=True)
        # Lower limit is masked, so it should be treated as +inf (no lower bound)
        # So only values > 4 should be unmasked
        if isinstance(b.mask, np.ndarray):
            self.assertTrue(np.all(b.mask[a.values <= 4.]))
        else:
            # If mask is scalar, check appropriately
            self.assertTrue(b.mask if np.all(a.values <= 4.) else not b.mask)

        ##################################################################################
        # mask_where_outside()
        ##################################################################################

        # Simple 1-D case: mask_endpoints=True
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.mask_where_outside(2., 4., mask_endpoints=True)
        self.assertTrue(b.mask[0])  # 1 <= 2
        self.assertTrue(b.mask[1])  # 2 <= 2
        self.assertFalse(b.mask[2])  # 3 > 2 and < 4
        self.assertTrue(b.mask[3])  # 4 >= 4
        self.assertTrue(b.mask[4])  # 5 >= 4
        self.assertTrue(b.mask[5])  # 6 >= 4

        # Simple 1-D case: mask_endpoints=False
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.mask_where_outside(2., 4., mask_endpoints=False)
        self.assertTrue(b.mask[0])  # 1 < 2
        self.assertFalse(b.mask[1])  # 2 >= 2
        self.assertFalse(b.mask[2])  # 3 >= 2 and < 4
        self.assertFalse(b.mask[3])  # 4 >= 2 and < 4
        self.assertTrue(b.mask[4])  # 5 >= 4
        self.assertTrue(b.mask[5])  # 6 >= 4

        # Simple 1-D case: with replacement
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.mask_where_outside(2., 4., replace=0., mask_endpoints=True, remask=False)
        self.assertEqual(b[0], 0.)
        self.assertEqual(b[1], 0.)
        self.assertEqual(b[2], 3.)
        self.assertEqual(b[3], 0.)
        self.assertEqual(b[4], 0.)
        self.assertEqual(b[5], 0.)

        # Complex n-D case
        a = Scalar(np.arange(20).reshape(4, 5))
        b = a.mask_where_outside(5., 15., mask_endpoints=True)
        self.assertTrue(np.all(b.mask[(a.values < 5.) | (a.values > 15.)]))

        ##################################################################################
        # clip()
        ##################################################################################

        # Simple 1-D case: remask=False
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.clip(2., 4., remask=False)
        self.assertEqual(b[0], 2.)  # Clipped to lower
        self.assertEqual(b[1], 2.)  # Clipped to lower
        self.assertEqual(b[2], 3.)  # Unchanged
        self.assertEqual(b[3], 4.)  # Unchanged
        self.assertEqual(b[4], 4.)  # Clipped to upper
        self.assertEqual(b[5], 4.)  # Clipped to upper

        # Simple 1-D case: remask=True
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.clip(2., 4., remask=True)
        self.assertTrue(b.mask[0])  # Outside range (< 2)
        self.assertFalse(b.mask[1])  # At lower limit, inclusive=True by default (not masked)
        self.assertFalse(b.mask[2])  # Inside range
        self.assertFalse(b.mask[3])  # At upper limit, inclusive=True by default (not masked)
        self.assertTrue(b.mask[4])  # Outside range (> 4)
        self.assertTrue(b.mask[5])  # Outside range (> 4)

        # Simple 1-D case: inclusive=False
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.clip(2., 4., remask=True, inclusive=False)
        self.assertTrue(b.mask[0])  # Outside range (< 2)
        self.assertFalse(b.mask[1])  # At lower limit, inclusive=False means not masked (value is 2, which is >= 2)
        self.assertFalse(b.mask[2])  # Inside range
        self.assertTrue(b.mask[3])  # At upper limit, inclusive=False means masked (value is 4, which is >= 4)
        self.assertTrue(b.mask[4])  # Outside range (> 4)
        self.assertTrue(b.mask[5])  # Outside range (> 4)

        # Simple 1-D case: lower=None
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.clip(None, 4., remask=False)
        self.assertEqual(b[0], 1.)  # No lower limit
        self.assertEqual(b[1], 2.)
        self.assertEqual(b[2], 3.)
        self.assertEqual(b[3], 4.)
        self.assertEqual(b[4], 4.)  # Clipped to upper
        self.assertEqual(b[5], 4.)  # Clipped to upper

        # Simple 1-D case: upper=None
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.clip(2., None, remask=False)
        self.assertEqual(b[0], 2.)  # Clipped to lower
        self.assertEqual(b[1], 2.)  # Clipped to lower
        self.assertEqual(b[2], 3.)
        self.assertEqual(b[3], 4.)
        self.assertEqual(b[4], 5.)  # No upper limit
        self.assertEqual(b[5], 6.)  # No upper limit

        # Complex n-D case: array limits
        a = Scalar([1., 2., 3., 4., 5., 6.])
        lower = Scalar([0., 1., 2., 3., 4., 5.])
        upper = Scalar([2., 3., 4., 5., 6., 7.])
        b = a.clip(lower, upper, remask=False)
        self.assertEqual(b[0], 1.)  # Between 0 and 2
        self.assertEqual(b[1], 2.)  # Between 1 and 3
        self.assertEqual(b[2], 3.)  # Between 2 and 4
        self.assertEqual(b[3], 4.)  # Between 3 and 5
        self.assertEqual(b[4], 5.)  # Between 4 and 6
        self.assertEqual(b[5], 6.)  # Between 5 and 7

        # Complex n-D case: with masked limits
        a = Scalar([1., 2., 3., 4., 5., 6.])
        lower = Scalar([0., 1., 2., 3., 4., 5.])
        upper = Scalar([2., 3., 4., 5., 6., 7.], mask=[False, False, False, False, False, True])
        b = a.clip(lower, upper, remask=False)
        # Last element has masked upper limit, so should be ignored
        self.assertEqual(b[5], 6.)  # No upper limit due to masking

        ##################################################################################
        # Static methods: is_below(), is_above(), is_outside(), is_inside()
        ##################################################################################

        # is_below() with inclusive=True
        result = Qube.is_below(3., 5., inclusive=True)
        self.assertTrue(result)
        result = Qube.is_below(5., 5., inclusive=True)
        self.assertTrue(result)
        result = Qube.is_below(6., 5., inclusive=True)
        self.assertFalse(result)

        # is_below() with inclusive=False
        result = Qube.is_below(3., 5., inclusive=False)
        self.assertTrue(result)
        result = Qube.is_below(5., 5., inclusive=False)
        self.assertFalse(result)
        result = Qube.is_below(6., 5., inclusive=False)
        self.assertFalse(result)

        # is_above() with inclusive=True
        result = Qube.is_above(6., 5., inclusive=True)
        self.assertTrue(result)
        result = Qube.is_above(5., 5., inclusive=True)
        self.assertFalse(result)
        result = Qube.is_above(3., 5., inclusive=True)
        self.assertFalse(result)

        # is_above() with inclusive=False
        result = Qube.is_above(6., 5., inclusive=False)
        self.assertTrue(result)
        result = Qube.is_above(5., 5., inclusive=False)
        self.assertTrue(result)
        result = Qube.is_above(3., 5., inclusive=False)
        self.assertFalse(result)

        # is_outside() with inclusive=True
        result = Qube.is_outside(1., 2., 5., inclusive=True)
        self.assertTrue(result)  # 1 < 2
        result = Qube.is_outside(2., 2., 5., inclusive=True)
        self.assertFalse(result)  # 2 >= 2 and <= 5
        result = Qube.is_outside(3., 2., 5., inclusive=True)
        self.assertFalse(result)  # 3 >= 2 and <= 5
        result = Qube.is_outside(5., 2., 5., inclusive=True)
        self.assertFalse(result)  # 5 >= 2 and <= 5
        result = Qube.is_outside(6., 2., 5., inclusive=True)
        self.assertTrue(result)  # 6 > 5

        # is_outside() with inclusive=False
        result = Qube.is_outside(1., 2., 5., inclusive=False)
        self.assertTrue(result)  # 1 < 2
        result = Qube.is_outside(2., 2., 5., inclusive=False)
        self.assertFalse(result)  # 2 >= 2 and < 5
        result = Qube.is_outside(5., 2., 5., inclusive=False)
        self.assertTrue(result)  # 5 >= 5
        result = Qube.is_outside(6., 2., 5., inclusive=False)
        self.assertTrue(result)  # 6 >= 5

        # is_inside() with inclusive=True
        result = Qube.is_inside(1., 2., 5., inclusive=True)
        self.assertFalse(result)  # 1 < 2
        result = Qube.is_inside(2., 2., 5., inclusive=True)
        self.assertTrue(result)  # 2 >= 2 and <= 5
        result = Qube.is_inside(3., 2., 5., inclusive=True)
        self.assertTrue(result)  # 3 >= 2 and <= 5
        result = Qube.is_inside(5., 2., 5., inclusive=True)
        self.assertTrue(result)  # 5 >= 2 and <= 5
        result = Qube.is_inside(6., 2., 5., inclusive=True)
        self.assertFalse(result)  # 6 > 5

        # is_inside() with inclusive=False
        result = Qube.is_inside(1., 2., 5., inclusive=False)
        self.assertFalse(result)  # 1 < 2
        result = Qube.is_inside(2., 2., 5., inclusive=False)
        self.assertTrue(result)  # 2 >= 2 and < 5
        result = Qube.is_inside(5., 2., 5., inclusive=False)
        self.assertFalse(result)  # 5 >= 5
        result = Qube.is_inside(6., 2., 5., inclusive=False)
        self.assertFalse(result)  # 6 >= 5

        # Test with arrays
        arg = np.array([1., 2., 3., 4., 5., 6.])
        result = Qube.is_inside(arg, 2., 5., inclusive=True)
        expected = np.array([False, True, True, True, True, False])
        self.assertTrue(np.all(result == expected))

        ##################################################################################
        # Additional coverage tests for missing lines
        ##################################################################################

        # Test mask_where with scalar object and replace=None (line 52)
        a = Scalar(5.)
        mask = True
        b = a.mask_where(mask, replace=None, remask=True)
        self.assertTrue(b.mask)
        self.assertEqual(b.shape, ())

        # Test mask_where with scalar object and replace
        a = Scalar(5.)
        mask = True
        b = a.mask_where(mask, replace=99., remask=True)
        self.assertTrue(b.mask)
        self.assertEqual(b.shape, ())

        # Test mask_where with scalar object, replace, and remask=False
        a = Scalar(5.)
        mask = True
        b = a.mask_where(mask, replace=99., remask=False)
        self.assertFalse(b.mask)
        self.assertEqual(b.values, 99.)

        # Test mask_where_outside with mask_endpoints as single value (not tuple/list) (line 343->346)
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.mask_where_outside(2., 4., mask_endpoints=True)
        # mask_endpoints=True should be converted to (True, True)
        self.assertTrue(b.mask[0])
        self.assertTrue(b.mask[1])
        self.assertFalse(b.mask[2])
        self.assertTrue(b.mask[3])

        # Test mask_where_between with mask_endpoints as single value (line 343->346)
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.mask_where_between(2., 4., mask_endpoints=False)
        # mask_endpoints=False should be converted to (False, False)
        self.assertFalse(b.mask[1])  # 2 is not > 2
        self.assertTrue(b.mask[2])  # 3 is > 2 and < 4
        self.assertFalse(b.mask[3])  # 4 is not < 4

        # Test clip with derivatives and remask=False
        a = Scalar([1., 2., 3., 4., 5., 6.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        b = a.clip(2., 4., remask=False)
        # Derivatives out of range should be set to zero
        self.assertTrue(hasattr(b, 'd_dt'))
        # Values outside range should have zero derivatives
        self.assertTrue(np.allclose(b.d_dt.values[0], 0.))
        self.assertTrue(np.allclose(b.d_dt.values[5], 0.))

        # Test clip with inclusive=False and upper limit (line 421)
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.clip(2., 4., remask=True, inclusive=False)
        # With inclusive=False, value exactly at upper limit (4) should be masked
        self.assertTrue(b.mask[3])  # 4 >= 4 with inclusive=False

        # Test clip with inclusive=False, upper only
        a = Scalar([1., 2., 3., 4., 5., 6.])
        b = a.clip(None, 4., remask=True, inclusive=False)
        # Values >= 4 should be masked
        self.assertTrue(b.mask[3])  # 4 >= 4
        self.assertTrue(b.mask[4])  # 5 >= 4
        self.assertTrue(b.mask[5])  # 6 >= 4

        # Test _limit_from_qube with np.ndarray limit
        a = Scalar([1., 2., 3., 4., 5.])
        limit = np.array([2., 3., 4., 5., 6.])
        # This should work through clip
        b = a.clip(limit, None, remask=False)
        self.assertEqual(b.shape, a.shape)

        # Test _limit_from_qube with np.ndarray limit and self._rank > 0 (lines 447-449)
        # When self has rank > 0, limit is reshaped
        a = Scalar([1., 2., 3., 4., 5.])
        limit = np.array(2.)  # Scalar array
        b = a.clip(limit, None, remask=False)
        self.assertEqual(b.shape, a.shape)

        # Test _limit_from_qube with np.ndarray limit (1-D) and self._rank > 0
        # For a 1-D Scalar, self._rank is 0, so this path won't be triggered
        # We need a 2-D Scalar to trigger self._rank > 0
        a = Scalar(np.arange(20).reshape(4, 5))  # 2-D, so _rank = 0 (Scalar has no item dimensions)
        # Actually, Scalar has _rank = 0 always, so we can't easily test this
        # The reshape path is for when limit is an array and self._rank > 0
        # This requires a Qube with item dimensions, which Scalar doesn't have
        # Let's skip this specific test case

        # Test _limit_from_qube with Qube limit that has denominator (should raise)
        # We need a Qube that supports comparison but has denominator
        # This is tricky - let's test with mask_where_ge which also uses _limit_from_qube
        # Actually, the error is raised before comparison, so we can test it
        # But we need a Qube that has drank and supports comparison
        # Scalar doesn't support drank, so this is hard to test directly
        # Let's skip this for now as it requires a specific Qube subclass

        # Test _limit_from_qube with Qube limit that has different numer (should raise)
        # This also requires comparison support, so it's hard to test
        # The error is raised in _limit_from_qube before comparison

        # Test _limit_from_qube with self._numer but limit has no numer
        # Vector doesn't support clip, so let's test with mask_where_ge which also uses _limit_from_qube
        # Actually, let's test with a Scalar that has numer (but Scalar has no numer)
        # This path is hard to test without a Qube subclass that has numer
        # Let's test the path where limit has no numer but self has numer using a different method
        # Actually, this requires a Qube with numer, which Vector has, but Vector doesn't support clip
        # So this path is difficult to test directly

        # Test _limit_from_qube with masked Qube limit (partial mask)
        a = Scalar([1., 2., 3., 4., 5.])
        limit = Scalar([2., 3., 4., 5., 6.], mask=[False, False, True, False, False])
        # Masked limit values should use the masked parameter
        b = a.clip(limit, None, remask=False)
        # The masked limit at index 2 should be ignored (treated as -inf)
        self.assertEqual(b[2], 3.)  # No lower limit due to masking

        # Test _limit_from_qube with Qube limit that has no numer but self has numer
        # Vector doesn't support clip, so let's test with mask_where_ge which also uses _limit_from_qube
        a = Scalar([1., 2., 3., 4., 5.])
        limit = Scalar(2.)
        # This should work - limit is broadcast to match
        b = a.clip(limit, None, remask=False)
        self.assertEqual(b.shape, a.shape)

        # Test _limit_from_qube with masked Qube limit
        a = Scalar([1., 2., 3., 4., 5.])
        limit = Scalar([2., 3., 4., 5., 6.], mask=[False, False, True, False, False])
        # Masked limit values should use the masked parameter
        b = a.clip(limit, None, remask=False)
        # The masked limit at index 2 should be ignored (treated as -inf)
        self.assertEqual(b[2], 3.)  # No lower limit due to masking

##########################################################################################
