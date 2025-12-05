##########################################################################################
# tests/test_polynomial_operations.py
# Polynomial special operations (deriv, eval, roots) and advanced tests
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Vector, Polynomial


class Test_Polynomial_Operations(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # Test deriv
        # Derivative of x^2 + 2x + 3 is 2x + 2
        p27 = Polynomial([1., 2., 3.])  # x^2 + 2x + 3
        p_deriv = p27.deriv()
        self.assertEqual(type(p_deriv), Polynomial)
        self.assertEqual(p_deriv.order, 1)
        self.assertAlmostEqual(p_deriv.values[0], 2., places=10)
        self.assertAlmostEqual(p_deriv.values[1], 2., places=10)

        # Derivative of constant is zero
        p_const = Polynomial([5.])
        p_deriv_const = p_const.deriv()
        self.assertEqual(p_deriv_const.order, 0)
        self.assertEqual(p_deriv_const.values[0], 0.)

        # Test eval
        # Evaluate x + 2 at x = 3 should give 5
        p28 = Polynomial([1., 2.])  # x + 2
        result = p28.eval(3.)
        self.assertEqual(type(result), Scalar)
        self.assertAlmostEqual(result.values, 5., places=10)

        # Evaluate x^2 + 2x + 3 at x = 2 should give 11
        # [1, 2, 3] with x_powers [x^2, x, 1] gives 1*x^2 + 2*x + 3*1 = x^2 + 2x + 3
        p29 = Polynomial([1., 2., 3.])  # x^2 + 2x + 3
        result2 = p29.eval(2.)
        self.assertAlmostEqual(result2.values, 11., places=10)

        # Test eval with array
        p30 = Polynomial([1., 2.])  # x + 2
        x_vals = Scalar([1., 2., 3.])
        result3 = p30.eval(x_vals)
        self.assertEqual(type(result3), Scalar)
        self.assertEqual(result3.shape, (3,))
        expected = np.array([3., 4., 5.])
        self.assertTrue(np.allclose(result3.values, expected))

        # Test roots for linear polynomial
        # x + 2 = 0 -> x = -2
        p31 = Polynomial([1., 2.])  # x + 2
        roots1 = p31.roots()
        self.assertEqual(type(roots1), Scalar)
        self.assertEqual(roots1.shape, (1,))
        self.assertAlmostEqual(roots1.values[0], -2., places=10)

        # Test roots for quadratic polynomial
        # x^2 - 5x + 6 = 0 -> x = 2 or x = 3
        p32 = Polynomial([1., -5., 6.])  # x^2 - 5x + 6
        roots2 = p32.roots()
        self.assertEqual(type(roots2), Scalar)
        self.assertEqual(roots2.shape, (2,))
        # Roots should be sorted
        self.assertAlmostEqual(roots2.values[0], 2., places=10)
        self.assertAlmostEqual(roots2.values[1], 3., places=10)

        # Test roots raises ValueError for order zero
        p_zero = Polynomial([5.])
        self.assertRaises(ValueError, p_zero.roots)

        # Test with n-D arrays (complicated cases)
        # Create array of polynomials
        coeffs = np.array([
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]]
        ])  # Shape (2, 2, 2) -> 2x2 array of linear polynomials
        p_array = Polynomial(coeffs)
        self.assertEqual(p_array.shape, (2, 2))
        self.assertEqual(p_array.numer, (2,))
        self.assertEqual(p_array.order, 1)

        # Test operations on array of polynomials
        p_array2 = p_array + 1.  # Add constant to each
        self.assertEqual(p_array2.shape, (2, 2))
        self.assertTrue(np.allclose(p_array2.values[..., 1], p_array.values[..., 1] + 1.))

        # Test eval on array of polynomials
        result_array = p_array.eval(2.)
        self.assertEqual(result_array.shape, (2, 2))
        # For polynomial [1, 2] at x=2: 2 + 2 = 4
        self.assertAlmostEqual(result_array.values[0, 0], 4., places=10)

        # Test roots on array of polynomials
        # Use simple linear polynomials: [1, 2] -> root at -2
        coeffs2 = np.array([
            [[1., 2.], [1., 2.]],
            [[1., 2.], [1., 2.]]
        ])
        p_array3 = Polynomial(coeffs2)
        roots_array = p_array3.roots()
        self.assertEqual(roots_array.shape, (1, 2, 2))
        self.assertTrue(np.allclose(roots_array.values[0], -2.))

        # Test with masks
        p_masked = Polynomial([1., 2., 3.], mask=True)
        self.assertTrue(p_masked.mask)
        p_masked2 = p_masked + Polynomial([1., 1., 1.])
        self.assertTrue(p_masked2.mask)

        # Test with partial mask
        mask_array = np.array([[False, True], [False, False]])
        coeffs3 = np.array([
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]]
        ])
        p_partial_mask = Polynomial(coeffs3, mask=mask_array)
        self.assertEqual(p_partial_mask.shape, (2, 2))
        self.assertTrue(np.any(p_partial_mask.mask))

        # Test recursive parameter
        # Create polynomial with derivatives
        p_base = Polynomial([1., 2., 3.])
        p_deriv = Polynomial([0., 2., 6.])  # derivative
        p_base.insert_deriv('t', p_deriv)

        # Test that deriv() respects recursive
        p_deriv_result = p_base.deriv(recursive=True)
        self.assertTrue(hasattr(p_deriv_result, 'd_dt'))

        p_deriv_result2 = p_base.deriv(recursive=False)
        self.assertFalse(hasattr(p_deriv_result2, 'd_dt'))

        # Test that eval respects recursive
        result_recursive = p_base.eval(2., recursive=True)
        self.assertTrue(hasattr(result_recursive, 'd_dt'))

        result_no_recursive = p_base.eval(2., recursive=False)
        self.assertFalse(hasattr(result_no_recursive, 'd_dt'))

        # Test that roots respects recursive
        p_linear_with_deriv = Polynomial([4., 2.])
        p_linear_with_deriv.insert_deriv('t', Polynomial([0., 1.]))
        roots_recursive = p_linear_with_deriv.roots(recursive=True)
        self.assertTrue(hasattr(roots_recursive, 'd_dt'))

        # Test higher order polynomial roots (cubic)
        # x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3) = 0
        p_cubic = Polynomial([1., -6., 11., -6.])  # x^3 - 6x^2 + 11x - 6
        roots_cubic = p_cubic.roots()
        self.assertEqual(type(roots_cubic), Scalar)
        self.assertEqual(roots_cubic.shape, (3,))
        # Roots should be 1, 2, 3 (sorted)
        roots_sorted = np.sort(roots_cubic.values)
        self.assertAlmostEqual(roots_sorted[0], 1., places=8)
        self.assertAlmostEqual(roots_sorted[1], 2., places=8)
        self.assertAlmostEqual(roots_sorted[2], 3., places=8)

        # Additional tests for coverage

        # Test eval with order == 0
        p_const2 = Polynomial([5.])
        result_const = p_const2.eval(10., recursive=True)
        self.assertEqual(type(result_const), Scalar)
        self.assertAlmostEqual(result_const.values, 5., places=10)

        # Test recursive=False path
        result_const_no_rec = p_const2.eval(10., recursive=False)
        self.assertEqual(type(result_const_no_rec), Scalar)
        self.assertAlmostEqual(result_const_no_rec.values, 5., places=10)

        # Test roots with scalar mask
        p_mask_scalar = Polynomial([1., 2.], mask=True)
        roots_masked = p_mask_scalar.roots()
        self.assertTrue(np.all(roots_masked.mask))

        # Test roots with all_zeros case
        p_zeros = Polynomial([0., 0., 1.])  # x^2 = 0
        roots_zeros = p_zeros.roots()
        # Should have root at 0 (masked duplicates)
        self.assertEqual(roots_zeros.shape, (2,))

        # Test roots with scalar shift case
        p_leading_zero = Polynomial([0., 1., 2.])  # x + 2 = 0, leading zero
        roots_shift = p_leading_zero.roots()
        # After shifting, the polynomial is effectively order 1, but roots()
        # returns shape (order,) with extraneous roots. After sort(), masked
        # values become inf, so we check for finite values
        self.assertEqual(roots_shift.shape, (2,))
        # The valid (finite) root should be -2
        valid_roots = roots_shift.values[np.isfinite(roots_shift.values)]
        self.assertEqual(len(valid_roots), 1)
        self.assertAlmostEqual(valid_roots[0], -2., places=10)

        # Test roots mask extraneous zeros
        p_extraneous = Polynomial([0., 0., 1., 2.])  # x + 2 = 0 with leading zeros
        roots_extraneous = p_extraneous.roots()
        # After shifting, the polynomial is effectively order 1, but roots()
        # returns shape (order,) = (3,) with extraneous roots. After sort(),
        # masked values become inf, so we check for finite values
        self.assertEqual(roots_extraneous.shape, (3,))
        # Should have 1 valid root at -2
        valid_roots = roots_extraneous.values[np.isfinite(roots_extraneous.values)]
        self.assertEqual(len(valid_roots), 1)
        self.assertAlmostEqual(valid_roots[0], -2., places=10)

        # Test roots mask duplicated values
        # Create polynomial with duplicate roots: (x-1)^2 = x^2 - 2x + 1
        p_duplicate = Polynomial([1., -2., 1.])
        roots_dup = p_duplicate.roots()
        self.assertEqual(roots_dup.shape, (2,))
        # One root should be masked as duplicate (after sort(), masked values become inf)
        # So we check for inf values instead of mask
        self.assertTrue(np.any(~np.isfinite(roots_dup.values)))

        # Test roots with derivatives
        p_roots_deriv = Polynomial([1., 2.])  # x + 2 = 0 -> x = -2
        p_roots_deriv.insert_deriv('t', Polynomial([0., 1.]))  # derivative: 1
        roots_with_deriv = p_roots_deriv.roots(recursive=True)
        self.assertTrue(hasattr(roots_with_deriv, 'd_dt'))
        # Derivative of root: if x + 2 = 0 and d/dt(x+2) = 1, then dx/dt = -1
        # At root x=-2, derivative of polynomial is 1, so dx/dt = -1/1 = -1
        self.assertAlmostEqual(roots_with_deriv.d_dt.values[0], -1., places=10)

        # Test roots with array mask
        mask_array = np.array([[False, True], [True, False]])
        coeffs_masked = np.array([
            [[1., 2.], [1., 2.]],
            [[1., 2.], [1., 2.]]
        ])
        p_array_mask = Polynomial(coeffs_masked, mask=mask_array)
        roots_array_mask = p_array_mask.roots()
        # Should have masked roots where mask is True
        self.assertEqual(roots_array_mask.shape, (1, 2, 2))

        # Test roots all_zeros with array case
        coeffs_all_zeros = np.array([
            [[0., 0., 1.], [0., 0., 1.]],
            [[0., 0., 1.], [0., 0., 1.]]
        ])
        p_all_zeros_array = Polynomial(coeffs_all_zeros)
        roots_all_zeros_array = p_all_zeros_array.roots()
        # Should handle all zeros case
        self.assertEqual(roots_all_zeros_array.shape, (2, 2, 2))

        # Test roots with array shift case
        coeffs_leading_zeros = np.array([
            [[0., 1., 2.], [0., 1., 2.]],
            [[0., 1., 2.], [0., 1., 2.]]
        ])
        p_array_shift = Polynomial(coeffs_leading_zeros)
        roots_array_shift = p_array_shift.roots()
        # After shifting, the polynomial is effectively order 1, but roots()
        # returns shape (order,) = (2,) with extraneous roots. After sort(),
        # masked values become inf
        self.assertEqual(roots_array_shift.shape, (2, 2, 2))
        # Should have 1 valid root per polynomial (check that finite values exist)
        finite_mask = np.isfinite(roots_array_shift.values)
        self.assertTrue(np.any(finite_mask))
        # Each of the 4 polynomials should have 1 valid root (sum along first axis)
        valid_per_poly = np.sum(finite_mask, axis=0)
        self.assertTrue(np.all(valid_per_poly == 1))

        # Test roots mask extraneous zeros with array
        coeffs_extraneous_array = np.array([
            [[0., 0., 1., 2.], [0., 0., 1., 2.]],
            [[0., 0., 1., 2.], [0., 0., 1., 2.]]
        ])
        p_extraneous_array = Polynomial(coeffs_extraneous_array)
        roots_extraneous_array = p_extraneous_array.roots()
        # After shifting, the polynomial is effectively order 1, but roots()
        # returns shape (order,) = (3,) with extraneous roots
        self.assertEqual(roots_extraneous_array.shape, (3, 2, 2))
        # Should have 1 valid root per polynomial
        finite_mask = np.isfinite(roots_extraneous_array.values)
        valid_per_poly = np.sum(finite_mask, axis=0)
        self.assertTrue(np.all(valid_per_poly == 1))

        # Test roots mask duplicated values with array
        coeffs_dup_array = np.array([
            [[1., -2., 1.], [1., -2., 1.]],
            [[1., -2., 1.], [1., -2., 1.]]
        ])
        p_dup_array = Polynomial(coeffs_dup_array)
        roots_dup_array = p_dup_array.roots()
        # Should mask duplicates (after sort(), masked values become inf)
        self.assertEqual(roots_dup_array.shape, (2, 2, 2))
        self.assertTrue(np.any(~np.isfinite(roots_dup_array.values)))

        # Test eval with order 0 and nested derivatives
        # Create a constant polynomial with derivatives that have derivatives
        p_const_deriv = Polynomial([5.])
        p_deriv1 = Polynomial([0.])  # derivative is constant
        p_deriv1.insert_deriv('s', Polynomial([1.]))  # derivative of derivative
        p_const_deriv.insert_deriv('t', p_deriv1)
        result_const_deriv = p_const_deriv.eval(10., recursive=True)
        self.assertEqual(type(result_const_deriv), Scalar)
        self.assertAlmostEqual(result_const_deriv.values, 5., places=10)
        self.assertTrue(hasattr(result_const_deriv, 'd_dt'))
        # The nested derivative conversion code should execute
        # When converting a constant derivative with nested derivatives, the nested
        # derivative is also constant, so it gets converted to a Scalar
        self.assertEqual(type(result_const_deriv.d_dt), Scalar)

        # Test eval with order 0, derivative with tail
        # This requires a polynomial with drank > 0
        # Create a Vector with drank=1 first
        v_const_drank = Vector(np.array([[5.]]), drank=1)  # shape (), numer (1,), denom (1,)
        p_const_drank = Polynomial(v_const_drank)
        result_const_drank = p_const_drank.eval(10., recursive=False)
        self.assertEqual(type(result_const_drank), Scalar)
        self.assertAlmostEqual(result_const_drank.values, 5., places=10)

        # Test eval with order 0, derivative with tail (drank > 0)
        v_const_deriv_drank = Vector(np.array([[7.]]), drank=1)
        p_const_deriv_drank = Polynomial(v_const_deriv_drank)
        v_deriv_drank = Vector(np.array([[0.]]), drank=1)
        p_deriv_drank = Polynomial(v_deriv_drank)
        p_const_deriv_drank.insert_deriv('t', p_deriv_drank)
        result_const_deriv_drank = p_const_deriv_drank.eval(20., recursive=True)
        self.assertEqual(type(result_const_deriv_drank), Scalar)
        self.assertAlmostEqual(result_const_deriv_drank.values, 7., places=10)
        self.assertTrue(hasattr(result_const_deriv_drank, 'd_dt'))

        # Test eval with order 0, nested derivatives with tail (drank > 0)
        # This tests the full nested derivative conversion path
        v_const_nested_drank = Vector(np.array([[9.]]), drank=1)
        p_const_nested_drank = Polynomial(v_const_nested_drank)
        v_deriv_nested = Vector(np.array([[0.]]), drank=1)
        p_deriv_nested = Polynomial(v_deriv_nested)
        # Test nested derivative that is constant (order 0) with tail
        v_deriv_nested2 = Vector(np.array([[1.]]), drank=1)
        p_deriv_nested2 = Polynomial(v_deriv_nested2)
        p_deriv_nested.insert_deriv('s', p_deriv_nested2)
        p_const_nested_drank.insert_deriv('t', p_deriv_nested)
        result_const_nested_drank = p_const_nested_drank.eval(30., recursive=True)
        self.assertEqual(type(result_const_nested_drank), Scalar)
        self.assertAlmostEqual(result_const_nested_drank.values, 9., places=10)
        self.assertTrue(hasattr(result_const_nested_drank, 'd_dt'))
        # The nested derivative 's' should be converted to a Scalar
        self.assertEqual(type(result_const_nested_drank.d_dt), Scalar)

        # Also test nested derivative that is constant (order 0) with no tail (drank=0)
        # This tests the else branch when dvalue_tail is empty
        v_const_nested_drank2 = Vector(np.array([[11.]]), drank=1)
        p_const_nested_drank2 = Polynomial(v_const_nested_drank2)
        v_deriv_nested3 = Vector(np.array([[0.]]), drank=1)
        p_deriv_nested3 = Polynomial(v_deriv_nested3)
        # Create a nested derivative that is constant with no tail (drank=0)
        p_deriv_nested5 = Polynomial([1.])  # constant, no tail
        p_deriv_nested3.insert_deriv('s', p_deriv_nested5)
        p_const_nested_drank2.insert_deriv('t', p_deriv_nested3)
        result_const_nested_drank2 = p_const_nested_drank2.eval(40., recursive=True)
        self.assertEqual(type(result_const_nested_drank2), Scalar)
        self.assertAlmostEqual(result_const_nested_drank2.values, 11., places=10)
        self.assertTrue(hasattr(result_const_nested_drank2, 'd_dt'))

        # Test roots with scalar mask True
        p_mask_true = Polynomial([1., 2.], mask=True)
        roots_mask_true = p_mask_true.roots()
        # After sort(), masked values become inf, so check for inf instead
        self.assertTrue(np.all(~np.isfinite(roots_mask_true.values)) or np.all(roots_mask_true.mask))

        # Test roots with scalar mask False
        p_mask_false = Polynomial([1., 2.], mask=False)
        roots_mask_false = p_mask_false.roots()
        self.assertFalse(np.any(roots_mask_false.mask))

        # Test roots with all_zeros case
        # Create polynomial where all coefficients are zero for some elements
        coeffs_all_zeros = np.array([
            [[0., 0., 0.], [1., 2., 3.]],
            [[0., 0., 0.], [1., 2., 3.]]
        ])
        p_all_zeros = Polynomial(coeffs_all_zeros)
        roots_all_zeros = p_all_zeros.roots()
        # Should handle all zeros case
        self.assertEqual(roots_all_zeros.shape, (2, 2, 2))

        # Test roots with array shifts and mask_indices
        # Create array where some elements need different numbers of shifts
        # This tests the array case (shift_shape is not empty)
        coeffs_array_shifts = np.array([
            [[0., 1., 2., 0.], [1., 2., 3., 0.]],  # First needs 1 shift, second needs 0
            [[0., 0., 1., 2.], [1., 2., 3., 0.]]  # First needs 2 shifts, second needs 0
        ])
        p_array_shifts = Polynomial(coeffs_array_shifts)
        roots_array_shifts = p_array_shifts.roots()
        # Should handle array shifts correctly
        # The order is 3 (4 coefficients), so roots shape is (3, 2, 2)
        self.assertEqual(roots_array_shifts.shape, (3, 2, 2))

        # Test roots duplicate detection scalar case
        # Create polynomial with duplicate roots in scalar case
        p_dup_scalar = Polynomial([1., -2., 1.])  # (x-1)^2, duplicate root at 1
        roots_dup_scalar = p_dup_scalar.roots()
        # Should have duplicate masked (becomes inf after sort)
        self.assertTrue(np.any(~np.isfinite(roots_dup_scalar.values)))

        # Test roots with derivatives
        # This tests the code path for adding derivatives to roots
        p_roots_deriv2 = Polynomial([1., -3., 2.])  # (x-1)(x-2) = x^2 - 3x + 2
        p_roots_deriv2.insert_deriv('t', Polynomial([0., -1., 0.]))  # derivative: -x
        roots_with_deriv2 = p_roots_deriv2.roots(recursive=True)
        # The code path for adding derivatives should execute
        # The derivative calculation involves evaluating the polynomial derivative
        # at the roots and dividing, which tests the code path
        self.assertEqual(roots_with_deriv2.shape, (2,))

        # Test roots with scalar mask True (duplicate test)
        p_mask_true2 = Polynomial([1., 2.], mask=True)
        roots_mask_true2 = p_mask_true2.roots()
        # After sort(), masked values become inf, so check for inf or mask
        self.assertTrue(np.all(~np.isfinite(roots_mask_true2.values)) or np.all(roots_mask_true2.mask))

        # Test roots with scalar mask False (duplicate test)
        p_mask_false2 = Polynomial([1., 2.], mask=False)
        roots_mask_false2 = p_mask_false2.roots()
        # Should have no mask
        if isinstance(roots_mask_false2.mask, np.ndarray):
            self.assertFalse(np.any(roots_mask_false2.mask))
        else:
            self.assertFalse(roots_mask_false2.mask)

        # Test roots with all_zeros case
        # Create polynomial where all coefficients are zero
        p_all_zeros2 = Polynomial([0., 0., 0.])
        roots_all_zeros2 = p_all_zeros2.roots()
        # Should handle all zeros case - the code sets leading coefficient to 1 and masks
        self.assertEqual(roots_all_zeros2.shape, (2,))
        # The all_zeros case should be masked (code sets poly_mask |= all_zeros)
        # After sort(), masked values become inf, so check for inf or mask
        if isinstance(roots_all_zeros2.mask, np.ndarray):
            # Check that mask is set (all True or all inf)
            self.assertTrue(np.all(roots_all_zeros2.mask) or np.all(~np.isfinite(roots_all_zeros2.values)))
        else:
            # Scalar mask case
            self.assertTrue(roots_all_zeros2.mask or not np.any(np.isfinite(roots_all_zeros2.values)))

        # Test roots with array shifts and mask_indices
        # Create array where some elements need different numbers of shifts
        # This tests the array case (shift_shape is not empty)
        coeffs_array_shifts2 = np.array([
            [[0., 0., 1., 2.], [0., 1., 2., 3.]],  # First needs 2 shifts, second needs 1 shift
            [[1., 2., 3., 4.], [0., 0., 0., 1.]]   # First needs 0 shifts, second needs 3 shifts
        ])
        p_array_shifts2 = Polynomial(coeffs_array_shifts2)
        roots_array_shifts2 = p_array_shifts2.roots()
        # Should handle array shifts correctly
        self.assertEqual(roots_array_shifts2.shape, (3, 2, 2))
        # The mask_indices code path should execute when total_shifts.size > 0
        # and len(mask_indices) > 0

        # Test roots duplicate detection scalar case
        # Create polynomial with duplicate roots in scalar case
        p_dup_scalar2 = Polynomial([1., -4., 4.])  # (x-2)^2, duplicate root at 2
        roots_dup_scalar2 = p_dup_scalar2.roots()
        # Should have duplicate masked (becomes inf after sort)
        # In scalar case, the code checks if root_values[k] == root_values[k-1] and not root_mask
        # If true, it sets root_mask = True and breaks
        self.assertTrue(np.any(~np.isfinite(roots_dup_scalar2.values)) or
                       (isinstance(roots_dup_scalar2.mask, bool) and roots_dup_scalar2.mask))

        # Test roots with derivatives
        # This tests the code path for adding derivatives to roots
        # Use a linear polynomial for simplicity: x + 2 = 0, root at -2
        # Derivative of polynomial: 1 (constant, nonzero at root)
        # Derivative of polynomial w.r.t. t: some constant
        p_roots_deriv3 = Polynomial([1., 2.])  # x + 2
        p_roots_deriv3.insert_deriv('t', Polynomial([0., 1.]))  # derivative w.r.t. t: 1
        roots_with_deriv3 = p_roots_deriv3.roots(recursive=True)
        # The code path for adding derivatives should execute
        # The derivative calculation: deriv = -value.eval(roots) / self.deriv().eval(roots)
        # = -1 / 1 = -1
        self.assertEqual(roots_with_deriv3.shape, (1,))
        # Derivatives should be added
        self.assertTrue(hasattr(roots_with_deriv3, 'd_dt'))
        self.assertAlmostEqual(roots_with_deriv3.d_dt.values[0], -1., places=10)

##########################################################################################
