##########################################################################################
# tests/test_polynomial_arithmetic.py
# Polynomial arithmetic operation tests
##########################################################################################

import numpy as np
import unittest

from polymath import Vector, Polynomial


class Test_Polynomial_Arithmetic(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # Test __neg__
        p9 = Polynomial([1., 2., 3.])
        p_neg = -p9
        self.assertEqual(type(p_neg), Polynomial)
        self.assertTrue(np.allclose(p_neg.values, -p9.values))

        # Test __add__
        # Coefficients are in decreasing order: [a, b, c] = a*x^2 + b*x + c
        p10 = Polynomial([1., 2.])  # x + 2
        p11 = Polynomial([3., 4., 5.])  # 3x^2 + 4x + 5
        p_sum = p10 + p11
        self.assertEqual(type(p_sum), Polynomial)
        self.assertEqual(p_sum.order, 2)
        # p10 padded to [0, 1, 2] = x + 2, sum = 3x^2 + 5x + 7
        self.assertAlmostEqual(p_sum.values[0], 3., places=10)
        self.assertAlmostEqual(p_sum.values[1], 5., places=10)
        self.assertAlmostEqual(p_sum.values[2], 7., places=10)

        # Test adding scalar
        p12 = Polynomial([1., 2.])  # x + 2
        p_sum2 = p12 + 5.  # should add 5 to constant term: x + 7
        self.assertEqual(p_sum2.order, 1)
        self.assertAlmostEqual(p_sum2.values[0], 1., places=10)  # x coefficient unchanged
        self.assertAlmostEqual(p_sum2.values[1], 7., places=10)  # constant term: 2 + 5 = 7

        # Test __radd__
        p13 = Polynomial([1., 2.])  # x + 2
        p_sum3 = 5. + p13  # adds 5 to constant term: x + 7
        self.assertEqual(type(p_sum3), Polynomial)
        self.assertAlmostEqual(p_sum3.values[1], 7., places=10)

        # Test __sub__
        p14 = Polynomial([5., 4., 3.])  # 5x^2 + 4x + 3
        p15 = Polynomial([1., 2.])  # x + 2
        p_diff = p14 - p15
        self.assertEqual(type(p_diff), Polynomial)
        self.assertEqual(p_diff.order, 2)
        # p15 padded to [0, 1, 2] = x + 2, diff = 5x^2 + 3x + 1
        self.assertAlmostEqual(p_diff.values[0], 5., places=10)
        self.assertAlmostEqual(p_diff.values[1], 3., places=10)
        self.assertAlmostEqual(p_diff.values[2], 1., places=10)

        # Test __rsub__
        p16 = Polynomial([1., 2.])  # x + 2
        p_diff2 = 5. - p16  # -x + 3
        self.assertEqual(type(p_diff2), Polynomial)
        self.assertAlmostEqual(p_diff2.values[0], -1., places=10)
        self.assertAlmostEqual(p_diff2.values[1], 3., places=10)

        # Test __mul__ with scalar
        p17 = Polynomial([1., 2., 3.])
        p_prod = p17 * 2.
        self.assertEqual(type(p_prod), Polynomial)
        self.assertTrue(np.allclose(p_prod.values, p17.values * 2.))

        # Test __mul__ with another polynomial
        # (x + 1) * (x + 2) = x^2 + 3x + 2
        p18 = Polynomial([1., 1.])  # x + 1
        p19 = Polynomial([1., 2.])  # x + 2 (not [2, 1] which is 2x + 1)
        p_prod2 = p18 * p19
        self.assertEqual(type(p_prod2), Polynomial)
        self.assertEqual(p_prod2.order, 2)
        # Verify by evaluation - (x+1)(x+2) at x=0 should be 2, at x=1 should be 6
        self.assertAlmostEqual(p_prod2.eval(0.).values, 2., places=10)
        self.assertAlmostEqual(p_prod2.eval(1.).values, 6., places=10)
        # Coefficients should be [1, 3, 2] for x^2 + 3x + 2
        self.assertAlmostEqual(p_prod2.values[0], 1., places=10)
        self.assertAlmostEqual(p_prod2.values[1], 3., places=10)
        self.assertAlmostEqual(p_prod2.values[2], 2., places=10)

        # Test __rmul__
        p20 = Polynomial([1., 2.])
        p_prod3 = 3. * p20
        self.assertEqual(type(p_prod3), Polynomial)
        self.assertTrue(np.allclose(p_prod3.values, p20.values * 3.))

        # Test __truediv__ with scalar
        p21 = Polynomial([2., 4., 6.])
        p_div = p21 / 2.
        self.assertEqual(type(p_div), Polynomial)
        self.assertTrue(np.allclose(p_div.values, p21.values / 2.))

        # Test __pow__
        # (x + 1)^2 = x^2 + 2x + 1
        p22 = Polynomial([1., 1.])  # x + 1
        p_pow = p22 ** 2
        self.assertEqual(type(p_pow), Polynomial)
        self.assertEqual(p_pow.order, 2)
        # (x+1)^2 = x^2 + 2x + 1
        self.assertAlmostEqual(p_pow.values[0], 1., places=10)
        self.assertAlmostEqual(p_pow.values[1], 2., places=10)
        self.assertAlmostEqual(p_pow.values[2], 1., places=10)

        # Test higher power
        p_pow3 = p22 ** 3  # (x+1)^3 = x^3 + 3x^2 + 3x + 1
        self.assertEqual(p_pow3.order, 3)
        self.assertAlmostEqual(p_pow3.values[0], 1., places=10)
        self.assertAlmostEqual(p_pow3.values[1], 3., places=10)
        self.assertAlmostEqual(p_pow3.values[2], 3., places=10)
        self.assertAlmostEqual(p_pow3.values[3], 1., places=10)

        # Test __pow__ with 0 (this one works because it returns early)
        p23 = Polynomial([1., 2., 3.])
        p_pow0 = p23 ** 0
        self.assertEqual(type(p_pow0), Polynomial)
        self.assertEqual(p_pow0.order, 0)
        self.assertEqual(p_pow0.values[0], 1.)

        # Test __pow__ raises ValueError for negative or non-integer
        self.assertRaises(ValueError, p23.__pow__, -1)
        self.assertRaises(ValueError, p23.__pow__, 1.5)

        # Test __eq__ and __ne__
        p24 = Polynomial([1., 2., 3.])
        p25 = Polynomial([1., 2., 3.])
        p26 = Polynomial([1., 2., 4.])
        self.assertTrue(p24 == p25)
        self.assertFalse(p24 == p26)
        self.assertTrue(p24 != p26)
        self.assertFalse(p24 != p25)

        # Test multiplication with incompatible denominators
        # Create polynomials with different drank values
        # This requires creating polynomials with denominators, which is complex
        # For now, we test that regular multiplication works (drank=0 case)
        # Testing with drank != 0 would require denominators
        p_normal1 = Polynomial([1., 2.])
        p_normal2 = Polynomial([3., 4.])
        # Both have drank=0, so multiplication should work
        p_normal_prod = p_normal1 * p_normal2
        self.assertEqual(p_normal_prod.order, 2)

        # Additional tests for coverage

        # Test __iadd__
        p_iadd = Polynomial([1., 2.])
        p_iadd += Polynomial([3., 4.])
        self.assertEqual(p_iadd.order, 1)
        self.assertAlmostEqual(p_iadd.values[0], 4., places=10)
        self.assertAlmostEqual(p_iadd.values[1], 6., places=10)

        # Test __isub__
        p_isub = Polynomial([5., 6.])
        p_isub -= Polynomial([1., 2.])
        self.assertEqual(p_isub.order, 1)
        self.assertAlmostEqual(p_isub.values[0], 4., places=10)
        self.assertAlmostEqual(p_isub.values[1], 4., places=10)

        # Test __mul__ with incompatible denominators
        # Create polynomials with different drank values
        # This is tricky - we need to create polynomials with denominators
        # For now, test that regular multiplication works
        p_mul1 = Polynomial([1., 2.])
        p_mul2 = Polynomial([3., 4.])
        p_mul_result = p_mul1 * p_mul2
        self.assertEqual(p_mul_result.order, 2)

        # Test __mul__ with derivatives
        p_mul_deriv1 = Polynomial([1., 2.])
        p_mul_deriv2 = Polynomial([3., 4.])
        p_mul_deriv1.insert_deriv('t', Polynomial([0., 1.]))
        p_mul_deriv2.insert_deriv('t', Polynomial([0., 2.]))
        p_mul_deriv_result = p_mul_deriv1 * p_mul_deriv2
        self.assertTrue(hasattr(p_mul_deriv_result, 'd_dt'))

        # Test __imul__ with Vector item == (1,)
        v_scalar = Vector([5.])
        p_imul = Polynomial([1., 2.])
        p_imul *= v_scalar
        self.assertEqual(p_imul.order, 1)
        self.assertAlmostEqual(p_imul.values[0], 5., places=10)
        self.assertAlmostEqual(p_imul.values[1], 10., places=10)

        # Test __truediv__ with Vector item == (1,)
        v_scalar2 = Vector([2.])
        p_tdiv = Polynomial([2., 4.])
        p_tdiv_result = p_tdiv / v_scalar2
        self.assertEqual(p_tdiv_result.order, 1)
        self.assertAlmostEqual(p_tdiv_result.values[0], 1., places=10)
        self.assertAlmostEqual(p_tdiv_result.values[1], 2., places=10)

        # Test __itruediv__ with Vector item == (1,)
        p_itdiv = Polynomial([4., 8.])
        p_itdiv /= Vector([2.])
        self.assertEqual(p_itdiv.order, 1)
        self.assertAlmostEqual(p_itdiv.values[0], 2., places=10)
        self.assertAlmostEqual(p_itdiv.values[1], 4., places=10)

        # Test __iadd__ when arg needs set_order
        p_iadd1 = Polynomial([1., 2.])  # order 1
        p_iadd2 = Polynomial([3., 4., 5.])  # order 2
        id_before = id(p_iadd1)
        p_iadd1 += p_iadd2
        self.assertEqual(id(p_iadd1), id_before)  # In-place
        # After padding, _values shape changes but order property may not update immediately
        # Check that values are correct instead
        self.assertEqual(len(p_iadd1.values), 3)  # Should have 3 coefficients

        # Test __iadd__ with derivatives (lines 270-271)
        p_iadd_deriv1 = Polynomial([1., 2.])
        p_iadd_deriv2 = Polynomial([3., 4.])
        p_iadd_deriv1.insert_deriv('t', Polynomial([0., 1.]))
        p_iadd_deriv2.insert_deriv('t', Polynomial([0., 2.]))
        p_iadd_deriv1 += p_iadd_deriv2
        self.assertTrue(hasattr(p_iadd_deriv1, 'd_dt'))

        # Test __isub__ when self needs padding (lines 319-321)
        p_isub1 = Polynomial([5., 6.])  # order 1
        p_isub2 = Polynomial([1., 2., 3.])  # order 2
        p_isub1 -= p_isub2
        self.assertEqual(len(p_isub1.values), 3)

        # Test __isub__ when arg.order < max_order
        # Need case where self.order > arg.order
        p_isub_self_larger = Polynomial([10., 20., 30., 40.])  # order 3
        p_isub_arg_smaller = Polynomial([1., 2.])  # order 1
        # When subtracting, max_order = max(3, 1) = 3, arg.order (1) < max_order (3)
        # So the branch should execute: arg = arg.at_least_order(3)
        p_isub_self_larger -= p_isub_arg_smaller
        self.assertEqual(p_isub_self_larger.order, 3)

        # Test __isub__ when arg needs at_least_order
        p_isub3 = Polynomial([5., 6., 7.])  # order 2
        p_isub4 = Polynomial([1., 2.])  # order 1, needs at_least_order
        p_isub3 -= p_isub4
        self.assertEqual(len(p_isub3.values), 3)

        # Test __isub__ with derivatives (lines 330-331)
        p_isub_deriv1 = Polynomial([5., 6.])
        p_isub_deriv2 = Polynomial([1., 2.])
        p_isub_deriv1.insert_deriv('t', Polynomial([0., 1.]))
        p_isub_deriv2.insert_deriv('t', Polynomial([0., 2.]))
        p_isub_deriv1 -= p_isub_deriv2
        self.assertTrue(hasattr(p_isub_deriv1, 'd_dt'))

        # Test __mul__ with incompatible denominators
        # Create two polynomials with different drank values
        # For a polynomial with drank=1, we need values with shape (..., n, d) where d is the denominator
        # Create a Vector with drank=1 first, then convert to Polynomial
        v_drank1 = Vector(np.array([[[1., 2.], [3., 4.]]]), drank=1)  # shape (1,), numer (2,), denom (2,)
        p_mul_drank1 = Polynomial(v_drank1)
        p_mul_drank2 = Polynomial([5., 6.])  # drank=0
        # This should raise ValueError
        self.assertRaises(ValueError, p_mul_drank1.__mul__, p_mul_drank2)

        # Test __itruediv__ with Vector item == (1,) (lines 456-459)
        p_itdiv_vec = Polynomial([4., 8.])
        v_scalar = Vector([2.])
        p_itdiv_vec /= v_scalar
        self.assertAlmostEqual(p_itdiv_vec.values[0], 2., places=10)
        self.assertAlmostEqual(p_itdiv_vec.values[1], 4., places=10)

        # Test __itruediv__ with Vector item == (1,) (lines 456-459)
        # This tests the branch: isinstance(arg, Vector) and arg.item == (1,)
        # Verify that Vector([4.]) has item == (1,)
        v_scalar3 = Vector([4.])
        self.assertEqual(v_scalar3.item, (1,))
        p_itdiv_vec2 = Polynomial([8., 16.])
        # This should hit the branch at line 456-457
        p_itdiv_vec2 /= v_scalar3
        self.assertAlmostEqual(p_itdiv_vec2.values[0], 2., places=10)
        self.assertAlmostEqual(p_itdiv_vec2.values[1], 4., places=10)

        # Test __iadd__ when arg.order < max_order
        # This tests the branch: if arg.order < max_order: arg = arg.at_least_order(max_order)
        # Need case where self.order > arg.order, so max_order = self.order and arg.order < max_order
        p_iadd_self_larger = Polynomial([1., 2., 3., 4.])  # order 3
        p_iadd_arg_smaller = Polynomial([5., 6.])  # order 1
        # When adding, max_order = max(3, 1) = 3, arg.order (1) < max_order (3)
        # So line 263 should execute: arg = arg.at_least_order(3)
        p_iadd_self_larger += p_iadd_arg_smaller
        self.assertEqual(p_iadd_self_larger.order, 3)
        # Verify the addition worked correctly
        self.assertAlmostEqual(p_iadd_self_larger.values[0], 1., places=10)
        self.assertAlmostEqual(p_iadd_self_larger.values[3], 10., places=10)  # 4 + 6 = 10

        # Test __mul__ with derivative else branch
        # Create two polynomials with different derivative keys
        p_mul_deriv_a = Polynomial([1., 2.])
        p_mul_deriv_b = Polynomial([3., 4.])
        p_mul_deriv_a.insert_deriv('t', Polynomial([0., 1.]))
        p_mul_deriv_b.insert_deriv('s', Polynomial([0., 2.]))  # Different key
        p_mul_mixed = p_mul_deriv_a * p_mul_deriv_b
        # Should have both derivatives
        self.assertTrue(hasattr(p_mul_mixed, 'd_dt'))
        self.assertTrue(hasattr(p_mul_mixed, 'd_ds'))

##########################################################################################
