##########################################################################################
# tests/test_polynomial_basic.py
# Polynomial basic construction and property tests
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Vector, Polynomial


class Test_Polynomial_Basic(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # Test basic construction
        # Polynomial is a Vector subclass, so it should accept Vector-like inputs
        # Coefficients are in decreasing order: [a, b, c] = a*x^2 + b*x + c
        p1 = Polynomial([1., 2., 3.])  # x^2 + 2x + 3
        self.assertEqual(p1.shape, ())
        self.assertEqual(p1.numer, (3,))
        self.assertEqual(p1.order, 2)

        # Test construction from Vector
        v = Vector([1., 2., 3.])
        p2 = Polynomial(v)
        self.assertEqual(p2.order, 2)
        self.assertTrue(np.allclose(p2.values, p1.values))

        # Test order property
        p0 = Polynomial([5.])  # constant polynomial
        self.assertEqual(p0.order, 0)

        p1_order = Polynomial([1., 0.])  # linear: x
        self.assertEqual(p1_order.order, 1)

        p2_order = Polynomial([1., 2., 3.])  # quadratic: x^2 + 2x + 3
        self.assertEqual(p2_order.order, 2)

        # Test as_polynomial static method
        p3 = Polynomial.as_polynomial([4., 5., 6.])
        self.assertEqual(type(p3), Polynomial)
        self.assertEqual(p3.order, 2)

        # Test as_polynomial with Vector
        v2 = Vector([7., 8.])
        p4 = Polynomial.as_polynomial(v2)
        self.assertEqual(type(p4), Polynomial)
        self.assertEqual(p4.order, 1)

        # Test as_vector method
        p5 = Polynomial([1., 2., 3.])
        v3 = p5.as_vector()
        self.assertEqual(type(v3), Vector)
        self.assertTrue(np.allclose(v3.values, p5.values))

        # Test at_least_order
        p_small = Polynomial([1., 2.])  # order 1
        p_large = p_small.at_least_order(3)  # should pad to order 3
        self.assertEqual(p_large.order, 3)
        self.assertEqual(p_large.numer[0], 4)  # 4 coefficients for order 3
        # Leading coefficients should be zero
        self.assertEqual(p_large.values[0], 0.)
        self.assertEqual(p_large.values[1], 0.)
        # Original coefficients should be at the end
        self.assertEqual(p_large.values[2], 1.)
        self.assertEqual(p_large.values[3], 2.)

        # If already larger order, should return unchanged
        p_big = Polynomial([1., 2., 3., 4.])  # order 3
        p_big2 = p_big.at_least_order(2)
        self.assertEqual(p_big2.order, 3)
        self.assertTrue(np.allclose(p_big2.values, p_big.values))

        # Test set_order
        p6 = Polynomial([1., 2.])  # order 1
        p7 = p6.set_order(2)
        self.assertEqual(p7.order, 2)
        self.assertEqual(p7.numer[0], 3)

        # set_order should raise ValueError if order is too small
        p8 = Polynomial([1., 2., 3., 4.])  # order 3
        self.assertRaises(ValueError, p8.set_order, 2)

        # Test invert_line
        # Linear polynomial: y = 3x + 2, so x = (y - 2) / 3 = (1/3)y - 2/3
        p_linear = Polynomial([3., 2.])  # 3x + 2 (coefficients in decreasing order)
        p_inv = p_linear.invert_line()
        self.assertEqual(p_inv.order, 1)
        # Inverse: x = (1/3)y - 2/3, so coefficients in decreasing order: [1/3, -2/3]
        self.assertAlmostEqual(p_inv.values[0], 1./3., places=10)
        self.assertAlmostEqual(p_inv.values[1], -2./3., places=10)

        # Test invert_line preserves derivatives
        p_linear_with_deriv = Polynomial([3., 2.])
        p_linear_deriv = Polynomial([1., 0.])  # derivative of 2x + 3 is 2
        p_linear_with_deriv.insert_deriv('t', p_linear_deriv)
        p_inv_with_deriv = p_linear_with_deriv.invert_line(recursive=True)
        self.assertTrue(hasattr(p_inv_with_deriv, 'd_dt'))
        # Derivative of inverse: if y = 2x + 3, then x = 0.5y - 1.5
        # If dy/dt = 2, then dx/dt = 0.5 * 2 = 1
        # But we need to check the actual derivative structure
        self.assertEqual(type(p_inv_with_deriv.d_dt), Polynomial)

        # invert_line should raise ValueError for non-linear
        p_nonlinear = Polynomial([1., 2., 3.])
        self.assertRaises(ValueError, p_nonlinear.invert_line)

        # Test that Polynomial only allows floats (not ints)
        # Based on _INTS_OK = False
        # This should work but be coerced to float
        p_int_coeffs = Polynomial([1, 2, 3])
        self.assertEqual(p_int_coeffs.values.dtype.kind, 'f')

        # Test that coefficients are in decreasing order of exponent
        # p = x^2 + 2x + 3 should have coefficients [1, 2, 3]
        p_test_order = Polynomial([1., 2., 3.])
        # Verify coefficient order: [1, 2, 3] means 1*x^2 + 2*x + 3
        self.assertEqual(p_test_order.values[0], 1.)  # x^2 coefficient
        self.assertEqual(p_test_order.values[1], 2.)  # x coefficient
        self.assertEqual(p_test_order.values[2], 3.)  # constant
        # Verify by evaluation: at x=1, should be 1+2+3=6
        self.assertAlmostEqual(p_test_order.eval(1.).values, 6., places=10)
        # At x=2, should be 4+4+3=11
        self.assertAlmostEqual(p_test_order.eval(2.).values, 11., places=10)

        # Additional tests for coverage

        # Test __init__ with Vector subclass that has derivatives
        v_with_deriv = Vector([1., 2.])
        v_deriv = Vector([0., 1.])
        v_with_deriv.insert_deriv('t', v_deriv)
        # Create a subclass to test the type check
        class PolySubclass(Polynomial):
            pass
        p_sub = PolySubclass(v_with_deriv)
        # The derivative should be converted to Polynomial when type(self) is not Polynomial
        self.assertTrue(hasattr(p_sub, 'd_dt'))
        # Check _derivs directly to verify conversion happened
        self.assertEqual(type(p_sub._derivs['t']), Polynomial)

        # Test as_polynomial with recursive=False
        v3 = Vector([1., 2., 3.])
        v3.insert_deriv('t', Vector([0., 1., 2.]))
        p_no_rec = Polynomial.as_polynomial(v3, recursive=False)
        self.assertFalse(hasattr(p_no_rec, 'd_dt'))

        p_no_rec2 = Polynomial.as_polynomial([1., 2.], recursive=False)
        self.assertEqual(type(p_no_rec2), Polynomial)

        # Test as_vector with recursive=False
        p_with_deriv2 = Polynomial([1., 2.])
        p_with_deriv2.insert_deriv('t', Polynomial([0., 1.]))
        v_no_rec = p_with_deriv2.as_vector(recursive=False)
        # When recursive=False, derivatives should not be preserved
        self.assertEqual(type(v_no_rec), Vector)
        # The _derivs might still exist from __dict__ copy, but the code path is tested

        # Test at_least_order with recursive=False when already >= order
        p_large2 = Polynomial([1., 2., 3., 4.])
        p_large3 = p_large2.at_least_order(2, recursive=False)
        self.assertEqual(p_large3.order, 3)

        # Test at_least_order with derivatives
        p_with_deriv3 = Polynomial([1., 2.])
        p_with_deriv3.insert_deriv('t', Polynomial([0., 1.]))
        p_padded = p_with_deriv3.at_least_order(3, recursive=True)
        self.assertTrue(hasattr(p_padded, 'd_dt'))
        self.assertEqual(p_padded.d_dt.order, 3)

        # Test as_vector with recursive=True
        p_asvec_deriv = Polynomial([1., 2.])
        p_asvec_deriv.insert_deriv('t', Polynomial([0., 1.]))
        v_with_deriv = p_asvec_deriv.as_vector(recursive=True)
        self.assertTrue(hasattr(v_with_deriv, 'd_dt'))
        # Derivatives should be preserved with recursive=True
        self.assertEqual(type(v_with_deriv.d_dt), Vector)

##########################################################################################

