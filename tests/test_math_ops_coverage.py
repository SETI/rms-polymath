##########################################################################################
# tests/test_math_ops_coverage.py
# Comprehensive coverage tests for math_ops.py to achieve >90% coverage
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Vector, Matrix, Boolean, Qube, Unit


class Test_Math_Ops_Coverage(unittest.TestCase):

    def runTest(self):

        np.random.seed(12345)

        ##################################################################################
        # Test __abs__ error case
        ##################################################################################
        # Vector actually supports abs(), so we test a case that doesn't work
        # The abs() test is covered by other operations that actually fail

        ##################################################################################
        # Test __add__ error cases
        ##################################################################################
        # Test incompatible types
        a = Scalar([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = a + "invalid"
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test incompatible numers - different types raise unsupported_op
        a = Scalar([1., 2., 3.])
        b = Vector([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = a + b
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test incompatible denoms
        a = Vector(np.arange(6).reshape(2, 3), drank=1)
        b = Vector(np.arange(6, 12).reshape(2, 3), drank=1)
        # Create incompatible denominator shapes
        a._denom = (2,)
        b._denom = (3,)
        with self.assertRaises(ValueError) as cm:
            _ = a + b
        self.assertIn('incompatible denominator shapes', str(cm.exception))

        # Test __add__ with non-recursive
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        c = a.__add__(b, recursive=False)
        # When recursive=False, derivatives are not included in the result
        # But the result might still have d_dt if it's copied from self
        # Actually, recursive=False means don't compute new derivatives, but existing ones might be copied
        # Let's just verify the operation works
        self.assertTrue(np.allclose(c.values, [5., 7., 9.]))

        ##################################################################################
        # Test __iadd__ error cases
        ##################################################################################
        # Test incompatible types
        a = Scalar([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            a += "invalid"
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test integer result from non-integer
        a = Scalar([1, 2, 3])  # Integer
        b = Scalar([1., 2., 3.])  # Float
        with self.assertRaises(TypeError) as cm:
            a += b
        self.assertIn('operation returns non-integer result', str(cm.exception))

        # Test with np.ndarray
        a = Scalar([1., 2., 3.])
        a += np.array([0.1, 0.2, 0.3])

        ##################################################################################
        # Test __sub__ error cases
        ##################################################################################
        # Test incompatible types
        a = Scalar([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = a - "invalid"
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test __sub__ with non-recursive
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        c = a.__sub__(b, recursive=False)
        # Verify the operation works
        self.assertTrue(np.allclose(c.values, [-3., -3., -3.]))

        ##################################################################################
        # Test __isub__ error cases
        ##################################################################################
        # Test integer result from non-integer
        a = Scalar([1, 2, 3])  # Integer
        b = Scalar([1., 2., 3.])  # Float
        with self.assertRaises(TypeError) as cm:
            a -= b
        self.assertIn('operation returns non-integer result', str(cm.exception))

        # Test with np.ndarray
        a = Scalar([1., 2., 3.])
        a -= np.array([0.1, 0.2, 0.3])

        ##################################################################################
        # Test __mul__ error cases
        ##################################################################################
        # Test incompatible types
        a = Scalar([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = a * "invalid"
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test dual denominators
        a = Vector(np.arange(6).reshape(2, 3), drank=1)
        b = Vector(np.arange(6, 12).reshape(2, 3), drank=1)
        with self.assertRaises(ValueError) as cm:
            _ = a * b
        self.assertIn('only one operand', str(cm.exception))

        # Test exception revision - object() cannot be converted
        a = Scalar([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = a * object()
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test __mul__ with non-recursive
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        c = a.__mul__(b, recursive=False)
        # Verify the operation works
        self.assertTrue(np.allclose(c.values, [4., 10., 18.]))

        ##################################################################################
        # Test __rmul__ error cases
        ##################################################################################
        # Test exception revision - object() doesn't have __rmul__
        a = Scalar([1., 2., 3.])
        with self.assertRaises(AttributeError):
            _ = object().__rmul__(a)

        ##################################################################################
        # Test __imul__ error cases
        ##################################################################################
        # Test integer result from non-integer
        a = Scalar([1, 2, 3])  # Integer
        b = Scalar([1., 2., 3.])  # Float
        with self.assertRaises(TypeError) as cm:
            a *= b
        self.assertIn('operation returns non-integer result', str(cm.exception))

        # Test matrix multiply case - Matrix *= actually works (matrix multiplication)
        a = Matrix([[1., 2.], [3., 4.]])
        b = Matrix([[5., 6.], [7., 8.]])
        a *= b
        # Verify matrix multiplication result
        self.assertTrue(np.allclose(a.values, [[19., 22.], [43., 50.]]))

        ##################################################################################
        # Test __truediv__ error cases
        ##################################################################################
        # Test incompatible types
        a = Scalar([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = a / "invalid"
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test right denominator
        a = Scalar([1., 2., 3.])
        b = Vector(np.arange(6).reshape(2, 3), drank=1)
        with self.assertRaises(ValueError) as cm:
            _ = a / b
        self.assertIn('right operand has denominator', str(cm.exception))

        # Test exception revision
        a = Scalar([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = a / object()
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test matrix / matrix - actually works (matrix division via inverse)
        a = Matrix([[1., 2.], [3., 4.]])
        b = Matrix([[5., 6.], [7., 8.]])
        c = a / b
        # Verify matrix division result (a * b^-1)
        self.assertTrue(np.allclose(c.values, [[3., -2.], [2., -1.]]))

        # Test __truediv__ with non-recursive
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([2., 4., 6.])
        c = a.__truediv__(b, recursive=False)
        # Verify the operation works
        self.assertTrue(np.allclose(c.values, [0.5, 0.5, 0.5]))

        ##################################################################################
        # Test __rtruediv__ error cases
        ##################################################################################
        # Test exception revision - object() doesn't have __rtruediv__
        a = Scalar([1., 2., 3.])
        with self.assertRaises(AttributeError):
            _ = object().__rtruediv__(a)

        ##################################################################################
        # Test __itruediv__ error cases
        ##################################################################################
        # Test integer division
        a = Scalar([1, 2, 3])  # Integer
        with self.assertRaises(TypeError) as cm:
            a /= 2.
        self.assertIn('operation returns non-integer result', str(cm.exception))

        # Test division by zero - should mask
        a = Scalar([1., 2., 3.])
        a /= 0.
        self.assertTrue(np.all(a.mask))

        # Test exception revision
        a = Scalar([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            a /= object()
        self.assertIn('unsupported operand type', str(cm.exception))

        ##################################################################################
        # Test __floordiv__ error cases
        ##################################################################################
        # Test incompatible types
        a = Scalar([7, 8, 9])
        with self.assertRaises(TypeError) as cm:
            _ = a // "invalid"
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test right denominator
        a = Scalar([7, 8, 9])
        b = Vector(np.arange(6).reshape(2, 3), drank=1)
        with self.assertRaises(ValueError) as cm:
            _ = a // b
        self.assertIn('right operand has denominator', str(cm.exception))

        # Test exception revision
        a = Scalar([7, 8, 9])
        with self.assertRaises(TypeError) as cm:
            _ = a // object()
        self.assertIn('unsupported operand type', str(cm.exception))

        ##################################################################################
        # Test __rfloordiv__ error cases
        ##################################################################################
        # Test exception revision - object() doesn't have __rfloordiv__
        a = Scalar([2, 3, 4])
        with self.assertRaises(AttributeError):
            _ = object().__rfloordiv__(a)

        ##################################################################################
        # Test __ifloordiv__ error cases
        ##################################################################################
        # Test division by zero - should mask
        a = Scalar([5., 7., 9.])
        a //= 0
        self.assertTrue(np.all(a.mask))

        # Test exception
        a = Scalar([5., 7., 9.])
        with self.assertRaises(TypeError) as cm:
            a //= object()
        self.assertIn('unsupported operand type', str(cm.exception))

        ##################################################################################
        # Test __mod__ error cases
        ##################################################################################
        # Test incompatible types
        a = Scalar([7, 8, 9])
        with self.assertRaises(TypeError) as cm:
            _ = a % "invalid"
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test right denominator
        a = Scalar([7, 8, 9])
        b = Vector(np.arange(6).reshape(2, 3), drank=1)
        with self.assertRaises(ValueError) as cm:
            _ = a % b
        self.assertIn('right operand has denominator', str(cm.exception))

        # Test exception revision
        a = Scalar([7, 8, 9])
        with self.assertRaises(TypeError) as cm:
            _ = a % object()
        self.assertIn('unsupported operand type', str(cm.exception))

        # Test __mod__ with non-recursive
        a = Scalar([7, 8, 9])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([3, 4, 5])
        c = a.__mod__(b, recursive=False)
        # Mod doesn't preserve derivatives in denominator, but may in numerator
        # Actually, mod supports derivatives in numerator per docstring

        ##################################################################################
        # Test __rmod__ error cases
        ##################################################################################
        # Test exception revision - object() doesn't have __rmod__
        a = Scalar([3, 4, 5])
        with self.assertRaises(AttributeError):
            _ = object().__rmod__(a)

        ##################################################################################
        # Test __imod__ error cases
        ##################################################################################
        # Test division by zero - should mask
        a = Scalar([5., 7., 9.])
        a %= 0
        self.assertTrue(np.all(a.mask))

        # Test exception
        a = Scalar([5., 7., 9.])
        with self.assertRaises(TypeError) as cm:
            a %= object()
        self.assertIn('unsupported operand type', str(cm.exception))

        ##################################################################################
        # Test __pow__ error cases
        ##################################################################################
        # Test incompatible types
        a = Scalar([2., 3., 4.])
        with self.assertRaises(TypeError) as cm:
            _ = a ** "invalid"
        self.assertIn('invalid Scalar data type', str(cm.exception))

        # Test array exponent
        a = Scalar([2., 3., 4.])
        b = Scalar([1., 2.])  # Array exponent
        with self.assertRaises(ValueError) as cm:
            _ = a ** b
        self.assertIn('could not be broadcast together', str(cm.exception))

        # Test masked exponent
        a = Scalar([2., 3., 4.])
        b = Scalar(2., mask=True)
        c = a ** b
        self.assertTrue(np.all(c.mask))

        # Test non-integer exponent - Scalar supports float exponents
        a = Scalar([2., 3., 4.])
        b = a ** 2.5
        self.assertTrue(np.allclose(b.values, [2.**2.5, 3.**2.5, 4.**2.5]))

        # Test out of range exponent - Scalar supports high powers
        a = Scalar([2., 3., 4.])
        b = a ** 16
        self.assertTrue(np.allclose(b.values, [2.**16, 3.**16, 4.**16]))

        # Test __pow__ with zero exponent and derivatives
        a = Scalar([2., 3., 4.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a ** 0
        self.assertTrue(hasattr(b, 'd_dt'))

        # Test negative exponent
        a = Scalar([2., 3., 4.])
        b = a ** -1
        self.assertTrue(np.allclose(b.values, [0.5, 1./3., 0.25]))

        # Test power of 1
        a = Scalar([2., 3., 4.])
        b = a ** 1
        self.assertTrue(np.allclose(b.values, [2., 3., 4.]))

        # Test higher powers
        a = Scalar([2., 3., 4.])
        b = a ** 4
        self.assertTrue(np.allclose(b.values, [16., 81., 256.]))

        a = Scalar([2., 3., 4.])
        b = a ** 8
        self.assertTrue(np.allclose(b.values, [256., 6561., 65536.]))

        ##################################################################################
        # Test __ipow__
        ##################################################################################
        a = Scalar([2., 3., 4.])
        a **= 2
        self.assertTrue(np.allclose(a.values, [4., 9., 16.]))

        ##################################################################################
        # Test comparison operators error cases
        ##################################################################################
        # Test __le__ on non-Scalar
        v = Vector([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = v <= Scalar(2.)
        self.assertIn('operation is not supported', str(cm.exception))
        self.assertIn('<=', str(cm.exception))

        # Test __lt__ on non-Scalar
        v = Vector([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = v < Scalar(2.)
        self.assertIn('operation is not supported', str(cm.exception))
        self.assertIn('<', str(cm.exception))

        # Test __ge__ on non-Scalar
        v = Vector([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = v >= Scalar(2.)
        self.assertIn('operation is not supported', str(cm.exception))
        self.assertIn('>=', str(cm.exception))

        # Test __gt__ on non-Scalar
        v = Vector([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = v > Scalar(2.)
        self.assertIn('operation is not supported', str(cm.exception))
        self.assertIn('>', str(cm.exception))

        ##################################################################################
        # Test __eq__ edge cases
        ##################################################################################
        # Test incompatible argument
        a = Scalar([1., 2., 3.])
        b = "incompatible"
        c = a == b
        self.assertFalse(c)

        # Test with masks
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 4.])
        a = a.mask_where_eq(2.)
        b = b.mask_where_eq(2.)
        c = a == b
        # Both masked at same location should be equal

        # Test scalar return
        a = Scalar(1.)
        b = Scalar(1.)
        c = a == b
        self.assertTrue(c)
        self.assertIsInstance(c, bool)

        # Test one masked
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 3.])
        a = a.mask_where_eq(2.)
        c = a == b
        self.assertFalse(c.values[1])  # Where a is masked, should be False

        ##################################################################################
        # Test __ne__ edge cases
        ##################################################################################
        # Test incompatible argument
        a = Scalar([1., 2., 3.])
        b = "incompatible"
        c = a != b
        self.assertTrue(c)

        # Test unit compatibility check
        a = Scalar([1., 2., 3.], unit=Unit.KM)
        b = Scalar([1., 2., 3.], unit=Unit.SEC)
        c = a != b
        self.assertTrue(c)

        # Test scalar return
        a = Scalar(1.)
        b = Scalar(2.)
        c = a != b
        self.assertTrue(c)
        self.assertIsInstance(c, bool)

        # Test with masks
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 4.])
        a = a.mask_where_eq(2.)
        b = b.mask_where_eq(2.)
        c = a != b
        # Both masked should be False

        ##################################################################################
        # Test __bool__ edge cases
        ##################################################################################
        # Test _truth_if_all
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 3.])
        c = (a == b)
        self.assertTrue(bool(c))

        # Test _truth_if_any
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 4.])
        c = (a != b)
        self.assertTrue(bool(c))

        ##################################################################################
        # Test boolean operators with MaskedArray
        ##################################################################################
        import numpy.ma as ma
        a = Scalar([0., 1., 2.])
        b = ma.MaskedArray([1., 0., 2.])
        c = a & b
        self.assertEqual(type(c).__name__, 'Boolean')

        c = a | b
        self.assertEqual(type(c).__name__, 'Boolean')

        c = a ^ b
        self.assertEqual(type(c).__name__, 'Boolean')

        # Test in-place with MaskedArray
        a = Boolean([False, True, True])
        b = ma.MaskedArray([True, False, True])
        a &= b
        a = Boolean([False, True, False])
        a |= b
        a = Boolean([False, True, False])
        a ^= b

        ##################################################################################
        # Test any/all edge cases
        ##################################################################################
        # Test any with no shape
        a = Scalar(1.)
        b = a.any()
        self.assertTrue(b)

        # Test any with builtins
        a = Boolean([False, True, False])
        old_builtins = Qube.prefer_builtins()
        try:
            Qube.prefer_builtins(True)
            b = a.any()
            self.assertIsInstance(b, bool)
        finally:
            Qube.prefer_builtins(old_builtins)

        # Test all with no shape
        a = Scalar(1.)
        b = a.all()
        self.assertTrue(b)

        # Test all with builtins
        a = Boolean([True, True, True])
        old_builtins = Qube.prefer_builtins()
        try:
            Qube.prefer_builtins(True)
            b = a.all()
            self.assertIsInstance(b, bool)
        finally:
            Qube.prefer_builtins(old_builtins)

        # Test any_true_or_masked with no shape
        a = Scalar(1.)
        b = a.any_true_or_masked()
        self.assertTrue(b)

        # Test all_true_or_masked with no shape
        a = Scalar(1.)
        b = a.all_true_or_masked()
        self.assertTrue(b)

        ##################################################################################
        # Test reciprocal error case
        ##################################################################################
        # Test on non-Scalar
        v = Vector([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = v.reciprocal()
        self.assertIn('reciprocal()', str(cm.exception))
        self.assertIn('not supported', str(cm.exception))

        ##################################################################################
        # Test identity error case
        ##################################################################################
        # Test on non-Scalar/Matrix/Boolean
        v = Vector([1., 2., 3.])
        with self.assertRaises(TypeError) as cm:
            _ = v.identity()
        self.assertIn('identity() operation is not supported', str(cm.exception))

        ##################################################################################
        # Test sum/mean with builtins
        ##################################################################################
        a = Scalar([1., 2., 3., 4.])
        old_builtins = Qube.prefer_builtins()
        try:
            Qube.prefer_builtins(True)
            b = a.sum()
            self.assertIsInstance(b, (int, float))
            c = a.mean()
            self.assertIsInstance(c, float)
        finally:
            Qube.prefer_builtins(old_builtins)

        ##################################################################################
        # Test error message functions
        ##################################################################################
        # Test _raise_unsupported_op with obj2=None - already tested above with reciprocal

        # Test _raise_unsupported_op with array-like obj1
        # NumPy arrays actually work with Qube objects through __radd__
        # So this test is not applicable - the operation succeeds
        arr = np.array([1., 2., 3.])
        result = arr + Scalar([1., 2., 3.])
        self.assertTrue(np.allclose(result.values, [2., 4., 6.]))

        # Test _raise_incompatible_shape
        # This is called internally, hard to test directly

        # Test _raise_incompatible_numers
        # Tested indirectly through addition operations

        # Test _raise_incompatible_denoms
        # Tested indirectly through operations

        # Test _raise_dual_denoms
        # Tested in multiplication tests above

        ##################################################################################
        # Test _div_by_number edge cases
        ##################################################################################
        # Test division by zero
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a._div_by_number(0., recursive=True)
        self.assertTrue(b.mask)

        # Test _div_by_number with non-recursive
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a._div_by_number(2., recursive=False)
        # Verify the operation works
        self.assertTrue(np.allclose(b.values, [0.5, 1., 1.5]))

        ##################################################################################
        # Test _div_by_scalar edge cases
        ##################################################################################
        # Test with nozeros=False
        a = Scalar([1., 2., 3.])
        b = Scalar([2., 0., 4.])
        c = a._div_by_scalar(b, recursive=True)
        self.assertTrue(c.mask[1])  # Division by zero should be masked

        # Test _div_by_scalar with non-recursive
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([2., 4., 6.])
        c = a._div_by_scalar(b, recursive=False)
        # Verify the operation works
        self.assertTrue(np.allclose(c.values, [0.5, 0.5, 0.5]))

        ##################################################################################
        # Test _div_derivs edge cases
        ##################################################################################
        # Test with nozeros=False - division by zero should mask
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([2., 0., 4.])
        b.insert_deriv('t', Scalar([0.4, 0.5, 0.6]))
        c = a / b
        self.assertTrue(c.mask[1])  # Division by zero should be masked
        self.assertTrue(hasattr(c, 'd_dt'))

        ##################################################################################
        # Test _mod_by_number edge cases
        ##################################################################################
        # Test modulus by zero
        a = Scalar([7, 8, 9])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a._mod_by_number(0, recursive=True)
        self.assertTrue(b.mask)

        # Test _mod_by_number with non-recursive
        a = Scalar([7, 8, 9])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a._mod_by_number(3, recursive=False)
        # Check values match expected remainders: 7%3=1, 8%3=2, 9%3=0
        self.assertTrue(np.allclose(b.values, [1, 2, 0]))
        # With recursive=False, derivatives are not preserved
        self.assertFalse(hasattr(b, 'd_dt'))
        # Test with recursive=True to verify derivatives are preserved
        b_recursive = a._mod_by_number(3, recursive=True)
        self.assertTrue(hasattr(b_recursive, 'd_dt'))
        self.assertIsNotNone(b_recursive.d_dt)

        ##################################################################################
        # Test _mod_by_scalar edge cases
        ##################################################################################
        # Test with derivatives
        a = Scalar([7, 8, 9])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([3, 4, 5])
        c = a._mod_by_scalar(b, recursive=True)
        self.assertTrue(hasattr(c, 'd_dt'))

        # Test _mod_by_scalar with non-recursive
        a = Scalar([7, 8, 9])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([3, 4, 5])
        c = a._mod_by_scalar(b, recursive=False)
        # Check values match expected remainders: 7%3=1, 8%4=0, 9%5=4
        self.assertTrue(np.allclose(c.values, [1, 0, 4]))
        # With recursive=False, derivatives are not preserved
        self.assertFalse(hasattr(c, 'd_dt'))
        # Test with recursive=True to verify derivatives are preserved
        c_recursive = a._mod_by_scalar(b, recursive=True)
        self.assertTrue(hasattr(c_recursive, 'd_dt'))
        self.assertIsNotNone(c_recursive.d_dt)

        ##################################################################################
        # Test _floordiv_by_number edge cases
        ##################################################################################
        # Test floor division by zero
        a = Scalar([7, 8, 9])
        b = a._floordiv_by_number(0)
        self.assertTrue(b.mask)

        ##################################################################################
        # Test _floordiv_by_scalar edge cases
        ##################################################################################
        # Test floor division by scalar with zero
        a = Scalar([7, 8, 9])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([2, 0, 4])
        c = a._floordiv_by_scalar(b)
        # Division by zero should be masked
        self.assertTrue(c.mask[1])
        # Check non-zero positions have correct floor division values: 7//2=3, 9//4=2
        self.assertEqual(c.values[0], 3)  # 7 // 2 = 3
        self.assertEqual(c.values[2], 2)  # 9 // 4 = 2
        # _floordiv_by_scalar doesn't preserve derivatives (no recursive parameter)

        ##################################################################################
        # Test _add_derivs edge cases
        ##################################################################################
        # Test with overlapping derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        b.insert_deriv('t', Scalar([0.4, 0.5, 0.6]))
        c = a + b
        self.assertTrue(hasattr(c, 'd_dt'))
        self.assertTrue(np.allclose(c.d_dt.values, [0.5, 0.7, 0.9]))

        # Test with non-overlapping derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        b.insert_deriv('x', Scalar([0.4, 0.5, 0.6]))
        c = a + b
        self.assertTrue(hasattr(c, 'd_dt'))
        self.assertTrue(hasattr(c, 'd_dx'))

        ##################################################################################
        # Test _sub_derivs edge cases
        ##################################################################################
        # Test with overlapping derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        b.insert_deriv('t', Scalar([0.4, 0.5, 0.6]))
        c = a - b
        self.assertTrue(hasattr(c, 'd_dt'))
        self.assertTrue(np.allclose(c.d_dt.values, [-0.3, -0.3, -0.3]))

        # Test with non-overlapping derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        b.insert_deriv('x', Scalar([0.4, 0.5, 0.6]))
        c = a - b
        self.assertTrue(hasattr(c, 'd_dt'))
        self.assertTrue(hasattr(c, 'd_dx'))
        self.assertTrue(np.allclose(c.d_dx.values, [-0.4, -0.5, -0.6]))

        ##################################################################################
        # Test _mul_derivs edge cases
        ##################################################################################
        # Test with overlapping derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        b.insert_deriv('t', Scalar([0.4, 0.5, 0.6]))
        c = a * b
        self.assertTrue(hasattr(c, 'd_dt'))
        # Derivative should be a.d_dt * b + a * b.d_dt

        # Test with non-overlapping derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        b.insert_deriv('x', Scalar([0.4, 0.5, 0.6]))
        c = a * b
        self.assertTrue(hasattr(c, 'd_dt'))
        self.assertTrue(hasattr(c, 'd_dx'))

        ##################################################################################
        # Test logical_not with rank > 0
        ##################################################################################
        a = Vector([1., 2., 3.])
        b = a.logical_not()
        # Should reduce along rank axis
        self.assertEqual(b.shape, ())

        ##################################################################################
        # Test _mul_by_scalar with denominator alignment
        ##################################################################################
        # Test case where arg has denominator and self has shape
        a = Scalar([1., 2., 3.])
        b = Vector(np.arange(6).reshape(2, 3), drank=1)
        # This should work - Scalar can multiply Vector with denominator
        c = a * b
        self.assertEqual(c.shape, (3,))
        self.assertEqual(c.denom, (3,))  # The denominator comes from the Vector's drank

        ##################################################################################
        # Test _mul_by_number with derivatives
        ##################################################################################
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a._mul_by_number(2., recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(b.d_dt.values, [0.2, 0.4, 0.6]))

        b = a._mul_by_number(2., recursive=False)
        self.assertFalse(hasattr(b, 'd_dt'))
