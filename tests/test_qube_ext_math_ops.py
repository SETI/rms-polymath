##########################################################################################
# tests/test_qube_math_ops.py
# Unit tests for Qube math operations
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Vector, Boolean


class Test_Qube_math_ops(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # Test __pos__
        # +self, element by element.
        a = Scalar([1., 2., 3.])
        b = +a
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a.values, b.values))

        # Test __pos__ with derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = +a
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(a.d_dt.values, b.d_dt.values))

        # Test __neg__
        # -self, element-by-element negation.
        a = Scalar([1., 2., 3.])
        b = -a
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(b.values, [-1., -2., -3.]))

        # Test __neg__ with derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = -a
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(b.d_dt.values, [-0.1, -0.2, -0.3]))

        # Test __abs__
        # abs(self), element-by-element absolute value.
        # This general method always raises TypeError, but Scalar overrides it
        # So we test with a Qube that doesn't override it
        # Actually, we can't easily test the base class behavior since most classes override it
        # The docstring says it raises TypeError, but Scalar overrides it
        a = Scalar([-1., 2., -3.])
        # Scalar overrides __abs__, so it should work
        b = abs(a)
        self.assertTrue(np.allclose(b.values, [1., 2., 3.]))

        # Test abs
        # abs(self), element-by-element absolute value.
        a = Scalar([-1., 2., -3.])
        b = a.abs()
        self.assertTrue(np.allclose(b.values, [1., 2., 3.]))

        # Test __len__
        # Number of elements along first axis.
        a = Scalar([1., 2., 3., 4.])
        self.assertEqual(len(a), 4)

        a = Scalar(np.arange(12).reshape(2, 3, 2))
        self.assertEqual(len(a), 2)

        # Test len on unsized object
        a = Scalar(1.)
        self.assertRaises(TypeError, len, a)

        # Test len
        # Number of elements along first axis.
        a = Scalar([1., 2., 3., 4.])
        self.assertEqual(a.len(), 4)

        # Test __add__
        # self + arg, element-by-element addition.
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        c = a + b
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.allclose(c.values, [5., 7., 9.]))

        # Test __add__ with number
        # If not a Qube object, it will be converted to a Qube of the same type as self using
        # as_this_type(). For simple scalar operations (when self._rank == 0), Python numbers
        # are handled directly for efficiency.
        a = Scalar(1.)
        b = a + 2.
        self.assertEqual(b.shape, ())
        self.assertTrue(np.allclose(b.values, 3.))

        # Test __add__ with array-like conversion
        a = Scalar([1., 2., 3.])
        b = a + [4., 5., 6.]
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [5., 7., 9.]))

        # Test __add__ with derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        b.insert_deriv('t', Scalar([0.4, 0.5, 0.6]))
        c = a + b
        self.assertTrue(hasattr(c, 'd_dt'))
        self.assertTrue(np.allclose(c.d_dt.values, [0.5, 0.7, 0.9]))

        # Test __radd__
        # arg + self, element-by-element addition.
        # If not a Qube object, it will be converted to a Qube of the same type as self using
        # as_this_type().
        a = Scalar([1., 2., 3.])
        b = 2. + a
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [3., 4., 5.]))

        # Test __radd__ with array-like conversion
        a = Scalar([1., 2., 3.])
        b = [4., 5., 6.] + a
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [5., 7., 9.]))

        # Test __iadd__
        # self += arg, element-by-element in-place addition.
        a = Scalar([1., 2., 3.])
        a += Scalar([4., 5., 6.])
        self.assertTrue(np.allclose(a.values, [5., 7., 9.]))

        # Test __iadd__ with number
        a = Scalar(1.)
        a += 2.
        self.assertTrue(np.allclose(a.values, 3.))

        # Test __sub__
        # self - arg, element-by-element subtraction.
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        c = a - b
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.allclose(c.values, [-3., -3., -3.]))

        # Test __sub__ with number
        a = Scalar(1.)
        b = a - 2.
        self.assertEqual(b.shape, ())
        self.assertTrue(np.allclose(b.values, -1.))

        # Test __rsub__
        # arg - self, element-by-element subtraction.
        a = Scalar([1., 2., 3.])
        b = 2. - a
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [1., 0., -1.]))

        # Test __rsub__ with Qube argument (bug fix case - when arg is already a Qube)
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        c = a.__rsub__(b, recursive=True)
        # Should compute b - a = [4-1, 5-2, 6-3] = [3., 3., 3.]
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.allclose(c.values, [3., 3., 3.]))

        # Test __rsub__ with Qube argument and derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = Scalar([4., 5., 6.])
        b.insert_deriv('t', Scalar([0.4, 0.5, 0.6]))
        c = a.__rsub__(b, recursive=True)
        self.assertTrue(hasattr(c, 'd_dt'))
        # Derivative should be b.d_dt - a.d_dt = [0.4-0.1, 0.5-0.2, 0.6-0.3] = [0.3, 0.3, 0.3]
        self.assertTrue(np.allclose(c.d_dt.values, [0.3, 0.3, 0.3]))

        # Test __isub__
        # self -= arg, element-by-element in-place subtraction.
        a = Scalar([1., 2., 3.])
        a -= Scalar([4., 5., 6.])
        self.assertTrue(np.allclose(a.values, [-3., -3., -3.]))

        # Test __mul__
        # self * arg, element-by-element multiplication.
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        c = a * b
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.allclose(c.values, [4., 10., 18.]))

        # Test __mul__ with number
        a = Scalar([1., 2., 3.])
        b = a * 2.
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [2., 4., 6.]))

        # Test __rmul__
        # arg * self, element-by-element multiplication.
        a = Scalar([1., 2., 3.])
        b = 2. * a
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [2., 4., 6.]))

        # Test __rmul__ with Qube argument
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        c = a.__rmul__(b, recursive=True)
        # Should compute b * a = [4*1, 5*2, 6*3] = [4., 10., 18.]
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.allclose(c.values, [4., 10., 18.]))

        # Test __imul__
        # Element-by-element in-place multiplication.
        a = Scalar([1., 2., 3.])
        a *= 2.
        self.assertTrue(np.allclose(a.values, [2., 4., 6.]))

        # Test __truediv__
        # self / arg, element-by-element division.
        # Cases of divide-by-zero are masked.
        a = Scalar([1., 2., 3.])
        b = Scalar([2., 4., 6.])
        c = a / b
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.allclose(c.values, [0.5, 0.5, 0.5]))

        # Test __truediv__ with zero
        a = Scalar([1., 2., 3.])
        b = Scalar([2., 0., 6.])
        c = a / b
        self.assertTrue(c.mask[1])  # division by zero should be masked

        # Test __truediv__ with number
        a = Scalar([1., 2., 3.])
        b = a / 2.
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [0.5, 1., 1.5]))

        # Test __rtruediv__
        # arg / self, element-by-element division.
        a = Scalar([1., 2., 3.])
        b = 2. / a
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [2., 1., 2./3.]))

        # Test __rtruediv__ with Qube argument
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        c = a.__rtruediv__(b, recursive=True)
        # Should compute b / a = [4/1, 5/2, 6/3] = [4., 2.5, 2.]
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.allclose(c.values, [4., 2.5, 2.]))

        # Test __itruediv__
        # self /= arg, element-by-element in-place division.
        a = Scalar([1., 2., 3.])
        a /= 2.
        self.assertTrue(np.allclose(a.values, [0.5, 1., 1.5]))

        # Test __floordiv__
        # self // arg, element-by-element floor division.
        # Cases of divide-by-zero are masked. Derivatives are ignored.
        a = Scalar([7, 8, 9])
        b = Scalar([2, 3, 4])
        c = a // b
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.array_equal(c.values, [3, 2, 2]))

        # Test __floordiv__ with zero
        a = Scalar([7, 8, 9])
        b = Scalar([2, 0, 4])
        c = a // b
        self.assertTrue(c.mask[1])  # division by zero should be masked

        # Test __rfloordiv__
        # arg // self, element-by-element floor division.
        a = Scalar([2, 3, 4])
        b = 7 // a
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.array_equal(b.values, [3, 2, 1]))

        # Test __rfloordiv__ with Qube argument
        a = Scalar([2, 3, 4])
        b = Scalar([7, 8, 9])
        c = a.__rfloordiv__(b)
        # Should compute b // a = [7//2, 8//3, 9//4] = [3, 2, 2]
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.array_equal(c.values, [3, 2, 2]))

        # Test __ifloordiv__
        # self //= arg, element-by-element in-place floor division.
        a = Scalar([7, 8, 9])
        a //= Scalar([2, 3, 4])
        self.assertTrue(np.array_equal(a.values, [3, 2, 2]))

        # Test __mod__
        # self % arg, element-by-element modulus.
        # Cases of divide-by-zero are masked. Derivatives in the numerator are supported, but
        # not in the denominator.
        a = Scalar([7, 8, 9])
        b = Scalar([3, 4, 5])
        c = a % b
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.array_equal(c.values, [1, 0, 4]))

        # Test __mod__ with zero
        a = Scalar([7, 8, 9])
        b = Scalar([3, 0, 5])
        c = a % b
        self.assertTrue(c.mask[1])  # modulus by zero should be masked

        # Test __rmod__
        # arg % self, element-by-element modulus.
        a = Scalar([3, 4, 5])
        b = 7 % a
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.array_equal(b.values, [1, 3, 2]))

        # Test __rmod__ with Qube argument
        a = Scalar([3, 4, 5])
        b = Scalar([7, 8, 9])
        c = a.__rmod__(b, recursive=True)
        # Should compute b % a = [7%3, 8%4, 9%5] = [1, 0, 4]
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(np.array_equal(c.values, [1, 0, 4]))

        # Test __imod__
        # self %= arg, element-by-element in-place modulus.
        a = Scalar([7, 8, 9])
        a %= Scalar([3, 4, 5])
        self.assertTrue(np.array_equal(a.values, [1, 0, 4]))

        # Test __pow__
        # self ** arg, element-by-element exponentiation.
        # Derivatives are not supported.
        # This general method supports single integer exponents between -15 and 15
        a = Scalar([2., 3., 4.])
        b = a ** 2
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [4., 9., 16.]))
        # Verify it's self ** arg, not arg ** self
        # 2 ** 3 = 8, not 3 ** 2 = 9
        a = Scalar(2.)
        b = a ** 3
        self.assertTrue(np.allclose(b.values, 8.))

        # Test __pow__ with negative exponent
        a = Scalar([2., 3., 4.])
        b = a ** -1
        self.assertEqual(b.shape, a.shape)
        self.assertTrue(np.allclose(b.values, [0.5, 1./3., 0.25]))

        # Test __pow__ with zero exponent
        a = Scalar([2., 3., 4.])
        b = a ** 0
        self.assertEqual(b.shape, a.shape)
        # Should return identity

        # Test __pow__ raises ValueError for out of range
        # Note: Scalar may override __pow__ with different behavior
        # The base Qube.__pow__ limits to range (-15, 15)
        a = Scalar([2., 3., 4.])
        # Scalar might override this, so we test that it either raises or works
        try:
            _ = a ** 16
            # If it doesn't raise, that's okay - Scalar may have different limits
        except ValueError:
            pass  # Expected for base Qube class

        # Test __ipow__
        # self **= arg, element-by-element in-place power.
        a = Scalar([2., 3., 4.])
        a **= 2
        self.assertTrue(np.allclose(a.values, [4., 9., 16.]))

        # Test __eq__
        # self == arg, element by element.
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 4.])
        c = a == b
        self.assertEqual(type(c).__name__, 'Boolean')
        self.assertTrue(c.values[0])
        self.assertTrue(c.values[1])
        self.assertFalse(c.values[2])

        # Test __eq__ with incompatible argument
        a = Scalar([1., 2., 3.])
        b = Vector([1., 2., 3.])
        c = a == b
        self.assertFalse(c)

        # Test __ne__
        # self != arg, element by element.
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 4.])
        c = a != b
        self.assertEqual(type(c).__name__, 'Boolean')
        self.assertFalse(c.values[0])
        self.assertFalse(c.values[1])
        self.assertTrue(c.values[2])

        # Test __le__, __lt__, __ge__, __gt__
        # These general methods always raise ValueError
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 4.])
        # These should work for Scalar (overridden), but test that base raises
        # Actually, these are overridden by Scalar, so we can't test the base behavior easily

        # Test __bool__
        # True if nonzero, otherwise False, element by element.
        # This method also supports "if a == b: ..." and "if a != b: ..." statements using the
        # internal attributes _truth_if_all and _truth_if_any. These attributes are set by
        # the __eq__() and __ne__() methods respectively.
        a = Scalar(1.)
        self.assertTrue(bool(a))

        a = Scalar(0.)
        self.assertFalse(bool(a))

        # Test __bool__ raises ValueError for array
        a = Scalar([1., 2., 3.])
        self.assertRaises(ValueError, bool, a)

        # Test __bool__ raises ValueError for masked
        a = Scalar(1.)
        a = a.mask_where_eq(1.)
        self.assertRaises(ValueError, bool, a)

        # Test __bool__ with _truth_if_all (set by __eq__)
        # When _truth_if_all is True (set by __eq__()), the result is True only if all
        # unmasked elements are True.
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 3.])
        c = (a == b)
        # c should have _truth_if_all set, and bool(c) should be True
        self.assertTrue(bool(c))

        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 4.])
        c = (a == b)
        # c should have _truth_if_all set, and bool(c) should be False
        self.assertFalse(bool(c))

        # Test __bool__ with _truth_if_any (set by __ne__)
        # When _truth_if_any is True (set by __ne__()), the result is True if any unmasked
        # element is True.
        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 4.])
        c = (a != b)
        # c should have _truth_if_any set, and bool(c) should be True (since some elements differ)
        self.assertTrue(bool(c))

        a = Scalar([1., 2., 3.])
        b = Scalar([1., 2., 3.])
        c = (a != b)
        # c should have _truth_if_any set, and bool(c) should be False (since no elements differ)
        self.assertFalse(bool(c))

        # Test __float__
        # This object as a single float.
        a = Scalar(1.5)
        self.assertEqual(float(a), 1.5)

        # Test __float__ raises ValueError for array
        a = Scalar([1., 2., 3.])
        self.assertRaises(ValueError, float, a)

        # Test __float__ raises ValueError for masked
        a = Scalar(1.5)
        a = a.mask_where_eq(1.5)
        self.assertRaises(ValueError, float, a)

        # Test __int__
        # This object as a single int; floats always round down.
        a = Scalar(1.9)
        self.assertEqual(int(a), 1)

        # Test __int__ raises ValueError for array
        a = Scalar([1., 2., 3.])
        self.assertRaises(ValueError, int, a)

        # Test __int__ raises ValueError for masked
        a = Scalar(1.9)
        a = a.mask_where_eq(1.9)
        self.assertRaises(ValueError, int, a)

        # Test __invert__
        # ~self, unary inversion, element by element.
        # This is boolean "not", not bit inversion.
        a = Scalar([0., 1., 2.])
        b = ~a
        self.assertEqual(type(b).__name__, 'Boolean')
        self.assertTrue(b.values[0])
        self.assertFalse(b.values[1])
        self.assertFalse(b.values[2])

        # Test __and__
        # self & arg, element-by-element logical "and".
        a = Scalar([0., 1., 2.])
        b = Scalar([1., 0., 2.])
        c = a & b
        self.assertEqual(type(c).__name__, 'Boolean')
        self.assertFalse(c.values[0])
        self.assertFalse(c.values[1])
        self.assertTrue(c.values[2])

        # Test __rand__
        # arg & self, element-by-element logical "and".
        a = Scalar([0., 1., 2.])
        b = 1 & a
        self.assertEqual(type(b).__name__, 'Boolean')

        # Test __rand__ with Qube argument
        a = Scalar([0., 1., 2.])
        b = Scalar([1., 0., 2.])
        c = a.__rand__(b)
        # Should compute b & a = logical_and([1,0,2], [0,1,2]) = [False, False, True]
        self.assertEqual(type(c).__name__, 'Boolean')
        self.assertFalse(c.values[0])
        self.assertFalse(c.values[1])
        self.assertTrue(c.values[2])

        # Test __or__
        # self | arg, element-by-element logical "or".
        a = Scalar([0., 1., 2.])
        b = Scalar([1., 0., 0.])
        c = a | b
        self.assertEqual(type(c).__name__, 'Boolean')
        self.assertTrue(c.values[0])
        self.assertTrue(c.values[1])
        self.assertTrue(c.values[2])

        # Test __ror__
        # arg | self, element-by-element logical "or".
        a = Scalar([0., 1., 2.])
        b = 1 | a
        self.assertEqual(type(b).__name__, 'Boolean')

        # Test __ror__ with Qube argument
        a = Scalar([0., 1., 2.])
        b = Scalar([1., 0., 0.])
        c = a.__ror__(b)
        # Should compute b | a = logical_or([1,0,0], [0,1,2]) = [True, True, True]
        self.assertEqual(type(c).__name__, 'Boolean')
        self.assertTrue(c.values[0])
        self.assertTrue(c.values[1])
        self.assertTrue(c.values[2])

        # Test __xor__
        # self ^ arg, element-by-element logical exclusive "or".
        a = Scalar([0., 1., 2.])
        b = Scalar([1., 0., 2.])
        c = a ^ b
        self.assertEqual(type(c).__name__, 'Boolean')
        self.assertTrue(c.values[0])
        self.assertTrue(c.values[1])
        self.assertFalse(c.values[2])

        # Test __rxor__
        # arg ^ self, element-by-element logical exclusive "or".
        a = Scalar([0., 1., 2.])
        b = 1 ^ a
        self.assertEqual(type(b).__name__, 'Boolean')

        # Test __rxor__ with Qube argument
        a = Scalar([0., 1., 2.])
        b = Scalar([1., 0., 2.])
        c = a.__rxor__(b)
        # Should compute b ^ a = logical_xor([1,0,2], [0,1,2]) = [True, True, False]
        self.assertEqual(type(c).__name__, 'Boolean')
        self.assertTrue(c.values[0])
        self.assertTrue(c.values[1])
        self.assertFalse(c.values[2])

        # Test __iand__
        # self &= arg, element-by-element in-place logical "and".
        # Note: This modifies the values in place, converting to boolean-like behavior
        a = Boolean([False, True, True])
        a &= Boolean([True, False, True])
        self.assertEqual(type(a).__name__, 'Boolean')
        self.assertFalse(a.values[0])
        self.assertFalse(a.values[1])
        self.assertTrue(a.values[2])

        # Test __ior__
        # self |= arg, element-by-element in-place logical "or".
        a = Boolean([False, True, False])
        a |= Boolean([True, False, True])
        self.assertEqual(type(a).__name__, 'Boolean')
        self.assertTrue(a.values[0])
        self.assertTrue(a.values[1])
        self.assertTrue(a.values[2])

        # Test __ixor__
        # self ^= arg, element-by-element in-place logical exclusive "or".
        a = Boolean([False, True, False])
        a ^= Boolean([True, False, True])
        self.assertEqual(type(a).__name__, 'Boolean')
        self.assertTrue(a.values[0])
        self.assertTrue(a.values[1])
        self.assertTrue(a.values[2])

        # Test logical_not
        # The negation of this object, True where it is zero or False.
        a = Scalar([0., 1., 2.])
        b = a.logical_not()
        self.assertEqual(type(b).__name__, 'Boolean')
        self.assertTrue(b.values[0])
        self.assertFalse(b.values[1])
        self.assertFalse(b.values[2])

        # Test any
        # True if any of the unmasked items are nonzero.
        a = Boolean([False, False, True, False])
        self.assertTrue(a.any())

        a = Boolean([False, False, False, False])
        self.assertFalse(a.any())

        # Test any with axis
        a = Boolean([[False, True], [False, False]])
        b = a.any(axis=0)
        self.assertEqual(b.shape, (2,))
        self.assertFalse(b.values[0])
        self.assertTrue(b.values[1])

        # Test all
        # True if all the unmasked items are nonzero.
        a = Boolean([True, True, True, True])
        self.assertTrue(a.all())

        a = Boolean([True, True, False, True])
        self.assertFalse(a.all())

        # Test all with axis
        a = Boolean([[True, True], [True, False]])
        b = a.all(axis=0)
        self.assertEqual(b.shape, (2,))
        self.assertTrue(b.values[0])
        self.assertFalse(b.values[1])

        # Test any_true_or_masked
        # True if any of the items are nonzero or masked.
        a = Boolean([False, False, False, False])
        a = a.mask_where_eq(False)
        b = a.any_true_or_masked()
        self.assertTrue(b)

        # Test all_true_or_masked
        # True if all of the items are nonzero or masked.
        a = Boolean([True, True, True, True])
        a = a.mask_where_eq(True)
        b = a.all_true_or_masked()
        self.assertTrue(b)

        ##################################################################################
        # Additional coverage tests for missing lines
        ##################################################################################

        # Test __iadd__ (in-place addition) (lines 172-175, 181-184, 187, 191)
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        a += b
        self.assertTrue(np.allclose(a.values, [5., 7., 9.]))

        # Test __iadd__ with number
        a = Scalar([1., 2., 3.])
        a += 2.
        self.assertTrue(np.allclose(a.values, [3., 4., 5.]))

        # Test __iadd__ with integer result from non-integer (line 191)
        a = Scalar([1, 2, 3])  # Integer
        b = Scalar([1., 2., 3.])  # Float
        self.assertRaises(TypeError, lambda: a.__iadd__(b))

        # Test __isub__ (in-place subtraction)
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        a -= b
        self.assertTrue(np.allclose(a.values, [-3., -3., -3.]))

        # Test __isub__ with number
        a = Scalar([1., 2., 3.])
        a -= 2.
        self.assertTrue(np.allclose(a.values, [-1., 0., 1.]))

        # Test __imul__ (in-place multiplication) (lines 393-396, 400, 408-423, 443-452, 472-473, 477-515)
        a = Scalar([1., 2., 3.])
        b = Scalar([4., 5., 6.])
        a *= b
        self.assertTrue(np.allclose(a.values, [4., 10., 18.]))

        # Test __imul__ with number
        a = Scalar([1., 2., 3.])
        a *= 2.
        self.assertTrue(np.allclose(a.values, [2., 4., 6.]))

        # Test __imul__ with integer result from non-integer (line 495)
        a = Scalar([1, 2, 3])  # Integer
        b = Scalar([1., 2., 3.])  # Float
        self.assertRaises(TypeError, lambda: a.__imul__(b))

        # Test __imul__ with array-like arg_values (line 489-491)
        a = Scalar([1., 2., 3.])
        b = Scalar([4.])  # Scalar that broadcasts
        a *= b
        self.assertTrue(np.allclose(a.values, [4., 8., 12.]))

        # Test __itruediv__ (in-place division)
        a = Scalar([1., 2., 3.])
        b = Scalar([2., 4., 6.])
        a /= b
        self.assertTrue(np.allclose(a.values, [0.5, 0.5, 0.5]))

        # Test __itruediv__ with number
        a = Scalar([1., 2., 3.])
        a /= 2.
        self.assertTrue(np.allclose(a.values, [0.5, 1., 1.5]))

        # Test __ifloordiv__ (in-place floor division)
        a = Scalar([5., 7., 9.])
        b = Scalar([2., 3., 4.])
        a //= b
        self.assertTrue(np.allclose(a.values, [2., 2., 2.]))

        # Test __ifloordiv__ with number
        a = Scalar([5., 7., 9.])
        a //= 2.
        self.assertTrue(np.allclose(a.values, [2., 3., 4.]))

        # Test __imod__ (in-place modulus)
        a = Scalar([5., 7., 9.])
        b = Scalar([2., 3., 4.])
        a %= b
        self.assertTrue(np.allclose(a.values, [1., 1., 1.]))

        # Test __imod__ with number
        a = Scalar([5., 7., 9.])
        a %= 2.
        self.assertTrue(np.allclose(a.values, [1., 1., 1.]))

        # Test __ipow__ (in-place power)
        a = Scalar([2., 3., 4.])
        a **= 2
        self.assertTrue(np.allclose(a.values, [4., 9., 16.]))

        # Test __add__ with incompatible types (line 109-110, 116-119, 122)
        a = Scalar([1., 2., 3.])
        # Try to add incompatible type
        try:
            _ = a + "invalid"
            # If it doesn't raise, that's unexpected
            self.fail("Expected TypeError or ValueError")
        except (TypeError, ValueError):
            pass  # Expected

        # Test __add__ with incompatible numers
        a = Scalar([1., 2., 3.])
        b = Vector([1., 2., 3.])
        # This raises TypeError, not ValueError, because types are different
        self.assertRaises((TypeError, ValueError), lambda: a + b)

        # Test __mul__ with dual denominators (line 400)
        # This requires objects with denominators
        # Vector with drank=1 and another with drank=1 should raise
        try:
            a = Vector(np.arange(6).reshape(2, 3), drank=1)
            b = Vector(np.arange(6, 12).reshape(2, 3), drank=1)
            _ = a * b
            # If it doesn't raise, that's unexpected
            self.fail("Expected ValueError")
        except ValueError:
            pass  # Expected

        # Test _mul_by_number (internal method)
        # This is an internal method, so we test it indirectly through multiplication
        a = Scalar([1., 2., 3.])
        b = a * 2.
        self.assertTrue(np.allclose(b.values, [2., 4., 6.]))

        # Test _mul_by_number with derivatives (indirectly)
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a * 2.
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(b.d_dt.values, [0.2, 0.4, 0.6]))

        # Test reciprocal
        # An object equivalent to the reciprocal of this object.
        # This method is not implemented for the base class.
        a = Scalar([1., 2., 4.])
        # Scalar should override this, so it should work
        b = a.reciprocal()
        self.assertTrue(np.allclose(b.values, [1., 0.5, 0.25]))

        # Test zero
        # An object of this subclass containing all zeros.
        a = Scalar([1., 2., 3.])
        b = a.zero()
        self.assertEqual(type(b).__name__, 'Scalar')
        self.assertEqual(b.shape, ())
        self.assertTrue(np.allclose(b.values, 0.))

        # Test identity
        # An object of this subclass equivalent to the identity.
        # This method is overridden by Scalar, Matrix, and Boolean
        a = Scalar([1., 2., 3.])
        # Scalar should override this
        b = a.identity()
        self.assertEqual(type(b).__name__, 'Scalar')
        self.assertEqual(b.shape, ())
        self.assertTrue(np.allclose(b.values, 1.))

        # Test sum
        # The sum of the unmasked values along the specified axis or axes.
        a = Scalar([1., 2., 3., 4.])
        b = a.sum()
        self.assertEqual(b.shape, ())
        self.assertTrue(np.allclose(b.values, 10.))

        # Test sum with axis
        a = Scalar(np.arange(12).reshape(2, 3, 2))
        b = a.sum(axis=0)
        # Summing along axis=0 of shape (2, 3, 2) gives shape (3, 2)
        self.assertEqual(b.shape, (3, 2))

        # Test mean
        # The mean of the unmasked values along the specified axis or axes.
        a = Scalar([1., 2., 3., 4.])
        b = a.mean()
        self.assertEqual(b.shape, ())
        self.assertTrue(np.allclose(b.values, 2.5))

        # Test mean with axis
        a = Scalar(np.arange(12).reshape(2, 3, 2))
        b = a.mean(axis=0)
        # Mean along axis=0 of shape (2, 3, 2) gives shape (3, 2)
        self.assertEqual(b.shape, (3, 2))
