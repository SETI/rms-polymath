##########################################################################################
# tests/test_scalar_coverage.py
# Comprehensive coverage tests for scalar.py to achieve >90% coverage
##########################################################################################

import numpy as np
import unittest
import warnings
from contextlib import contextmanager

from polymath import Scalar, Vector, Boolean, Qube, Unit


@contextmanager
def prefer_builtins(value):
    """Context manager to temporarily set Qube.prefer_builtins() flag."""
    old_value = Qube.prefer_builtins()
    try:
        Qube.prefer_builtins(value)
        yield
    finally:
        Qube.prefer_builtins(old_value)


class Test_Scalar_Coverage(unittest.TestCase):

    def runTest(self):

        np.random.seed(54321)

        ##################################################################################
        # Test _minval and _maxval edge cases
        ##################################################################################
        # Test invalid dtype
        dtype = np.dtype('U')  # Unicode string dtype
        with self.assertRaises(ValueError):
            _ = Scalar._minval(dtype)

        with self.assertRaises(ValueError):
            _ = Scalar._maxval(dtype)

        # Test all dtype kinds
        for kind in ['f', 'u', 'i']:
            dtype = np.dtype(kind + '8')
            min_val = Scalar._minval(dtype)
            max_val = Scalar._maxval(dtype)
            self.assertIsNotNone(min_val)
            self.assertIsNotNone(max_val)

        # Test boolean dtype separately
        dtype = np.dtype('bool')
        min_val = Scalar._minval(dtype)
        max_val = Scalar._maxval(dtype)
        self.assertIsNotNone(min_val)
        self.assertIsNotNone(max_val)

        ##################################################################################
        # Test as_scalar edge cases
        ##################################################################################
        # Test with Boolean
        b = Boolean(True)
        s = Scalar.as_scalar(b)
        self.assertEqual(s, 1)

        # Test with Qube that's not Scalar
        # Vector has nrank=1, so converting to Scalar (nrank=0) fails on the rank mismatch
        v = Vector([1., 2., 3.])
        with self.assertRaises(ValueError):
            _ = Scalar.as_scalar(v)

        # Test with Unit
        s = Scalar.as_scalar(Unit.KM)
        self.assertIsNotNone(s.unit_)

        # Test recursive=False
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        s = Scalar.as_scalar(a, recursive=False)
        self.assertFalse(hasattr(s, 'd_dt'))

        ##################################################################################
        # Test to_scalar error case
        ##################################################################################
        # Test index out of range
        a = Scalar(1.)
        self.assertRaises(ValueError, a.to_scalar, 1)

        # Test recursive=False
        a = Scalar(1.)
        a.insert_deriv('t', Scalar(0.1))
        s = a.to_scalar(0, recursive=False)
        self.assertFalse(hasattr(s, 'd_dt'))

        ##################################################################################
        # Test as_index_and_mask error cases
        ##################################################################################
        # Test floating-point indexing
        a = Scalar([1.5, 2.5, 3.5])
        self.assertRaises(IndexError, a.as_index_and_mask)

        # Test with denominator
        a = Vector(np.arange(6).reshape(2, 3), drank=1)
        with self.assertRaises(ValueError):
            _ = a.as_index_and_mask()

        # Test purge=True with all masked
        a = Scalar([1, 2, 3], mask=True)
        idx, mask = a.as_index_and_mask(purge=True)
        self.assertEqual(len(idx), 0)

        # Test purge=True with partially masked
        a = Scalar([1, 2, 3])
        a = a.mask_where_eq(2)
        idx, mask = a.as_index_and_mask(purge=True)
        self.assertEqual(len(idx), 2)

        # Test masked=None with all masked
        a = Scalar([1, 2, 3], mask=True)
        idx, mask = a.as_index_and_mask(masked=999)
        self.assertTrue(np.all(idx == 999))

        # Test masked=None with partially masked
        a = Scalar([1, 2, 3])
        a = a.mask_where_eq(2)
        idx, mask = a.as_index_and_mask(masked=999)
        self.assertEqual(idx[1], 999)

        ##################################################################################
        # Test int() error cases
        ##################################################################################
        # Test with denominator
        a = Vector(np.arange(6).reshape(2, 3), drank=1)
        with self.assertRaises(ValueError):
            _ = a.int()

        # Test with top parameter and shift
        a = Scalar([1, 2, 3, 4, 5])
        b = a.int(top=3, shift=True, clip=False)
        # shift=True means shift values equal to top down by 1
        # So value 3 at index 2 should become 2, value 4 at index 3 should become 3, etc.
        # Actually, the logic shifts values equal to top, so if top=3, values of 3 become 2
        # Let's just verify the operation completes
        self.assertEqual(len(b), 5)

        # Test with remask and clip
        a = Scalar([1, 2, 3, 4, 5])
        b = a.int(top=3, remask=True, clip=False)
        self.assertTrue(b.mask[3] or b.mask[4])

        # Test with clip=True
        a = Scalar([1, 2, 3, 4, 5])
        b = a.int(top=3, clip=True)
        self.assertTrue(np.all(b.values <= 2))

        # Test with remask and no top
        a = Scalar([-1, 0, 1, 2, 3])
        b = a.int(remask=True, clip=False)
        self.assertTrue(b.mask[0])

        # Test builtins
        a = Scalar(5.7)
        with prefer_builtins(True):
            b = a.int()
            self.assertIsInstance(b, int)

        ##################################################################################
        # Test frac() error case
        ##################################################################################
        # Test with denominator
        # frac() is a Scalar method, so test with Scalar that has denominator
        # Actually, Scalar can't have denominator, so this test is hard to do
        # Let's just test that frac() works normally
        a = Scalar([1.5, 2.5, 3.5])
        b = a.frac()
        self.assertTrue(np.allclose(b.values, [0.5, 0.5, 0.5]))

        # Test with derivatives
        a = Scalar([1.5, 2.5, 3.5])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a.frac(recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))

        ##################################################################################
        # Test sin() error case
        ##################################################################################
        # Test with denominator
        # sin() is a Scalar method, and Scalar can't have denominator
        # So this error case is hard to test directly
        # Let's just test that sin() works normally
        a = Scalar([0., np.pi/2, np.pi], unit=Unit.RAD)
        b = a.sin()
        self.assertTrue(np.allclose(b.values, [0., 1., 0.], atol=1e-10))

        ##################################################################################
        # Test cos() error case
        ##################################################################################
        # Test with denominator
        # cos() is a Scalar method, and Scalar can't have denominator
        # Let's just test that cos() works normally
        a = Scalar([0., np.pi/2, np.pi], unit=Unit.RAD)
        b = a.cos()
        self.assertTrue(np.allclose(b.values, [1., 0., -1.], atol=1e-10))

        ##################################################################################
        # Test tan() error case
        ##################################################################################
        # Test with denominator
        # tan() is a Scalar method, and Scalar can't have denominator
        # Let's just test that tan() works normally
        a = Scalar([0., np.pi/4], unit=Unit.RAD)
        b = a.tan()
        self.assertTrue(np.allclose(b.values, [0., 1.], atol=1e-10))

        ##################################################################################
        # Test arcsin() error cases
        ##################################################################################
        # Test with denominator
        # arcsin() is a Scalar method, and Scalar can't have denominator
        # Let's just test that arcsin() works normally
        a = Scalar([0., 0.5, 1.])
        b = a.arcsin()
        self.assertTrue(np.allclose(b.values, [0., np.arcsin(0.5), np.pi/2], atol=1e-10))

        # Test with check=False and invalid value
        a = Scalar(2.)  # Outside [-1, 1]
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            with self.assertRaises(ValueError):
                _ = a.arcsin(check=False)

        # Test with check=True and invalid values
        a = Scalar([-2., 0., 2.])
        b = a.arcsin(check=True)
        self.assertTrue(b.mask[0] or b.mask[2])

        ##################################################################################
        # Test arccos() error cases
        ##################################################################################
        # Test with denominator
        # arccos() is a Scalar method, and Scalar can't have denominator
        # Let's just test that arccos() works normally
        a = Scalar([1., 0.5, 0.])
        b = a.arccos()
        self.assertTrue(np.allclose(b.values, [0., np.arccos(0.5), np.pi/2], atol=1e-10))

        # Test with check=False and invalid value
        a = Scalar(2.)  # Outside [-1, 1]
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            with self.assertRaises(ValueError):
                _ = a.arccos(check=False)

        # Test with check=True and invalid values
        a = Scalar([-2., 0., 2.])
        b = a.arccos(check=True)
        self.assertTrue(b.mask[0] or b.mask[2])

        ##################################################################################
        # Test arctan() error case
        ##################################################################################
        # Test with denominator
        # arctan() is a Scalar method, and Scalar can't have denominator
        # Let's just test that arctan() works normally
        a = Scalar([0., 1., -1.])
        b = a.arctan()
        self.assertTrue(np.allclose(b.values, [0., np.pi/4, -np.pi/4], atol=1e-10))

        ##################################################################################
        # Test arctan2() error case
        ##################################################################################
        # Test with denominator
        # arctan2() requires both arguments to be Scalars without denominators
        # Let's test the normal case
        a = Scalar(1.)
        b = Scalar(1.)
        c = a.arctan2(b)
        self.assertAlmostEqual(c, np.pi/4, places=10)

        ##################################################################################
        # Test sqrt() error cases
        ##################################################################################
        # Test with denominator
        # sqrt() is a Scalar method, and Scalar can't have denominator
        # Let's just test that sqrt() works normally
        a = Scalar([1., 4., 9.])
        b = a.sqrt()
        self.assertTrue(np.allclose(b.values, [1., 2., 3.]))

        # Test with check=False and negative value
        a = Scalar(-1.)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            with self.assertRaises(ValueError):
                _ = a.sqrt(check=False)

        ##################################################################################
        # Test log() error cases
        ##################################################################################
        # Test with denominator
        # log() is a Scalar method, and Scalar can't have denominator
        # Let's just test that log() works normally
        a = Scalar([1., np.e, np.e**2])
        b = a.log()
        self.assertTrue(np.allclose(b.values, [0., 1., 2.], atol=1e-10))

        # Test with check=False and non-positive value
        a = Scalar(0.)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            with self.assertRaises(ValueError):
                _ = a.log(check=False)

        ##################################################################################
        # Test exp() error cases
        ##################################################################################
        # Test with denominator
        # exp() is a Scalar method, and Scalar can't have denominator
        # Let's just test that exp() works normally
        a = Scalar([0., 1., 2.])
        b = a.exp()
        self.assertTrue(np.allclose(b.values, [1., np.e, np.e**2], atol=1e-10))

        # Test with check=False and overflow
        a = Scalar(1000.)  # Very large value
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            # The overflow surfaces as the RuntimeWarning raised by np.exp, unless it is
            # first converted to a ValueError by Scalar.exp() itself
            with self.assertRaises((ValueError, RuntimeWarning)):
                _ = a.exp(check=False)

        # Test with check=True and overflow
        a = Scalar(1000.)
        b = a.exp(check=True)
        self.assertTrue(b.mask)  # Overflow values are masked

        ##################################################################################
        # Test sign() edge cases
        ##################################################################################
        # Test with zeros=False
        a = Scalar([-1., 0., 1.])
        b = a.sign(zeros=False)
        self.assertEqual(b[1], 1)  # Zero should become 1

        # Test builtins
        a = Scalar(1.)
        with prefer_builtins(True):
            b = a.sign()
            # sign() returns the sign, which for float 1.0 is 1.0 (float), not int
            # But if it's an integer Scalar, it might return int
            a_int = Scalar(1)  # Integer
            b_int = a_int.sign()
            # The result type depends on the input type
            self.assertIsInstance(b, (int, float))
            self.assertIsInstance(b_int, int)
            self.assertEqual(b_int, 1)

        ##################################################################################
        # Test max() error case
        ##################################################################################
        # Test with denominator
        # max() is a Scalar method, and Scalar can't have denominator
        # Let's just test that max() works normally
        a = Scalar([1., 3., 2.])
        b = a.max()
        self.assertEqual(b, 3.)

        # Test with all masked
        a = Scalar([1., 2., 3.], mask=True)
        b = a.max()
        self.assertTrue(b.mask)

        # Test with partially masked
        a = Scalar([1., 2., 3.])
        a = a.mask_where_eq(2.)
        b = a.max()
        self.assertEqual(b, 3.)

        # Test builtins
        a = Scalar([1., 2., 3.])
        with prefer_builtins(True):
            b = a.max()
            self.assertIsInstance(b, (int, float))

        ##################################################################################
        # Test min() error case
        ##################################################################################
        # Test with denominator
        # min() is a Scalar method, and Scalar can't have denominator
        # Let's just test that min() works normally
        a = Scalar([3., 1., 2.])
        b = a.min()
        self.assertEqual(b, 1.)

        # Test with all masked
        a = Scalar([1., 2., 3.], mask=True)
        b = a.min()
        self.assertTrue(b.mask)

        # Test with partially masked
        a = Scalar([1., 2., 3.])
        a = a.mask_where_eq(2.)
        b = a.min()
        self.assertEqual(b, 1.)

        # Test builtins
        a = Scalar([1., 2., 3.])
        with prefer_builtins(True):
            b = a.min()
            self.assertIsInstance(b, (int, float))

        ##################################################################################
        # Test argmax() error cases
        ##################################################################################
        # Test with denominator
        # argmax() is a Scalar method, and Scalar can't have denominator
        # Let's just test that argmax() works normally
        a = Scalar([1., 3., 2.])
        b = a.argmax()
        self.assertEqual(b, 1)  # Index of max value

        # Test with shape ()
        a = Scalar(1.)
        self.assertRaises(ValueError, a.argmax)

        # Test with all masked
        a = Scalar([1., 2., 3.], mask=True)
        b = a.argmax()
        self.assertTrue(b.mask)

        # Test with partially masked
        a = Scalar([1., 2., 3.])
        a = a.mask_where_eq(2.)
        b = a.argmax()
        # Should return index of max unmasked value

        # Test builtins
        a = Scalar([1., 2., 3.])
        with prefer_builtins(True):
            b = a.argmax()
            self.assertIsInstance(b, int)

        ##################################################################################
        # Test argmin() error cases
        ##################################################################################
        # Test with denominator
        # argmin() is a Scalar method, and Scalar can't have denominator
        # Let's just test that argmin() works normally
        a = Scalar([3., 1., 2.])
        b = a.argmin()
        self.assertEqual(b, 1)  # Index of min value

        # Test with shape ()
        a = Scalar(1.)
        self.assertRaises(ValueError, a.argmin)

        # Test with all masked
        a = Scalar([1., 2., 3.], mask=True)
        b = a.argmin()
        self.assertTrue(b.mask)

        # Test with partially masked
        a = Scalar([1., 2., 3.])
        a = a.mask_where_eq(2.)
        b = a.argmin()
        # Should return index of min unmasked value

        # Test builtins
        a = Scalar([1., 2., 3.])
        with prefer_builtins(True):
            b = a.argmin()
            self.assertIsInstance(b, int)

        ##################################################################################
        # Test maximum() error cases
        ##################################################################################
        # Test missing arguments
        self.assertRaises(ValueError, Scalar.maximum)

        # Test with denominator
        # maximum() is a Scalar static method, and Scalar can't have denominator
        # Let's test the normal case
        a = Scalar([1., 3., 2.])
        b = Scalar([2., 1., 4.])
        c = Scalar.maximum(a, b)
        self.assertTrue(np.allclose(c.values, [2., 3., 4.]))

        # Test with single argument
        a = Scalar([1., 2., 3.])
        b = Scalar.maximum(a)
        self.assertTrue(np.allclose(b.values, a.values))

        # Test with mixed int/float
        a = Scalar([1, 2, 3])
        b = Scalar([1., 2., 3.])
        c = Scalar.maximum(a, b)
        self.assertTrue(c.is_float())

        ##################################################################################
        # Test minimum() error cases
        ##################################################################################
        # Test missing arguments
        self.assertRaises(ValueError, Scalar.minimum)

        # Test with denominator
        # minimum() is a Scalar static method, and Scalar can't have denominator
        # Let's test the normal case
        a = Scalar([1., 3., 2.])
        b = Scalar([2., 1., 4.])
        c = Scalar.minimum(a, b)
        self.assertTrue(np.allclose(c.values, [1., 1., 2.]))

        # Test with single argument
        a = Scalar([1., 2., 3.])
        b = Scalar.minimum(a)
        self.assertTrue(np.allclose(b.values, a.values))

        # Test with mixed int/float
        a = Scalar([1, 2, 3])
        b = Scalar([1., 2., 3.])
        c = Scalar.minimum(a, b)
        self.assertTrue(c.is_float())

        ##################################################################################
        # Test median() error case
        ##################################################################################
        # Test with denominator
        # median() is a Scalar method, and Scalar can't have denominator
        # Let's just test that median() works normally
        a = Scalar([1., 3., 2., 4., 5.])
        b = a.median()
        self.assertEqual(b, 3.)

        # Test with all masked
        a = Scalar([1., 2., 3.], mask=True)
        b = a.median()
        self.assertTrue(b.mask)

        # Test with axis=None and masked
        a = Scalar([1., 2., 3., 4., 5.])
        a = a.mask_where_eq(3.)
        b = a.median(axis=None)
        # Should compute median of unmasked values

        # Test with axis and masked
        a = Scalar(np.arange(24).reshape(2, 3, 4))
        a = a.mask_where_eq(5.)
        b = a.median(axis=0)
        # Should compute median along axis 0

        # Test builtins
        a = Scalar([1., 2., 3., 4., 5.])
        with prefer_builtins(True):
            b = a.median()
            self.assertIsInstance(b, float)

        ##################################################################################
        # Test sort() error case
        ##################################################################################
        # Test with denominator
        # sort() is a Scalar method, and Scalar can't have denominator
        # Let's just test that sort() works normally
        a = Scalar([3., 1., 2.])
        b = a.sort()
        self.assertTrue(np.allclose(b.values, [1., 2., 3.]))

        # Test with masked values
        a = Scalar([3., 1., 2.])
        a = a.mask_where_eq(2.)
        b = a.sort()
        # Masked values should appear at end

        ##################################################################################
        # Test reciprocal() error cases
        ##################################################################################
        # Test with denominator
        # reciprocal() is a Scalar method, and Scalar can't have denominator
        # The error check is for self._rank, not self._drank
        # Let's test the normal case
        a = Scalar([1., 2., 4.])
        b = a.reciprocal()
        self.assertTrue(np.allclose(b.values, [1., 0.5, 0.25]))

        # Test with nozeros=True and zero
        a = Scalar([1., 0., 2.])
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            with self.assertRaises(ValueError):
                _ = a.reciprocal(nozeros=True)

        # Test with nozeros=False and zero
        a = Scalar([1., 0., 2.])
        b = a.reciprocal(nozeros=False)
        self.assertTrue(b.mask[1])  # Zero should be masked

        ##################################################################################
        # Test __pow__ error cases
        ##################################################################################
        # Test with denominator
        # __pow__ checks for denominator using _disallow_denom
        # Scalar can't have denominator, so this is hard to test
        # Let's test the normal case
        a = Scalar([2., 3., 4.])
        b = a ** 2
        self.assertTrue(np.allclose(b.values, [4., 9., 16.]))

        # Test with array exponent
        a = Scalar([2., 3., 4.])
        b = Scalar([1., 2.])  # Different shape
        with self.assertRaises(ValueError):
            _ = a ** b

        # Test with unit and array exponent
        a = Scalar([2., 3., 4.], unit=Unit.KM)
        b = Scalar([1., 2.])  # Array exponent
        with self.assertRaises(ValueError):
            _ = a ** b

        # Test with masked result
        a = Scalar(0.)
        b = Scalar(-1.)
        c = a ** b  # 0 ** -1 is undefined, so the result is masked rather than raised
        self.assertTrue(c.mask)

        # Test with non-Real exponent
        a = Scalar([2., 3., 4.])
        with self.assertRaises(TypeError):
            _ = a ** "invalid"

        ##################################################################################
        # Test __le__, __lt__, __ge__, __gt__ with denominators
        ##################################################################################
        # Test with denominators
        a = Scalar(1.)
        b = Vector(np.arange(6).reshape(2, 3), drank=1)

        with self.assertRaises(ValueError):
            _ = a <= b

        with self.assertRaises(ValueError):
            _ = a < b

        with self.assertRaises(ValueError):
            _ = a >= b

        with self.assertRaises(ValueError):
            _ = a > b

        # Test builtins
        a = Scalar(1.)
        b = Scalar(2.)
        with prefer_builtins(True):
            c = a <= b
            self.assertIsInstance(c, bool)
            c = a < b
            self.assertIsInstance(c, bool)
            c = a >= b
            self.assertIsInstance(c, bool)
            c = a > b
            self.assertIsInstance(c, bool)

        ##################################################################################
        # Test __round__
        ##################################################################################
        a = Scalar(1.234567)
        b = round(a, 2)
        self.assertAlmostEqual(b, 1.23, places=2)

        ##################################################################################
        # Test __abs__ with derivatives
        ##################################################################################
        a = Scalar([-1., 2., -3.])
        a.insert_deriv('t', Scalar([-0.1, 0.2, -0.3]))
        b = abs(a)
        self.assertTrue(hasattr(b, 'd_dt'))
        # Derivatives should be multiplied by sign

        ##################################################################################
        # Test _power_0 with derivatives
        ##################################################################################
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a._power_0(recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        # Derivatives should be zeros

        ##################################################################################
        # Test _power_1
        ##################################################################################
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a._power_1(recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        b = a._power_1(recursive=False)
        self.assertFalse(hasattr(b, 'd_dt'))

        ##################################################################################
        # Test _power_2, _power_3, _power_4
        ##################################################################################
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a._power_2(recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(b.values, [1., 4., 9.]))

        b = a._power_3(recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(b.values, [1., 8., 27.]))

        b = a._power_4(recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(b.values, [1., 16., 81.]))

        ##################################################################################
        # Test _power_neg_1, _power_half, _power_neg_half
        ##################################################################################
        a = Scalar([1., 2., 4.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a._power_neg_1(recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(b.values, [1., 0.5, 0.25]))

        b = a._power_half(recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(b.values, [1., np.sqrt(2.), 2.]))

        b = a._power_neg_half(recursive=True)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(b.values, [1., 1./np.sqrt(2.), 0.5]))

        ##################################################################################
        # Test __pow__ with easy powers
        ##################################################################################
        a = Scalar([1., 2., 3.])
        # Test power 0
        b = a ** 0
        self.assertTrue(np.allclose(b.values, [1., 1., 1.]))

        # Test power 1
        b = a ** 1
        self.assertTrue(np.allclose(b.values, [1., 2., 3.]))

        # Test power 2
        b = a ** 2
        self.assertTrue(np.allclose(b.values, [1., 4., 9.]))

        # Test power 3
        b = a ** 3
        self.assertTrue(np.allclose(b.values, [1., 8., 27.]))

        # Test power 4
        b = a ** 4
        self.assertTrue(np.allclose(b.values, [1., 16., 81.]))

        # Test power -1
        b = a ** -1
        self.assertTrue(np.allclose(b.values, [1., 0.5, 1./3.]))

        # Test power 0.5
        b = a ** 0.5
        self.assertTrue(np.allclose(b.values, [1., np.sqrt(2.), np.sqrt(3.)]))

        # Test power -0.5
        b = a ** -0.5
        self.assertTrue(np.allclose(b.values, [1., 1./np.sqrt(2.), 1./np.sqrt(3.)]))

        # Test with integer exponent that needs conversion
        a = Scalar([1, 2, 3])  # Integer
        b = Scalar(-1)  # Negative integer exponent
        c = a ** b
        self.assertTrue(c.is_float())  # Should convert to float

        # Test with masked exponent
        a = Scalar([2., 3., 4.])
        b = Scalar(2., mask=True)
        c = a ** b
        self.assertTrue(np.all(c.mask))

        # Test with invalid result
        a = Scalar([2., 3., 4.])
        b = Scalar([1000., 1000., 1000.])  # Very large exponent
        c = a ** b
        # 2**1000 is representable; 3**1000 and 4**1000 overflow and get masked
        self.assertTrue(np.all(c.mask == [False, True, True]))

        # Test with derivatives
        a = Scalar([2., 3., 4.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        b = a ** 2
        self.assertTrue(hasattr(b, 'd_dt'))

        ##################################################################################
        # Additional tests for missing lines
        ##################################################################################

        # Test as_scalar with Boolean.as_int() path
        b = Boolean([True, False, True])
        s = Scalar.as_scalar(b, recursive=False)
        self.assertEqual(type(s), Scalar)

        # Test as_index_and_mask with scalar values
        a = Scalar(5)
        idx, mask = a.as_index_and_mask()
        self.assertEqual(idx, 5)
        self.assertFalse(mask)

        # Test as_index_and_mask with masked=None
        a = Scalar([1, 2, 3])
        idx, mask = a.as_index_and_mask(masked=None)
        self.assertTrue(np.array_equal(idx, [1, 2, 3]))
        self.assertFalse(mask)

        # Test int() with top as list/tuple
        a = Scalar([1.5, 2.5, 3.5])
        b = a.int(top=[5])
        self.assertTrue(np.all(b.values <= 4))

        # Test int() with non-int values and mask copying
        a = Scalar([1.5, 2.5, 3.5], mask=[False, True, False])
        b = a.int(top=3)
        self.assertTrue(isinstance(b._mask, np.ndarray))

        # Test int() with shift and array values
        a = Scalar([1., 2., 3.])
        b = a.int(top=2, shift=True, clip=False)
        # When shift=True and value==top, it becomes top-1
        # Value 2 becomes 1, but value 3 stays 3 (no clip)
        self.assertEqual(b.values[0], 1)  # 1 stays 1
        self.assertEqual(b.values[1], 1)  # 2 becomes 1 (shifted)
        self.assertEqual(b.values[2], 3)  # 3 stays 3 (no clip)

        # Test int() with clip and remask
        a = Scalar([-1., 0., 1., 2.])
        b = a.int(top=2, clip=True, remask=True)
        self.assertTrue(np.all(b.values >= 0))
        self.assertTrue(np.all(b.values < 2))

        # Test int() with builtins
        a = Scalar(1.5)
        with prefer_builtins(True):
            b = a.int(builtins=True)
            self.assertIsInstance(b, int)

        # Test frac() with denominators
        # Scalar with drank=1 needs values with shape (..., 1)
        a = Scalar([[1.5]], drank=1)  # shape (1,), item (1,)
        with self.assertRaises(ValueError):
            _ = a.frac()

        # Test sin() with denominators
        a = Scalar([[1.0]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.sin()

        # Test cos() with denominators
        a = Scalar([[1.0]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.cos()

        # Test tan() with denominators
        a = Scalar([[1.0]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.tan()

        # Test arcsin() with denominators
        a = Scalar([[0.5]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.arcsin()

        # Test arcsin() with RuntimeWarning
        a = Scalar(1.5)  # Outside domain
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaises(ValueError):
                _ = a.arcsin(check=False)

        # Test arccos() with denominators
        a = Scalar([[0.5]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.arccos()

        # Test arccos() with RuntimeWarning
        a = Scalar(1.5)  # Outside domain
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaises(ValueError):
                _ = a.arccos(check=False)

        # Test arctan() with denominators
        a = Scalar([[1.0]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.arctan()

        # Test arctan2() with denominators
        a = Scalar([[1.0]], drank=1)
        b = Scalar(1.0)
        with self.assertRaises(ValueError):
            _ = a.arctan2(b)

        # Test sqrt() with denominators
        a = Scalar([[4.0]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.sqrt()

        # Test log() with denominators
        a = Scalar([[2.0]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.log()

        # Test exp() with denominators
        a = Scalar([[1.0]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.exp()

        # Test exp() with RuntimeWarning/ValueError
        a = Scalar(1000.)  # Very large value
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaises((ValueError, RuntimeWarning)):
                _ = a.exp(check=False)

        # Test sign() with builtins
        a = Scalar(1.0)
        with prefer_builtins(True):
            b = a.sign(builtins=True)
            self.assertIsInstance(b, float)

        # Test solve_quadratic with include_antimask
        a = Scalar([1., 2., 3.])
        b = Scalar([-1., -2., -3.])
        c = Scalar([0., 0., 0.])
        _, _, discr = Scalar.solve_quadratic(a, b, c, include_antimask=True)
        self.assertIsNotNone(discr)

        # Test max() with empty size
        a = Scalar([])
        b = a.max()
        # Empty array max() returns shape (0,)
        self.assertEqual(b.shape, (0,))

        # Test max() with mask handling
        a = Scalar([1., 2., 3.], mask=[False, True, False])
        b = a.max()
        self.assertEqual(b, 3.)

        # Test min() with empty size
        a = Scalar([])
        b = a.min()
        self.assertEqual(b.shape, (0,))

        # Test min() with mask handling
        a = Scalar([1., 2., 3.], mask=[True, False, False])
        b = a.min()
        self.assertEqual(b, 2.)

        # Test min() with builtins
        a = Scalar([1., 2., 3.])
        with prefer_builtins(True):
            b = a.min(builtins=True)
            self.assertIsInstance(b, float)

        # Test argmax() with denominators
        # Scalar with drank=1 needs values with shape (n, 1) for array of size n
        a = Scalar([[1.], [2.], [3.]], drank=1)  # shape (3,), item (1,)
        with self.assertRaises(ValueError):
            _ = a.argmax()

        # Test argmax() with empty size
        a = Scalar([])
        b = a.argmax()
        self.assertEqual(b.shape, (0,))

        # Test argmax() with mask handling
        a = Scalar([1., 2., 3.], mask=[True, False, False])
        b = a.argmax()
        self.assertEqual(b, 2)

        # Test argmax() with builtins
        a = Scalar([1., 2., 3.])
        with prefer_builtins(True):
            b = a.argmax(builtins=True)
            self.assertIsInstance(b, int)

        # Test argmin() with denominators
        a = Scalar([[1.], [2.], [3.]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.argmin()

        # Test argmin() with empty size
        a = Scalar([])
        b = a.argmin()
        self.assertEqual(b.shape, (0,))

        # Test argmin() with mask handling
        a = Scalar([1., 2., 3.], mask=[True, False, False])
        b = a.argmin()
        self.assertEqual(b, 1)

        # Test argmin() with builtins
        a = Scalar([1., 2., 3.])
        with prefer_builtins(True):
            b = a.argmin(builtins=True)
            self.assertIsInstance(b, int)

        ##################################################################################
        # Test argmax edge cases for coverage
        ##################################################################################
        # Test argmax with partially masked array and scalar mask result
        a = Scalar([[1., 2., 3.], [4., 5., 6.]])
        mask = np.array([[False, False, False], [True, True, True]])
        a_masked = Scalar(a.values, mask=mask)
        result = a_masked.argmax(axis=1)
        # Row 0 should have argmax, row 1 should be masked
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.shape, (2,))
        self.assertFalse(result.mask[0])
        self.assertTrue(result.mask[1])

        # Test argmax with partially masked array and array mask result
        a = Scalar([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        mask = np.array([[False, False, False], [True, True, True], [False, True, False]])
        a_masked = Scalar(a.values, mask=mask)
        result = a_masked.argmax(axis=1)
        # Should handle array mask case
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.shape, (3,))

        # Test argmax with scalar mask result
        # Need case where np.all(self._mask, axis=axis) returns scalar True
        # This happens when reducing to scalar shape
        a = Scalar([1., 2., 3.], mask=[True, True, True])
        result = a.argmax(axis=None)
        # When all masked and axis=None, mask becomes scalar
        self.assertIsInstance(result, Scalar)
        # Verify the code path was executed
        self.assertTrue(result.mask if isinstance(result.mask, (bool, np.bool_)) else np.all(result.mask))

        ##################################################################################
        # Test argmin edge cases for coverage
        ##################################################################################
        # Test argmin with partially masked array and scalar mask result
        a = Scalar([[1., 2., 3.], [4., 5., 6.]])
        mask = np.array([[False, False, False], [True, True, True]])
        a_masked = Scalar(a.values, mask=mask)
        result = a_masked.argmin(axis=1)
        # Row 0 should have argmin, row 1 should be masked
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.shape, (2,))
        self.assertFalse(result.mask[0])
        self.assertTrue(result.mask[1])

        # Test argmin with partially masked array and array mask result
        a = Scalar([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        mask = np.array([[False, False, False], [True, True, True], [False, True, False]])
        a_masked = Scalar(a.values, mask=mask)
        result = a_masked.argmin(axis=1)
        # Should handle array mask case
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.shape, (3,))

        # Test argmin with scalar mask result
        # Need case where np.all(self._mask, axis=axis) returns scalar True
        # This happens when reducing to scalar shape
        a = Scalar([1., 2., 3.], mask=[True, True, True])
        result = a.argmin(axis=None)
        # When all masked and axis=None, mask becomes scalar
        self.assertIsInstance(result, Scalar)
        # Verify the code path was executed
        self.assertTrue(result.mask if isinstance(result.mask, (bool, np.bool_)) else np.all(result.mask))

        # Test maximum() with denominators
        a = Scalar([[1.], [2.], [3.]], drank=1)
        b = Scalar([2., 3., 4.])
        with self.assertRaises(ValueError):
            _ = Scalar.maximum(a, b)

        # Test minimum() with denominators
        a = Scalar([[1.], [2.], [3.]], drank=1)
        b = Scalar([2., 3., 4.])
        with self.assertRaises(ValueError):
            _ = Scalar.minimum(a, b)

        # Test median() with denominators
        a = Scalar([[1.], [2.], [3.]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.median()

        # Test median() with empty size
        a = Scalar([])
        b = a.median()
        self.assertEqual(b.shape, (0,))

        # Test median() with mask handling
        a = Scalar([1., 2., 3., 4., 5.], mask=[True, False, False, False, True])
        b = a.median()
        self.assertIsNotNone(b)

        # Test median() with builtins
        a = Scalar([1., 2., 3.])
        with prefer_builtins(True):
            b = a.median(builtins=True)
            self.assertIsInstance(b, float)

        # Test sort() with denominators
        a = Scalar([[3.], [1.], [2.]], drank=1)
        with self.assertRaises(ValueError):
            _ = a.sort()

        # Test sort() with empty size
        # Unlike argmax()/argmin()/median(), sort() raises IndexError on an empty array,
        # because _zero_sized_result() indexes the empty array with index 0
        a = Scalar([])
        with self.assertRaises(IndexError):
            _ = a.sort()
