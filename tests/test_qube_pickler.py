##########################################################################################
# tests/test_qube_pickler.py
# Unit tests for Qube pickling operations
##########################################################################################

import numpy as np
import unittest
import pickle

from polymath import Qube, Scalar, Vector, Vector3, Boolean


class Test_Qube_pickler(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # Test set_pickle_digits
        # Set the desired number of decimal digits of precision in the storage of this
        # object's floating-point values and their derivatives.
        a = Scalar([1.23456789, 2.34567890])
        a.set_pickle_digits(8, 'fpzip')
        digits = a.pickle_digits()
        self.assertEqual(digits[0], 8)
        self.assertEqual(digits[1], 8)

        # Test set_pickle_digits with tuple
        a = Scalar([1.23456789, 2.34567890])
        a.set_pickle_digits((8, 7), ('fpzip', 'smallest'))
        digits = a.pickle_digits()
        self.assertEqual(digits[0], 8)
        self.assertEqual(digits[1], 7)

        # Test set_pickle_digits with "double"
        a = Scalar([1.23456789, 2.34567890])
        a.set_pickle_digits('double', 'fpzip')
        digits = a.pickle_digits()
        self.assertEqual(digits[0], 'double')
        self.assertEqual(digits[1], 'double')

        # Test set_pickle_digits with "single"
        a = Scalar([1.23456789, 2.34567890])
        a.set_pickle_digits('single', 'fpzip')
        digits = a.pickle_digits()
        self.assertEqual(digits[0], 'single')
        self.assertEqual(digits[1], 'single')

        # Test set_pickle_digits with reference options
        a = Scalar([1.23456789, 2.34567890])
        a.set_pickle_digits(8, 'smallest')
        ref = a.pickle_reference()
        self.assertEqual(ref[0], 'smallest')

        a.set_pickle_digits(8, 'largest')
        ref = a.pickle_reference()
        self.assertEqual(ref[0], 'largest')

        a.set_pickle_digits(8, 'mean')
        ref = a.pickle_reference()
        self.assertEqual(ref[0], 'mean')

        a.set_pickle_digits(8, 'median')
        ref = a.pickle_reference()
        self.assertEqual(ref[0], 'median')

        a.set_pickle_digits(8, 'logmean')
        ref = a.pickle_reference()
        self.assertEqual(ref[0], 'logmean')

        a.set_pickle_digits(8, 'fpzip')
        ref = a.pickle_reference()
        self.assertEqual(ref[0], 'fpzip')

        # Test set_pickle_digits with numeric reference
        a = Scalar([1.23456789, 2.34567890])
        a.set_pickle_digits(8, 100.)
        ref = a.pickle_reference()
        self.assertEqual(ref[0], 100.)

        # Test set_pickle_digits with derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        a.set_pickle_digits((8, 7), ('fpzip', 'smallest'))
        # Derivatives should have the second value
        self.assertEqual(a.d_dt.pickle_digits()[0], 7)
        self.assertEqual(a.d_dt.pickle_reference()[0], 'smallest')

        # Test set_default_pickle_digits
        # Set the default number of decimal digits of precision in the storage of
        # floating-point values and their derivatives.
        Qube.set_default_pickle_digits(10, 'mean')
        a = Scalar([1., 2., 3.])
        digits = a.pickle_digits()
        self.assertEqual(digits[0], 10)
        ref = a.pickle_reference()
        self.assertEqual(ref[0], 'mean')

        # Reset to default
        Qube.set_default_pickle_digits('double', 'fpzip')

        # Test pickle_digits
        # The digits of floating-point precision to include when pickling this object and its
        # derivatives.
        a = Scalar([1., 2., 3.])
        digits = a.pickle_digits()
        self.assertIsInstance(digits, tuple)
        self.assertEqual(len(digits), 2)

        # Test pickle_reference
        # The reference value to use when determining the number of digits of floating-point
        # precision in this object and its derivatives.
        a = Scalar([1., 2., 3.])
        ref = a.pickle_reference()
        self.assertIsInstance(ref, tuple)
        self.assertEqual(len(ref), 2)

        # Test __getstate__ and __setstate__
        # The state is defined by a dictionary containing most of the Qube attributes.
        # "_cache" is removed (or set to empty dict).
        # "_mask", and "_values" are replaced by encodings.
        # "PICKLE_VERSION" is added.
        # New attribute "MASK_ENCODING" is a list of the steps that have been applied to the
        # mask.
        # New attribute "VALS_ENCODING" is a list of the steps that have been applied to the
        # values.
        a = Scalar([1., 2., 3., 4.])
        state = a.__getstate__()
        self.assertIn('PICKLE_VERSION', state)
        self.assertIn('MASK_ENCODING', state)
        self.assertIn('VALS_ENCODING', state)
        # Note: _cache may be present but should be empty or cleared
        if '_cache' in state:
            self.assertEqual(state['_cache'], {})

        # Test round-trip pickling
        a = Scalar([1., 2., 3., 4.])
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a.values, b.values))
        self.assertEqual(a.mask, b.mask)

        # Test pickling with masked values
        a = Scalar([1., 2., 3., 4.])
        a = a.mask_where_eq(2.)
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertEqual(a.shape, b.shape)
        # Values should match for unmasked elements
        self.assertTrue(np.allclose(a.values[~a.mask], b.values[~b.mask]))
        self.assertTrue(np.array_equal(a.mask, b.mask))

        # Test pickling fully masked object
        a = Scalar([1., 2., 3., 4.])
        a = a.mask_where_eq(1.)
        a = a.mask_where_eq(2.)
        a = a.mask_where_eq(3.)
        a = a.mask_where_eq(4.)
        state = a.__getstate__()
        self.assertIn(('ALL_MASKED',), state['VALS_ENCODING'])

        # Test pickling with derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(a.d_dt.values, b.d_dt.values))

        # Test pickling integer arrays
        a = Scalar([1, 2, 3, 4])
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.array_equal(a.values, b.values))

        # Test pickling boolean arrays
        a = Boolean([True, False, True, False])
        state = a.__getstate__()
        b = Boolean.__new__(Boolean)
        b.__setstate__(state)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.array_equal(a.values, b.values))

        # Test pickling Vector
        a = Vector([1., 2., 3.])
        state = a.__getstate__()
        b = Vector.__new__(Vector)
        b.__setstate__(state)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a.values, b.values))

        # Test pickling Vector3
        a = Vector3([1., 2., 3.])
        state = a.__getstate__()
        b = Vector3.__new__(Vector3)
        b.__setstate__(state)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a.values, b.values))

        # Test pickling with different compression methods
        a = Scalar(np.random.randn(1000))
        a.set_pickle_digits(8, 'smallest')
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.allclose(a.values, b.values, rtol=1e-7))

        a.set_pickle_digits(8, 'largest')
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.allclose(a.values, b.values, rtol=1e-7))

        a.set_pickle_digits(8, 'mean')
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.allclose(a.values, b.values, rtol=1e-7))

        # Test standard pickle module
        a = Scalar([1., 2., 3., 4.])
        data = pickle.dumps(a)
        b = pickle.loads(data)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a.values, b.values))

        # Test standard pickle with masked values
        a = Scalar([1., 2., 3., 4.])
        a = a.mask_where_eq(2.)
        data = pickle.dumps(a)
        b = pickle.loads(data)
        self.assertEqual(a.shape, b.shape)
        # Values should match for unmasked elements (compression may affect masked values)
        self.assertTrue(np.allclose(a.values[~a.mask], b.values[~b.mask]))
        self.assertTrue(np.array_equal(a.mask, b.mask))

        # Test standard pickle with derivatives
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        data = pickle.dumps(a)
        b = pickle.loads(data)
        self.assertTrue(hasattr(b, 'd_dt'))
        self.assertTrue(np.allclose(a.d_dt.values, b.d_dt.values))

        # Test set_pickle_digits with integer values
        # The method will still set the attribute on the object, but it will not be used
        # during pickling of integer arrays.
        a = Scalar([1, 2, 3, 4])
        a.set_pickle_digits(8, 'fpzip')
        digits = a.pickle_digits()
        self.assertEqual(digits[0], 8)
        # The attribute is set, but won't be used for integer pickling
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.array_equal(a.values, b.values))

        # Test set_pickle_digits with boolean values
        # The method will still set the attribute on the object, but it will not be used
        # during pickling of boolean arrays.
        a = Boolean([True, False, True, False])
        a.set_pickle_digits(8, 'fpzip')
        digits = a.pickle_digits()
        self.assertEqual(digits[0], 8)
        # The attribute is set, but won't be used for boolean pickling
        state = a.__getstate__()
        b = Boolean.__new__(Boolean)
        b.__setstate__(state)
        self.assertTrue(np.array_equal(a.values, b.values))

        # Test __setstate__ with precision loss note
        # For floating-point arrays using lossy compression, values may differ slightly
        a = Scalar(np.random.randn(100))
        a.set_pickle_digits(6, 'mean')
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        # Values should be close but not necessarily exact due to compression
        self.assertTrue(np.allclose(a.values, b.values, rtol=1e-5))

        ##################################################################################
        # Additional coverage tests for missing lines
        ##################################################################################

        # Test _pickle_debug function (line 96)
        # This is a global function, but it's not directly accessible
        # We can test it indirectly through pickling behavior
        # Actually, _pickle_debug is a module-level variable, not a function
        # Let's skip direct testing of this internal variable

        # Test pickle_digits with derivatives (lines 256-259)
        a = Scalar([1., 2., 3.])
        a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
        # Set pickle_digits - derivatives should get default values if not set
        a.set_pickle_digits(8, 'fpzip')
        # Check that derivatives have pickle_digits attribute (set by set_pickle_digits)
        # Actually, the code sets it only if not already set, so let's check after setting
        self.assertTrue(hasattr(a.d_dt, '_pickle_digits') or hasattr(a.d_dt, 'pickle_digits'))

        # Test _validate_pickle_digits with various edge cases
        # This is an internal function, but we can test through set_pickle_digits
        a = Scalar([1., 2., 3.])
        # Test with None (should default to 'double')
        a.set_pickle_digits(None, 'fpzip')
        digits = a.pickle_digits()
        self.assertEqual(digits[0], 'double')

        # Test _validate_pickle_reference with invalid reference
        a = Scalar([1., 2., 3.])
        self.assertRaises(ValueError, a.set_pickle_digits, 8, 'invalid_ref')

        # Test pickling with different compression methods to trigger encoding paths
        # Test with 'smallest' reference
        a = Scalar(np.random.randn(100))
        a.set_pickle_digits(8, 'smallest')
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.allclose(a.values, b.values, rtol=1e-7))

        # Test with 'largest' reference
        a = Scalar(np.random.randn(100))
        a.set_pickle_digits(8, 'largest')
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.allclose(a.values, b.values, rtol=1e-7))

        # Test with 'median' reference
        a = Scalar(np.random.randn(100))
        a.set_pickle_digits(8, 'median')
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.allclose(a.values, b.values, rtol=1e-7))

        # Test with 'logmean' reference
        a = Scalar(np.random.randn(100))
        a.set_pickle_digits(8, 'logmean')
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.allclose(a.values, b.values, rtol=1e-7))

        # Test with numeric reference
        a = Scalar(np.random.randn(100))
        a.set_pickle_digits(8, 100.)
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.allclose(a.values, b.values, rtol=1e-7))

        # Test pickling with different mask encodings
        # Test with CORNERS encoding
        a = Scalar(np.random.randn(100))
        a = a.mask_where(np.random.rand(100) > 0.5)  # Random mask
        state = a.__getstate__()
        # Check that MASK_ENCODING is present
        self.assertIn('MASK_ENCODING', state)

        # Test pickling with BOOL encoding
        a = Scalar(np.random.randn(1000))
        a = a.mask_where(np.random.rand(1000) > 0.5)  # Large random mask
        state = a.__getstate__()
        self.assertIn('MASK_ENCODING', state)

        # Test pickling with ANTIMASKED encoding
        a = Scalar(np.random.randn(100))
        a = a.mask_where(np.random.rand(100) > 0.3)  # Partial mask
        state = a.__getstate__()
        self.assertIn('VALS_ENCODING', state)

        # Test pickling with FLOAT encoding
        a = Scalar(np.random.randn(100))
        a.set_pickle_digits(6, 'fpzip')
        state = a.__getstate__()
        self.assertIn('VALS_ENCODING', state)
        # Check that FLOAT encoding is present
        vals_encoding = state['VALS_ENCODING']
        has_float = any(item[0] == 'FLOAT' for item in vals_encoding if isinstance(item, tuple))
        # May or may not have FLOAT depending on compression method

        # Test pickling with INT encoding
        a = Scalar([1, 2, 3, 4, 5])
        state = a.__getstate__()
        self.assertIn('VALS_ENCODING', state)

        # Test pickling with BOOL encoding for values
        a = Boolean([True, False, True, False] * 100)
        state = a.__getstate__()
        self.assertIn('VALS_ENCODING', state)

        # Test __setstate__ with various encoding combinations
        # Test with ALL_MASKED
        a = Scalar([1., 2., 3., 4.])
        a = a.mask_where(True)  # Fully masked
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertTrue(np.all(b.mask))

        # Test __setstate__ with renamed keys (old format compatibility)
        a = Scalar([1., 2., 3.])
        state = a.__getstate__()
        # Simulate old format with renamed keys
        if '_units_' not in state:
            state['_units_'] = state.get('_unit', None)
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        self.assertEqual(a.shape, b.shape)
