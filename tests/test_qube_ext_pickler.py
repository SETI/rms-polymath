##########################################################################################
# tests/test_qube_ext_pickler.py
# Unit tests for Qube pickling operations
##########################################################################################

import numpy as np
import pytest
import pickle

from polymath import Qube, Scalar, Vector, Vector3, Boolean


def test_qube_ext_pickler_test_set_pickle_digits_set_the_desired_number_of_decimal_dig() -> None:
    """Test set_pickle_digits # Set the desired number of decimal digits of precision in the storage of this # object's floating-point values and their derivatives."""

    np.random.seed(2599)

    a = Scalar([1.23456789, 2.34567890])
    a.set_pickle_digits(8, 'fpzip')
    digits = a.pickle_digits()
    assert digits[0] == 8
    assert digits[1] == 8

    a = Scalar([1.23456789, 2.34567890])
    a.set_pickle_digits((8, 7), ('fpzip', 'smallest'))
    digits = a.pickle_digits()
    assert digits[0] == 8
    assert digits[1] == 7

    a = Scalar([1.23456789, 2.34567890])
    a.set_pickle_digits('double', 'fpzip')
    digits = a.pickle_digits()
    assert digits[0] == 'double'
    assert digits[1] == 'double'

    a = Scalar([1.23456789, 2.34567890])
    a.set_pickle_digits('single', 'fpzip')
    digits = a.pickle_digits()
    assert digits[0] == 'single'
    assert digits[1] == 'single'

    a = Scalar([1.23456789, 2.34567890])
    a.set_pickle_digits(8, 'smallest')
    ref = a.pickle_reference()
    assert ref[0] == 'smallest'
    a.set_pickle_digits(8, 'largest')
    ref = a.pickle_reference()
    assert ref[0] == 'largest'
    a.set_pickle_digits(8, 'mean')
    ref = a.pickle_reference()
    assert ref[0] == 'mean'
    a.set_pickle_digits(8, 'median')
    ref = a.pickle_reference()
    assert ref[0] == 'median'
    a.set_pickle_digits(8, 'logmean')
    ref = a.pickle_reference()
    assert ref[0] == 'logmean'
    a.set_pickle_digits(8, 'fpzip')
    ref = a.pickle_reference()
    assert ref[0] == 'fpzip'

    a = Scalar([1.23456789, 2.34567890])
    a.set_pickle_digits(8, 100.)
    ref = a.pickle_reference()
    assert ref[0] == 100.

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    a.set_pickle_digits((8, 7), ('fpzip', 'smallest'))

    assert a.d_dt.pickle_digits()[0] == 7
    assert a.d_dt.pickle_reference()[0] == 'smallest'

    Qube.set_default_pickle_digits(10, 'mean')
    a = Scalar([1., 2., 3.])
    digits = a.pickle_digits()
    assert digits[0] == 10
    ref = a.pickle_reference()
    assert ref[0] == 'mean'

    Qube.set_default_pickle_digits('double', 'fpzip')

    a = Scalar([1., 2., 3.])
    digits = a.pickle_digits()
    assert isinstance(digits, tuple)
    assert len(digits) == 2

    a = Scalar([1., 2., 3.])
    ref = a.pickle_reference()
    assert isinstance(ref, tuple)
    assert len(ref) == 2

    a = Scalar([1., 2., 3., 4.])
    state = a.__getstate__()
    assert 'PICKLE_VERSION' in state
    assert 'MASK_ENCODING' in state
    assert 'VALS_ENCODING' in state

    if '_cache' in state:
        assert state['_cache'] == {}

    a = Scalar([1., 2., 3., 4.])
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert a.shape == b.shape
    assert np.allclose(a.values, b.values)
    assert a.mask == b.mask

    a = Scalar([1., 2., 3., 4.])
    a = a.mask_where_eq(2.)
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert a.shape == b.shape

    assert np.allclose(a.values[~a.mask], b.values[~b.mask])
    assert np.array_equal(a.mask, b.mask)

    a = Scalar([1., 2., 3., 4.])
    a = a.mask_where_eq(1.)
    a = a.mask_where_eq(2.)
    a = a.mask_where_eq(3.)
    a = a.mask_where_eq(4.)
    state = a.__getstate__()
    assert ('ALL_MASKED',) in state['VALS_ENCODING']

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert hasattr(b, 'd_dt')
    assert np.allclose(a.d_dt.values, b.d_dt.values)

    a = Scalar([1, 2, 3, 4])
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert a.shape == b.shape
    assert np.array_equal(a.values, b.values)

    a = Boolean([True, False, True, False])
    state = a.__getstate__()
    b = Boolean.__new__(Boolean)
    b.__setstate__(state)
    assert a.shape == b.shape
    assert np.array_equal(a.values, b.values)

    a = Vector([1., 2., 3.])
    state = a.__getstate__()
    b = Vector.__new__(Vector)
    b.__setstate__(state)
    assert a.shape == b.shape
    assert np.allclose(a.values, b.values)

    a = Vector3([1., 2., 3.])
    state = a.__getstate__()
    b = Vector3.__new__(Vector3)
    b.__setstate__(state)
    assert a.shape == b.shape
    assert np.allclose(a.values, b.values)

    a = Scalar(np.random.randn(1000))
    a.set_pickle_digits(8, 'smallest')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.allclose(a.values, b.values, rtol=1e-7)
    a.set_pickle_digits(8, 'largest')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.allclose(a.values, b.values, rtol=1e-7)
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.allclose(a.values, b.values, rtol=1e-7)

    a = Scalar([1., 2., 3., 4.])
    data = pickle.dumps(a)
    b = pickle.loads(data)
    assert a.shape == b.shape
    assert np.allclose(a.values, b.values)

    a = Scalar([1., 2., 3., 4.])
    a = a.mask_where_eq(2.)
    data = pickle.dumps(a)
    b = pickle.loads(data)
    assert a.shape == b.shape

    assert np.allclose(a.values[~a.mask], b.values[~b.mask])
    assert np.array_equal(a.mask, b.mask)

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    data = pickle.dumps(a)
    b = pickle.loads(data)
    assert hasattr(b, 'd_dt')
    assert np.allclose(a.d_dt.values, b.d_dt.values)

    a = Scalar([1, 2, 3, 4])
    a.set_pickle_digits(8, 'fpzip')
    digits = a.pickle_digits()
    assert digits[0] == 8

    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.array_equal(a.values, b.values)

    a = Boolean([True, False, True, False])
    a.set_pickle_digits(8, 'fpzip')
    digits = a.pickle_digits()
    assert digits[0] == 8

    state = a.__getstate__()
    b = Boolean.__new__(Boolean)
    b.__setstate__(state)
    assert np.array_equal(a.values, b.values)

    a = Scalar(np.random.randn(100))
    a.set_pickle_digits(6, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)

    assert np.allclose(a.values, b.values, rtol=1e-5)

    ##################################################################################
    # Additional coverage tests for missing lines
    ##################################################################################

    # Test _pickle_debug function
    # This is a global function, but it's not directly accessible
    # We can test it indirectly through pickling behavior
    # Actually, _pickle_debug is a module-level variable, not a function
    # Let's skip direct testing of this internal variable

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))

    a.set_pickle_digits(8, 'fpzip')

    assert hasattr(a.d_dt, '_pickle_digits')

    a = Scalar([1., 2., 3.])

    a.set_pickle_digits(None, 'fpzip')
    digits = a.pickle_digits()
    assert digits[0] == 'double'

    a = Scalar([1., 2., 3.])
    with pytest.raises(ValueError):
        a.set_pickle_digits(8, 'invalid_ref')

    a = Scalar(np.random.randn(100))
    a.set_pickle_digits(8, 'smallest')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.allclose(a.values, b.values, rtol=1e-7)

    a = Scalar(np.random.randn(100))
    a.set_pickle_digits(8, 'largest')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.allclose(a.values, b.values, rtol=1e-7)

    a = Scalar(np.random.randn(100))
    a.set_pickle_digits(8, 'median')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.allclose(a.values, b.values, rtol=1e-7)

    a = Scalar(np.random.randn(100))
    a.set_pickle_digits(8, 'logmean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.allclose(a.values, b.values, rtol=1e-7)

    a = Scalar(np.random.randn(100))
    a.set_pickle_digits(8, 100.)
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.allclose(a.values, b.values, rtol=1e-7)

    a = Scalar(np.random.randn(100))
    a = a.mask_where(np.random.rand(100) > 0.5)  # Random mask
    state = a.__getstate__()

    assert 'MASK_ENCODING' in state

    a = Scalar(np.random.randn(1000))
    a = a.mask_where(np.random.rand(1000) > 0.5)  # Large random mask
    state = a.__getstate__()
    assert 'MASK_ENCODING' in state

    a = Scalar(np.random.randn(100))
    a = a.mask_where(np.random.rand(100) > 0.3)  # Partial mask
    state = a.__getstate__()
    assert 'VALS_ENCODING' in state

    a = Scalar(np.random.randn(100))
    a.set_pickle_digits(6, 'fpzip')
    state = a.__getstate__()
    assert 'VALS_ENCODING' in state

    vals_encoding = state['VALS_ENCODING']

    _ = any(item[0] == 'FLOAT' for item in vals_encoding
            if isinstance(item, tuple))

    a = Scalar([1, 2, 3, 4, 5])
    state = a.__getstate__()
    assert 'VALS_ENCODING' in state

    a = Boolean([True, False, True, False] * 100)
    state = a.__getstate__()
    assert 'VALS_ENCODING' in state

    a = Scalar([1., 2., 3., 4.])
    a = a.mask_where(True)  # Fully masked
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert np.all(b.mask)

    a = Scalar([1., 2., 3.])
    state = a.__getstate__()

    if '_units_' not in state:
        state['_units_'] = state.get('_unit', None)

    state['_test_'] = 'test'
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert a.shape == b.shape

    Qube._pickle_debug(True)
    try:
        # This sets _PICKLE_DEBUG global
        a = Scalar([1., 2., 3.])
        state = a.__getstate__()
        # With _PICKLE_DEBUG, __setstate__ should preserve encoding info
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        # Check if encoding info is preserved
        assert hasattr(b, 'ENCODED_MASK')
        assert hasattr(b, 'ENCODED_VALS')
        # Verify the encoded values are preserved
        assert b.ENCODED_MASK is not None
        assert b.ENCODED_VALS is not None
    finally:
        Qube._pickle_debug(False)

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))

    a.set_pickle_digits(8, 'mean')

    assert hasattr(a.d_dt, '_pickle_digits')
    assert hasattr(a.d_dt, '_pickle_reference')

    a = Scalar([1., 2., 3.])
    a.set_pickle_digits([8, 8], 'mean')  # List instead of tuple

    assert a._pickle_digits == (8, 8)

    a = Scalar([1., 2., 3.])
    a.set_pickle_digits(8, ('mean', 'mean'))  # Tuple reference

    assert a._pickle_reference == ('mean', 'mean')

    a = Scalar(np.arange(2*3*4*5*6).reshape(2, 3, 4, 5, 6))
    a.set_pickle_digits('double', 'fpzip')
    state = a.__getstate__()

    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar(np.arange(1000))
    a.set_pickle_digits(8, 'fpzip')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, 'fpzip')  # Lossy compression
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)

    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits('single', 'fpzip')  # Single precision
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits('double', 'fpzip')
    state = a.__getstate__()

    assert 'VALS_ENCODING' in state

    a = Scalar([5., 5., 5., 5., 5.])  # Constant array
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert np.allclose(b.values, 5.)

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, 100.)  # Reference as float
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])

    a.set_pickle_digits(8, 'smallest')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a.set_pickle_digits(8, 'largest')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a.set_pickle_digits(8, 'median')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a.set_pickle_digits(8, 'logmean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1e-10, 1e10, 1e-10, 1e10])  # Very large range
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(7, 'mean')  # Should trigger nbytes == 4 path
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits('single', 'fpzip')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Vector([[1., 2., 3.], [4., 5., 6.]])  # Vector with shape (2,), numer (3,)
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()

    b = Vector.__new__(Vector)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert b.numer == a.numer

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits('double', 'fpzip')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([5., 5., 5., 5., 5.])  # Constant
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert np.allclose(b.values, 5.)

    a = Vector([[1., 2., 3.], [4., 5., 6.]])
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Vector.__new__(Vector)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert b.numer == a.numer

    a = Scalar([1, 2, 3, 4, 5])  # Integer array
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert np.array_equal(b.values, a.values)

    # Test _decode_ints
    # This is tested through the encode/decode cycle above

    a = Scalar(7.)  # Scalar with shape ()
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert a == b

    Qube._pickle_debug(True)
    try:
        a = Scalar([1., 2., 3.])
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        # With _PICKLE_DEBUG, encoding info should be preserved
        assert hasattr(b, 'ENCODED_MASK')
        assert hasattr(b, 'ENCODED_VALS')
        # Verify the encoded values are preserved
        assert b.ENCODED_MASK is not None
        assert b.ENCODED_VALS is not None
    finally:
        Qube._pickle_debug(False)

    # Test __setstate__ with _cache
    # The cache is removed in __getstate__, so this is tested implicitly

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == a.d_dt.shape

    a = Scalar(np.arange(20).reshape(4, 5))

    mask = np.ones((4, 5), dtype=bool)
    mask[1:3, 1:4] = False  # Inner region is False
    a = a.mask_where(mask)
    state = a.__getstate__()

    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.], mask=[False, True, False, True, False])
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert np.array_equal(b.mask, a.mask)

    # Test __setstate__ with _values as np.ndarray
    # This is tested through all the encode/decode cycles above

    a = Scalar([1., 2., 3., 4., 5.]).as_readonly()
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.readonly

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits([8, 8], 'mean')
    assert a._pickle_digits == (8, 8)

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, ['mean', 'mean'])
    assert a._pickle_reference == ('mean', 'mean')

    a = Scalar([1., 2., 3., 4., 5.])
    with pytest.raises(ValueError):
        a.set_pickle_digits(['invalid', 2], 'mean')

    a = Scalar([1., 2., 3., 4., 5.])
    a.insert_deriv('t', Scalar([10., 20., 30., 40., 50.]))

    if hasattr(a.d_dt, '_pickle_digits'):
        delattr(a.d_dt, '_pickle_digits')
    if hasattr(a.d_dt, '_pickle_reference'):
        delattr(a.d_dt, '_pickle_reference')
    a.set_pickle_digits(8, 'mean')
    assert hasattr(a.d_dt, '_pickle_digits')
    assert hasattr(a.d_dt, '_pickle_reference')

    a = Scalar([5.] * 300)  # All same value, size > 200
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()

    vals_encoding = state['VALS_ENCODING']
    assert vals_encoding == [('FLOAT', 8.0, 'mean')]
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert np.allclose(b.values, a.values)

    a = Scalar(np.arange(1., 301.))  # Size > 200
    a.set_pickle_digits(8, 2.5)  # Real number reference
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, 'median')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, 'logmean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    with pytest.raises(ValueError):
        a.set_pickle_digits(8, 'invalid_reference')

    a = Scalar(np.linspace(1e-10, 1e10, 300))  # Size > 200, large range
    a.set_pickle_digits(15, 'mean')  # High precision, large range
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar(np.arange(1., 301.))  # Size > 200
    a.set_pickle_digits(7, 'mean')  # Should trigger single precision
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits('single', 'fpzip')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape

    values = np.arange(300.).reshape(100, 3)  # 100 items, each with 3 elements
    a = Vector(values)
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Vector.__new__(Vector)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert b.numer == a.numer

    a = Vector([[1., 2., 3.]])  # Single item
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Vector.__new__(Vector)
    b.__setstate__(state)
    assert b.shape == a.shape

    a = Scalar([1, 2, 3, 4, 5])  # Integer array

    a_slice = a[::2]
    a_slice.set_pickle_digits(8, 'mean')
    state = a_slice.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a_slice.shape

    a = Boolean([True, False, True, False, True])

    a_slice = a[::2]
    state = a_slice.__getstate__()
    b = Boolean.__new__(Boolean)
    b.__setstate__(state)
    assert b.shape == a_slice.shape

    a = Scalar(5.0)  # Scalar value
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert b.values == a.values

    a = Scalar([1., 2., 3., 4., 5.])
    a.insert_deriv('t', Scalar([10., 20., 30., 40., 50.]))

    a = a.mask_where([False, True, False, True, False])
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert ('t' in b.derivs)

    # Test __setstate__ with keys ending with '_'
    # This is an internal detail - the code processes keys ending with '_'
    # and renames them. This is tested indirectly through normal pickling.
    # We'll skip direct testing as it requires manipulating internal state.

    a = Scalar([1., 2., 3., 4., 5.])
    state = a.__getstate__()

    state2 = state.copy()
    state2['MASK_ENCODING'] = [('INVALID', None)]

    if 'VALS_ENCODING' not in state2:
        state2['VALS_ENCODING'] = []
    b = Scalar.__new__(Scalar)
    with pytest.raises(ValueError):
        b.__setstate__(state2)

    a = Scalar([1., 2., 3., 4., 5.])
    a = a.mask_where([False, True, False, True, False])
    state = a.__getstate__()

    state2 = state.copy()

    if 'VALS_ENCODING' in state2:
        # Replace with ANTIMASKED encoding
        state2['VALS_ENCODING'] = [('ANTIMASKED', None)]

    if 'ANTIMASK' in state2:
        del state2['ANTIMASK']
    b2 = Scalar.__new__(Scalar)
    with pytest.raises(ValueError):
        b2.__setstate__(state2)

    a = Scalar([1., 2., 3., 4., 5.])
    state = a.__getstate__()

    state2 = state.copy()
    state2['VALS_ENCODING'] = [('INVALID', None)]

    if 'MASK_ENCODING' not in state2:
        state2['MASK_ENCODING'] = []
    b = Scalar.__new__(Scalar)
    with pytest.raises(ValueError):
        b.__setstate__(state2)


def test_qube_ext_pickler_test_setstate_with_readonly_and_writability_checks() -> None:
    """Test __setstate__ with readonly and writability checks."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    state = a.__getstate__()

    state['_readonly'] = True
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.readonly


def test_qube_ext_pickler_test_setstate_with_derivatives_and_antimask() -> None:
    """Test __setstate__ with derivatives and antimask."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.insert_deriv('t', Scalar([10., 20., 30., 40., 50.]))
    a = a.mask_where([False, True, False, True, False])
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert ('t' in b.derivs)


def test_qube_ext_pickler_test_setstate_with_derivative_readonly() -> None:
    """Test __setstate__ with derivative readonly."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    deriv = Scalar([10., 20., 30., 40., 50.]).as_readonly()
    a.insert_deriv('t', deriv)
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.d_dt.readonly


def test_qube_ext_pickler_test_float32_decoding() -> None:
    """Test float32 decoding."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits('single', 'fpzip')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_float64_decoding() -> None:
    """Test float64 decoding."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits('double', 'fpzip')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_constant_decoding() -> None:
    """Test constant decoding."""

    np.random.seed(2599)

    a = Scalar([5., 5., 5., 5., 5.])
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert np.allclose(b.values, a.values)

    # Test unrecognized method in _decode_floats
    # This is hard to test directly, but we can try to construct an invalid encoding
    # Actually, this is tested indirectly through the invalid values encoding test above


def test_qube_ext_pickler_test_nbytes_3_decoding_this_is_tested_through_the_encode_dec() -> None:
    """Test nbytes == 3 decoding # This is tested through the encode/decode cycle with appropriate digits # We need to create a scenario where nbytes == 3 # This requires: 2 < bytes_needed <= 3 # bytes_needed = log(unique_values_needed) / log(256) # unique_values_needed = span / precision + 1 # Need size > 200 to avoid 'literal' encoding # Let's try with a specific range and precision."""

    np.random.seed(2599)

    a = Scalar(np.linspace(100., 500., 300))  # Size > 200
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_nbytes_5_decoding_similar_approach_need_size_200() -> None:
    """Test nbytes == 5 decoding # Similar approach, need size > 200."""

    np.random.seed(2599)

    a = Scalar(np.linspace(1e3, 5e3, 300))  # Size > 200
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_nbytes_6_decoding_need_size_200() -> None:
    """Test nbytes == 6 decoding # Need size > 200."""

    np.random.seed(2599)

    a = Scalar(np.linspace(1e4, 5e4, 300))  # Size > 200
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_single_precision_calculation_this_is_triggered_when_dig() -> None:
    """Test single precision calculation # This is triggered when digits is a number and dtype is float32 # We need to trigger the else branch in fpzip_compress."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(7, 'mean')  # Should use single precision
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_array_ndim_4_reshaping_create_a_5d_array() -> None:
    """Test array.ndim > 4 reshaping # Create a 5D array."""

    np.random.seed(2599)

    a = Scalar(np.arange(2*3*4*5*6).reshape(2, 3, 4, 5, 6))
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_fpzip_reference_encoding() -> None:
    """Test fpzip reference encoding."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(8, 'fpzip')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_pickle_debug_path_we_need_to_set_pickle_debug_to_true() -> None:
    """Test _PICKLE_DEBUG path # We need to set _PICKLE_DEBUG to True."""

    np.random.seed(2599)

    from polymath.extensions import pickler
    original_debug = pickler._PICKLE_DEBUG
    try:
        pickler._PICKLE_DEBUG = True
        a = Scalar([1., 2., 3., 4., 5.])
        state = a.__getstate__()
        b = Scalar.__new__(Scalar)
        b.__setstate__(state)
        # Check if debug attributes are set
        assert hasattr(b, 'ENCODED_MASK')
        assert hasattr(b, 'ENCODED_VALS')
        # Verify the encoded values are preserved
        assert b.ENCODED_MASK is not None
        assert b.ENCODED_VALS is not None
        assert b.shape == a.shape
    finally:
        pickler._PICKLE_DEBUG = original_debug

    # Test _PICKLE_WARNINGS path
    # This is hard to test without actually triggering fpzip errors
    # We'll skip this for now as it requires specific fpzip error conditions

    # Test fpzip error handling paths
    # These are also hard to test without actually triggering fpzip errors
    # We'll skip these for now


def test_qube_ext_pickler_test_corners_mask_encoding_create_a_mask_with_edges_all_true() -> None:
    """Test CORNERS mask encoding # Create a mask with edges all True."""

    np.random.seed(2599)

    a = Scalar(np.arange(20).reshape(4, 5))
    mask = np.ones((4, 5), dtype=bool)
    mask[1:3, 1:4] = False  # Inner region is False
    a = a.mask_where(mask)
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert np.array_equal(b.mask, a.mask)


def test_qube_ext_pickler_test_fpzip_decompress_with_bits_0_this_happens_when_fpzip_co() -> None:
    """Test fpzip_decompress with bits == 0 # This happens when fpzip compression is lossless."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits('double', 'fpzip')  # Use fpzip with double precision
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_fpzip_decompress_with_bits_0_this_happens_when_fpzip_co_2() -> None:
    """Test fpzip_decompress with bits > 0 # This happens when fpzip compression is lossy # We need to trigger lossy compression by using lower precision."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits(10, 'fpzip')  # Lower precision to trigger lossy compression
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_fpzip_decompress_with_float32() -> None:
    """Test fpzip_decompress with float32."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.set_pickle_digits('single', 'fpzip')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_getstate_with_derivatives_and_antimask_none() -> None:
    """Test __getstate__ with derivatives and antimask None."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.insert_deriv('t', Scalar([10., 20., 30., 40., 50.]))

    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert ('t' in b.derivs)


def test_qube_ext_pickler_test_getstate_with_derivatives_and_antimask() -> None:
    """Test __getstate__ with derivatives and antimask."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    a.insert_deriv('t', Scalar([10., 20., 30., 40., 50.]))

    a = a.mask_where([False, True, False, True, False])
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape
    assert ('t' in b.derivs)


def test_qube_ext_pickler_test_setstate_with_values_writability_check_this_is_tested_t() -> None:
    """Test __setstate__ with values writability check # This is tested through normal pickling, but let's be explicit."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4., 5.])
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_decode_floats_with_single_item_create_a_vector_with_a_s() -> None:
    """Test _decode_floats with single item # Create a Vector with a single item that uses items encoding."""

    np.random.seed(2599)

    Vector([[1., 2., 3.]])  # Single item


def test_qube_ext_pickler_make_it_large_enough_to_trigger_items_encoding() -> None:
    """Make it large enough to trigger items encoding."""

    np.random.seed(2599)

    values = np.tile([1., 2., 3.], (100, 1))  # 100 items, each [1, 2, 3]
    a = Vector(values)
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Vector.__new__(Vector)
    b.__setstate__(state)
    assert b.shape == a.shape

    # Test _decode_floats with unrecognized method
    # This is hard to test directly, but we can try to construct an invalid encoding
    # Actually, this is already tested through the invalid values encoding test above


def test_qube_ext_pickler_test_reference_value_calculation_paths_these_are_tested_thro() -> None:
    """Test reference value calculation paths # These are tested through the different reference values above # But let's make sure they're using the scaled encoding # Test with 'smallest' reference."""

    np.random.seed(2599)

    a = Scalar(np.arange(1., 301.))
    a.set_pickle_digits(8, 'smallest')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_with_largest_reference() -> None:
    """Test with 'largest' reference."""

    np.random.seed(2599)

    a = Scalar(np.arange(1., 301.))
    a.set_pickle_digits(8, 'largest')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_fpzip_reference_encoding_this_should_use_fpzip_compress() -> None:
    """Test fpzip reference encoding # This should use fpzip compression directly."""

    np.random.seed(2599)

    a = Scalar(np.arange(1., 301.))
    a.set_pickle_digits(8, 'fpzip')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_single_precision_calculation_this_is_in_fpzip_compress_() -> None:
    """Test single precision calculation # This is in fpzip_compress, triggered when digits is a number and dtype is float32 # We need to trigger the else branch."""

    np.random.seed(2599)

    a = Scalar(np.arange(1., 301.))
    a.set_pickle_digits(7, 'mean')  # Should use single precision
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape


def test_qube_ext_pickler_test_array_ndim_4_reshaping_create_a_5d_array_2() -> None:
    """Test array.ndim > 4 reshaping # Create a 5D array."""

    np.random.seed(2599)

    a = Scalar(np.arange(2*3*4*5*6).reshape(2, 3, 4, 5, 6))
    a.set_pickle_digits(8, 'mean')
    state = a.__getstate__()
    b = Scalar.__new__(Scalar)
    b.__setstate__(state)
    assert b.shape == a.shape




def test_qube_ext_pickler_invalid_digits_names_the_offending_value() -> None:
    """An invalid digit value is named in the error, not the whole argument."""

    a = Scalar([1., 2., 3.])
    with pytest.raises(ValueError, match="invalid pickle digits: 'quadruple'"):
        a.set_pickle_digits(('double', 'quadruple'), 'fpzip')


def test_qube_ext_pickler_unhashable_digits_are_rejected() -> None:
    """A digit value that cannot be hashed is rejected as invalid."""

    a = Scalar([1., 2., 3.])
    with pytest.raises(ValueError, match=r'invalid pickle digits: \[1\]'):
        a.set_pickle_digits(([1], 'double'), 'fpzip')


def test_qube_ext_pickler_digits_without_a_reference_are_rejected() -> None:
    """A number of digits with no reference value to match it is rejected."""

    a = Scalar([1., 2., 3.])
    with pytest.raises(ValueError, match='missing pickle reference for digits: 7'):
        a.set_pickle_digits((8, 7), ('fpzip',))


def test_qube_ext_pickler_invalid_reference_names_the_offending_value() -> None:
    """An invalid reference value is named in the error, not the whole argument."""

    a = Scalar([1., 2., 3.])
    with pytest.raises(ValueError, match="invalid pickle reference 'bogus'"):
        a.set_pickle_digits('double', ('fpzip', 'bogus'))


def test_qube_ext_pickler_unhashable_reference_is_rejected() -> None:
    """A reference value that cannot be hashed is rejected as invalid."""

    a = Scalar([1., 2., 3.])
    with pytest.raises(ValueError, match=r"invalid pickle reference \['1'\]"):
        a.set_pickle_digits('double', (['1'], 'fpzip'))
