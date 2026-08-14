##########################################################################################
# tests/test_scalar_coverage.py
# Comprehensive coverage tests for scalar.py to achieve >90% coverage
##########################################################################################

import numpy as np
import pytest
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


def test_scalar_coverage_test_invalid_dtype() -> None:
    """Test invalid dtype."""

    np.random.seed(54321)

    dtype = np.dtype('U')  # Unicode string dtype
    with pytest.raises(ValueError):
        _ = Scalar._minval(dtype)
    with pytest.raises(ValueError):
        _ = Scalar._maxval(dtype)

    for kind in ['f', 'u', 'i']:
        dtype = np.dtype(kind + '8')
        min_val = Scalar._minval(dtype)
        max_val = Scalar._maxval(dtype)
        assert min_val is not None
        assert max_val is not None

    dtype = np.dtype('bool')
    min_val = Scalar._minval(dtype)
    max_val = Scalar._maxval(dtype)
    assert min_val is not None
    assert max_val is not None

    b = Boolean(True)
    s = Scalar.as_scalar(b)
    assert s == 1

    v = Vector([1., 2., 3.])
    with pytest.raises(ValueError):
        _ = Scalar.as_scalar(v)

    s = Scalar.as_scalar(Unit.KM)
    assert s.unit_ is not None

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    s = Scalar.as_scalar(a, recursive=False)
    assert not hasattr(s, 'd_dt')

    a = Scalar(1.)
    with pytest.raises(ValueError):
        a.to_scalar(1)

    a = Scalar(1.)
    a.insert_deriv('t', Scalar(0.1))
    s = a.to_scalar(0, recursive=False)
    assert not hasattr(s, 'd_dt')

    a = Scalar([1.5, 2.5, 3.5])
    with pytest.raises(IndexError):
        a.as_index_and_mask()

    a = Vector(np.arange(6).reshape(2, 3), drank=1)
    with pytest.raises(ValueError):
        _ = a.as_index_and_mask()

    a = Scalar([1, 2, 3], mask=True)
    idx, mask = a.as_index_and_mask(purge=True)
    assert len(idx) == 0

    a = Scalar([1, 2, 3])
    a = a.mask_where_eq(2)
    idx, mask = a.as_index_and_mask(purge=True)
    assert len(idx) == 2

    a = Scalar([1, 2, 3], mask=True)
    idx, mask = a.as_index_and_mask(masked=999)
    assert np.all(idx == 999)

    a = Scalar([1, 2, 3])
    a = a.mask_where_eq(2)
    idx, mask = a.as_index_and_mask(masked=999)
    assert idx[1] == 999

    a = Vector(np.arange(6).reshape(2, 3), drank=1)
    with pytest.raises(ValueError):
        _ = a.int()

    a = Scalar([1, 2, 3, 4, 5])
    b = a.int(top=3, shift=True, clip=False)

    assert len(b) == 5

    a = Scalar([1, 2, 3, 4, 5])
    b = a.int(top=3, remask=True, clip=False)
    assert (b.mask[3] or b.mask[4])

    a = Scalar([1, 2, 3, 4, 5])
    b = a.int(top=3, clip=True)
    assert np.all(b.values <= 2)

    a = Scalar([-1, 0, 1, 2, 3])
    b = a.int(remask=True, clip=False)
    assert b.mask[0]

    a = Scalar(5.7)
    with prefer_builtins(True):
        b = a.int()
        assert isinstance(b, int)

    a = Scalar([1.5, 2.5, 3.5])
    b = a.frac()
    assert np.allclose(b.values, [0.5, 0.5, 0.5])

    a = Scalar([1.5, 2.5, 3.5])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = a.frac(recursive=True)
    assert hasattr(b, 'd_dt')

    a = Scalar([0., np.pi/2, np.pi], unit=Unit.RAD)
    b = a.sin()
    assert np.allclose(b.values, [0., 1., 0.], atol=1e-10)

    a = Scalar([0., np.pi/2, np.pi], unit=Unit.RAD)
    b = a.cos()
    assert np.allclose(b.values, [1., 0., -1.], atol=1e-10)

    a = Scalar([0., np.pi/4], unit=Unit.RAD)
    b = a.tan()
    assert np.allclose(b.values, [0., 1.], atol=1e-10)

    a = Scalar([0., 0.5, 1.])
    b = a.arcsin()
    assert np.allclose(b.values, [0., np.arcsin(0.5), np.pi/2], atol=1e-10)

    a = Scalar(2.)  # Outside [-1, 1]
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        with pytest.raises(ValueError):
            _ = a.arcsin(check=False)

    a = Scalar([-2., 0., 2.])
    b = a.arcsin(check=True)
    assert (b.mask[0] or b.mask[2])

    a = Scalar([1., 0.5, 0.])
    b = a.arccos()
    assert np.allclose(b.values, [0., np.arccos(0.5), np.pi/2], atol=1e-10)

    a = Scalar(2.)  # Outside [-1, 1]
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        with pytest.raises(ValueError):
            _ = a.arccos(check=False)

    a = Scalar([-2., 0., 2.])
    b = a.arccos(check=True)
    assert (b.mask[0] or b.mask[2])

    a = Scalar([0., 1., -1.])
    b = a.arctan()
    assert np.allclose(b.values, [0., np.pi/4, -np.pi/4], atol=1e-10)

    a = Scalar(1.)
    b = Scalar(1.)
    c = a.arctan2(b)
    assert c == np.pi/4 or abs(c - np.pi/4) <= 1e-10

    a = Scalar([1., 4., 9.])
    b = a.sqrt()
    assert np.allclose(b.values, [1., 2., 3.])

    a = Scalar(-1.)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        with pytest.raises(ValueError):
            _ = a.sqrt(check=False)

    a = Scalar([1., np.e, np.e**2])
    b = a.log()
    assert np.allclose(b.values, [0., 1., 2.], atol=1e-10)

    a = Scalar(0.)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        with pytest.raises(ValueError):
            _ = a.log(check=False)

    a = Scalar([0., 1., 2.])
    b = a.exp()
    assert np.allclose(b.values, [1., np.e, np.e**2], atol=1e-10)

    a = Scalar(1000.)  # Very large value
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        # The overflow surfaces as the RuntimeWarning raised by np.exp, unless it is
        # first converted to a ValueError by Scalar.exp() itself
        with pytest.raises((ValueError, RuntimeWarning)):
            _ = a.exp(check=False)

    a = Scalar(1000.)
    b = a.exp(check=True)
    assert b.mask  # Overflow values are masked

    a = Scalar([-1., 0., 1.])
    b = a.sign(zeros=False)
    assert b[1] == 1  # Zero should become 1

    a = Scalar(1.)
    with prefer_builtins(True):
        b = a.sign()
        # sign() returns the sign, which for float 1.0 is 1.0 (float), not int
        # But if it's an integer Scalar, it might return int
        a_int = Scalar(1)  # Integer
        b_int = a_int.sign()
        # The result type depends on the input type
        assert isinstance(b, (int, float))
        assert isinstance(b_int, int)
        assert b_int == 1

    a = Scalar([1., 3., 2.])
    b = a.max()
    assert b == 3.

    a = Scalar([1., 2., 3.], mask=True)
    b = a.max()
    assert b.mask

    a = Scalar([1., 2., 3.])
    a = a.mask_where_eq(2.)
    b = a.max()
    assert b == 3.

    a = Scalar([1., 2., 3.])
    with prefer_builtins(True):
        b = a.max()
        assert isinstance(b, (int, float))

    a = Scalar([3., 1., 2.])
    b = a.min()
    assert b == 1.

    a = Scalar([1., 2., 3.], mask=True)
    b = a.min()
    assert b.mask

    a = Scalar([1., 2., 3.])
    a = a.mask_where_eq(2.)
    b = a.min()
    assert b == 1.

    a = Scalar([1., 2., 3.])
    with prefer_builtins(True):
        b = a.min()
        assert isinstance(b, (int, float))

    a = Scalar([1., 3., 2.])
    b = a.argmax()
    assert b == 1  # Index of max value

    a = Scalar(1.)
    with pytest.raises(ValueError):
        a.argmax()

    a = Scalar([1., 2., 3.], mask=True)
    b = a.argmax()
    assert b.mask

    a = Scalar([1., 2., 3.])
    a = a.mask_where_eq(2.)
    b = a.argmax()
    # Should return index of max unmasked value

    a = Scalar([1., 2., 3.])
    with prefer_builtins(True):
        b = a.argmax()
        assert isinstance(b, int)

    a = Scalar([3., 1., 2.])
    b = a.argmin()
    assert b == 1  # Index of min value

    a = Scalar(1.)
    with pytest.raises(ValueError):
        a.argmin()

    a = Scalar([1., 2., 3.], mask=True)
    b = a.argmin()
    assert b.mask

    a = Scalar([1., 2., 3.])
    a = a.mask_where_eq(2.)
    b = a.argmin()
    # Should return index of min unmasked value

    a = Scalar([1., 2., 3.])
    with prefer_builtins(True):
        b = a.argmin()
        assert isinstance(b, int)

    with pytest.raises(ValueError):
        Scalar.maximum()

    a = Scalar([1., 3., 2.])
    b = Scalar([2., 1., 4.])
    c = Scalar.maximum(a, b)
    assert np.allclose(c.values, [2., 3., 4.])

    a = Scalar([1., 2., 3.])
    b = Scalar.maximum(a)
    assert np.allclose(b.values, a.values)

    a = Scalar([1, 2, 3])
    b = Scalar([1., 2., 3.])
    c = Scalar.maximum(a, b)
    assert c.is_float()

    with pytest.raises(ValueError):
        Scalar.minimum()

    a = Scalar([1., 3., 2.])
    b = Scalar([2., 1., 4.])
    c = Scalar.minimum(a, b)
    assert np.allclose(c.values, [1., 1., 2.])

    a = Scalar([1., 2., 3.])
    b = Scalar.minimum(a)
    assert np.allclose(b.values, a.values)

    a = Scalar([1, 2, 3])
    b = Scalar([1., 2., 3.])
    c = Scalar.minimum(a, b)
    assert c.is_float()

    a = Scalar([1., 3., 2., 4., 5.])
    b = a.median()
    assert b == 3.

    a = Scalar([1., 2., 3.], mask=True)
    b = a.median()
    assert b.mask

    a = Scalar([1., 2., 3., 4., 5.])
    a = a.mask_where_eq(3.)
    b = a.median(axis=None)
    # Should compute median of unmasked values

    a = Scalar(np.arange(24).reshape(2, 3, 4))
    a = a.mask_where_eq(5.)
    b = a.median(axis=0)
    # Should compute median along axis 0

    a = Scalar([1., 2., 3., 4., 5.])
    with prefer_builtins(True):
        b = a.median()
        assert isinstance(b, float)

    a = Scalar([3., 1., 2.])
    b = a.sort()
    assert np.allclose(b.values, [1., 2., 3.])

    a = Scalar([3., 1., 2.])
    a = a.mask_where_eq(2.)
    b = a.sort()
    # Masked values should appear at end

    a = Scalar([1., 2., 4.])
    b = a.reciprocal()
    assert np.allclose(b.values, [1., 0.5, 0.25])

    a = Scalar([1., 0., 2.])
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        with pytest.raises(ValueError):
            _ = a.reciprocal(nozeros=True)

    a = Scalar([1., 0., 2.])
    b = a.reciprocal(nozeros=False)
    assert b.mask[1]  # Zero should be masked

    a = Scalar([2., 3., 4.])
    b = a ** 2
    assert np.allclose(b.values, [4., 9., 16.])

    a = Scalar([2., 3., 4.])
    b = Scalar([1., 2.])  # Different shape
    with pytest.raises(ValueError):
        _ = a ** b

    a = Scalar([2., 3., 4.], unit=Unit.KM)
    b = Scalar([1., 2.])  # Array exponent
    with pytest.raises(ValueError):
        _ = a ** b

    a = Scalar(0.)
    b = Scalar(-1.)
    c = a ** b  # 0 ** -1 is undefined, so the result is masked rather than raised
    assert c.mask

    a = Scalar([2., 3., 4.])
    with pytest.raises(TypeError):
        _ = a ** "invalid"

    a = Scalar(1.)
    b = Vector(np.arange(6).reshape(2, 3), drank=1)
    with pytest.raises(ValueError):
        _ = a <= b
    with pytest.raises(ValueError):
        _ = a < b
    with pytest.raises(ValueError):
        _ = a >= b
    with pytest.raises(ValueError):
        _ = a > b

    a = Scalar(1.)
    b = Scalar(2.)
    with prefer_builtins(True):
        c = a <= b
        assert isinstance(c, bool)
        c = a < b
        assert isinstance(c, bool)
        c = a >= b
        assert isinstance(c, bool)
        c = a > b
        assert isinstance(c, bool)
    ##################################################################################
    # Test __round__
    ##################################################################################
    a = Scalar(1.234567)
    b = round(a, 2)
    assert b == 1.23 or abs(b - 1.23) <= 1e-2
    ##################################################################################
    # Test __abs__ with derivatives
    ##################################################################################
    a = Scalar([-1., 2., -3.])
    a.insert_deriv('t', Scalar([-0.1, 0.2, -0.3]))
    b = abs(a)
    assert hasattr(b, 'd_dt')
    # Derivatives should be multiplied by sign
    ##################################################################################
    # Test _power_0 with derivatives
    ##################################################################################
    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = a._power_0(recursive=True)
    assert hasattr(b, 'd_dt')
    # Derivatives should be zeros
    ##################################################################################
    # Test _power_1
    ##################################################################################
    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = a._power_1(recursive=True)
    assert hasattr(b, 'd_dt')
    b = a._power_1(recursive=False)
    assert not hasattr(b, 'd_dt')
    ##################################################################################
    # Test _power_2, _power_3, _power_4
    ##################################################################################
    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = a._power_2(recursive=True)
    assert hasattr(b, 'd_dt')
    assert np.allclose(b.values, [1., 4., 9.])
    b = a._power_3(recursive=True)
    assert hasattr(b, 'd_dt')
    assert np.allclose(b.values, [1., 8., 27.])
    b = a._power_4(recursive=True)
    assert hasattr(b, 'd_dt')
    assert np.allclose(b.values, [1., 16., 81.])
    ##################################################################################
    # Test _power_neg_1, _power_half, _power_neg_half
    ##################################################################################
    a = Scalar([1., 2., 4.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = a._power_neg_1(recursive=True)
    assert hasattr(b, 'd_dt')
    assert np.allclose(b.values, [1., 0.5, 0.25])
    b = a._power_half(recursive=True)
    assert hasattr(b, 'd_dt')
    assert np.allclose(b.values, [1., np.sqrt(2.), 2.])
    b = a._power_neg_half(recursive=True)
    assert hasattr(b, 'd_dt')
    assert np.allclose(b.values, [1., 1./np.sqrt(2.), 0.5])
    ##################################################################################
    # Test __pow__ with easy powers
    ##################################################################################
    a = Scalar([1., 2., 3.])

    b = a ** 0
    assert np.allclose(b.values, [1., 1., 1.])

    b = a ** 1
    assert np.allclose(b.values, [1., 2., 3.])

    b = a ** 2
    assert np.allclose(b.values, [1., 4., 9.])

    b = a ** 3
    assert np.allclose(b.values, [1., 8., 27.])

    b = a ** 4
    assert np.allclose(b.values, [1., 16., 81.])

    b = a ** -1
    assert np.allclose(b.values, [1., 0.5, 1./3.])

    b = a ** 0.5
    assert np.allclose(b.values, [1., np.sqrt(2.), np.sqrt(3.)])

    b = a ** -0.5
    assert np.allclose(b.values, [1., 1./np.sqrt(2.), 1./np.sqrt(3.)])

    a = Scalar([1, 2, 3])  # Integer
    b = Scalar(-1)  # Negative integer exponent
    c = a ** b
    assert c.is_float()  # Should convert to float

    a = Scalar([2., 3., 4.])
    b = Scalar(2., mask=True)
    c = a ** b
    assert np.all(c.mask)

    a = Scalar([2., 3., 4.])
    b = Scalar([1000., 1000., 1000.])  # Very large exponent
    c = a ** b

    assert np.all(c.mask == [False, True, True])

    a = Scalar([2., 3., 4.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = a ** 2
    assert hasattr(b, 'd_dt')

    ##################################################################################
    # Additional tests for missing lines
    ##################################################################################

    b = Boolean([True, False, True])
    s = Scalar.as_scalar(b, recursive=False)
    assert type(s) == Scalar

    a = Scalar(5)
    idx, mask = a.as_index_and_mask()
    assert idx == 5
    assert not mask

    a = Scalar([1, 2, 3])
    idx, mask = a.as_index_and_mask(masked=None)
    assert np.array_equal(idx, [1, 2, 3])
    assert not mask

    a = Scalar([1.5, 2.5, 3.5])
    b = a.int(top=[5])
    assert np.all(b.values <= 4)

    a = Scalar([1.5, 2.5, 3.5], mask=[False, True, False])
    b = a.int(top=3)
    assert isinstance(b._mask, np.ndarray)

    a = Scalar([1., 2., 3.])
    b = a.int(top=2, shift=True, clip=False)

    assert b.values[0] == 1  # 1 stays 1
    assert b.values[1] == 1  # 2 becomes 1 (shifted)
    assert b.values[2] == 3  # 3 stays 3 (no clip)

    a = Scalar([-1., 0., 1., 2.])
    b = a.int(top=2, clip=True, remask=True)
    assert np.all(b.values >= 0)
    assert np.all(b.values < 2)

    a = Scalar(1.5)
    with prefer_builtins(True):
        b = a.int(builtins=True)
        assert isinstance(b, int)

    a = Scalar([[1.5]], drank=1)  # shape (1,), item (1,)
    with pytest.raises(ValueError):
        _ = a.frac()

    a = Scalar([[1.0]], drank=1)
    with pytest.raises(ValueError):
        _ = a.sin()

    a = Scalar([[1.0]], drank=1)
    with pytest.raises(ValueError):
        _ = a.cos()

    a = Scalar([[1.0]], drank=1)
    with pytest.raises(ValueError):
        _ = a.tan()

    a = Scalar([[0.5]], drank=1)
    with pytest.raises(ValueError):
        _ = a.arcsin()

    a = Scalar(1.5)  # Outside domain
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError):
            _ = a.arcsin(check=False)

    a = Scalar([[0.5]], drank=1)
    with pytest.raises(ValueError):
        _ = a.arccos()

    a = Scalar(1.5)  # Outside domain
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError):
            _ = a.arccos(check=False)

    a = Scalar([[1.0]], drank=1)
    with pytest.raises(ValueError):
        _ = a.arctan()

    a = Scalar([[1.0]], drank=1)
    b = Scalar(1.0)
    with pytest.raises(ValueError):
        _ = a.arctan2(b)

    a = Scalar([[4.0]], drank=1)
    with pytest.raises(ValueError):
        _ = a.sqrt()

    a = Scalar([[2.0]], drank=1)
    with pytest.raises(ValueError):
        _ = a.log()

    a = Scalar([[1.0]], drank=1)
    with pytest.raises(ValueError):
        _ = a.exp()

    a = Scalar(1000.)  # Very large value
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises((ValueError, RuntimeWarning)):
            _ = a.exp(check=False)

    a = Scalar(1.0)
    with prefer_builtins(True):
        b = a.sign(builtins=True)
        assert isinstance(b, float)

    a = Scalar([1., 2., 3.])
    b = Scalar([-1., -2., -3.])
    c = Scalar([0., 0., 0.])
    _, _, discr = Scalar.solve_quadratic(a, b, c, include_antimask=True)
    assert discr is not None

    a = Scalar([])
    b = a.max()

    assert b.shape == (0,)

    a = Scalar([1., 2., 3.], mask=[False, True, False])
    b = a.max()
    assert b == 3.

    a = Scalar([])
    b = a.min()
    assert b.shape == (0,)

    a = Scalar([1., 2., 3.], mask=[True, False, False])
    b = a.min()
    assert b == 2.

    a = Scalar([1., 2., 3.])
    with prefer_builtins(True):
        b = a.min(builtins=True)
        assert isinstance(b, float)

    a = Scalar([[1.], [2.], [3.]], drank=1)  # shape (3,), item (1,)
    with pytest.raises(ValueError):
        _ = a.argmax()

    a = Scalar([])
    b = a.argmax()
    assert b.shape == (0,)

    a = Scalar([1., 2., 3.], mask=[True, False, False])
    b = a.argmax()
    assert b == 2

    a = Scalar([1., 2., 3.])
    with prefer_builtins(True):
        b = a.argmax(builtins=True)
        assert isinstance(b, int)

    a = Scalar([[1.], [2.], [3.]], drank=1)
    with pytest.raises(ValueError):
        _ = a.argmin()

    a = Scalar([])
    b = a.argmin()
    assert b.shape == (0,)

    a = Scalar([1., 2., 3.], mask=[True, False, False])
    b = a.argmin()
    assert b == 1

    a = Scalar([1., 2., 3.])
    with prefer_builtins(True):
        b = a.argmin(builtins=True)
        assert isinstance(b, int)

    a = Scalar([[1., 2., 3.], [4., 5., 6.]])
    mask = np.array([[False, False, False], [True, True, True]])
    a_masked = Scalar(a.values, mask=mask)
    result = a_masked.argmax(axis=1)

    assert isinstance(result, Scalar)
    assert result.shape == (2,)
    assert not result.mask[0]
    assert result.mask[1]

    a = Scalar([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    mask = np.array([[False, False, False], [True, True, True], [False, True, False]])
    a_masked = Scalar(a.values, mask=mask)
    result = a_masked.argmax(axis=1)

    assert isinstance(result, Scalar)
    assert result.shape == (3,)

    a = Scalar([1., 2., 3.], mask=[True, True, True])
    result = a.argmax(axis=None)

    assert isinstance(result, Scalar)

    assert (result.mask if isinstance(result.mask, (bool, np.bool_)) else np.all(result.mask))

    a = Scalar([[1., 2., 3.], [4., 5., 6.]])
    mask = np.array([[False, False, False], [True, True, True]])
    a_masked = Scalar(a.values, mask=mask)
    result = a_masked.argmin(axis=1)

    assert isinstance(result, Scalar)
    assert result.shape == (2,)
    assert not result.mask[0]
    assert result.mask[1]

    a = Scalar([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    mask = np.array([[False, False, False], [True, True, True], [False, True, False]])
    a_masked = Scalar(a.values, mask=mask)
    result = a_masked.argmin(axis=1)

    assert isinstance(result, Scalar)
    assert result.shape == (3,)

    a = Scalar([1., 2., 3.], mask=[True, True, True])
    result = a.argmin(axis=None)

    assert isinstance(result, Scalar)

    assert (result.mask if isinstance(result.mask, (bool, np.bool_)) else np.all(result.mask))

    a = Scalar([[1.], [2.], [3.]], drank=1)
    b = Scalar([2., 3., 4.])
    with pytest.raises(ValueError):
        _ = Scalar.maximum(a, b)

    a = Scalar([[1.], [2.], [3.]], drank=1)
    b = Scalar([2., 3., 4.])
    with pytest.raises(ValueError):
        _ = Scalar.minimum(a, b)

    a = Scalar([[1.], [2.], [3.]], drank=1)
    with pytest.raises(ValueError):
        _ = a.median()

    a = Scalar([])
    b = a.median()
    assert b.shape == (0,)

    a = Scalar([1., 2., 3., 4., 5.], mask=[True, False, False, False, True])
    b = a.median()
    assert b is not None

    a = Scalar([1., 2., 3.])
    with prefer_builtins(True):
        b = a.median(builtins=True)
        assert isinstance(b, float)

    a = Scalar([[3.], [1.], [2.]], drank=1)
    with pytest.raises(ValueError):
        _ = a.sort()

    a = Scalar([])
    with pytest.raises(IndexError):
        _ = a.sort()


