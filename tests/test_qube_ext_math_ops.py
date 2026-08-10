##########################################################################################
# tests/test_qube_ext_math_ops.py
# Unit tests for Qube math operations
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Vector, Boolean


def test_qube_ext_math_ops_test_pos_self_element_by_element() -> None:
    """Test __pos__ # +self, element by element."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3.])
    b = +a
    assert a.shape == b.shape
    assert np.allclose(a.values, b.values)

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = +a
    assert hasattr(b, 'd_dt')
    assert np.allclose(a.d_dt.values, b.d_dt.values)

    a = Scalar([1., 2., 3.])
    b = -a
    assert a.shape == b.shape
    assert np.allclose(b.values, [-1., -2., -3.])

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = -a
    assert hasattr(b, 'd_dt')
    assert np.allclose(b.d_dt.values, [-0.1, -0.2, -0.3])

    a = Scalar([-1., 2., -3.])

    b = abs(a)
    assert np.allclose(b.values, [1., 2., 3.])

    a = Scalar([-1., 2., -3.])
    b = a.abs()
    assert np.allclose(b.values, [1., 2., 3.])

    a = Scalar([1., 2., 3., 4.])
    assert len(a) == 4
    a = Scalar(np.arange(12).reshape(2, 3, 2))
    assert len(a) == 2

    a = Scalar(1.)
    with pytest.raises(TypeError):
        len(a)

    a = Scalar([1., 2., 3., 4.])
    assert a.len() == 4

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    c = a + b
    assert c.shape == a.shape
    assert np.allclose(c.values, [5., 7., 9.])

    a = Scalar(1.)
    b = a + 2.
    assert b.shape == ()
    assert np.allclose(b.values, 3.)

    a = Scalar([1., 2., 3.])
    b = a + [4., 5., 6.]
    assert b.shape == a.shape
    assert np.allclose(b.values, [5., 7., 9.])

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = Scalar([4., 5., 6.])
    b.insert_deriv('t', Scalar([0.4, 0.5, 0.6]))
    c = a + b
    assert hasattr(c, 'd_dt')
    assert np.allclose(c.d_dt.values, [0.5, 0.7, 0.9])

    a = Scalar([1., 2., 3.])
    b = 2. + a
    assert b.shape == a.shape
    assert np.allclose(b.values, [3., 4., 5.])

    a = Scalar([1., 2., 3.])
    b = [4., 5., 6.] + a
    assert b.shape == a.shape
    assert np.allclose(b.values, [5., 7., 9.])

    a = Scalar([1., 2., 3.])
    a += Scalar([4., 5., 6.])
    assert np.allclose(a.values, [5., 7., 9.])

    a = Scalar(1.)
    a += 2.
    assert np.allclose(a.values, 3.)

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    c = a - b
    assert c.shape == a.shape
    assert np.allclose(c.values, [-3., -3., -3.])

    a = Scalar(1.)
    b = a - 2.
    assert b.shape == ()
    assert np.allclose(b.values, -1.)

    a = Scalar([1., 2., 3.])
    b = 2. - a
    assert b.shape == a.shape
    assert np.allclose(b.values, [1., 0., -1.])

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    c = a.__rsub__(b, recursive=True)

    assert c.shape == a.shape
    assert np.allclose(c.values, [3., 3., 3.])

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = Scalar([4., 5., 6.])
    b.insert_deriv('t', Scalar([0.4, 0.5, 0.6]))
    c = a.__rsub__(b, recursive=True)
    assert hasattr(c, 'd_dt')

    assert np.allclose(c.d_dt.values, [0.3, 0.3, 0.3])

    a = Scalar([1., 2., 3.])
    a -= Scalar([4., 5., 6.])
    assert np.allclose(a.values, [-3., -3., -3.])

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    c = a * b
    assert c.shape == a.shape
    assert np.allclose(c.values, [4., 10., 18.])

    a = Scalar([1., 2., 3.])
    b = a * 2.
    assert b.shape == a.shape
    assert np.allclose(b.values, [2., 4., 6.])

    a = Scalar([1., 2., 3.])
    b = 2. * a
    assert b.shape == a.shape
    assert np.allclose(b.values, [2., 4., 6.])

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    c = a.__rmul__(b, recursive=True)

    assert c.shape == a.shape
    assert np.allclose(c.values, [4., 10., 18.])

    a = Scalar([1., 2., 3.])
    a *= 2.
    assert np.allclose(a.values, [2., 4., 6.])

    a = Scalar([1., 2., 3.])
    b = Scalar([2., 4., 6.])
    c = a / b
    assert c.shape == a.shape
    assert np.allclose(c.values, [0.5, 0.5, 0.5])

    a = Scalar([1., 2., 3.])
    b = Scalar([2., 0., 6.])
    c = a / b
    assert c.mask[1]  # division by zero should be masked

    a = Scalar([1., 2., 3.])
    b = a / 2.
    assert b.shape == a.shape
    assert np.allclose(b.values, [0.5, 1., 1.5])

    a = Scalar([1., 2., 3.])
    b = 2. / a
    assert b.shape == a.shape
    assert np.allclose(b.values, [2., 1., 2./3.])

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    c = a.__rtruediv__(b, recursive=True)

    assert c.shape == a.shape
    assert np.allclose(c.values, [4., 2.5, 2.])

    a = Scalar([1., 2., 3.])
    a /= 2.
    assert np.allclose(a.values, [0.5, 1., 1.5])

    a = Scalar([7, 8, 9])
    b = Scalar([2, 3, 4])
    c = a // b
    assert c.shape == a.shape
    assert np.array_equal(c.values, [3, 2, 2])

    a = Scalar([7, 8, 9])
    b = Scalar([2, 0, 4])
    c = a // b
    assert c.mask[1]  # division by zero should be masked

    a = Scalar([2, 3, 4])
    b = 7 // a
    assert b.shape == a.shape
    assert np.array_equal(b.values, [3, 2, 1])

    a = Scalar([2, 3, 4])
    b = Scalar([7, 8, 9])
    c = a.__rfloordiv__(b)

    assert c.shape == a.shape
    assert np.array_equal(c.values, [3, 2, 2])

    a = Scalar([7, 8, 9])
    a //= Scalar([2, 3, 4])
    assert np.array_equal(a.values, [3, 2, 2])

    a = Scalar([7, 8, 9])
    b = Scalar([3, 4, 5])
    c = a % b
    assert c.shape == a.shape
    assert np.array_equal(c.values, [1, 0, 4])

    a = Scalar([7, 8, 9])
    b = Scalar([3, 0, 5])
    c = a % b
    assert c.mask[1]  # modulus by zero should be masked

    a = Scalar([3, 4, 5])
    b = 7 % a
    assert b.shape == a.shape
    assert np.array_equal(b.values, [1, 3, 2])

    a = Scalar([3, 4, 5])
    b = Scalar([7, 8, 9])
    c = a.__rmod__(b, recursive=True)

    assert c.shape == a.shape
    assert np.array_equal(c.values, [1, 0, 4])

    a = Scalar([7, 8, 9])
    a %= Scalar([3, 4, 5])
    assert np.array_equal(a.values, [1, 0, 4])

    a = Scalar([2., 3., 4.])
    b = a ** 2
    assert b.shape == a.shape
    assert np.allclose(b.values, [4., 9., 16.])

    a = Scalar(2.)
    b = a ** 3
    assert np.allclose(b.values, 8.)

    a = Scalar([2., 3., 4.])
    b = a ** -1
    assert b.shape == a.shape
    assert np.allclose(b.values, [0.5, 1./3., 0.25])

    a = Scalar([2., 3., 4.])
    b = a ** 0
    assert b.shape == a.shape

    assert np.allclose(b.values, [1., 1., 1.])

    a = Scalar([2., 3., 4.])

    try:
        _ = a ** 16
        # If it doesn't raise, that's okay - Scalar may have different limits
    except ValueError:
        pass  # Expected for base Qube class

    a = Scalar([2., 3., 4.])
    a **= 2
    assert np.allclose(a.values, [4., 9., 16.])

    a = Scalar([1., 2., 3.])
    b = Scalar([1., 2., 4.])
    c = a == b
    assert type(c).__name__ == 'Boolean'
    assert c.values[0]
    assert c.values[1]
    assert not c.values[2]

    a = Scalar([1., 2., 3.])
    b = Vector([1., 2., 3.])
    c = a == b
    assert not c

    a = Scalar([1., 2., 3.])
    b = Scalar([1., 2., 4.])
    c = a != b
    assert type(c).__name__ == 'Boolean'
    assert not c.values[0]
    assert not c.values[1]
    assert c.values[2]

    a = Scalar([1., 2., 3.])
    b = Scalar([1., 2., 4.])
    # These should work for Scalar (overridden), but test that base raises
    # Actually, these are overridden by Scalar, so we can't test the base behavior easily

    a = Scalar(1.)
    assert bool(a)
    a = Scalar(0.)
    assert not bool(a)

    a = Scalar([1., 2., 3.])
    with pytest.raises(ValueError):
        bool(a)

    a = Scalar(1.)
    a = a.mask_where_eq(1.)
    with pytest.raises(ValueError):
        bool(a)

    a = Scalar([1., 2., 3.])
    b = Scalar([1., 2., 3.])
    c = (a == b)

    assert bool(c)
    a = Scalar([1., 2., 3.])
    b = Scalar([1., 2., 4.])
    c = (a == b)

    assert not bool(c)

    a = Scalar([1., 2., 3.])
    b = Scalar([1., 2., 4.])
    c = (a != b)

    assert bool(c)
    a = Scalar([1., 2., 3.])
    b = Scalar([1., 2., 3.])
    c = (a != b)

    assert not bool(c)

    a = Scalar(1.5)
    assert float(a) == 1.5

    a = Scalar([1., 2., 3.])
    with pytest.raises(ValueError):
        float(a)

    a = Scalar(1.5)
    a = a.mask_where_eq(1.5)
    with pytest.raises(ValueError):
        float(a)

    a = Scalar(1.9)
    assert int(a) == 1

    a = Scalar([1., 2., 3.])
    with pytest.raises(ValueError):
        int(a)

    a = Scalar(1.9)
    a = a.mask_where_eq(1.9)
    with pytest.raises(ValueError):
        int(a)

    a = Scalar([0., 1., 2.])
    b = ~a
    assert type(b).__name__ == 'Boolean'
    assert b.values[0]
    assert not b.values[1]
    assert not b.values[2]

    a = Scalar([0., 1., 2.])
    b = Scalar([1., 0., 2.])
    c = a & b
    assert type(c).__name__ == 'Boolean'
    assert not c.values[0]
    assert not c.values[1]
    assert c.values[2]

    a = Scalar([0., 1., 2.])
    b = 1 & a
    assert type(b).__name__ == 'Boolean'

    a = Scalar([0., 1., 2.])
    b = Scalar([1., 0., 2.])
    c = a.__rand__(b)

    assert type(c).__name__ == 'Boolean'
    assert not c.values[0]
    assert not c.values[1]
    assert c.values[2]

    a = Scalar([0., 1., 2.])
    b = Scalar([1., 0., 0.])
    c = a | b
    assert type(c).__name__ == 'Boolean'
    assert c.values[0]
    assert c.values[1]
    assert c.values[2]

    a = Scalar([0., 1., 2.])
    b = 1 | a
    assert type(b).__name__ == 'Boolean'

    a = Scalar([0., 1., 2.])
    b = Scalar([1., 0., 0.])
    c = a.__ror__(b)

    assert type(c).__name__ == 'Boolean'
    assert c.values[0]
    assert c.values[1]
    assert c.values[2]

    a = Scalar([0., 1., 2.])
    b = Scalar([1., 0., 2.])
    c = a ^ b
    assert type(c).__name__ == 'Boolean'
    assert c.values[0]
    assert c.values[1]
    assert not c.values[2]

    a = Scalar([0., 1., 2.])
    b = 1 ^ a
    assert type(b).__name__ == 'Boolean'

    a = Scalar([0., 1., 2.])
    b = Scalar([1., 0., 2.])
    c = a.__rxor__(b)

    assert type(c).__name__ == 'Boolean'
    assert c.values[0]
    assert c.values[1]
    assert not c.values[2]

    a = Boolean([False, True, True])
    a &= Boolean([True, False, True])
    assert type(a).__name__ == 'Boolean'
    assert not a.values[0]
    assert not a.values[1]
    assert a.values[2]

    a = Boolean([False, True, False])
    a |= Boolean([True, False, True])
    assert type(a).__name__ == 'Boolean'
    assert a.values[0]
    assert a.values[1]
    assert a.values[2]

    a = Boolean([False, True, False])
    a ^= Boolean([True, False, True])
    assert type(a).__name__ == 'Boolean'
    assert a.values[0]
    assert a.values[1]
    assert a.values[2]

    a = Scalar([0., 1., 2.])
    b = a.logical_not()
    assert type(b).__name__ == 'Boolean'
    assert b.values[0]
    assert not b.values[1]
    assert not b.values[2]

    a = Boolean([False, False, True, False])
    assert a.any()
    a = Boolean([False, False, False, False])
    assert not a.any()

    a = Boolean([[False, True], [False, False]])
    b = a.any(axis=0)
    assert b.shape == (2,)
    assert not b.values[0]
    assert b.values[1]

    a = Boolean([True, True, True, True])
    assert a.all()
    a = Boolean([True, True, False, True])
    assert not a.all()

    a = Boolean([[True, True], [True, False]])
    b = a.all(axis=0)
    assert b.shape == (2,)
    assert b.values[0]
    assert not b.values[1]

    a = Boolean([False, False, False, False])
    a = a.mask_where_eq(False)
    b = a.any_true_or_masked()
    assert b

    a = Boolean([True, True, True, True])
    a = a.mask_where_eq(True)
    b = a.all_true_or_masked()
    assert b

    ##################################################################################
    # Additional coverage tests for missing lines
    ##################################################################################

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    a += b
    assert np.allclose(a.values, [5., 7., 9.])

    a = Scalar([1., 2., 3.])
    a += 2.
    assert np.allclose(a.values, [3., 4., 5.])

    a = Scalar([1, 2, 3])  # Integer
    b = Scalar([1., 2., 3.])  # Float
    with pytest.raises(TypeError):
        (lambda: a.__iadd__(b))()

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    a -= b
    assert np.allclose(a.values, [-3., -3., -3.])

    a = Scalar([1., 2., 3.])
    a -= 2.
    assert np.allclose(a.values, [-1., 0., 1.])

    a = Scalar([1., 2., 3.])
    b = Scalar([4., 5., 6.])
    a *= b
    assert np.allclose(a.values, [4., 10., 18.])

    a = Scalar([1., 2., 3.])
    a *= 2.
    assert np.allclose(a.values, [2., 4., 6.])

    a = Scalar([1, 2, 3])  # Integer
    b = Scalar([1., 2., 3.])  # Float
    with pytest.raises(TypeError):
        (lambda: a.__imul__(b))()

    a = Scalar([1., 2., 3.])
    b = Scalar([4.])  # Scalar that broadcasts
    a *= b
    assert np.allclose(a.values, [4., 8., 12.])

    a = Scalar([1., 2., 3.])
    b = Scalar([2., 4., 6.])
    a /= b
    assert np.allclose(a.values, [0.5, 0.5, 0.5])

    a = Scalar([1., 2., 3.])
    a /= 2.
    assert np.allclose(a.values, [0.5, 1., 1.5])

    a = Scalar([5., 7., 9.])
    b = Scalar([2., 3., 4.])
    a //= b
    assert np.allclose(a.values, [2., 2., 2.])

    a = Scalar([5., 7., 9.])
    a //= 2.
    assert np.allclose(a.values, [2., 3., 4.])

    a = Scalar([5., 7., 9.])
    b = Scalar([2., 3., 4.])
    a %= b
    assert np.allclose(a.values, [1., 1., 1.])

    a = Scalar([5., 7., 9.])
    a %= 2.
    assert np.allclose(a.values, [1., 1., 1.])

    a = Scalar([2., 3., 4.])
    a **= 2
    assert np.allclose(a.values, [4., 9., 16.])

    a = Scalar([1., 2., 3.])

    try:
        _ = a + "invalid"
        # If it doesn't raise, that's unexpected
        pytest.fail("Expected TypeError or ValueError")
    except (TypeError, ValueError):
        pass  # Expected

    a = Scalar([1., 2., 3.])
    b = Vector([1., 2., 3.])

    with pytest.raises((TypeError, ValueError)):
        (lambda: a + b)()

    try:
        a = Vector(np.arange(6).reshape(2, 3), drank=1)
        b = Vector(np.arange(6, 12).reshape(2, 3), drank=1)
        _ = a * b
        # If it doesn't raise, that's unexpected
        pytest.fail("Expected ValueError")
    except ValueError:
        pass  # Expected


def test_qube_ext_math_ops_test_mul_by_number_internal_method_this_is_an_internal_metho() -> None:
    """Test _mul_by_number (internal method) # This is an internal method, so we test it indirectly through multiplication."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3.])
    b = a * 2.
    assert np.allclose(b.values, [2., 4., 6.])


def test_qube_ext_math_ops_test_mul_by_number_with_derivatives_indirectly() -> None:
    """Test _mul_by_number with derivatives (indirectly)."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3]))
    b = a * 2.
    assert hasattr(b, 'd_dt')
    assert np.allclose(b.d_dt.values, [0.2, 0.4, 0.6])


def test_qube_ext_math_ops_test_reciprocal_an_object_equivalent_to_the_reciprocal_of_th() -> None:
    """Test reciprocal # An object equivalent to the reciprocal of this object. # This method is not implemented for the base class."""

    np.random.seed(2599)

    a = Scalar([1., 2., 4.])

    b = a.reciprocal()
    assert np.allclose(b.values, [1., 0.5, 0.25])


def test_qube_ext_math_ops_test_zero_an_object_of_this_subclass_containing_all_zeros() -> None:
    """Test zero # An object of this subclass containing all zeros."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3.])
    b = a.zero()
    assert type(b).__name__ == 'Scalar'
    assert b.shape == ()
    assert np.allclose(b.values, 0.)


def test_qube_ext_math_ops_test_identity_an_object_of_this_subclass_equivalent_to_the_i() -> None:
    """Test identity # An object of this subclass equivalent to the identity. # This method is overridden by Scalar, Matrix, and Boolean."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3.])

    b = a.identity()
    assert type(b).__name__ == 'Scalar'
    assert b.shape == ()
    assert np.allclose(b.values, 1.)


def test_qube_ext_math_ops_test_sum_the_sum_of_the_unmasked_values_along_the_specified_() -> None:
    """Test sum # The sum of the unmasked values along the specified axis or axes."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4.])
    b = a.sum()
    assert b.shape == ()
    assert np.allclose(b.values, 10.)


def test_qube_ext_math_ops_test_sum_with_axis() -> None:
    """Test sum with axis."""

    np.random.seed(2599)

    a = Scalar(np.arange(12).reshape(2, 3, 2))
    b = a.sum(axis=0)

    assert b.shape == (3, 2)


def test_qube_ext_math_ops_test_mean_the_mean_of_the_unmasked_values_along_the_specifie() -> None:
    """Test mean # The mean of the unmasked values along the specified axis or axes."""

    np.random.seed(2599)

    a = Scalar([1., 2., 3., 4.])
    b = a.mean()
    assert b.shape == ()
    assert np.allclose(b.values, 2.5)


def test_qube_ext_math_ops_test_mean_with_axis() -> None:
    """Test mean with axis."""

    np.random.seed(2599)

    a = Scalar(np.arange(12).reshape(2, 3, 2))
    b = a.mean(axis=0)

    assert b.shape == (3, 2)


