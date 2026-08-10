##########################################################################################
# tests/test_polynomial_basic.py
# Polynomial basic construction and property tests
##########################################################################################

import numpy as np
import pytest

from polymath import Vector, Polynomial


def test_polynomial_basic_test_basic_construction_polynomial_is_a_vector_subclass_so_i() -> None:
    """Test basic construction # Polynomial is a Vector subclass, so it should accept Vector-like inputs # Coefficients are in decreasing order: [a, b, c] = a*x^2 + b*x + c."""

    np.random.seed(2599)

    p1 = Polynomial([1., 2., 3.])  # x^2 + 2x + 3
    assert p1.shape == ()
    assert p1.numer == (3,)
    assert p1.order == 2

    v = Vector([1., 2., 3.])
    p2 = Polynomial(v)
    assert p2.order == 2
    assert np.allclose(p2.values, p1.values)

    p0 = Polynomial([5.])  # constant polynomial
    assert p0.order == 0
    p1_order = Polynomial([1., 0.])  # linear: x
    assert p1_order.order == 1
    p2_order = Polynomial([1., 2., 3.])  # quadratic: x^2 + 2x + 3
    assert p2_order.order == 2

    p3 = Polynomial.as_polynomial([4., 5., 6.])
    assert type(p3) == Polynomial
    assert p3.order == 2

    v2 = Vector([7., 8.])
    p4 = Polynomial.as_polynomial(v2)
    assert type(p4) == Polynomial
    assert p4.order == 1

    p5 = Polynomial([1., 2., 3.])
    v3 = p5.as_vector()
    assert type(v3) == Vector
    assert np.allclose(v3.values, p5.values)

    p_small = Polynomial([1., 2.])  # order 1
    p_large = p_small.at_least_order(3)  # should pad to order 3
    assert p_large.order == 3
    assert p_large.numer[0] == 4  # 4 coefficients for order 3

    assert p_large.values[0] == 0.
    assert p_large.values[1] == 0.

    assert p_large.values[2] == 1.
    assert p_large.values[3] == 2.

    p_big = Polynomial([1., 2., 3., 4.])  # order 3
    p_big2 = p_big.at_least_order(2)
    assert p_big2.order == 3
    assert np.allclose(p_big2.values, p_big.values)

    p6 = Polynomial([1., 2.])  # order 1
    p7 = p6.set_order(2)
    assert p7.order == 2
    assert p7.numer[0] == 3

    p8 = Polynomial([1., 2., 3., 4.])  # order 3
    with pytest.raises(ValueError):
        p8.set_order(2)

    p_linear = Polynomial([3., 2.])  # 3x + 2 (coefficients in decreasing order)
    p_inv = p_linear.invert_line()
    assert p_inv.order == 1

    assert p_inv.values[0] == 1./3. or abs(p_inv.values[0] - 1./3.) <= 1e-10
    assert p_inv.values[1] == -2./3. or abs(p_inv.values[1] - -2./3.) <= 1e-10

    p_linear_with_deriv = Polynomial([3., 2.])
    p_linear_deriv = Polynomial([1., 0.])  # derivative of 3 + 2x is 2
    p_linear_with_deriv.insert_deriv('t', p_linear_deriv)
    p_inv_with_deriv = p_linear_with_deriv.invert_line(recursive=True)
    assert hasattr(p_inv_with_deriv, 'd_dt')

    assert type(p_inv_with_deriv.d_dt) == Polynomial

    p_nonlinear = Polynomial([1., 2., 3.])
    with pytest.raises(ValueError):
        p_nonlinear.invert_line()

    p_int_coeffs = Polynomial([1, 2, 3])
    assert p_int_coeffs.values.dtype.kind == 'f'

    p_test_order = Polynomial([1., 2., 3.])

    assert p_test_order.values[0] == 1.  # x^2 coefficient
    assert p_test_order.values[1] == 2.  # x coefficient
    assert p_test_order.values[2] == 3.  # constant

    assert p_test_order.eval(1.).values == 6. or abs(p_test_order.eval(1.).values - 6.) <= 1e-10

    assert p_test_order.eval(2.).values == 11. or abs(p_test_order.eval(2.).values - 11.) <= 1e-10

    # Additional tests for coverage

    v_with_deriv = Vector([1., 2.])
    v_deriv = Vector([0., 1.])
    v_with_deriv.insert_deriv('t', v_deriv)
    # Create a subclass to test the type check

    class PolySubclass(Polynomial):
        pass
    p_sub = PolySubclass(v_with_deriv)

    assert hasattr(p_sub, 'd_dt')

    assert type(p_sub._derivs['t']) == Polynomial


def test_polynomial_basic_test_as_polynomial_with_recursive_false() -> None:
    """Test as_polynomial with recursive=False."""

    np.random.seed(2599)

    v3 = Vector([1., 2., 3.])
    v3.insert_deriv('t', Vector([0., 1., 2.]))
    p_no_rec = Polynomial.as_polynomial(v3, recursive=False)
    assert not hasattr(p_no_rec, 'd_dt')
    p_no_rec2 = Polynomial.as_polynomial([1., 2.], recursive=False)
    assert type(p_no_rec2) == Polynomial


def test_polynomial_basic_test_as_vector_with_recursive_false() -> None:
    """Test as_vector with recursive=False."""

    np.random.seed(2599)

    p_with_deriv2 = Polynomial([1., 2.])
    p_with_deriv2.insert_deriv('t', Polynomial([0., 1.]))
    v_no_rec = p_with_deriv2.as_vector(recursive=False)

    assert type(v_no_rec) == Vector
    # The _derivs might still exist from __dict__ copy, but the code path is tested


def test_polynomial_basic_test_at_least_order_with_recursive_false_when_already_order() -> None:
    """Test at_least_order with recursive=False when already >= order."""

    np.random.seed(2599)

    p_large2 = Polynomial([1., 2., 3., 4.])
    p_large3 = p_large2.at_least_order(2, recursive=False)
    assert p_large3.order == 3


def test_polynomial_basic_test_at_least_order_with_derivatives() -> None:
    """Test at_least_order with derivatives."""

    np.random.seed(2599)

    p_with_deriv3 = Polynomial([1., 2.])
    p_with_deriv3.insert_deriv('t', Polynomial([0., 1.]))
    p_padded = p_with_deriv3.at_least_order(3, recursive=True)
    assert hasattr(p_padded, 'd_dt')
    assert p_padded.d_dt.order == 3


def test_polynomial_basic_test_as_vector_with_recursive_true() -> None:
    """Test as_vector with recursive=True."""

    np.random.seed(2599)

    p_asvec_deriv = Polynomial([1., 2.])
    p_asvec_deriv.insert_deriv('t', Polynomial([0., 1.]))
    v_with_deriv = p_asvec_deriv.as_vector(recursive=True)
    assert hasattr(v_with_deriv, 'd_dt')

    assert type(v_with_deriv.d_dt) == Vector


def test_polynomial_basic_test_eval_with_zero_order_polynomial_and_zero_order_derivati() -> None:
    """Test eval with zero-order polynomial and zero-order derivative."""

    np.random.seed(2599)

    p_const = Polynomial([5.])
    p_deriv = Polynomial([3.])
    p_const.insert_deriv('t', p_deriv)
    result = p_const.eval(10., recursive=True)
    assert result.values == 5.
    assert result.d_dt.values == 3.


def test_polynomial_basic_test_eval_with_zero_order_polynomial_and_non_zero_order_deri() -> None:
    """Test eval with zero-order polynomial and non-zero-order derivative # Manually set derivative to bypass numerator shape check."""

    np.random.seed(2599)

    p_const3 = Polynomial([9.])
    p_deriv3 = Polynomial([2., 1.])  # 2x + 1, order 1
    p_const3._derivs['t'] = p_deriv3
    result3 = p_const3.eval(8., recursive=True)
    assert result3.values == 9.
    assert result3.d_dt.values == 1.


def test_polynomial_basic_test_eval_with_zero_order_polynomial_zero_order_derivative_w() -> None:
    """Test eval with zero-order polynomial, zero-order derivative with zero-order nested derivative."""

    np.random.seed(2599)

    p_const2 = Polynomial([7.])
    p_deriv2 = Polynomial([4.])
    p_const2.insert_deriv('t', p_deriv2)
    p_const2._derivs['t']._derivs = {'s': Polynomial([0.5])}
    result2 = p_const2.eval(5., recursive=True)
    assert result2.values == 7.
    assert result2.d_dt.values == 4.


def test_polynomial_basic_test_eval_with_zero_order_polynomial_non_zero_order_derivati() -> None:
    """Test eval with zero-order polynomial, non-zero-order derivative with nested derivatives."""

    np.random.seed(2599)

    p_const4 = Polynomial([11.])
    p_deriv4 = Polynomial([1., 5.])  # x + 5, order 1
    p_nested_zero = Polynomial([6.])  # zero-order nested
    p_nested_nonzero = Polynomial([2., 3.])  # 2x + 3, order 1 nested
    p_deriv4._derivs = {'v': p_nested_zero, 'w': p_nested_nonzero}
    p_const4._derivs['t'] = p_deriv4
    result4 = p_const4.eval(12., recursive=True)
    assert result4.values == 11.
    assert result4.d_dt.values == 5.


def test_polynomial_basic_test_eval_with_zero_order_polynomial_non_zero_order_derivati_2() -> None:
    """Test eval with zero-order polynomial, non-zero-order derivative with nested derivative that has drank > 0."""

    np.random.seed(2599)

    p_const5 = Polynomial([13.])
    p_deriv5 = Polynomial([3., 7.])  # 3x + 7, order 1
    p_nested_with_drank = Polynomial(np.array([8.]).reshape(1, 1), drank=1)  # zero-order with drank > 0
    p_deriv5._derivs = {'u': p_nested_with_drank}
    p_const5._derivs['t'] = p_deriv5
    result5 = p_const5.eval(14., recursive=True)
    assert result5.values == 13.
    assert result5.d_dt.values == 7.


##########################################################################################
