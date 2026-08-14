##########################################################################################
# tests/test_polynomial_operations.py
# Polynomial special operations (deriv, eval, roots) and advanced tests
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Vector, Polynomial


def test_polynomial_operations_test_deriv_derivative_of_x_2_2x_3_is_2x_2() -> None:
    """Test deriv # Derivative of x^2 + 2x + 3 is 2x + 2."""

    np.random.seed(2599)

    p27 = Polynomial([1., 2., 3.])  # x^2 + 2x + 3
    p_deriv = p27.deriv()
    assert type(p_deriv) == Polynomial
    assert p_deriv.order == 1
    assert p_deriv.values[0] == 2. or abs(p_deriv.values[0] - 2.) <= 1e-10
    assert p_deriv.values[1] == 2. or abs(p_deriv.values[1] - 2.) <= 1e-10

    p_const = Polynomial([5.])
    p_deriv_const = p_const.deriv()
    assert p_deriv_const.order == 0
    assert p_deriv_const.values[0] == 0.

    p28 = Polynomial([1., 2.])  # x + 2
    result = p28.eval(3.)
    assert type(result) == Scalar
    assert result.values == 5. or abs(result.values - 5.) <= 1e-10

    p29 = Polynomial([1., 2., 3.])  # x^2 + 2x + 3
    result2 = p29.eval(2.)
    assert result2.values == 11. or abs(result2.values - 11.) <= 1e-10

    p30 = Polynomial([1., 2.])  # x + 2
    x_vals = Scalar([1., 2., 3.])
    result3 = p30.eval(x_vals)
    assert type(result3) == Scalar
    assert result3.shape == (3,)
    expected = np.array([3., 4., 5.])
    assert np.allclose(result3.values, expected)

    p31 = Polynomial([1., 2.])  # x + 2
    roots1 = p31.roots()
    assert type(roots1) == Scalar
    assert roots1.shape == (1,)
    assert roots1.values[0] == -2. or abs(roots1.values[0] - -2.) <= 1e-10

    p32 = Polynomial([1., -5., 6.])  # x^2 - 5x + 6
    roots2 = p32.roots()
    assert type(roots2) == Scalar
    assert roots2.shape == (2,)

    assert roots2.values[0] == 2. or abs(roots2.values[0] - 2.) <= 1e-10
    assert roots2.values[1] == 3. or abs(roots2.values[1] - 3.) <= 1e-10

    p_zero = Polynomial([5.])
    with pytest.raises(ValueError):
        p_zero.roots()


def test_polynomial_operations_test_with_n_d_arrays_complicated_cases_create_array_of_polyn() -> None:
    """Test with n-D arrays (complicated cases) # Create array of polynomials."""

    np.random.seed(2599)

    coeffs = np.array([
        [[1., 2.], [3., 4.]],
        [[5., 6.], [7., 8.]]
    ])  # Shape (2, 2, 2) -> 2x2 array of linear polynomials
    p_array = Polynomial(coeffs)
    assert p_array.shape == (2, 2)
    assert p_array.numer == (2,)
    assert p_array.order == 1

    p_array2 = p_array + 1.  # Add constant to each
    assert p_array2.shape == (2, 2)
    assert np.allclose(p_array2.values[..., 1], p_array.values[..., 1] + 1.)

    result_array = p_array.eval(2.)
    assert result_array.shape == (2, 2)

    assert result_array.values[0, 0] == 4. or abs(result_array.values[0, 0] - 4.) <= 1e-10


def test_polynomial_operations_test_roots_with_scalar_mask_true() -> None:
    """Test roots with scalar mask=True."""

    np.random.seed(2599)

    p_masked = Polynomial([1., 2.], mask=True)
    roots_masked = p_masked.roots()
    assert np.all(roots_masked.mask)


def test_polynomial_operations_test_roots_with_scalar_mask_false() -> None:
    """Test roots with scalar mask=False."""

    np.random.seed(2599)

    p_unmasked = Polynomial([1., 2.], mask=False)
    roots_unmasked = p_unmasked.roots()
    assert not np.any(roots_unmasked.mask)


def test_polynomial_operations_test_roots_with_array_mask_for_array_of_polynomials() -> None:
    """Test roots with array mask (for array of polynomials)."""

    np.random.seed(2599)

    coeffs_mask = np.array([[[1., 2.]], [[3., 4.]]])  # Shape (2, 1, 2)
    mask_array = np.array([[False], [True]])  # Match shape (2, 1)
    p_array_mask = Polynomial(coeffs_mask, mask=mask_array)
    roots_array_mask = p_array_mask.roots()
    assert isinstance(roots_array_mask, Scalar)


def test_polynomial_operations_test_roots_with_all_coefficients_zero_this_tests_the_all_zer() -> None:
    """Test roots with all coefficients zero # This tests the all_zeros code path."""

    np.random.seed(2599)

    p_all_zeros = Polynomial([0., 0., 0.])
    roots_all_zeros = p_all_zeros.roots()

    assert isinstance(roots_all_zeros, Scalar)
    assert roots_all_zeros.shape == (2,)


def test_polynomial_operations_test_roots_with_leading_coefficient_zero_requires_shifting() -> None:
    """Test roots with leading coefficient zero (requires shifting)."""

    np.random.seed(2599)

    p_leading_zero = Polynomial([0., 1., 2.])  # x + 2 = 0, root at -2
    roots_leading_zero = p_leading_zero.roots()
    assert roots_leading_zero.shape == (2,)

    unmasked_roots = roots_leading_zero[~roots_leading_zero.mask]
    assert unmasked_roots.values[0] == -2. or abs(unmasked_roots.values[0] - -2.) <= 1e-10


def test_polynomial_operations_test_roots_with_multiple_leading_zeros_scalar_case() -> None:
    """Test roots with multiple leading zeros (scalar case)."""

    np.random.seed(2599)

    p_multi_zero = Polynomial([0., 0., 1., 2.])  # x + 2 = 0 after shifting
    roots_multi_zero = p_multi_zero.roots()

    assert isinstance(roots_multi_zero, Scalar)
    assert roots_multi_zero.shape == (3,)


def test_polynomial_operations_test_roots_with_array_of_polynomials_requiring_shifts_use_sa() -> None:
    """Test roots with array of polynomials requiring shifts # Use same order for both to avoid shape mismatch."""

    np.random.seed(2599)

    coeffs_shift = np.array([
        [[0., 0., 1., 2.]],  # x + 2 = 0 after double shift
        [[0., 1., 2., 0.]]  # x + 2 = 0 after single shift (pad to same size)
    ])
    p_shift_array = Polynomial(coeffs_shift)
    roots_shift_array = p_shift_array.roots()

    assert isinstance(roots_shift_array, Scalar)
    assert roots_shift_array.shape[1:] == (2, 1)


def test_polynomial_operations_test_roots_with_recursive_derivatives_use_a_higher_order_pol() -> None:
    """Test roots with recursive derivatives # Use a higher order polynomial to ensure we hit the recursive path."""

    np.random.seed(2599)

    p_with_deriv = Polynomial([1., 0., -1., 0.])  # x^3 - x = 0, roots at -1, 0, 1
    p_with_deriv.insert_deriv('t', Polynomial([0., 0., -1., 0.]))  # derivative: -x
    roots_with_deriv = p_with_deriv.roots(recursive=True)

    assert isinstance(roots_with_deriv, Scalar)
    assert roots_with_deriv.shape[0] == 3


def test_polynomial_operations_test_roots_with_array_mask_not_scalar_to_hit_array_mask_copy() -> None:
    """Test roots with array mask (not scalar) to hit array mask copy path."""

    np.random.seed(2599)

    coeffs_array_mask = np.array([[[1., 2.]], [[3., 4.]]])
    mask_array_not_scalar = np.array([[False], [True]])
    p_array_mask_not_scalar = Polynomial(coeffs_array_mask, mask=mask_array_not_scalar)
    roots_array_mask_not_scalar = p_array_mask_not_scalar.roots()
    assert isinstance(roots_array_mask_not_scalar, Scalar)


def test_polynomial_operations_test_roots_with_all_coefficients_zero_in_array_case() -> None:
    """Test roots with all coefficients zero in array case."""

    np.random.seed(2599)

    coeffs_all_zeros_array = np.array([
        [[0., 0., 0.]],  # All zeros
        [[1., 2., 3.]]   # Normal polynomial
    ])
    p_all_zeros_array = Polynomial(coeffs_all_zeros_array)
    roots_all_zeros_array = p_all_zeros_array.roots()

    assert isinstance(roots_all_zeros_array, Scalar)


def test_polynomial_operations_test_roots_with_array_requiring_shifts_and_mask_indices_crea() -> None:
    """Test roots with array requiring shifts and mask_indices # Create array where some polynomials need different numbers of shifts."""

    np.random.seed(2599)

    coeffs_shift_array = np.array([
        [[0., 0., 1., 2.]],  # Needs 2 shifts
        [[0., 1., 2., 0.]]   # Needs 1 shift
    ])
    p_shift_array2 = Polynomial(coeffs_shift_array)
    roots_shift_array2 = p_shift_array2.roots()

    assert isinstance(roots_shift_array2, Scalar)
    assert roots_shift_array2.shape[1:] == (2, 1)


def test_polynomial_operations_test_roots_on_array_of_polynomials_use_simple_linear_polynom() -> None:
    """Test roots on array of polynomials # Use simple linear polynomials: [1, 2] -> root at -2."""

    np.random.seed(2599)

    coeffs2 = np.array([
        [[1., 2.], [1., 2.]],
        [[1., 2.], [1., 2.]]
    ])
    p_array3 = Polynomial(coeffs2)
    roots_array = p_array3.roots()
    assert roots_array.shape == (1, 2, 2)
    assert np.allclose(roots_array.values[0], -2.)


def test_polynomial_operations_test_with_masks() -> None:
    """Test with masks."""

    np.random.seed(2599)

    p_masked = Polynomial([1., 2., 3.], mask=True)
    assert p_masked.mask
    p_masked2 = p_masked + Polynomial([1., 1., 1.])
    assert p_masked2.mask


def test_polynomial_operations_test_with_partial_mask() -> None:
    """Test with partial mask."""

    np.random.seed(2599)

    mask_array = np.array([[False, True], [False, False]])
    coeffs3 = np.array([
        [[1., 2.], [3., 4.]],
        [[5., 6.], [7., 8.]]
    ])
    p_partial_mask = Polynomial(coeffs3, mask=mask_array)
    assert p_partial_mask.shape == (2, 2)
    assert np.any(p_partial_mask.mask)


def test_polynomial_operations_test_recursive_parameter_create_polynomial_with_derivatives() -> None:
    """Test recursive parameter # Create polynomial with derivatives."""

    np.random.seed(2599)

    p_base = Polynomial([1., 2., 3.])
    p_deriv = Polynomial([0., 2., 6.])  # derivative
    p_base.insert_deriv('t', p_deriv)

    p_deriv_result = p_base.deriv(recursive=True)
    assert hasattr(p_deriv_result, 'd_dt')
    p_deriv_result2 = p_base.deriv(recursive=False)
    assert not hasattr(p_deriv_result2, 'd_dt')

    result_recursive = p_base.eval(2., recursive=True)
    assert hasattr(result_recursive, 'd_dt')
    result_no_recursive = p_base.eval(2., recursive=False)
    assert not hasattr(result_no_recursive, 'd_dt')


def test_polynomial_operations_test_that_roots_respects_recursive() -> None:
    """Test that roots respects recursive."""

    np.random.seed(2599)

    p_linear_with_deriv = Polynomial([4., 2.])
    p_linear_with_deriv.insert_deriv('t', Polynomial([0., 1.]))
    roots_recursive = p_linear_with_deriv.roots(recursive=True)
    assert hasattr(roots_recursive, 'd_dt')


def test_polynomial_operations_test_higher_order_polynomial_roots_cubic_x_3_6x_2_11x_6_x_1_() -> None:
    """Test higher order polynomial roots (cubic) # x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3) = 0."""

    np.random.seed(2599)

    p_cubic = Polynomial([1., -6., 11., -6.])  # x^3 - 6x^2 + 11x - 6
    roots_cubic = p_cubic.roots()
    assert type(roots_cubic) == Scalar
    assert roots_cubic.shape == (3,)

    roots_sorted = np.sort(roots_cubic.values)
    assert roots_sorted[0] == 1. or abs(roots_sorted[0] - 1.) <= 1e-8
    assert roots_sorted[1] == 2. or abs(roots_sorted[1] - 2.) <= 1e-8
    assert roots_sorted[2] == 3. or abs(roots_sorted[2] - 3.) <= 1e-8

    # Additional tests for coverage


def test_polynomial_operations_test_eval_with_order_0() -> None:
    """Test eval with order == 0."""

    np.random.seed(2599)

    p_const2 = Polynomial([5.])
    result_const = p_const2.eval(10., recursive=True)
    assert type(result_const) == Scalar
    assert result_const.values == 5. or abs(result_const.values - 5.) <= 1e-10

    result_const_no_rec = p_const2.eval(10., recursive=False)
    assert type(result_const_no_rec) == Scalar
    assert result_const_no_rec.values == 5. or abs(result_const_no_rec.values - 5.) <= 1e-10


def test_polynomial_operations_test_roots_with_scalar_mask() -> None:
    """Test roots with scalar mask."""

    np.random.seed(2599)

    p_mask_scalar = Polynomial([1., 2.], mask=True)
    roots_masked = p_mask_scalar.roots()
    assert np.all(roots_masked.mask)


def test_polynomial_operations_test_roots_with_all_zeros_case() -> None:
    """Test roots with all_zeros case."""

    np.random.seed(2599)

    p_zeros = Polynomial([0., 0., 1.])  # x^2 = 0
    roots_zeros = p_zeros.roots()

    assert roots_zeros.shape == (2,)


def test_polynomial_operations_test_roots_with_scalar_shift_case() -> None:
    """Test roots with scalar shift case."""

    np.random.seed(2599)

    p_leading_zero = Polynomial([0., 1., 2.])  # x + 2 = 0, leading zero
    roots_shift = p_leading_zero.roots()

    assert roots_shift.shape == (2,)

    valid_roots = roots_shift.values[np.isfinite(roots_shift.values)]
    assert len(valid_roots) == 1
    assert valid_roots[0] == -2. or abs(valid_roots[0] - -2.) <= 1e-10


def test_polynomial_operations_test_roots_mask_extraneous_zeros() -> None:
    """Test roots mask extraneous zeros."""

    np.random.seed(2599)

    p_extraneous = Polynomial([0., 0., 1., 2.])  # x + 2 = 0 with leading zeros
    roots_extraneous = p_extraneous.roots()

    assert roots_extraneous.shape == (3,)

    valid_roots = roots_extraneous.values[np.isfinite(roots_extraneous.values)]
    assert len(valid_roots) == 1
    assert valid_roots[0] == -2. or abs(valid_roots[0] - -2.) <= 1e-10


def test_polynomial_operations_test_roots_mask_duplicated_values_create_polynomial_with_dup() -> None:
    """Test roots mask duplicated values # Create polynomial with duplicate roots: (x-1)^2 = x^2 - 2x + 1."""

    np.random.seed(2599)

    p_duplicate = Polynomial([1., -2., 1.])
    roots_dup = p_duplicate.roots()
    assert roots_dup.shape == (2,)

    assert np.any(~np.isfinite(roots_dup.values))


def test_polynomial_operations_test_roots_with_derivatives() -> None:
    """Test roots with derivatives."""

    np.random.seed(2599)

    p_roots_deriv = Polynomial([1., 2.])  # x + 2 = 0 -> x = -2
    p_roots_deriv.insert_deriv('t', Polynomial([0., 1.]))  # derivative: 1
    roots_with_deriv = p_roots_deriv.roots(recursive=True)
    assert hasattr(roots_with_deriv, 'd_dt')

    assert roots_with_deriv.d_dt.values[0] == -1. or abs(roots_with_deriv.d_dt.values[0] - -1.) <= 1e-10


def test_polynomial_operations_test_roots_with_array_mask() -> None:
    """Test roots with array mask."""

    np.random.seed(2599)

    mask_array = np.array([[False, True], [True, False]])
    coeffs_masked = np.array([
        [[1., 2.], [1., 2.]],
        [[1., 2.], [1., 2.]]
    ])
    p_array_mask = Polynomial(coeffs_masked, mask=mask_array)
    roots_array_mask = p_array_mask.roots()

    assert roots_array_mask.shape == (1, 2, 2)


def test_polynomial_operations_test_roots_all_zeros_with_array_case() -> None:
    """Test roots all_zeros with array case."""

    np.random.seed(2599)

    coeffs_all_zeros = np.array([
        [[0., 0., 1.], [0., 0., 1.]],
        [[0., 0., 1.], [0., 0., 1.]]
    ])
    p_all_zeros_array = Polynomial(coeffs_all_zeros)
    roots_all_zeros_array = p_all_zeros_array.roots()

    assert roots_all_zeros_array.shape == (2, 2, 2)


def test_polynomial_operations_test_roots_with_array_shift_case() -> None:
    """Test roots with array shift case."""

    np.random.seed(2599)

    coeffs_leading_zeros = np.array([
        [[0., 1., 2.], [0., 1., 2.]],
        [[0., 1., 2.], [0., 1., 2.]]
    ])
    p_array_shift = Polynomial(coeffs_leading_zeros)
    roots_array_shift = p_array_shift.roots()

    assert roots_array_shift.shape == (2, 2, 2)

    finite_mask = np.isfinite(roots_array_shift.values)
    assert np.any(finite_mask)

    valid_per_poly = np.sum(finite_mask, axis=0)
    assert np.all(valid_per_poly == 1)


def test_polynomial_operations_test_roots_mask_extraneous_zeros_with_array() -> None:
    """Test roots mask extraneous zeros with array."""

    np.random.seed(2599)

    coeffs_extraneous_array = np.array([
        [[0., 0., 1., 2.], [0., 0., 1., 2.]],
        [[0., 0., 1., 2.], [0., 0., 1., 2.]]
    ])
    p_extraneous_array = Polynomial(coeffs_extraneous_array)
    roots_extraneous_array = p_extraneous_array.roots()

    assert roots_extraneous_array.shape == (3, 2, 2)

    finite_mask = np.isfinite(roots_extraneous_array.values)
    valid_per_poly = np.sum(finite_mask, axis=0)
    assert np.all(valid_per_poly == 1)


def test_polynomial_operations_test_roots_mask_duplicated_values_with_array() -> None:
    """Test roots mask duplicated values with array."""

    np.random.seed(2599)

    coeffs_dup_array = np.array([
        [[1., -2., 1.], [1., -2., 1.]],
        [[1., -2., 1.], [1., -2., 1.]]
    ])
    p_dup_array = Polynomial(coeffs_dup_array)
    roots_dup_array = p_dup_array.roots()

    assert roots_dup_array.shape == (2, 2, 2)
    assert np.any(~np.isfinite(roots_dup_array.values))


def test_polynomial_operations_test_eval_with_order_0_and_nested_derivatives_create_a_const() -> None:
    """Test eval with order 0 and nested derivatives # Create a constant polynomial with derivatives that have derivatives."""

    np.random.seed(2599)

    p_const_deriv = Polynomial([5.])
    p_deriv1 = Polynomial([0.])  # derivative is constant
    p_deriv1.insert_deriv('s', Polynomial([1.]))  # derivative of derivative
    p_const_deriv.insert_deriv('t', p_deriv1)
    result_const_deriv = p_const_deriv.eval(10., recursive=True)
    assert type(result_const_deriv) == Scalar
    assert result_const_deriv.values == 5. or abs(result_const_deriv.values - 5.) <= 1e-10
    assert hasattr(result_const_deriv, 'd_dt')

    assert type(result_const_deriv.d_dt) == Scalar


def test_polynomial_operations_test_eval_with_order_0_derivative_with_tail_this_requires_a_() -> None:
    """Test eval with order 0, derivative with tail # This requires a polynomial with drank > 0 # Create a Vector with drank=1 first."""

    np.random.seed(2599)

    v_const_drank = Vector(np.array([[5.]]), drank=1)  # shape (), numer (1,), denom (1,)
    p_const_drank = Polynomial(v_const_drank)
    result_const_drank = p_const_drank.eval(10., recursive=False)
    assert type(result_const_drank) == Scalar
    assert result_const_drank.values == 5. or abs(result_const_drank.values - 5.) <= 1e-10


def test_polynomial_operations_test_eval_with_order_0_derivative_with_tail_drank_0() -> None:
    """Test eval with order 0, derivative with tail (drank > 0)."""

    np.random.seed(2599)

    v_const_deriv_drank = Vector(np.array([[7.]]), drank=1)
    p_const_deriv_drank = Polynomial(v_const_deriv_drank)
    v_deriv_drank = Vector(np.array([[0.]]), drank=1)
    p_deriv_drank = Polynomial(v_deriv_drank)
    p_const_deriv_drank.insert_deriv('t', p_deriv_drank)
    result_const_deriv_drank = p_const_deriv_drank.eval(20., recursive=True)
    assert type(result_const_deriv_drank) == Scalar
    assert result_const_deriv_drank.values == 7. or abs(result_const_deriv_drank.values - 7.) <= 1e-10
    assert hasattr(result_const_deriv_drank, 'd_dt')


def test_polynomial_operations_test_eval_with_order_0_nested_derivatives_with_tail_drank_0_() -> None:
    """Test eval with order 0, nested derivatives with tail (drank > 0) # This tests the full nested derivative conversion path."""

    np.random.seed(2599)

    v_const_nested_drank = Vector(np.array([[9.]]), drank=1)
    p_const_nested_drank = Polynomial(v_const_nested_drank)
    v_deriv_nested = Vector(np.array([[0.]]), drank=1)
    p_deriv_nested = Polynomial(v_deriv_nested)

    v_deriv_nested2 = Vector(np.array([[1.]]), drank=1)
    p_deriv_nested2 = Polynomial(v_deriv_nested2)
    p_deriv_nested.insert_deriv('s', p_deriv_nested2)
    p_const_nested_drank.insert_deriv('t', p_deriv_nested)
    result_const_nested_drank = p_const_nested_drank.eval(30., recursive=True)
    assert type(result_const_nested_drank) == Scalar
    assert result_const_nested_drank.values == 9. or abs(result_const_nested_drank.values - 9.) <= 1e-10
    assert hasattr(result_const_nested_drank, 'd_dt')

    assert type(result_const_nested_drank.d_dt) == Scalar


def test_polynomial_operations_also_test_nested_derivative_that_is_constant_order_0_with_no() -> None:
    """Also test nested derivative that is constant (order 0) with no tail (drank=0) # This tests the else branch when dvalue_tail is empty."""

    np.random.seed(2599)

    v_const_nested_drank2 = Vector(np.array([[11.]]), drank=1)
    p_const_nested_drank2 = Polynomial(v_const_nested_drank2)
    v_deriv_nested3 = Vector(np.array([[0.]]), drank=1)
    p_deriv_nested3 = Polynomial(v_deriv_nested3)

    p_deriv_nested5 = Polynomial([1.])  # constant, no tail
    p_deriv_nested3.insert_deriv('s', p_deriv_nested5)
    p_const_nested_drank2.insert_deriv('t', p_deriv_nested3)
    result_const_nested_drank2 = p_const_nested_drank2.eval(40., recursive=True)
    assert type(result_const_nested_drank2) == Scalar
    assert result_const_nested_drank2.values == 11. or abs(result_const_nested_drank2.values - 11.) <= 1e-10
    assert hasattr(result_const_nested_drank2, 'd_dt')


def test_polynomial_operations_test_roots_with_scalar_mask_true_2() -> None:
    """Test roots with scalar mask True."""

    np.random.seed(2599)

    p_mask_true = Polynomial([1., 2.], mask=True)
    roots_mask_true = p_mask_true.roots()

    assert (np.all(~np.isfinite(roots_mask_true.values)) or np.all(roots_mask_true.mask))


def test_polynomial_operations_test_roots_with_scalar_mask_false_2() -> None:
    """Test roots with scalar mask False."""

    np.random.seed(2599)

    p_mask_false = Polynomial([1., 2.], mask=False)
    roots_mask_false = p_mask_false.roots()
    assert not np.any(roots_mask_false.mask)


def test_polynomial_operations_test_roots_with_all_zeros_case_create_polynomial_where_all_c() -> None:
    """Test roots with all_zeros case # Create polynomial where all coefficients are zero for some elements."""

    np.random.seed(2599)

    coeffs_all_zeros = np.array([
        [[0., 0., 0.], [1., 2., 3.]],
        [[0., 0., 0.], [1., 2., 3.]]
    ])
    p_all_zeros = Polynomial(coeffs_all_zeros)
    roots_all_zeros = p_all_zeros.roots()

    assert roots_all_zeros.shape == (2, 2, 2)


def test_polynomial_operations_test_roots_with_array_shifts_and_mask_indices_create_array_w() -> None:
    """Test roots with array shifts and mask_indices # Create array where some elements need different numbers of shifts # This tests the array case (shift_shape is not empty)."""

    np.random.seed(2599)

    coeffs_array_shifts = np.array([
        [[0., 1., 2., 0.], [1., 2., 3., 0.]],  # First needs 1 shift, second needs 0
        [[0., 0., 1., 2.], [1., 2., 3., 0.]]  # First needs 2 shifts, second needs 0
    ])
    p_array_shifts = Polynomial(coeffs_array_shifts)
    roots_array_shifts = p_array_shifts.roots()

    assert roots_array_shifts.shape == (3, 2, 2)


def test_polynomial_operations_test_roots_duplicate_detection_scalar_case_create_polynomial() -> None:
    """Test roots duplicate detection scalar case # Create polynomial with duplicate roots in scalar case."""

    np.random.seed(2599)

    p_dup_scalar = Polynomial([1., -2., 1.])  # (x-1)^2, duplicate root at 1
    roots_dup_scalar = p_dup_scalar.roots()

    assert np.any(~np.isfinite(roots_dup_scalar.values))


def test_polynomial_operations_test_roots_with_derivatives_this_tests_the_code_path_for_add() -> None:
    """Test roots with derivatives # This tests the code path for adding derivatives to roots."""

    np.random.seed(2599)

    p_roots_deriv2 = Polynomial([1., -3., 2.])  # (x-1)(x-2) = x^2 - 3x + 2
    p_roots_deriv2.insert_deriv('t', Polynomial([0., -1., 0.]))  # derivative: -x
    roots_with_deriv2 = p_roots_deriv2.roots(recursive=True)

    assert roots_with_deriv2.shape == (2,)


def test_polynomial_operations_test_roots_with_scalar_mask_true_duplicate_test() -> None:
    """Test roots with scalar mask True (duplicate test)."""

    np.random.seed(2599)

    p_mask_true2 = Polynomial([1., 2.], mask=True)
    roots_mask_true2 = p_mask_true2.roots()

    assert (np.all(~np.isfinite(roots_mask_true2.values)) or np.all(roots_mask_true2.mask))


def test_polynomial_operations_test_roots_with_scalar_mask_false_duplicate_test() -> None:
    """Test roots with scalar mask False (duplicate test)."""

    np.random.seed(2599)

    p_mask_false2 = Polynomial([1., 2.], mask=False)
    roots_mask_false2 = p_mask_false2.roots()

    if isinstance(roots_mask_false2.mask, np.ndarray):
        assert not np.any(roots_mask_false2.mask)
    else:
        assert not roots_mask_false2.mask


def test_polynomial_operations_test_roots_with_all_zeros_case_create_polynomial_where_all_c_2() -> None:
    """Test roots with all_zeros case # Create polynomial where all coefficients are zero."""

    np.random.seed(2599)

    p_all_zeros2 = Polynomial([0., 0., 0.])
    roots_all_zeros2 = p_all_zeros2.roots()

    assert roots_all_zeros2.shape == (2,)

    if isinstance(roots_all_zeros2.mask, np.ndarray):
        # Check that mask is set (all True or all inf)
        assert (np.all(roots_all_zeros2.mask) or np.all(~np.isfinite(roots_all_zeros2.values)))
    else:
        # Scalar mask case
        assert (roots_all_zeros2.mask or not np.any(np.isfinite(roots_all_zeros2.values)))


def test_polynomial_operations_test_roots_with_array_shifts_and_mask_indices_create_array_w_2() -> None:
    """Test roots with array shifts and mask_indices # Create array where some elements need different numbers of shifts # This tests the array case (shift_shape is not empty)."""

    np.random.seed(2599)

    coeffs_array_shifts2 = np.array([
        [[0., 0., 1., 2.], [0., 1., 2., 3.]],  # First needs 2 shifts, second needs 1 shift
        [[1., 2., 3., 4.], [0., 0., 0., 1.]]   # First needs 0 shifts, second needs 3 shifts
    ])
    p_array_shifts2 = Polynomial(coeffs_array_shifts2)
    roots_array_shifts2 = p_array_shifts2.roots()

    assert roots_array_shifts2.shape == (3, 2, 2)
    # The mask_indices code path should execute when total_shifts.size > 0
    # and len(mask_indices) > 0


def test_polynomial_operations_test_roots_duplicate_detection_scalar_case_create_polynomial_2() -> None:
    """Test roots duplicate detection scalar case # Create polynomial with duplicate roots in scalar case."""

    np.random.seed(2599)

    p_dup_scalar2 = Polynomial([1., -4., 4.])  # (x-2)^2, duplicate root at 2
    roots_dup_scalar2 = p_dup_scalar2.roots()

    assert (np.any(~np.isfinite(roots_dup_scalar2.values)) or
                   (isinstance(roots_dup_scalar2.mask, bool) and roots_dup_scalar2.mask))


def test_polynomial_operations_test_roots_with_derivatives_this_tests_the_code_path_for_add_2() -> None:
    """Test roots with derivatives # This tests the code path for adding derivatives to roots # Use a linear polynomial for simplicity: x + 2 = 0, root at -2 # Derivative of polynomial: 1 (constant, nonzero at root) # Derivative of polynomial w.r.t. t: some constant."""

    np.random.seed(2599)

    p_roots_deriv3 = Polynomial([1., 2.])  # x + 2
    p_roots_deriv3.insert_deriv('t', Polynomial([0., 1.]))  # derivative w.r.t. t: 1
    roots_with_deriv3 = p_roots_deriv3.roots(recursive=True)

    assert roots_with_deriv3.shape == (1,)

    assert hasattr(roots_with_deriv3, 'd_dt')
    assert roots_with_deriv3.d_dt.values[0] == -1. or abs(roots_with_deriv3.d_dt.values[0] - -1.) <= 1e-10


##########################################################################################
