##########################################################################################
# tests/test_polynomial_arithmetic.py
# Polynomial arithmetic operation tests
##########################################################################################

import numpy as np
import pytest

from polymath import Vector, Polynomial


def test_polynomial_arithmetic_test_neg() -> None:
    """Test __neg__."""

    np.random.seed(2599)

    p9 = Polynomial([1., 2., 3.])
    p_neg = -p9
    assert type(p_neg) == Polynomial
    assert np.allclose(p_neg.values, -p9.values)

    p10 = Polynomial([1., 2.])  # x + 2
    p11 = Polynomial([3., 4., 5.])  # 3x^2 + 4x + 5
    p_sum = p10 + p11
    assert type(p_sum) == Polynomial
    assert p_sum.order == 2

    assert p_sum.values[0] == 3. or abs(p_sum.values[0] - 3.) <= 1e-10
    assert p_sum.values[1] == 5. or abs(p_sum.values[1] - 5.) <= 1e-10
    assert p_sum.values[2] == 7. or abs(p_sum.values[2] - 7.) <= 1e-10

    p12 = Polynomial([1., 2.])  # x + 2
    p_sum2 = p12 + 5.  # should add 5 to constant term: x + 7
    assert p_sum2.order == 1
    assert p_sum2.values[0] == 1. or abs(p_sum2.values[0] - 1.) <= 1e-10  # x coefficient unchanged
    assert p_sum2.values[1] == 7. or abs(p_sum2.values[1] - 7.) <= 1e-10  # constant term: 2 + 5 = 7

    p13 = Polynomial([1., 2.])  # x + 2
    p_sum3 = 5. + p13  # adds 5 to constant term: x + 7
    assert type(p_sum3) == Polynomial
    assert p_sum3.values[1] == 7. or abs(p_sum3.values[1] - 7.) <= 1e-10

    p14 = Polynomial([5., 4., 3.])  # 5x^2 + 4x + 3
    p15 = Polynomial([1., 2.])  # x + 2
    p_diff = p14 - p15
    assert type(p_diff) == Polynomial
    assert p_diff.order == 2

    assert p_diff.values[0] == 5. or abs(p_diff.values[0] - 5.) <= 1e-10
    assert p_diff.values[1] == 3. or abs(p_diff.values[1] - 3.) <= 1e-10
    assert p_diff.values[2] == 1. or abs(p_diff.values[2] - 1.) <= 1e-10

    p16 = Polynomial([1., 2.])  # x + 2
    p_diff2 = 5. - p16  # -x + 3
    assert type(p_diff2) == Polynomial
    assert p_diff2.values[0] == -1. or abs(p_diff2.values[0] - -1.) <= 1e-10
    assert p_diff2.values[1] == 3. or abs(p_diff2.values[1] - 3.) <= 1e-10

    p17 = Polynomial([1., 2., 3.])
    p_prod = p17 * 2.
    assert type(p_prod) == Polynomial
    assert np.allclose(p_prod.values, p17.values * 2.)

    p18 = Polynomial([1., 1.])  # x + 1
    p19 = Polynomial([1., 2.])  # x + 2 (not [2, 1] which is 2x + 1)
    p_prod2 = p18 * p19
    assert type(p_prod2) == Polynomial
    assert p_prod2.order == 2

    assert p_prod2.eval(0.).values == 2. or abs(p_prod2.eval(0.).values - 2.) <= 1e-10
    assert p_prod2.eval(1.).values == 6. or abs(p_prod2.eval(1.).values - 6.) <= 1e-10

    assert p_prod2.values[0] == 1. or abs(p_prod2.values[0] - 1.) <= 1e-10
    assert p_prod2.values[1] == 3. or abs(p_prod2.values[1] - 3.) <= 1e-10
    assert p_prod2.values[2] == 2. or abs(p_prod2.values[2] - 2.) <= 1e-10

    p20 = Polynomial([1., 2.])
    p_prod3 = 3. * p20
    assert type(p_prod3) == Polynomial
    assert np.allclose(p_prod3.values, p20.values * 3.)

    p21 = Polynomial([2., 4., 6.])
    p_div = p21 / 2.
    assert type(p_div) == Polynomial
    assert np.allclose(p_div.values, p21.values / 2.)

    p22 = Polynomial([1., 1.])  # x + 1
    p_pow = p22 ** 2
    assert type(p_pow) == Polynomial
    assert p_pow.order == 2

    assert p_pow.values[0] == 1. or abs(p_pow.values[0] - 1.) <= 1e-10
    assert p_pow.values[1] == 2. or abs(p_pow.values[1] - 2.) <= 1e-10
    assert p_pow.values[2] == 1. or abs(p_pow.values[2] - 1.) <= 1e-10

    p_pow3 = p22 ** 3  # (x+1)^3 = x^3 + 3x^2 + 3x + 1
    assert p_pow3.order == 3
    assert p_pow3.values[0] == 1. or abs(p_pow3.values[0] - 1.) <= 1e-10
    assert p_pow3.values[1] == 3. or abs(p_pow3.values[1] - 3.) <= 1e-10
    assert p_pow3.values[2] == 3. or abs(p_pow3.values[2] - 3.) <= 1e-10
    assert p_pow3.values[3] == 1. or abs(p_pow3.values[3] - 1.) <= 1e-10

    p23 = Polynomial([1., 2., 3.])
    p_pow0 = p23 ** 0
    assert type(p_pow0) == Polynomial
    assert p_pow0.order == 0
    assert p_pow0.values[0] == 1.

    with pytest.raises(ValueError):
        p23.__pow__(-1)
    with pytest.raises(ValueError):
        p23.__pow__(1.5)

    p24 = Polynomial([1., 2., 3.])
    p25 = Polynomial([1., 2., 3.])
    p26 = Polynomial([1., 2., 4.])
    assert (p24 == p25)
    assert p24 != p26
    assert (p24 != p26)
    assert p24 == p25

    p_normal1 = Polynomial([1., 2.])
    p_normal2 = Polynomial([3., 4.])

    p_normal_prod = p_normal1 * p_normal2
    assert p_normal_prod.order == 2

    # Additional tests for coverage

    p_iadd = Polynomial([1., 2.])
    p_iadd += Polynomial([3., 4.])
    assert p_iadd.order == 1
    assert p_iadd.values[0] == 4. or abs(p_iadd.values[0] - 4.) <= 1e-10
    assert p_iadd.values[1] == 6. or abs(p_iadd.values[1] - 6.) <= 1e-10

    p_isub = Polynomial([5., 6.])
    p_isub -= Polynomial([1., 2.])
    assert p_isub.order == 1
    assert p_isub.values[0] == 4. or abs(p_isub.values[0] - 4.) <= 1e-10
    assert p_isub.values[1] == 4. or abs(p_isub.values[1] - 4.) <= 1e-10

    p_mul1 = Polynomial([1., 2.])
    p_mul2 = Polynomial([3., 4.])
    p_mul_result = p_mul1 * p_mul2
    assert p_mul_result.order == 2

    p_mul_deriv1 = Polynomial([1., 2.])
    p_mul_deriv2 = Polynomial([3., 4.])
    p_mul_deriv1.insert_deriv('t', Polynomial([0., 1.]))
    p_mul_deriv2.insert_deriv('t', Polynomial([0., 2.]))
    p_mul_deriv_result = p_mul_deriv1 * p_mul_deriv2
    assert hasattr(p_mul_deriv_result, 'd_dt')

    v_scalar = Vector([5.])
    p_imul = Polynomial([1., 2.])
    p_imul *= v_scalar
    assert p_imul.order == 1
    assert p_imul.values[0] == 5. or abs(p_imul.values[0] - 5.) <= 1e-10
    assert p_imul.values[1] == 10. or abs(p_imul.values[1] - 10.) <= 1e-10

    v_scalar2 = Vector([2.])
    p_tdiv = Polynomial([2., 4.])
    p_tdiv_result = p_tdiv / v_scalar2
    assert p_tdiv_result.order == 1
    assert p_tdiv_result.values[0] == 1. or abs(p_tdiv_result.values[0] - 1.) <= 1e-10
    assert p_tdiv_result.values[1] == 2. or abs(p_tdiv_result.values[1] - 2.) <= 1e-10

    p_itdiv = Polynomial([4., 8.])
    p_itdiv /= Vector([2.])
    assert p_itdiv.order == 1
    assert p_itdiv.values[0] == 2. or abs(p_itdiv.values[0] - 2.) <= 1e-10
    assert p_itdiv.values[1] == 4. or abs(p_itdiv.values[1] - 4.) <= 1e-10

    p_iadd1 = Polynomial([1., 2.])  # order 1
    p_iadd2 = Polynomial([3., 4., 5.])  # order 2
    id_before = id(p_iadd1)
    p_iadd1 += p_iadd2
    assert id(p_iadd1) == id_before  # In-place

    assert len(p_iadd1.values) == 3  # Should have 3 coefficients

    p_iadd_deriv1 = Polynomial([1., 2.])
    p_iadd_deriv2 = Polynomial([3., 4.])
    p_iadd_deriv1.insert_deriv('t', Polynomial([0., 1.]))
    p_iadd_deriv2.insert_deriv('t', Polynomial([0., 2.]))
    p_iadd_deriv1 += p_iadd_deriv2
    assert hasattr(p_iadd_deriv1, 'd_dt')

    p_isub1 = Polynomial([5., 6.])  # order 1
    p_isub2 = Polynomial([1., 2., 3.])  # order 2
    p_isub1 -= p_isub2
    assert len(p_isub1.values) == 3

    p_isub_self_larger = Polynomial([10., 20., 30., 40.])  # order 3
    p_isub_arg_smaller = Polynomial([1., 2.])  # order 1

    p_isub_self_larger -= p_isub_arg_smaller
    assert p_isub_self_larger.order == 3

    p_isub3 = Polynomial([5., 6., 7.])  # order 2
    p_isub4 = Polynomial([1., 2.])  # order 1, needs at_least_order
    p_isub3 -= p_isub4
    assert len(p_isub3.values) == 3

    p_isub_deriv1 = Polynomial([5., 6.])
    p_isub_deriv2 = Polynomial([1., 2.])
    p_isub_deriv1.insert_deriv('t', Polynomial([0., 1.]))
    p_isub_deriv2.insert_deriv('t', Polynomial([0., 2.]))
    p_isub_deriv1 -= p_isub_deriv2
    assert hasattr(p_isub_deriv1, 'd_dt')

    v_drank1 = Vector(np.array([[[1., 2.], [3., 4.]]]), drank=1)  # shape (1,), numer (2,), denom (2,)
    p_mul_drank1 = Polynomial(v_drank1)
    p_mul_drank2 = Polynomial([5., 6.])  # drank=0

    with pytest.raises(ValueError):
        p_mul_drank1.__mul__(p_mul_drank2)


def test_polynomial_arithmetic_test_itruediv_with_vector_item_1() -> None:
    """Test __itruediv__ with Vector item == (1,)."""

    np.random.seed(2599)

    p_itdiv_vec = Polynomial([4., 8.])
    v_scalar = Vector([2.])
    p_itdiv_vec /= v_scalar
    assert p_itdiv_vec.values[0] == 2. or abs(p_itdiv_vec.values[0] - 2.) <= 1e-10
    assert p_itdiv_vec.values[1] == 4. or abs(p_itdiv_vec.values[1] - 4.) <= 1e-10


def test_polynomial_arithmetic_test_itruediv_with_vector_item_1_this_tests_the_branch_isins() -> None:
    """Test __itruediv__ with Vector item == (1,) # This tests the branch: isinstance(arg, Vector) and arg.item == (1,) # Verify that Vector([4.]) has item == (1,)."""

    np.random.seed(2599)

    v_scalar3 = Vector([4.])
    assert v_scalar3.item == (1,)
    p_itdiv_vec2 = Polynomial([8., 16.])

    p_itdiv_vec2 /= v_scalar3
    assert p_itdiv_vec2.values[0] == 2. or abs(p_itdiv_vec2.values[0] - 2.) <= 1e-10
    assert p_itdiv_vec2.values[1] == 4. or abs(p_itdiv_vec2.values[1] - 4.) <= 1e-10


def test_polynomial_arithmetic_test_iadd_when_arg_order_max_order_this_tests_the_branch_if_() -> None:
    """Test __iadd__ when arg.order < max_order # This tests the branch: if arg.order < max_order: arg = arg.at_least_order(max_order) # Need case where self.order > arg.order, so max_order = self.order and arg.order < max_order."""

    np.random.seed(2599)

    p_iadd_self_larger = Polynomial([1., 2., 3., 4.])  # order 3
    p_iadd_arg_smaller = Polynomial([5., 6.])  # order 1

    p_iadd_self_larger += p_iadd_arg_smaller
    assert p_iadd_self_larger.order == 3

    assert p_iadd_self_larger.values[0] == 1. or abs(p_iadd_self_larger.values[0] - 1.) <= 1e-10
    assert p_iadd_self_larger.values[3] == 10. or abs(p_iadd_self_larger.values[3] - 10.) <= 1e-10  # 4 + 6 = 10


def test_polynomial_arithmetic_test_mul_with_derivative_else_branch_create_two_polynomials_() -> None:
    """Test __mul__ with derivative else branch # Create two polynomials with different derivative keys."""

    np.random.seed(2599)

    p_mul_deriv_a = Polynomial([1., 2.])
    p_mul_deriv_b = Polynomial([3., 4.])
    p_mul_deriv_a.insert_deriv('t', Polynomial([0., 1.]))
    p_mul_deriv_b.insert_deriv('s', Polynomial([0., 2.]))  # Different key
    p_mul_mixed = p_mul_deriv_a * p_mul_deriv_b

    assert hasattr(p_mul_mixed, 'd_dt')
    assert hasattr(p_mul_mixed, 'd_ds')


##########################################################################################
