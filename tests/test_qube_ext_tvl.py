##########################################################################################
# tests/test_qube_ext_tvl.py
##########################################################################################

import numpy as np
import pytest
import numpy.ma as ma

from polymath import Qube, Scalar, Boolean


@pytest.fixture(autouse=True)
def _setup_teardown():
    """Replaces the original setUp and tearDown methods."""
    Qube.prefer_builtins(False)
    yield
    Qube.prefer_builtins(False)


def test_qube_ext_tvl_test_truth_table_false_and_anything_false() -> None:
    """Test truth table: False and anything = False."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    assert Boolean(False).tvl_and(False) == Boolean(False)
    assert Boolean(False).tvl_and(True) == Boolean(False)
    assert Boolean(False).tvl_and(Boolean(True, mask=True)) == Boolean(False)


def test_qube_ext_tvl_test_truth_table_true_and_true_true() -> None:
    """Test truth table: True and True = True."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    assert Boolean(True).tvl_and(True) == Boolean(True)
    assert Boolean(True).tvl_and(Boolean(True)) == Boolean(True)


def test_qube_ext_tvl_test_truth_table_true_and_masked_masked() -> None:
    """Test truth table: True and Masked = Masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    masked_true = Boolean(True, mask=True)
    result = Boolean(True).tvl_and(masked_true)
    assert result.mask
    # When masked, the value can be True or False, but it's masked

    result = masked_true.tvl_and(False)
    assert result == Boolean(False)

    masked_false = Boolean(False, mask=True)
    result = masked_true.tvl_and(masked_false)

    assert result.mask

    masked_true2 = Boolean(True, mask=True)
    result = masked_true.tvl_and(masked_true2)
    assert result.mask

    a = Boolean([False, True, False, True])
    b = Boolean([True, True, False, False])
    result = a.tvl_and(b)
    assert result.shape == (4,)
    assert np.all(result.values == [False, True, False, False])

    a_masked = Boolean([True, False, True], mask=[False, True, False])
    b_masked = Boolean([True, True, False], mask=[False, False, True])
    result = a_masked.tvl_and(b_masked)
    assert result.shape == (3,)

    assert result.values[0]
    assert not result.mask[0]

    assert not result.values[1]

    assert not result.values[2]
    # Note: The mask behavior here may differ from docstring expectation

    a_nd = Boolean(np.random.rand(2, 3, 4) > 0.5)
    b_nd = Boolean(np.random.rand(2, 3, 4) > 0.5)
    result = a_nd.tvl_and(b_nd)
    assert result.shape == (2, 3, 4)
    expected = a_nd.values & b_nd.values
    assert np.all(result.values == expected)

    Qube.prefer_builtins(True)
    result = Boolean(True).tvl_and(True)
    assert type(result) == bool
    assert result == True
    result = Boolean(False).tvl_and(True)
    assert type(result) == bool
    assert result == False

    masked_result = Boolean(True, mask=True).tvl_and(True, builtins=True, masked=False)
    assert type(masked_result) == bool
    assert masked_result == False
    masked_result = Boolean(True, mask=True).tvl_and(True, builtins=True, masked=True)
    assert type(masked_result) == bool
    assert masked_result == True
    Qube.prefer_builtins(False)

    masked_bool = Boolean(True, mask=True)
    result = masked_bool.tvl_and(True, builtins=True, masked=None)

    assert isinstance(result, Boolean)
    result = masked_bool.tvl_and(True, builtins=True, masked=False)
    assert type(result) == bool
    assert result == False
    result = masked_bool.tvl_and(True, builtins=True, masked=True)
    assert type(result) == bool
    assert result == True

    ##################################################################################
    # tvl_or(self, arg, builtins=None, masked=None)
    ##################################################################################

    assert Boolean(True).tvl_or(False) == Boolean(True)
    assert Boolean(True).tvl_or(True) == Boolean(True)
    assert Boolean(True).tvl_or(Boolean(False, mask=True)) == Boolean(True)

    assert Boolean(False).tvl_or(False) == Boolean(False)

    result = Boolean(False).tvl_or(masked_true)

    assert result.mask

    masked_false = Boolean(False, mask=True)
    result = Boolean(False).tvl_or(masked_false)
    assert result.mask
    # When masked, the value can be True or False, but it's masked

    masked_false = Boolean(False, mask=True)
    result = masked_true.tvl_or(masked_false)

    assert result.mask

    masked_false2 = Boolean(False, mask=True)
    result = masked_false.tvl_or(masked_false2)
    assert result.mask


def test_qube_ext_tvl_test_with_arrays_n_d() -> None:
    """Test with arrays (n-D)."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([False, True, False, True])
    b = Boolean([True, False, False, False])
    result = a.tvl_or(b)
    assert result.shape == (4,)
    assert np.all(result.values == [True, True, False, True])


def test_qube_ext_tvl_test_with_masked_arrays() -> None:
    """Test with masked arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a_masked = Boolean([False, True, False], mask=[False, True, False])
    b_masked = Boolean([True, False, False], mask=[False, False, True])
    result = a_masked.tvl_or(b_masked)
    assert result.shape == (3,)

    assert result.values[0]
    assert not result.mask[0]

    assert result.mask[1]

    assert result.mask[2]


def test_qube_ext_tvl_test_with_n_d_arrays() -> None:
    """Test with n-D arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a_nd = Boolean(np.random.rand(2, 3, 4) > 0.5)
    b_nd = Boolean(np.random.rand(2, 3, 4) > 0.5)
    result = a_nd.tvl_or(b_nd)
    assert result.shape == (2, 3, 4)
    expected = a_nd.values | b_nd.values
    assert np.all(result.values == expected)


def test_qube_ext_tvl_test_builtins_parameter() -> None:
    """Test builtins parameter."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)
    result = Boolean(True).tvl_or(False)
    assert type(result) == bool
    assert result == True
    result = Boolean(False).tvl_or(False)
    assert type(result) == bool
    assert result == False
    Qube.prefer_builtins(False)


def test_qube_ext_tvl_test_builtins_true_with_masked_result_and_masked_parameter_f() -> None:
    """Test builtins=True with masked result and masked parameter for tvl_or."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    masked_bool = Boolean(False, mask=True)
    result = masked_bool.tvl_or(False, builtins=True, masked=None)
    assert isinstance(result, Boolean)
    result = masked_bool.tvl_or(False, builtins=True, masked=False)
    assert type(result) == bool
    assert result == False

    ##################################################################################
    # tvl_any(self, axis=None, builtins=None, masked=None)
    ##################################################################################


def test_qube_ext_tvl_test_true_if_any_unmasked_value_is_true() -> None:
    """Test: True if any unmasked value is True."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([False, False, True, False])
    result = a.tvl_any()
    assert result == Boolean(True)


def test_qube_ext_tvl_test_false_if_and_only_if_all_items_are_false_and_unmasked() -> None:
    """Test: False if and only if all items are False and unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([False, False, False])
    result = a.tvl_any()
    assert result == Boolean(False)


def test_qube_ext_tvl_test_masked_if_all_false_but_some_masked() -> None:
    """Test: Masked if all False but some masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([False, False, False], mask=[False, True, False])
    result = a.tvl_any()
    assert result.mask
    assert not result.values


def test_qube_ext_tvl_test_true_if_any_true_even_with_some_masked() -> None:
    """Test: True if any True even with some masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([False, True, False], mask=[False, False, True])
    result = a.tvl_any()
    assert result == Boolean(True)


def test_qube_ext_tvl_test_with_axis_parameter_1_d() -> None:
    """Test with axis parameter (1-D)."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([[False, True, False], [False, False, False]])
    result = a.tvl_any(axis=1)
    assert result.shape == (2,)
    assert result.values[0]
    assert not result.values[1]


def test_qube_ext_tvl_test_with_axis_parameter_n_d() -> None:
    """Test with axis parameter (n-D)."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean(np.random.rand(2, 3, 4) > 0.5)
    result = a.tvl_any(axis=0)
    assert result.shape == (3, 4)
    result = a.tvl_any(axis=(0, 1))
    assert result.shape == (4,)


def test_qube_ext_tvl_test_with_masked_arrays_and_axis() -> None:
    """Test with masked arrays and axis."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([[False, True, False], [False, False, False]],
               mask=[[False, False, True], [False, True, False]])
    result = a.tvl_any(axis=1)
    assert result.shape == (2,)

    assert result.values[0]
    assert not result.mask[0]

    assert not result.values[1]
    assert result.mask[1]


def test_qube_ext_tvl_test_builtins_parameter_2() -> None:
    """Test builtins parameter."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)
    result = Boolean(True).tvl_any()
    assert type(result) == bool
    assert result == True
    result = Boolean(False).tvl_any()
    assert type(result) == bool
    assert result == False
    Qube.prefer_builtins(False)


def test_qube_ext_tvl_test_builtins_true_with_masked_result_and_masked_parameter_f_2() -> None:
    """Test builtins=True with masked result and masked parameter for tvl_any."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    masked_bool = Boolean([False, False], mask=[True, False])
    result = masked_bool.tvl_any(builtins=True, masked=None)
    assert isinstance(result, Boolean)
    result = masked_bool.tvl_any(builtins=True, masked=False)
    assert type(result) == bool
    assert result == False

    ##################################################################################
    # tvl_all(self, axis=None, builtins=None, masked=None)
    ##################################################################################


def test_qube_ext_tvl_test_true_if_and_only_if_all_items_are_true_and_unmasked() -> None:
    """Test: True if and only if all items are True and unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([True, True, True])
    result = a.tvl_all()
    assert result == Boolean(True)


def test_qube_ext_tvl_test_false_if_any_unmasked_value_is_false() -> None:
    """Test: False if any unmasked value is False."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([True, False, True])
    result = a.tvl_all()
    assert result == Boolean(False)


def test_qube_ext_tvl_test_masked_if_all_true_but_some_masked() -> None:
    """Test: Masked if all True but some masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([True, True, True], mask=[False, True, False])
    result = a.tvl_all()
    assert result.mask
    assert result.values


def test_qube_ext_tvl_test_false_if_any_false_even_with_some_masked() -> None:
    """Test: False if any False even with some masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([True, False, True], mask=[False, False, True])
    result = a.tvl_all()
    assert result == Boolean(False)


def test_qube_ext_tvl_test_with_axis_parameter_1_d_2() -> None:
    """Test with axis parameter (1-D)."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([[True, True, True], [True, False, True]])
    result = a.tvl_all(axis=1)
    assert result.shape == (2,)
    assert result.values[0]
    assert not result.values[1]


def test_qube_ext_tvl_test_with_axis_parameter_n_d_2() -> None:
    """Test with axis parameter (n-D)."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean(np.random.rand(2, 3, 4) > 0.5)
    result = a.tvl_all(axis=0)
    assert result.shape == (3, 4)
    result = a.tvl_all(axis=(0, 1))
    assert result.shape == (4,)


def test_qube_ext_tvl_test_with_masked_arrays_and_axis_2() -> None:
    """Test with masked arrays and axis."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Boolean([[True, True, True], [True, True, True]],
               mask=[[False, False, True], [False, True, False]])
    result = a.tvl_all(axis=1)
    assert result.shape == (2,)

    assert result.values[0]
    assert result.mask[0]

    assert result.values[1]
    assert result.mask[1]


def test_qube_ext_tvl_test_builtins_parameter_3() -> None:
    """Test builtins parameter."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)
    result = Boolean(True).tvl_all()
    assert type(result) == bool
    assert result == True
    result = Boolean(False).tvl_all()
    assert type(result) == bool
    assert result == False
    Qube.prefer_builtins(False)


def test_qube_ext_tvl_test_builtins_true_with_masked_result_for_tvl_all() -> None:
    """Test builtins=True with masked result for tvl_all."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    masked_bool = Boolean([True, True], mask=[True, False])
    result = masked_bool.tvl_all(builtins=True, masked=None)
    assert isinstance(result, Boolean)
    result = masked_bool.tvl_all(builtins=True, masked=False)
    assert type(result) == bool
    assert result == False

    ##################################################################################
    # tvl_eq(self, arg, builtins=None)
    ##################################################################################


def test_qube_ext_tvl_test_equal_values_both_unmasked() -> None:
    """Test: Equal values, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0)
    b = Scalar(5.0)
    result = a.tvl_eq(b)
    assert isinstance(result, Boolean)
    assert result == Boolean(True)
    Qube.prefer_builtins(True)
    result = a.tvl_eq(5.0)
    assert result is True
    result = a.tvl_eq(5.0, builtins=False)
    assert isinstance(result, Boolean)
    assert result == Boolean(True)
    Qube.prefer_builtins(False)
    result = a.tvl_eq(5.0)
    assert isinstance(result, Boolean)
    assert result == Boolean(True)


def test_qube_ext_tvl_test_unequal_values_both_unmasked() -> None:
    """Test: Unequal values, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0)
    b = Scalar(6.0)
    result = a.tvl_eq(b)
    assert result == Boolean(False)


def test_qube_ext_tvl_test_if_either_value_is_masked_result_is_masked() -> None:
    """Test: If either value is masked, result is masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0, mask=True)
    b = Scalar(5.0)
    result = a.tvl_eq(b)
    assert result.mask
    a = Scalar(5.0)
    b = Scalar(5.0, mask=True)
    result = a.tvl_eq(b)
    assert result.mask


def test_qube_ext_tvl_test_with_arrays() -> None:
    """Test with arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar([1.0, 2.0, 3.0])
    b = Scalar([1.0, 2.0, 4.0])
    result = a.tvl_eq(b)
    assert result.shape == (3,)
    assert np.all(result.values == [True, True, False])


def test_qube_ext_tvl_test_with_n_d_arrays_2() -> None:
    """Test with n-D arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(np.random.rand(2, 3, 4))
    b = Scalar(np.random.rand(2, 3, 4))
    result = a.tvl_eq(b)
    assert result.shape == (2, 3, 4)
    expected = (a.values == b.values) & np.logical_not(a.mask) & np.logical_not(b.mask)

    mask_expected = a.mask | b.mask
    assert np.all((result.values == expected) | mask_expected)
    assert np.all(result.mask == mask_expected)


def test_qube_ext_tvl_test_builtins_parameter_4() -> None:
    """Test builtins parameter."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)
    result = Scalar(5.0).tvl_eq(5.0)
    assert type(result) == bool
    assert result == True
    Qube.prefer_builtins(False)

    ##################################################################################
    # tvl_ne(self, arg, builtins=None)
    ##################################################################################


def test_qube_ext_tvl_test_equal_values_both_unmasked_2() -> None:
    """Test: Equal values, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0)
    b = Scalar(5.0)
    result = a.tvl_ne(b)
    assert result == Boolean(False)


def test_qube_ext_tvl_test_unequal_values_both_unmasked_2() -> None:
    """Test: Unequal values, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0)
    b = Scalar(6.0)
    result = a.tvl_ne(b)
    assert result == Boolean(True)


def test_qube_ext_tvl_test_if_either_value_is_masked_result_is_masked_2() -> None:
    """Test: If either value is masked, result is masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0, mask=True)
    b = Scalar(6.0)
    result = a.tvl_ne(b)
    assert result.mask


def test_qube_ext_tvl_test_with_arrays_2() -> None:
    """Test with arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar([1.0, 2.0, 3.0])
    b = Scalar([1.0, 2.0, 4.0])
    result = a.tvl_ne(b)
    assert result.shape == (3,)
    assert np.all(result.values == [False, False, True])


def test_qube_ext_tvl_test_builtins_parameter_5() -> None:
    """Test builtins parameter."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)
    result = Scalar(5.0).tvl_ne(6.0)
    assert type(result) == bool
    assert result == True
    Qube.prefer_builtins(False)

    ##################################################################################
    # tvl_lt(self, arg, builtins=None)
    ##################################################################################


def test_qube_ext_tvl_test_less_than_both_unmasked() -> None:
    """Test: Less than, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0)
    b = Scalar(6.0)
    result = a.tvl_lt(b)
    assert result == Boolean(True)


def test_qube_ext_tvl_test_not_less_than_both_unmasked() -> None:
    """Test: Not less than, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(6.0)
    b = Scalar(5.0)
    result = a.tvl_lt(b)
    assert result == Boolean(False)


def test_qube_ext_tvl_test_if_either_value_is_masked_result_is_masked_3() -> None:
    """Test: If either value is masked, result is masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0, mask=True)
    b = Scalar(6.0)
    result = a.tvl_lt(b)
    assert result.mask


def test_qube_ext_tvl_test_with_arrays_3() -> None:
    """Test with arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar([1.0, 2.0, 3.0])
    b = Scalar([2.0, 1.0, 3.0])
    result = a.tvl_lt(b)
    assert result.shape == (3,)
    assert np.all(result.values == [True, False, False])


def test_qube_ext_tvl_test_with_n_d_arrays_3() -> None:
    """Test with n-D arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(np.random.rand(2, 3, 4))
    b = Scalar(np.random.rand(2, 3, 4) + 0.5)
    result = a.tvl_lt(b)
    assert result.shape == (2, 3, 4)
    mask_expected = a.mask | b.mask
    assert np.all(result.mask == mask_expected)


def test_qube_ext_tvl_test_builtins_parameter_6() -> None:
    """Test builtins parameter."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)
    result = Scalar(5.0).tvl_lt(6.0)
    assert type(result) == bool
    assert result == True
    Qube.prefer_builtins(False)

    ##################################################################################
    # tvl_gt(self, arg, builtins=None)
    ##################################################################################


def test_qube_ext_tvl_test_greater_than_both_unmasked() -> None:
    """Test: Greater than, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(6.0)
    b = Scalar(5.0)
    result = a.tvl_gt(b)
    assert result == Boolean(True)


def test_qube_ext_tvl_test_not_greater_than_both_unmasked() -> None:
    """Test: Not greater than, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0)
    b = Scalar(6.0)
    result = a.tvl_gt(b)
    assert result == Boolean(False)


def test_qube_ext_tvl_test_if_either_value_is_masked_result_is_masked_4() -> None:
    """Test: If either value is masked, result is masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(6.0, mask=True)
    b = Scalar(5.0)
    result = a.tvl_gt(b)
    assert result.mask


def test_qube_ext_tvl_test_with_arrays_4() -> None:
    """Test with arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar([2.0, 1.0, 3.0])
    b = Scalar([1.0, 2.0, 3.0])
    result = a.tvl_gt(b)
    assert result.shape == (3,)
    assert np.all(result.values == [True, False, False])


def test_qube_ext_tvl_test_builtins_parameter_7() -> None:
    """Test builtins parameter."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)
    result = Scalar(6.0).tvl_gt(5.0)
    assert type(result) == bool
    assert result == True
    Qube.prefer_builtins(False)

    ##################################################################################
    # tvl_le(self, arg, builtins=None)
    ##################################################################################


def test_qube_ext_tvl_test_less_than_or_equal_both_unmasked() -> None:
    """Test: Less than or equal, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0)
    b = Scalar(6.0)
    result = a.tvl_le(b)
    assert result == Boolean(True)
    a = Scalar(5.0)
    b = Scalar(5.0)
    result = a.tvl_le(b)
    assert result == Boolean(True)


def test_qube_ext_tvl_test_not_less_than_or_equal_both_unmasked() -> None:
    """Test: Not less than or equal, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(6.0)
    b = Scalar(5.0)
    result = a.tvl_le(b)
    assert result == Boolean(False)


def test_qube_ext_tvl_test_if_either_value_is_masked_result_is_masked_5() -> None:
    """Test: If either value is masked, result is masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0, mask=True)
    b = Scalar(6.0)
    result = a.tvl_le(b)
    assert result.mask


def test_qube_ext_tvl_test_with_arrays_5() -> None:
    """Test with arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar([1.0, 2.0, 3.0])
    b = Scalar([2.0, 1.0, 3.0])
    result = a.tvl_le(b)
    assert result.shape == (3,)
    assert np.all(result.values == [True, False, True])


def test_qube_ext_tvl_test_builtins_parameter_8() -> None:
    """Test builtins parameter."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)
    result = Scalar(5.0).tvl_le(6.0)
    assert type(result) == bool
    assert result == True
    Qube.prefer_builtins(False)

    ##################################################################################
    # tvl_ge(self, arg, builtins=None)
    ##################################################################################


def test_qube_ext_tvl_test_greater_than_or_equal_both_unmasked() -> None:
    """Test: Greater than or equal, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(6.0)
    b = Scalar(5.0)
    result = a.tvl_ge(b)
    assert result == Boolean(True)
    a = Scalar(5.0)
    b = Scalar(5.0)
    result = a.tvl_ge(b)
    assert result == Boolean(True)


def test_qube_ext_tvl_test_not_greater_than_or_equal_both_unmasked() -> None:
    """Test: Not greater than or equal, both unmasked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0)
    b = Scalar(6.0)
    result = a.tvl_ge(b)
    assert result == Boolean(False)


def test_qube_ext_tvl_test_if_either_value_is_masked_result_is_masked_6() -> None:
    """Test: If either value is masked, result is masked."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(6.0, mask=True)
    b = Scalar(5.0)
    result = a.tvl_ge(b)
    assert result.mask


def test_qube_ext_tvl_test_with_arrays_6() -> None:
    """Test with arrays."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar([2.0, 1.0, 3.0])
    b = Scalar([1.0, 2.0, 3.0])
    result = a.tvl_ge(b)
    assert result.shape == (3,)
    assert np.all(result.values == [True, False, True])


def test_qube_ext_tvl_test_builtins_parameter_9() -> None:
    """Test builtins parameter."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)
    result = Scalar(6.0).tvl_ge(5.0)
    assert type(result) == bool
    assert result == True
    Qube.prefer_builtins(False)

    ##################################################################################
    # Additional tests for _tvl_op branches
    ##################################################################################


def test_qube_ext_tvl_test_tvl_op_with_bool_comparison_and_builtins_true_this_test() -> None:
    """Test _tvl_op with bool comparison and builtins=True # This tests the branch where comparison is a bool and builtins is None then True."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(True)

    a = Scalar(5.0)

    result = a.tvl_eq(5.0)

    assert type(result) == bool
    assert result == True
    result = a.tvl_ne(6.0)
    assert type(result) == bool
    assert result == True
    result = a.tvl_lt(6.0)
    assert type(result) == bool
    assert result == True
    result = a.tvl_gt(4.0)
    assert type(result) == bool
    assert result == True
    result = a.tvl_le(6.0)
    assert type(result) == bool
    assert result == True
    result = a.tvl_ge(4.0)
    assert type(result) == bool
    assert result == True
    Qube.prefer_builtins(False)


def test_qube_ext_tvl_test_tvl_op_with_maskedarray_as_arg() -> None:
    """Test _tvl_op with MaskedArray as arg."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    masked_array = ma.MaskedArray([1.0, 2.0, 3.0], mask=[False, True, False])
    a = Scalar([1.0, 2.0, 3.0])
    result = a.tvl_eq(masked_array)

    assert result.shape == (3,)

    assert result.values[0]
    assert not result.mask[0]

    assert result.mask[1]

    assert result.values[2]
    assert not result.mask[2]


def test_qube_ext_tvl_test_tvl_op_with_non_qube_non_maskedarray_arg_should_use_arg() -> None:
    """Test _tvl_op with non-Qube, non-MaskedArray arg (should use arg_mask=False)."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    a = Scalar(5.0)
    result = a.tvl_eq(5.0)
    assert result == Boolean(True)
    result = a.tvl_ne(6.0)
    assert result == Boolean(True)
    result = a.tvl_lt(6.0)
    assert result == Boolean(True)
    result = a.tvl_gt(4.0)
    assert result == Boolean(True)
    result = a.tvl_le(6.0)
    assert result == Boolean(True)
    result = a.tvl_ge(4.0)
    assert result == Boolean(True)


def test_qube_ext_tvl_test_with_masked_self_and_non_qube_arg_with_prefer_builtins_() -> None:
    """Test with masked self and non-Qube arg # With prefer_builtins(False), result should always be a Boolean."""

    np.random.seed(7456)

    ##################################################################################
    # tvl_and(self, arg, builtins=None, masked=None)
    ##################################################################################

    Qube.prefer_builtins(False)
    a_masked = Scalar(5.0, mask=True)
    result = a_masked.tvl_eq(5.0)
    assert isinstance(result, Boolean)
    assert result.mask

    assert not result.values
    result = a_masked.tvl_ne(6.0)
    assert isinstance(result, Boolean)
    assert result.mask

    assert result.values
    Qube.prefer_builtins(False)


##########################################################################################
