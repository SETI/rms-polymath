##########################################################################################
# tests/test_qube_ext_item_ops.py
#
# Comprehensive unit tests for item operations based on docstrings in item_ops.py
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Matrix, Matrix3, Qube, Scalar, Vector, Vector3


def test_qube_ext_item_ops_simple_case_extract_from_1_d_numerator() -> None:
    """Simple case: extract from 1-D numerator."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector([1., 2., 3.])
    b = a.extract_numer(0, 1)
    assert b.shape == ()
    assert b.numer == ()
    assert b == 2.

    a = Matrix(np.arange(12).reshape(2, 3, 2))  # shape (2,), numer (3, 2)
    b = a.extract_numer(0, 1)  # Extract index 1 from first numerator axis
    assert b.shape == (2,)
    assert b.numer == (2,)
    assert np.allclose(b.values[0], a.values[0, 1, :])
    assert np.allclose(b.values[1], a.values[1, 1, :])

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    b = a.extract_numer(-2, 1)  # Same as axis 0
    assert b.shape == (2,)
    assert b.numer == (2,)
    assert np.allclose(b.values[0], a.values[0, 1, :])

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    b = a.extract_numer(0, 1, classes=Vector)
    assert type(b) == Vector

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    da_dt = Matrix(np.arange(12).reshape(2, 3, 2, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.extract_numer(0, 1, recursive=True)
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == (2,)
    assert b.d_dt.numer == (2,)

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    da_dt = Matrix(np.arange(12).reshape(2, 3, 2, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.extract_numer(0, 1, recursive=False)
    assert not hasattr(b, 'd_dt')

    a = Vector([1., 2., 3.])  # shape (), numer (3,), so only axis 0 exists
    with pytest.raises(ValueError):
        a.extract_numer(1, 0)  # axis 1 doesn't exist (only axis 0)

    ##################################################################################
    # extract_denom()
    ##################################################################################

    a = Vector(np.arange(9).reshape(3, 3), drank=1)  # shape (3,), numer (3,), denom (3,)
    assert a.denom == (3,)
    b = a.extract_denom(0, 1)
    assert b.shape == ()  # Extracting from denominator reduces shape
    assert b.numer == (3,)
    assert b.denom == ()  # After extraction, denom becomes empty

    assert np.allclose(b.values, a.values[:, 1])

    a = Vector(np.arange(24).reshape(2, 3, 2, 2), drank=2)  # shape (2,), numer (3,), denom (2, 2)
    b = a.extract_denom(0, 1)  # Extract index 1 from first denominator axis
    assert b.shape == (2,)
    assert b.numer == (3,)
    assert b.denom == (2,)
    assert np.allclose(b.values[0], a.values[0, :, 1, :])

    a = Vector(np.arange(24).reshape(2, 3, 2, 2), drank=2)
    b = a.extract_denom(-2, 1)  # Same as axis 0
    assert b.shape == (2,)
    assert b.denom == (2,)
    assert np.allclose(b.values[0], a.values[0, :, 1, :])

    a = Vector(np.arange(24).reshape(2, 3, 2, 2), drank=2)
    b = a.extract_denom(0, 1, classes=(Vector,))
    assert type(b) == Vector

    a = Vector(np.arange(9).reshape(3, 3), drank=1)
    with pytest.raises(ValueError):
        a.extract_denom(1, 0)  # axis 1 doesn't exist (only 1 denom axis)

    ##################################################################################
    # extract_denoms()
    ##################################################################################

    a = Vector(np.arange(9).reshape(3, 3), drank=1)  # shape (3,), numer (3,), denom (3,)
    objects = a.extract_denoms()
    assert len(objects) == 3
    assert np.allclose(objects[0].values, a.values[:, 0])
    assert np.allclose(objects[1].values, a.values[:, 1])
    assert np.allclose(objects[2].values, a.values[:, 2])
    assert objects[0].drank == 0
    assert objects[1].drank == 0
    assert objects[2].drank == 0

    a = Vector(np.arange(12).reshape(2, 3, 2), drank=1)  # shape (2,), numer (3,), denom (2,)
    objects = a.extract_denoms()
    assert len(objects) == 2
    assert objects[0].shape == (2,)
    assert objects[0].numer == (3,)
    assert objects[0].drank == 0
    assert np.allclose(objects[0].values, a.values[:, :, 0])
    assert np.allclose(objects[1].values, a.values[:, :, 1])

    a = Vector([1., 2., 3.])
    objects = a.extract_denoms()
    assert len(objects) == 1
    assert objects[0] == a

    a = Vector(np.arange(18).reshape(3, 2, 3), drank=2)  # shape (3,), numer (2,), denom (3, 3)
    with pytest.raises(ValueError):
        a.extract_denoms()  # extract_denoms requires drank=1

    ##################################################################################
    # slice_numer()
    ##################################################################################

    a = Vector([1., 2., 3., 4., 5.])
    b = a.slice_numer(0, 1, 3)  # Slice indices 1 to 3
    assert b.shape == ()
    assert b.numer == (2,)
    assert np.allclose(b.values, [2., 3.])

    a = Matrix(np.arange(24).reshape(2, 4, 3))  # shape (2,), numer (4, 3)
    b = a.slice_numer(0, 1, 3)  # Slice indices 1 to 3 from first numerator axis
    assert b.shape == (2,)
    assert b.numer == (2, 3)
    assert np.allclose(b.values[0], a.values[0, 1:3, :])
    assert np.allclose(b.values[1], a.values[1, 1:3, :])

    a = Matrix(np.arange(24).reshape(2, 4, 3))
    b = a.slice_numer(0, 1, 3, classes=Matrix)
    assert type(b) == Matrix

    a = Matrix(np.arange(24).reshape(2, 4, 3))
    da_dt = Matrix(np.arange(24).reshape(2, 4, 3, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.slice_numer(0, 1, 3, recursive=True)
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == (2,)
    assert b.d_dt.numer == (2, 3)

    a = Matrix(np.arange(24).reshape(2, 4, 3))
    da_dt = Matrix(np.arange(24).reshape(2, 4, 3, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.slice_numer(0, 1, 3, recursive=False)
    assert not hasattr(b, 'd_dt')

    a = Vector([1., 2., 3.])
    with pytest.raises(ValueError):
        a.slice_numer(1, 0, 1)

    ##################################################################################
    # transpose_numer()
    ##################################################################################

    a = Matrix(np.arange(12).reshape(2, 3, 2))  # shape (2,), numer (3, 2)
    b = a.transpose_numer(0, 1)
    assert b.shape == (2,)
    assert b.numer == (2, 3)
    assert np.allclose(b.values[0], a.values[0].T)
    assert np.allclose(b.values[1], a.values[1].T)

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    b = a.transpose_numer(-2, -1)  # Same as (0, 1)
    assert b.numer == (2, 3)
    assert np.allclose(b.values[0], a.values[0].T)

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    da_dt = Matrix(np.arange(12).reshape(2, 3, 2, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.transpose_numer(0, 1, recursive=True)
    assert hasattr(b, 'd_dt')
    assert b.d_dt.numer == (2, 3)

    expected = np.transpose(a.d_dt.values[0], (1, 0, 2))
    assert np.allclose(b.d_dt.values[0], expected)

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    da_dt = Matrix(np.arange(12).reshape(2, 3, 2, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.transpose_numer(0, 1, recursive=False)
    assert not hasattr(b, 'd_dt')

    a = Vector([1., 2., 3.])
    with pytest.raises(ValueError):
        a.transpose_numer(0, 1)  # Only 1 numerator axis

    ##################################################################################
    # reshape_numer()
    ##################################################################################

    a = Vector([1., 2., 3., 4., 5., 6.])
    b = a.reshape_numer((2, 3))
    assert b.shape == ()
    assert b.numer == (2, 3)
    assert np.allclose(b.values.reshape(6), a.values)

    a = Matrix(np.arange(24).reshape(2, 4, 3))  # shape (2,), numer (4, 3) = 12 elements
    b = a.reshape_numer((6, 2))
    assert b.shape == (2,)
    assert b.numer == (6, 2)
    assert np.allclose(b.values.reshape(2, 12), a.values.reshape(2, 12))

    a = Vector([1., 2., 3., 4., 5., 6.])
    b = a.reshape_numer((2, 3), classes=Matrix)
    assert type(b) == Matrix

    a = Matrix(np.arange(24).reshape(2, 4, 3))
    da_dt = Matrix(np.arange(24).reshape(2, 4, 3, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.reshape_numer((6, 2), recursive=True)
    assert hasattr(b, 'd_dt')
    assert b.d_dt.numer == (6, 2)

    a = Matrix(np.arange(24).reshape(2, 4, 3))
    da_dt = Matrix(np.arange(24).reshape(2, 4, 3, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.reshape_numer((6, 2), recursive=False)
    assert not hasattr(b, 'd_dt')

    a = Vector([1., 2., 3., 4., 5., 6.])
    with pytest.raises(ValueError):
        a.reshape_numer((2, 2))  # 4 != 6

    ##################################################################################
    # flatten_numer()
    ##################################################################################

    a = Matrix(np.arange(12).reshape(2, 3, 2))  # shape (2,), numer (3, 2)
    b = a.flatten_numer()
    assert b.shape == (2,)
    assert b.numer == (6,)
    assert np.allclose(b.values[0], a.values[0].flatten())
    assert np.allclose(b.values[1], a.values[1].flatten())

    a = Matrix(np.arange(24).reshape(2, 2, 3, 2), drank=1)  # shape (2,), numer (2, 3) = 6, denom (2,)
    b = a.flatten_numer()
    assert b.shape == (2,)
    assert b.numer == (6,)  # 2 * 3 = 6
    assert b.denom == (2,)

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    b = a.flatten_numer(classes=Vector)
    assert type(b) == Vector

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    da_dt = Matrix(np.arange(12).reshape(2, 3, 2, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.flatten_numer(recursive=True)
    assert hasattr(b, 'd_dt')
    assert b.d_dt.numer == (6,)

    a = Matrix(np.arange(12).reshape(2, 3, 2))
    da_dt = Matrix(np.arange(12).reshape(2, 3, 2, 1), drank=1)
    a.insert_deriv('t', da_dt)
    b = a.flatten_numer(recursive=False)
    assert not hasattr(b, 'd_dt')

    ##################################################################################
    # transpose_denom()
    ##################################################################################

    a = Vector(np.arange(24).reshape(2, 3, 2, 2), drank=2)  # shape (2,), numer (3,), denom (2, 2)
    b = a.transpose_denom(0, 1)
    assert b.shape == (2,)
    assert b.numer == (3,)
    assert b.denom == (2, 2)
    assert np.allclose(b.values[0, :, 0, 0], a.values[0, :, 0, 0])
    assert np.allclose(b.values[0, :, 0, 1], a.values[0, :, 1, 0])
    assert np.allclose(b.values[0, :, 1, 0], a.values[0, :, 0, 1])
    assert np.allclose(b.values[0, :, 1, 1], a.values[0, :, 1, 1])

    a = Vector(np.arange(24).reshape(2, 3, 2, 2), drank=2)
    b = a.transpose_denom(-2, -1)  # Same as (0, 1)
    assert b.denom == (2, 2)

    a = Vector(np.arange(9).reshape(3, 3), drank=1)  # shape (3,), numer (3,), denom (3,)
    with pytest.raises(ValueError):
        a.transpose_denom(0, 1)  # Only 1 denominator axis (axis 1 doesn't exist)

    ##################################################################################
    # reshape_denom()
    ##################################################################################

    a = Vector(np.arange(18).reshape(3, 6), drank=1)  # shape (), numer (3,), denom (6,)
    assert a.denom == (6,)
    b = a.reshape_denom((2, 3))
    assert b.shape == ()  # Shape is preserved (scalar)
    assert b.numer == (3,)  # Numer is preserved
    assert b.denom == (2, 3)  # Denom is reshaped

    assert np.allclose(b.values.reshape(18), a.values.reshape(18))

    a = Vector(np.arange(24).reshape(2, 3, 2, 2), drank=2)  # shape (2,), numer (3,), denom (2, 2) = 4
    b = a.reshape_denom((4,))
    assert b.shape == (2,)
    assert b.numer == (3,)
    assert b.denom == (4,)
    assert np.allclose(b.values.reshape(2, 3, 4), a.values.reshape(2, 3, 4))

    a = Vector(np.arange(18).reshape(3, 6), drank=1)  # shape (3,), numer (3,), denom (6,)
    with pytest.raises(ValueError):
        a.reshape_denom((2, 2))  # 4 != 6

    ##################################################################################
    # flatten_denom()
    ##################################################################################


def test_qube_ext_item_ops_simple_case_flatten_2_d_denominator() -> None:
    """Simple case: flatten 2-D denominator."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(24).reshape(2, 3, 2, 2), drank=2)  # shape (2,), numer (3,), denom (2, 2)
    b = a.flatten_denom()
    assert b.shape == (2,)
    assert b.numer == (3,)
    assert b.denom == (4,)

    assert np.allclose(b.values[0, :, 0], a.values[0, :, 0, 0])
    assert np.allclose(b.values[0, :, 1], a.values[0, :, 0, 1])
    assert np.allclose(b.values[0, :, 2], a.values[0, :, 1, 0])
    assert np.allclose(b.values[0, :, 3], a.values[0, :, 1, 1])


def test_qube_ext_item_ops_complex_n_d_case_flatten_3_d_denominator() -> None:
    """Complex n-D case: flatten 3-D denominator."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(48).reshape(2, 3, 2, 2, 2), drank=3)  # shape (2,), numer (3,), denom (2, 2, 2) = 8
    b = a.flatten_denom()
    assert b.shape == (2,)
    assert b.numer == (3,)
    assert b.denom == (8,)


def test_qube_ext_item_ops_test_with_drank_0() -> None:
    """Test with drank=0."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector([1., 2., 3.])  # shape (), numer (3,), denom ()
    b = a.flatten_denom()

    assert a.shape == b.shape
    assert a.numer == b.numer
    assert b.denom == (1,)  # dsize=0 becomes (1,) after reshape

    ##################################################################################
    # join_items()
    ##################################################################################


def test_qube_ext_item_ops_simple_case_join_1_d_denominator_to_numerator() -> None:
    """Simple case: join 1-D denominator to numerator."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(9).reshape(3, 3), drank=1)  # shape (), numer (3,), denom (3,)
    assert a.numer == (3,)
    assert a.denom == (3,)
    b = a.join_items(Matrix)
    assert b.shape == ()  # Shape is preserved
    assert b.numer == (3, 3)  # numer and denom are joined
    assert b.denom == ()
    assert type(b) == Matrix


def test_qube_ext_item_ops_complex_n_d_case_join_with_shape_for_shape_2_numer_3_denom_2() -> None:
    """Complex n-D case: join with shape # For shape (2,), numer (3,), denom (2,), we need values shape (2, 3, 2) # But 2*3*2 = 12, not 24. Let's use a different size."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(12).reshape(2, 3, 2), drank=1)  # shape (2,), numer (3,), denom (2,)
    b = a.join_items(Matrix)
    assert b.shape == (2,)  # Shape is preserved
    assert b.numer == (3, 2)  # numer and denom are joined
    assert b.denom == ()


def test_qube_ext_item_ops_test_with_classes_parameter_list() -> None:
    """Test with classes parameter (list)."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(9).reshape(3, 3), drank=1)
    b = a.join_items((Boolean, Scalar, Matrix3, Matrix))

    assert type(b) == Matrix3


def test_qube_ext_item_ops_test_with_drank_0_should_return_without_derivatives() -> None:
    """Test with drank=0 (should return without derivatives)."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector([1., 2., 3.])
    b = a.join_items(Matrix)
    assert a.wod == b  # Should return without derivatives

    ##################################################################################
    # split_items()
    ##################################################################################


def test_qube_ext_item_ops_simple_case_split_numerator_to_denominator_use_matrix_which_() -> None:
    """Simple case: split numerator to denominator # Use Matrix which has _NRANK=2, so we can split it."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Matrix(np.arange(24).reshape(2, 3, 4))  # shape (2,), numer (3, 4), denom ()
    b = a.split_items(1, Matrix)  # Keep first 1 numer axis, rest become denom
    assert b.shape == (2,)
    assert b.numer == (3,)  # First numer axis
    assert b.denom == (4,)  # Remaining becomes denom

    assert isinstance(b, Qube)


def test_qube_ext_item_ops_complex_n_d_case_split_with_shape_use_matrix_which_has_nrank() -> None:
    """Complex n-D case: split with shape # Use Matrix which has _NRANK=2, so we can split it properly."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Matrix(np.arange(24).reshape(2, 3, 4))  # shape (2,), numer (3, 4)
    b = a.split_items(1, Vector)  # Keep first 1 numer axis, rest become denom
    assert b.shape == (2,)
    assert b.numer == (3,)  # First numer axis
    assert b.denom == (4,)  # Remaining becomes denom


def test_qube_ext_item_ops_test_with_classes_parameter_use_matrix_which_has_nrank_2_so_() -> None:
    """Test with classes parameter # Use Matrix which has _NRANK=2, so we can split it properly."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Matrix(np.arange(24).reshape(2, 3, 4))  # shape (2,), numer (3, 4)
    b = a.split_items(1, (Boolean, Scalar, Vector3, Vector))

    assert isinstance(b, Qube)

    ##################################################################################
    # swap_items()
    ##################################################################################


def test_qube_ext_item_ops_simple_case_swap_numerator_and_denominator() -> None:
    """Simple case: swap numerator and denominator."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(9).reshape(3, 3), drank=1)  # shape (), numer (3,), denom (3,)
    assert a.numer == (3,)
    assert a.denom == (3,)
    b = a.swap_items(Matrix)
    assert b.shape == ()  # Shape is preserved
    assert b.numer == (3,)  # Swapped from denom
    assert b.denom == (3,)  # Swapped from numer

    assert isinstance(b, Qube)


def test_qube_ext_item_ops_complex_n_d_case_swap_with_different_sizes() -> None:
    """Complex n-D case: swap with different sizes."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(24).reshape(2, 3, 4), drank=1)  # shape (2,), numer (3,), denom (4,)
    b = a.swap_items(Matrix)
    assert b.shape == (2,)
    assert b.numer == (4,)  # Swapped from denom
    assert b.denom == (3,)  # Swapped from numer


def test_qube_ext_item_ops_test_with_classes_parameter() -> None:
    """Test with classes parameter."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(9).reshape(3, 3), drank=1)
    b = a.swap_items((Boolean, Scalar, Matrix3, Matrix))

    assert isinstance(b, Qube)

    ##################################################################################
    # chain()
    ##################################################################################


def test_qube_ext_item_ops_simple_case_chain_multiplication_for_chain_we_need_a_denom_t() -> None:
    """Simple case: chain multiplication # For chain, we need a.denom to match b.numer."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(12).reshape(2, 3, 2), drank=1)  # shape (2,), numer (3,), denom (2,)
    b = Vector(np.arange(12, 24).reshape(2, 2, 3), drank=1)  # shape (2,), numer (2,), denom (3,)
    a = Vector(np.arange(12).reshape(2, 3, 2), drank=1)  # shape (2,), numer (3,), denom (2,)
    b = Vector(np.arange(12).reshape(2, 2, 3), drank=1)  # shape (2,), numer (2,), denom (3,)
    c = a.chain(b)

    assert c.shape == (2,)
    assert type(c) == Vector
    c = a @ b

    assert c.shape == (2,)
    assert type(c) == Vector


def test_qube_ext_item_ops_test_with_matmul_operator_chain_multiplication_for_chain_to_() -> None:
    """Test with __matmul__ operator (chain multiplication) # For chain to work, a.denom must match b.numer."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(12).reshape(2, 3, 2), drank=1)  # shape (2,), numer (3,), denom (2,)
    b = Vector(np.arange(12, 24).reshape(2, 2, 3), drank=1)  # shape (2,), numer (2,), denom (3,)

    c = a.chain(b)

    assert c.shape == (2,)
    assert c.numer == (3,)
    assert c.denom == (3,)


def test_qube_ext_item_ops_complex_n_d_case_different_shapes() -> None:
    """Complex n-D case: different shapes."""

    np.random.seed(8736)

    ##################################################################################
    # extract_numer()
    ##################################################################################

    a = Vector(np.arange(60).reshape(5, 3, 4), drank=1)  # shape (5,), numer (3,), denom (4,)
    b = Vector(np.arange(80).reshape(5, 4, 2, 2), drank=2)  # shape (5,), numer (4,), denom (2, 2)
    c = a.chain(b)

    assert c.shape == (5,)
    assert c.numer == (3,)
    assert c.denom == (2, 2)


##########################################################################################
