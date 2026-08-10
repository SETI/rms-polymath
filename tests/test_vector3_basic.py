##########################################################################################
# tests/test_vector3_basic.py
# Vector3 basic construction, factory methods, static methods, and class constants
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Vector3, Matrix, Vector


def test_vector3_basic_test_basic_construction() -> None:
    """Test basic construction."""

    np.random.seed(2599)

    v1 = Vector3([1., 2., 3.])
    assert v1.shape == ()
    assert v1.item == (3,)
    assert v1.numer == (3,)
    assert np.allclose(v1.vals, [1., 2., 3.])

    v2 = Vector3([4., 5., 6.])
    assert np.allclose(v2.vals, [4., 5., 6.])

    v3 = Vector3((7., 8., 9.))
    assert np.allclose(v3.vals, [7., 8., 9.])

    v4 = Vector3(np.array([10., 11., 12.]))
    assert np.allclose(v4.vals, [10., 11., 12.])

    with pytest.raises(ValueError):
        Vector3(np.random.randn(3, 4, 5))
    with pytest.raises(ValueError):
        Vector3(1.)
    with pytest.raises(ValueError):
        Vector3([1., 2.])
    with pytest.raises(ValueError):
        Vector3([1., 2., 3., 4.])

    v_bool = Vector3([True, True, False])
    assert np.allclose(v_bool.vals, [1., 1., 0.])

    v7 = Vector3.zeros((2, 3))
    assert v7.shape == (2, 3)
    assert v7.vals.shape == (2, 3, 3)
    assert v7.vals.dtype.kind == 'f'
    assert np.all(v7.vals == 0)
    v8 = Vector3.zeros((2, 3), dtype='float')
    assert v8.shape == (2, 3)
    assert v8.vals.shape == (2, 3, 3)
    assert v8.vals.dtype.kind == 'f'
    assert np.all(v8.vals == 0)
    v9 = Vector3.zeros((2, 2), mask=[[0, 1], [0, 0]])
    assert v9.shape == (2, 2)
    assert v9.vals.shape == (2, 2, 3)
    assert np.all(v9.vals == 0)
    assert np.all(v9.mask == [[0, 1], [0, 0]])
    v10 = Vector3.zeros((2, 2), denom=(3, 3))
    assert v10.shape == (2, 2)
    assert v10.vals.shape == (2, 2, 3, 3, 3)
    assert np.all(v10.vals == 0)
    with pytest.raises(ValueError):
        Vector3.zeros((2, 3), numer=(4,))


def test_vector3_basic_test_ones() -> None:
    """Test ones."""

    np.random.seed(2599)

    v11 = Vector3.ones((2, 3))
    assert v11.shape == (2, 3)
    assert v11.vals.shape == (2, 3, 3)
    assert v11.vals.dtype.kind == 'f'
    assert np.all(v11.vals == 1)
    v12 = Vector3.ones((2, 2), mask=[[0, 1], [0, 0]])
    assert v12.shape == (2, 2)
    assert v12.vals.shape == (2, 2, 3)
    assert np.all(v12.vals == 1)
    assert np.all(v12.mask == [[0, 1], [0, 0]])


def test_vector3_basic_test_filled() -> None:
    """Test filled."""

    np.random.seed(2599)

    v13 = Vector3.filled((2, 3), 7.)
    assert v13.shape == (2, 3)
    assert v13.vals.shape == (2, 3, 3)
    assert np.all(v13.vals == 7)
    v14 = Vector3.filled((2, 2), (1., 2., 3.))
    assert v14.shape == (2, 2)
    assert v14.vals.shape == (2, 2, 3)
    assert np.all(v14.vals[..., 0] == 1)
    assert np.all(v14.vals[..., 1] == 2)
    assert np.all(v14.vals[..., 2] == 3)


def test_vector3_basic_test_as_vector3_static_method() -> None:
    """Test as_vector3 static method."""

    np.random.seed(2599)

    v15 = Vector3([1., 2., 3.])
    v15_conv = Vector3.as_vector3(v15)
    assert type(v15_conv) == Vector3
    assert np.allclose(v15_conv.vals, [1., 2., 3.])


def test_vector3_basic_test_as_vector3_with_vector() -> None:
    """Test as_vector3 with Vector."""

    np.random.seed(2599)

    v16 = Vector([1., 2., 3.])
    v16_conv = Vector3.as_vector3(v16)
    assert type(v16_conv) == Vector3
    assert np.allclose(v16_conv.vals, [1., 2., 3.])


def test_vector3_basic_test_as_vector3_with_array() -> None:
    """Test as_vector3 with array."""

    np.random.seed(2599)

    v17_conv = Vector3.as_vector3([4., 5., 6.])
    assert type(v17_conv) == Vector3
    assert np.allclose(v17_conv.vals, [4., 5., 6.])


def test_vector3_basic_test_as_vector3_with_1x3_matrix() -> None:
    """Test as_vector3 with 1x3 Matrix."""

    np.random.seed(2599)

    m1x3 = Matrix([[1., 2., 3.]])
    assert m1x3._numer == (1, 3)
    v1x3_conv = Vector3.as_vector3(m1x3)
    assert type(v1x3_conv) == Vector3
    assert np.allclose(v1x3_conv.vals, [1., 2., 3.])


def test_vector3_basic_test_as_vector3_with_3x1_matrix() -> None:
    """Test as_vector3 with 3x1 Matrix."""

    np.random.seed(2599)

    m3x1 = Matrix([[1.], [2.], [3.]])
    assert m3x1._numer == (3, 1)
    v3x1_conv = Vector3.as_vector3(m3x1)
    assert type(v3x1_conv) == Vector3
    assert np.allclose(v3x1_conv.vals, [1., 2., 3.])


def test_vector3_basic_test_as_vector3_with_n_d_1x3_matrix() -> None:
    """Test as_vector3 with n-D 1x3 Matrix."""

    np.random.seed(2599)

    m1x3_nd = Matrix([[[1., 2., 3.]], [[4., 5., 6.]]])
    assert m1x3_nd.shape == (2,)
    assert m1x3_nd._numer == (1, 3)
    v1x3_nd_conv = Vector3.as_vector3(m1x3_nd)
    assert type(v1x3_nd_conv) == Vector3
    assert v1x3_nd_conv.shape == (2,)
    assert np.allclose(v1x3_nd_conv.vals[0], [1., 2., 3.])
    assert np.allclose(v1x3_nd_conv.vals[1], [4., 5., 6.])


def test_vector3_basic_test_as_vector3_with_qube_rank_1_and_first_numerator_dimensi() -> None:
    """Test as_vector3 with Qube rank > 1 and first numerator dimension == 3 # Create a Vector with shape that has rank > 1 and first numer dim == 3 # This would be a Vector with drank > 0, where the first numer dim is 3 # Actually, let's create a Matrix with shape (3, N) where N > 1 # But wait, for line 53, we need arg.rank > 1 and arg._numer[0] == 3 # rank = nrank + drank, so we need nrank + drank > 1 and _numer[0] == 3 # For a Matrix with _numer = (3, 4), we have nrank=2, so rank=2 > 1, and _numer[0] == 3."""

    np.random.seed(2599)

    m3x4 = Matrix(np.random.randn(2, 3, 4))  # shape (2,), numer (3, 4)
    assert m3x4.shape == (2,)
    assert m3x4._numer == (3, 4)
    assert m3x4.rank == 2  # nrank=2
    assert m3x4._numer[0] == 3
    v3x4_conv = Vector3.as_vector3(m3x4)
    assert type(v3x4_conv) == Vector3

    assert v3x4_conv.shape == (2,)
    assert v3x4_conv.item == (3, 4)  # numer=(3,), denom=(4,)
    assert v3x4_conv.numer == (3,)
    assert v3x4_conv.denom == (4,)


def test_vector3_basic_test_from_scalars_static_method() -> None:
    """Test from_scalars static method."""

    np.random.seed(2599)

    x = Scalar(1.)
    y = Scalar(2.)
    z = Scalar(3.)
    v18 = Vector3.from_scalars(x, y, z)
    assert type(v18) == Vector3
    assert v18.shape == ()
    assert np.allclose(v18.vals, [1., 2., 3.])


def test_vector3_basic_test_from_scalars_with_n_d_scalars() -> None:
    """Test from_scalars with n-D scalars."""

    np.random.seed(2599)

    x_2d = Scalar([[1., 2.], [3., 4.]])
    y_2d = Scalar([[5., 6.], [7., 8.]])
    z_2d = Scalar([[9., 10.], [11., 12.]])
    v19 = Vector3.from_scalars(x_2d, y_2d, z_2d)
    assert v19.shape == (2, 2)
    assert np.allclose(v19.vals[0, 0], [1., 5., 9.])
    assert np.allclose(v19.vals[0, 1], [2., 6., 10.])


def test_vector3_basic_test_from_scalars_with_zero() -> None:
    """Test from_scalars with zero."""

    np.random.seed(2599)

    v20 = Vector3.from_scalars(1., 0., 3.)
    assert np.allclose(v20.vals, [1., 0., 3.])


def test_vector3_basic_test_from_scalars_with_none_docstring_says_none_is_converted() -> None:
    """Test from_scalars with None (docstring says None is converted to zero Scalar)."""

    np.random.seed(2599)

    v20_none = Vector3.from_scalars(1., None, 3.)
    assert np.allclose(v20_none.vals, [1., 0., 3.])


def test_vector3_basic_test_from_scalars_with_none_and_n_d_scalars() -> None:
    """Test from_scalars with None and n-D scalars."""

    np.random.seed(2599)

    x_nd = Scalar([[1., 2.], [3., 4.]], drank=1)
    y_nd = Scalar([[5., 6.], [7., 8.]], drank=1)
    v20_none_nd = Vector3.from_scalars(x_nd, None, y_nd)
    assert v20_none_nd.shape == (2,)
    assert v20_none_nd.denom == (2,)  # Should match the denominator of x_nd and y_nd

    assert np.allclose(v20_none_nd.vals[0, :, 0], [1., 0., 5.])


def test_vector3_basic_test_from_scalars_with_all_none() -> None:
    """Test from_scalars with all None."""

    np.random.seed(2599)

    v_all_none = Vector3.from_scalars(None, None, None)
    assert type(v_all_none) == Vector3
    assert v_all_none.shape == ()
    assert np.allclose(v_all_none.vals, [0., 0., 0.])


def test_vector3_basic_test_from_scalars_with_x_none() -> None:
    """Test from_scalars with x=None."""

    np.random.seed(2599)

    v_x_none = Vector3.from_scalars(None, 2., 3.)
    assert type(v_x_none) == Vector3
    assert v_x_none.shape == ()
    assert np.allclose(v_x_none.vals, [0., 2., 3.])


def test_vector3_basic_test_from_scalars_with_z_none() -> None:
    """Test from_scalars with z=None."""

    np.random.seed(2599)

    v_z_none = Vector3.from_scalars(1., 2., None)
    assert type(v_z_none) == Vector3
    assert v_z_none.shape == ()
    assert np.allclose(v_z_none.vals, [1., 2., 0.])


def test_vector3_basic_test_from_scalars_with_exactly_1_non_none_arg_skips_if_block() -> None:
    """Test from_scalars with exactly 1 non-None arg (skips if block at line 108, goes directly to 110) # This tests the case where len(scalars) = 1, so the if len(scalars) > 1: block is skipped."""

    np.random.seed(2599)

    v_one_arg = Vector3.from_scalars(None, 2., None)
    assert type(v_one_arg) == Vector3
    assert v_one_arg.shape == ()
    assert np.allclose(v_one_arg.vals, [0., 2., 0.])


def test_vector3_basic_test_from_scalars_with_multiple_scalars_requiring_broadcasti() -> None:
    """Test from_scalars with multiple scalars requiring broadcasting # Create scalars with different shapes that need broadcasting."""

    np.random.seed(2599)

    x_broad = Scalar([1., 2.])  # shape (2,)
    y_broad = Scalar([[3.], [4.]])  # shape (2, 1)
    z_broad = Scalar(5.)  # shape ()

    v_broad = Vector3.from_scalars(x_broad, y_broad, z_broad)
    assert type(v_broad) == Vector3
    assert v_broad.shape == (2, 2)

    assert np.allclose(v_broad.vals[0, 0], [1., 3., 5.])
    assert np.allclose(v_broad.vals[0, 1], [2., 3., 5.])
    assert np.allclose(v_broad.vals[1, 0], [1., 4., 5.])
    assert np.allclose(v_broad.vals[1, 1], [2., 4., 5.])


def test_vector3_basic_test_from_scalars_with_broadcasting_and_none_x_is_none_y_and() -> None:
    """Test from_scalars with broadcasting and None # x is None, y and z need broadcasting - this ensures len(scalars) = 2, triggering line 108."""

    np.random.seed(2599)

    y_broad2 = Scalar([3., 4.])  # shape (2,)
    z_broad2 = Scalar([[5.], [6.]])  # shape (2, 1)
    v_broad_none = Vector3.from_scalars(None, y_broad2, z_broad2)
    assert type(v_broad_none) == Vector3
    assert v_broad_none.shape == (2, 2)

    assert np.allclose(v_broad_none.vals[:, :, 0], 0.)

    assert np.allclose(v_broad_none.vals[0, 0], [0., 3., 5.])
    assert np.allclose(v_broad_none.vals[0, 1], [0., 4., 5.])


def test_vector3_basic_test_from_scalars_with_exactly_2_non_none_args_that_need_bro() -> None:
    """Test from_scalars with exactly 2 non-None args that need broadcasting # This explicitly tests the case where len(scalars) = 2, ensuring the if block is entered # Case 1: x=None, y and z have different shapes requiring broadcast."""

    np.random.seed(2599)

    y_broad3 = Scalar([1., 2.])  # shape (2,)
    z_broad3 = Scalar([[3.], [4.]])  # shape (2, 1) - different shape requires broadcast
    v_broad2 = Vector3.from_scalars(None, y_broad3, z_broad3)
    assert type(v_broad2) == Vector3
    assert v_broad2.shape == (2, 2)  # Broadcast result: (2,) and (2,1) -> (2,2)

    assert np.allclose(v_broad2.vals[0, 0], [0., 1., 3.])
    assert np.allclose(v_broad2.vals[0, 1], [0., 2., 3.])
    assert np.allclose(v_broad2.vals[1, 0], [0., 1., 4.])
    assert np.allclose(v_broad2.vals[1, 1], [0., 2., 4.])


def test_vector3_basic_case_2_y_none_x_and_z_have_different_shapes_requiring_broadc() -> None:
    """Case 2: y=None, x and z have different shapes requiring broadcast."""

    np.random.seed(2599)

    x_broad4 = Scalar([1., 2.])  # shape (2,)
    z_broad4 = Scalar([[3.], [4.]])  # shape (2, 1)
    v_broad3 = Vector3.from_scalars(x_broad4, None, z_broad4)
    assert type(v_broad3) == Vector3
    assert v_broad3.shape == (2, 2)

    assert np.allclose(v_broad3.vals[0, 0], [1., 0., 3.])
    assert np.allclose(v_broad3.vals[0, 1], [2., 0., 3.])
    assert np.allclose(v_broad3.vals[1, 0], [1., 0., 4.])
    assert np.allclose(v_broad3.vals[1, 1], [2., 0., 4.])


def test_vector3_basic_case_3_all_three_non_none_but_with_different_shapes_requirin() -> None:
    """Case 3: All three non-None, but with different shapes requiring broadcast # This ensures len(scalars) = 3, which is > 1, so should enter the if block."""

    np.random.seed(2599)

    x_broad5 = Scalar([1., 2.])  # shape (2,)
    y_broad5 = Scalar([[3.], [4.]])  # shape (2, 1)
    z_broad5 = Scalar(5.)  # shape ()
    v_broad4 = Vector3.from_scalars(x_broad5, y_broad5, z_broad5)
    assert type(v_broad4) == Vector3
    assert v_broad4.shape == (2, 2)  # Broadcast: (2,), (2,1), () -> (2,2)

    assert np.allclose(v_broad4.vals[0, 0], [1., 3., 5.])
    assert np.allclose(v_broad4.vals[0, 1], [2., 3., 5.])
    assert np.allclose(v_broad4.vals[1, 0], [1., 4., 5.])
    assert np.allclose(v_broad4.vals[1, 1], [2., 4., 5.])


def test_vector3_basic_test_from_ra_dec_length_static_method() -> None:
    """Test from_ra_dec_length static method."""

    np.random.seed(2599)

    ra = Scalar(0.)  # along x-axis
    dec = Scalar(0.)  # in equatorial plane
    length = Scalar(1.)
    v21 = Vector3.from_ra_dec_length(ra, dec, length)
    assert type(v21) == Vector3

    assert np.allclose(v21.vals, [1., 0., 0.], atol=1e-10)

    v22 = Vector3.from_ra_dec_length(ra, dec)
    assert np.allclose(v22.vals, [1., 0., 0.], atol=1e-10)


def test_vector3_basic_test_from_cylindrical_static_method() -> None:
    """Test from_cylindrical static method."""

    np.random.seed(2599)

    radius = Scalar(1.)
    longitude = Scalar(0.)  # along x-axis
    z_coord = Scalar(0.)
    v26 = Vector3.from_cylindrical(radius, longitude, z_coord)
    assert type(v26) == Vector3

    assert np.allclose(v26.vals, [1., 0., 0.], atol=1e-10)

    v27 = Vector3.from_cylindrical(radius, longitude)
    assert np.allclose(v27.vals, [1., 0., 0.], atol=1e-10)


def test_vector3_basic_test_class_constants() -> None:
    """Test class constants."""

    np.random.seed(2599)

    assert type(Vector3.ZERO) == Vector3
    assert np.allclose(Vector3.ZERO.vals, [0., 0., 0.])
    assert Vector3.ZERO.readonly
    assert type(Vector3.ONES) == Vector3
    assert np.allclose(Vector3.ONES.vals, [1., 1., 1.])
    assert Vector3.ONES.readonly
    assert type(Vector3.XAXIS) == Vector3
    assert np.allclose(Vector3.XAXIS.vals, [1., 0., 0.])
    assert Vector3.XAXIS.readonly
    assert type(Vector3.YAXIS) == Vector3
    assert np.allclose(Vector3.YAXIS.vals, [0., 1., 0.])
    assert Vector3.YAXIS.readonly
    assert type(Vector3.ZAXIS) == Vector3
    assert np.allclose(Vector3.ZAXIS.vals, [0., 0., 1.])
    assert Vector3.ZAXIS.readonly
    assert type(Vector3.MASKED) == Vector3
    assert Vector3.MASKED.mask
    assert Vector3.MASKED.readonly
    assert type(Vector3.AXES) == tuple
    assert len(Vector3.AXES) == 3
    assert Vector3.AXES[0] == Vector3.XAXIS
    assert Vector3.AXES[1] == Vector3.YAXIS
    assert Vector3.AXES[2] == Vector3.ZAXIS


def test_vector3_basic_test_that_vector3_only_accepts_floats_not_ints_integers_shou() -> None:
    """Test that Vector3 only accepts floats (not ints) # Integers should be coerced to float."""

    np.random.seed(2599)

    v84 = Vector3([1, 2, 3])
    assert v84.vals.dtype.kind == 'f'


def test_vector3_basic_test_with_mask() -> None:
    """Test with mask."""

    np.random.seed(2599)

    v85 = Vector3([1., 2., 3.], mask=False)
    assert not v85.mask
    v86 = Vector3([1., 2., 3.], mask=True)
    assert v86.mask


##########################################################################################
