##########################################################################################
# tests/test_pair.py
# Pair comprehensive tests
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Pair, Matrix, Vector


def test_pair_test_basic_construction() -> None:
    """Test basic construction."""

    np.random.seed(2599)

    p1 = Pair([1., 2.])
    assert p1.shape == ()
    assert p1.item == (2,)
    assert p1.numer == (2,)
    assert np.allclose(p1.vals, [1., 2.])

    p2 = Pair([4., 5.])
    assert np.allclose(p2.vals, [4., 5.])

    p3 = Pair((7., 8.))
    assert np.allclose(p3.vals, [7., 8.])

    p4 = Pair(np.array([10., 11.]))
    assert np.allclose(p4.vals, [10., 11.])

    p5 = Pair(np.random.randn(2, 3, 2))
    assert p5.shape == (2, 3)
    assert p5.item == (2,)
    assert p5.vals.shape == (2, 3, 2)

    p6 = Pair(np.random.randn(4, 5, 6, 2))
    assert p6.shape == (4, 5, 6)
    assert p6.item == (2,)
    assert p6.vals.shape == (4, 5, 6, 2)

    with pytest.raises(ValueError):
        Pair(np.random.randn(2, 3, 4))
    with pytest.raises(ValueError):
        Pair(1.)
    with pytest.raises(ValueError):
        Pair([1.])
    with pytest.raises(ValueError):
        Pair([1., 2., 3.])

    p7 = Pair.zeros((2, 3))
    assert p7.shape == (2, 3)
    assert p7.vals.shape == (2, 3, 2)
    assert p7.vals.dtype.kind == 'f'
    assert np.all(p7.vals == 0)
    p8 = Pair.zeros((2, 3), dtype='float')
    assert p8.shape == (2, 3)
    assert p8.vals.shape == (2, 3, 2)
    assert p8.vals.dtype.kind == 'f'
    assert np.all(p8.vals == 0)
    p9 = Pair.zeros((2, 2), mask=[[0, 1], [0, 0]])
    assert p9.shape == (2, 2)
    assert p9.vals.shape == (2, 2, 2)
    assert np.all(p9.vals == 0)
    assert np.all(p9.mask == [[0, 1], [0, 0]])
    p10 = Pair.zeros((2, 2), denom=(3, 3))
    assert p10.shape == (2, 2)
    assert p10.vals.shape == (2, 2, 2, 3, 3)
    assert np.all(p10.vals == 0)
    with pytest.raises(ValueError):
        Pair.zeros((2, 3), numer=(3,))

    p11 = Pair.ones((2, 3))
    assert p11.shape == (2, 3)
    assert p11.vals.shape == (2, 3, 2)
    assert p11.vals.dtype.kind == 'f'
    assert np.all(p11.vals == 1)
    p12 = Pair.ones((2, 2), mask=[[0, 1], [0, 0]])
    assert p12.shape == (2, 2)
    assert p12.vals.shape == (2, 2, 2)
    assert np.all(p12.vals == 1)
    assert np.all(p12.mask == [[0, 1], [0, 0]])

    p13 = Pair.filled((2, 3), 7.)
    assert p13.shape == (2, 3)
    assert p13.vals.shape == (2, 3, 2)
    assert np.all(p13.vals == 7)
    p14 = Pair.filled((2, 2), (1., 2.))
    assert p14.shape == (2, 2)
    assert p14.vals.shape == (2, 2, 2)
    assert np.all(p14.vals[..., 0] == 1)
    assert np.all(p14.vals[..., 1] == 2)

    p15 = Pair([1., 2.])
    p15_conv = Pair.as_pair(p15)
    assert type(p15_conv) == Pair
    assert np.allclose(p15_conv.vals, [1., 2.])

    v16 = Vector([1., 2.])
    p16_conv = Pair.as_pair(v16)
    assert type(p16_conv) == Pair
    assert np.allclose(p16_conv.vals, [1., 2.])

    p17_conv = Pair.as_pair([4., 5.])
    assert type(p17_conv) == Pair
    assert np.allclose(p17_conv.vals, [4., 5.])

    m1x2 = Matrix([[1., 2.]])
    assert m1x2._numer == (1, 2)
    p1x2_conv = Pair.as_pair(m1x2)
    assert type(p1x2_conv) == Pair
    assert np.allclose(p1x2_conv.vals, [1., 2.])

    m2x1 = Matrix([[1.], [2.]])
    assert m2x1._numer == (2, 1)
    p2x1_conv = Pair.as_pair(m2x1)
    assert type(p2x1_conv) == Pair
    assert np.allclose(p2x1_conv.vals, [1., 2.])

    m1x2_nd = Matrix([[[1., 2.]], [[4., 5.]]])
    assert m1x2_nd.shape == (2,)
    assert m1x2_nd._numer == (1, 2)
    p1x2_nd_conv = Pair.as_pair(m1x2_nd)
    assert type(p1x2_nd_conv) == Pair
    assert p1x2_nd_conv.shape == (2,)
    assert np.allclose(p1x2_nd_conv.vals[0], [1., 2.])
    assert np.allclose(p1x2_nd_conv.vals[1], [4., 5.])

    m2x4 = Matrix(np.random.randn(2, 2, 4))  # shape (2,), numer (2, 4)
    assert m2x4.shape == (2,)
    assert m2x4._numer == (2, 4)
    assert m2x4.rank == 2  # nrank=2
    assert m2x4._numer[0] == 2
    p2x4_conv = Pair.as_pair(m2x4)
    assert type(p2x4_conv) == Pair

    assert p2x4_conv.shape == (2,)
    assert p2x4_conv.item == (2, 4)  # numer=(2,), denom=(4,)
    assert p2x4_conv.numer == (2,)
    assert p2x4_conv.denom == (4,)

    p18_conv = Pair.as_pair(5.)
    assert type(p18_conv) == Pair
    assert np.allclose(p18_conv.vals, [5., 5.])

    p19 = Pair([1., 2.])
    p19.insert_deriv('t', Pair([3., 4.]))
    p19_conv = Pair.as_pair(p19, recursive=False)
    assert type(p19_conv) == Pair
    assert np.allclose(p19_conv.vals, [1., 2.])
    assert not hasattr(p19_conv, 'd_dt')

    x = Scalar(1.)
    y = Scalar(2.)
    p20 = Pair.from_scalars(x, y)
    assert type(p20) == Pair
    assert p20.shape == ()
    assert np.allclose(p20.vals, [1., 2.])

    x_2d = Scalar([[1., 2.], [3., 4.]])
    y_2d = Scalar([[5., 6.], [7., 8.]])
    p21 = Pair.from_scalars(x_2d, y_2d)
    assert p21.shape == (2, 2)
    assert np.allclose(p21.vals[0, 0], [1., 5.])
    assert np.allclose(p21.vals[0, 1], [2., 6.])

    p22 = Pair.from_scalars(1., 0.)
    assert np.allclose(p22.vals, [1., 0.])

    p22_none = Pair.from_scalars(1., None)
    assert np.allclose(p22_none.vals, [1., 0.])
    p22_none2 = Pair.from_scalars(None, 2.)
    assert np.allclose(p22_none2.vals, [0., 2.])

    x_nd = Scalar([[1., 2.], [3., 4.]], drank=1)
    p22_none_nd = Pair.from_scalars(x_nd, None)
    assert p22_none_nd.shape == (2,)
    assert p22_none_nd.denom == (2,)  # Should match the denominator of x_nd

    assert np.allclose(p22_none_nd.vals[0, :, 0], [1., 0.])

    p_all_none = Pair.from_scalars(None, None)
    assert type(p_all_none) == Pair
    assert p_all_none.shape == ()
    assert np.allclose(p_all_none.vals, [0., 0.])

    x_broad = Scalar([1., 2.])  # shape (2,)
    y_broad = Scalar([[3.], [4.]])  # shape (2, 1)

    p_broad = Pair.from_scalars(x_broad, y_broad)
    assert type(p_broad) == Pair
    assert p_broad.shape == (2, 2)

    assert np.allclose(p_broad.vals[0, 0], [1., 3.])
    assert np.allclose(p_broad.vals[0, 1], [2., 3.])
    assert np.allclose(p_broad.vals[1, 0], [1., 4.])
    assert np.allclose(p_broad.vals[1, 1], [2., 4.])

    p23 = Pair.from_scalars(1., 2., readonly=True)
    assert type(p23) == Pair
    # readonly may not be set by Qube.from_scalars, but parameter is accepted

    p24 = Pair([1., 2.])
    p24_swapped = p24.swapxy()
    assert type(p24_swapped) == Pair
    assert np.allclose(p24_swapped.vals, [2., 1.])

    p25 = Pair(np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
    p25_swapped = p25.swapxy()
    assert p25_swapped.shape == (2, 2)
    assert np.allclose(p25_swapped.vals[0, 0], [2., 1.])
    assert np.allclose(p25_swapped.vals[0, 1], [4., 3.])

    p26 = Pair([1., 2.])
    p26.insert_deriv('t', Pair([3., 4.]))
    p26_swapped = p26.swapxy(recursive=False)
    assert type(p26_swapped) == Pair
    assert np.allclose(p26_swapped.vals, [2., 1.])
    assert not hasattr(p26_swapped, 'd_dt')

    p27 = Pair([1., 2.])
    p27.insert_deriv('t', Pair([3., 4.]))
    p27_swapped = p27.swapxy(recursive=True)
    assert type(p27_swapped) == Pair
    assert np.allclose(p27_swapped.vals, [2., 1.])
    assert hasattr(p27_swapped, 'd_dt')
    assert np.allclose(p27_swapped.d_dt.vals, [4., 3.])

    p28 = Pair([1., 0.])  # along x-axis
    p28_rot = p28.rot90()
    assert type(p28_rot) == Pair

    assert np.allclose(p28_rot.vals, [0., -1.], atol=1e-10)

    p29 = Pair([0., 1.])  # along y-axis
    p29_rot = p29.rot90()

    assert np.allclose(p29_rot.vals, [1., 0.], atol=1e-10)

    p30 = Pair(np.array([[[1., 0.], [0., 1.]], [[-1., 0.], [0., -1.]]]))
    p30_rot = p30.rot90()
    assert p30_rot.shape == (2, 2)
    assert np.allclose(p30_rot.vals[0, 0], [0., -1.], atol=1e-10)
    assert np.allclose(p30_rot.vals[0, 1], [1., 0.], atol=1e-10)

    p31 = Pair([1., 0.])
    p31.insert_deriv('t', Pair([2., 3.]))
    p31_rot = p31.rot90(recursive=False)
    assert type(p31_rot) == Pair
    assert np.allclose(p31_rot.vals, [0., -1.], atol=1e-10)
    assert not hasattr(p31_rot, 'd_dt')

    p32 = Pair([1., 0.])
    p32.insert_deriv('t', Pair([2., 3.]))
    p32_rot = p32.rot90(recursive=True)
    assert type(p32_rot) == Pair
    assert np.allclose(p32_rot.vals, [0., -1.], atol=1e-10)
    assert hasattr(p32_rot, 'd_dt')

    assert np.allclose(p32_rot.d_dt.vals, [3., -2.], atol=1e-10)

    p33 = Pair([1., 0.])  # along x-axis
    angle33 = p33.angle()
    assert type(angle33) == Scalar
    assert np.allclose(angle33.vals, 0., atol=1e-10)
    p34 = Pair([0., 1.])  # along y-axis
    angle34 = p34.angle()
    assert np.allclose(angle34.vals, np.pi/2, atol=1e-10)

    p35 = Pair(np.array([[[1., 0.], [0., 1.]], [[-1., 0.], [0., -1.]]]))
    angle35 = p35.angle()
    assert angle35.shape == (2, 2)
    assert np.allclose(angle35.vals[0, 0], 0., atol=1e-10)
    assert np.allclose(angle35.vals[0, 1], np.pi/2, atol=1e-10)

    p36 = Pair([-1., 0.])  # negative x-axis
    angle36 = p36.angle()
    assert (angle36.vals >= 0)
    assert (angle36.vals <= 2*np.pi)

    assert np.allclose(angle36.vals, np.pi, atol=1e-10)

    p37 = Pair([1., 1.])
    p37.insert_deriv('t', Pair([2., 3.]))
    angle37 = p37.angle(recursive=False)
    assert type(angle37) == Scalar
    assert not hasattr(angle37, 'd_dt')

    p38 = Pair([5., 5.])
    lower = Pair([2., 2.])
    upper = Pair([4., 4.])
    p38_clipped = p38.clip2d(lower, upper)
    assert type(p38_clipped) == Pair

    assert np.allclose(p38_clipped.vals, [4., 4.], atol=1e-10)

    p39 = Pair([1., 5.])
    upper = Pair([4., 4.])
    p39_clipped = p39.clip2d(None, upper)
    assert type(p39_clipped) == Pair

    assert np.allclose(p39_clipped.vals, [1., 4.], atol=1e-10)

    p40 = Pair([1., 1.])
    lower = Pair([2., 2.])
    p40_clipped = p40.clip2d(lower, None)
    assert type(p40_clipped) == Pair

    assert np.allclose(p40_clipped.vals, [2., 2.], atol=1e-10)

    p41 = Pair(np.array([[[5., 5.], [1., 1.]], [[3., 3.], [6., 6.]]]))
    lower = Pair([2., 2.])
    upper = Pair([4., 4.])
    p41_clipped = p41.clip2d(lower, upper)
    assert p41_clipped.shape == (2, 2)

    assert np.allclose(p41_clipped.vals[0, 0], [4., 4.], atol=1e-10)
    assert np.allclose(p41_clipped.vals[0, 1], [2., 2.], atol=1e-10)

    p42 = Pair([5., 5.])
    lower = Pair([2., 2.])
    upper = Pair([4., 4.])
    p42_clipped = p42.clip2d(lower, upper, remask=True)
    assert type(p42_clipped) == Pair

    assert np.allclose(p42_clipped.vals, [4., 4.], atol=1e-10)
    # With remask=True, the original mask is kept

    p43 = Pair([1., 1.])
    lower_bad = Pair([[2., 2.], [3., 3.]])  # has shape
    upper = Pair([4., 4.])
    with pytest.raises(ValueError):
        p43.clip2d(lower_bad, upper)

    p44 = Pair([1., 1.])
    lower = Pair([2., 2.])
    upper_bad = Pair([[4., 4.], [5., 5.]])  # has shape
    with pytest.raises(ValueError):
        p44.clip2d(lower, upper_bad)


def test_pair_test_clip2d_with_masked_lower_limit_should_be_treated_as_non() -> None:
    """Test clip2d with masked lower limit (should be treated as None)."""

    np.random.seed(2599)

    p45 = Pair([5., 5.])
    lower_masked = Pair([2., 2.], mask=True)  # masked
    upper = Pair([4., 4.])
    p45_clipped = p45.clip2d(lower_masked, upper)
    assert type(p45_clipped) == Pair

    assert np.allclose(p45_clipped.vals, [4., 4.], atol=1e-10)


def test_pair_test_clip2d_with_masked_upper_limit_should_be_treated_as_non() -> None:
    """Test clip2d with masked upper limit (should be treated as None)."""

    np.random.seed(2599)

    p46 = Pair([1., 1.])
    lower = Pair([2., 2.])
    upper_masked = Pair([4., 4.], mask=True)  # masked
    p46_clipped = p46.clip2d(lower, upper_masked)
    assert type(p46_clipped) == Pair

    assert np.allclose(p46_clipped.vals, [2., 2.], atol=1e-10)


def test_pair_test_clip2d_with_both_limits_masked_both_should_be_ignored() -> None:
    """Test clip2d with both limits masked (both should be ignored)."""

    np.random.seed(2599)

    p47 = Pair([5., 5.])
    lower_masked2 = Pair([2., 2.], mask=True)
    upper_masked2 = Pair([4., 4.], mask=True)
    p47_clipped = p47.clip2d(lower_masked2, upper_masked2)
    assert type(p47_clipped) == Pair

    assert np.allclose(p47_clipped.vals, [5., 5.], atol=1e-10)


def test_pair_test_inherited_methods_from_vector_to_scalar() -> None:
    """Test inherited methods from Vector - to_scalar."""

    np.random.seed(2599)

    p_toscalar = Pair(np.random.randn(4, 1, 5, 2))
    s_toscalar = p_toscalar.to_scalar(0)
    assert type(s_toscalar) == Scalar
    assert s_toscalar.shape == p_toscalar.shape

    scalars_from_pair = p_toscalar.to_scalars()
    assert len(scalars_from_pair) == 2
    assert type(scalars_from_pair[0]) == Scalar
    assert scalars_from_pair[0].shape == p_toscalar.shape


def test_pair_test_dot() -> None:
    """Test dot."""

    np.random.seed(2599)

    p_dot_a = Pair([1., 2.])
    p_dot_b = Pair([3., 4.])
    dot_result = p_dot_a.dot(p_dot_b)
    assert type(dot_result) == Scalar

    assert np.allclose(dot_result.vals, 11.)


def test_pair_test_dot_with_n_d() -> None:
    """Test dot with n-D."""

    np.random.seed(2599)

    p_dot_nd_a = Pair(np.random.randn(4, 1, 5, 2))
    p_dot_nd_b = Pair(np.random.randn(8, 5, 2))
    dot_nd_result = p_dot_nd_a.dot(p_dot_nd_b)

    assert dot_nd_result.shape == (4, 8, 5)


def test_pair_test_norm() -> None:
    """Test norm."""

    np.random.seed(2599)

    p50 = Pair([3., 4.])
    norm50 = p50.norm()
    assert type(norm50) == Scalar

    assert np.allclose(norm50.vals, 5.)


def test_pair_test_norm_with_n_d() -> None:
    """Test norm with n-D."""

    np.random.seed(2599)

    p51 = Pair(np.random.randn(2, 3, 2))
    norm51 = p51.norm()
    assert norm51.shape == (2, 3)


def test_pair_test_unit() -> None:
    """Test unit."""

    np.random.seed(2599)

    p52 = Pair([3., 4.])
    unit52 = p52.unit()
    assert type(unit52) == Pair

    assert np.allclose(unit52.vals, [0.6, 0.8], atol=1e-10)
    assert np.allclose(unit52.norm().vals, 1., atol=1e-10)


def test_pair_test_unit_with_n_d() -> None:
    """Test unit with n-D."""

    np.random.seed(2599)

    p53 = Pair(np.random.randn(2, 3, 2))
    unit53 = p53.unit()
    assert unit53.shape == (2, 3)


def test_pair_test_class_constants() -> None:
    """Test class constants."""

    np.random.seed(2599)

    assert type(Pair.ZERO) == Pair
    assert np.allclose(Pair.ZERO.vals, [0., 0.])
    assert Pair.ZERO.readonly
    assert type(Pair.ZEROS) == Pair
    assert np.allclose(Pair.ZEROS.vals, [0., 0.])
    assert Pair.ZEROS.readonly
    assert type(Pair.ONES) == Pair
    assert np.allclose(Pair.ONES.vals, [1., 1.])
    assert Pair.ONES.readonly
    assert type(Pair.HALF) == Pair
    assert np.allclose(Pair.HALF.vals, [0.5, 0.5])
    assert Pair.HALF.readonly
    assert type(Pair.XAXIS) == Pair
    assert np.allclose(Pair.XAXIS.vals, [1., 0.])
    assert Pair.XAXIS.readonly
    assert type(Pair.YAXIS) == Pair
    assert np.allclose(Pair.YAXIS.vals, [0., 1.])
    assert Pair.YAXIS.readonly
    assert type(Pair.MASKED) == Pair
    assert Pair.MASKED.mask
    assert Pair.MASKED.readonly
    assert type(Pair.IDENTITY) == Pair
    assert Pair.IDENTITY.shape == ()
    assert Pair.IDENTITY.denom == (2,)
    assert Pair.IDENTITY.item == (2, 2)
    assert Pair.IDENTITY.readonly
    assert type(Pair.INT00) == Pair
    assert np.allclose(Pair.INT00.vals, [0, 0])
    assert Pair.INT00.readonly
    assert type(Pair.INT11) == Pair
    assert np.allclose(Pair.INT11.vals, [1, 1])
    assert Pair.INT11.readonly


def test_pair_test_that_pair_accepts_both_floats_and_ints() -> None:
    """Test that Pair accepts both floats and ints."""

    np.random.seed(2599)

    p54 = Pair([1, 2])
    assert p54.vals.dtype.kind == 'i'  # Should allow integers
    p55 = Pair([1., 2.])
    assert p55.vals.dtype.kind == 'f'


def test_pair_test_with_mask() -> None:
    """Test with mask."""

    np.random.seed(2599)

    p56 = Pair([1., 2.], mask=False)
    assert not p56.mask
    p57 = Pair([1., 2.], mask=True)
    assert p57.mask


def test_pair_test_complex_n_d_case() -> None:
    """Test complex n-D case."""

    np.random.seed(2599)

    p58 = Pair(np.random.randn(3, 4, 5, 6, 2))
    assert p58.shape == (3, 4, 5, 6)
    assert p58.item == (2,)
    assert p58.vals.shape == (3, 4, 5, 6, 2)


def test_pair_test_that_operations_preserve_type() -> None:
    """Test that operations preserve type."""

    np.random.seed(2599)

    p59 = Pair([1., 2.])
    p60 = Pair([3., 4.])
    p_result = p59 + p60
    assert type(p_result) == Pair
    p_result2 = p59 * 2.
    assert type(p_result2) == Pair


def test_pair_test_round_trip_swapxy_then_swapxy_should_return_original() -> None:
    """Test round-trip: swapxy then swapxy should return original."""

    np.random.seed(2599)

    p61 = Pair([1., 2.])
    p61_round = p61.swapxy().swapxy()
    assert np.allclose(p61.vals, p61_round.vals, atol=1e-10)


def test_pair_test_round_trip_rot90_four_times_should_return_original() -> None:
    """Test round-trip: rot90 four times should return original."""

    np.random.seed(2599)

    p62 = Pair([1., 2.])
    p62_round = p62.rot90().rot90().rot90().rot90()
    assert np.allclose(p62.vals, p62_round.vals, atol=1e-10)


def test_pair_test_angle_consistency_angle_of_rot90_note_rot90_does_x_y_y_() -> None:
    """Test angle consistency: angle of rot90 # Note: rot90 does (x,y) -> (y,-x), which rotates by 90 degrees counterclockwise # For (1,0) -> (0,-1), the angle goes from 0 to 3π/2 (270 degrees)."""

    np.random.seed(2599)

    p63 = Pair([1., 0.])
    angle63 = p63.angle()
    p63_rot = p63.rot90()
    angle63_rot = p63_rot.angle()

    expected_angle = (angle63.vals - np.pi/2) % (2*np.pi)
    assert np.allclose(angle63_rot.vals, expected_angle, atol=1e-10)


##########################################################################################
