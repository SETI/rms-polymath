##########################################################################################
# tests/test_scalar_comprehensive.py
# Comprehensive unit tests for Scalar class based on docstrings
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_comprehensive_test_as_scalar_static_method() -> None:
    """Test as_scalar static method."""

    np.random.seed(5678)

    s1 = Scalar.as_scalar(5.)
    assert type(s1) == Scalar
    assert s1 == 5.
    s2 = Scalar.as_scalar([1., 2., 3.])
    assert type(s2) == Scalar
    assert np.allclose(s2.vals, [1., 2., 3.])

    s3 = Scalar(5.)
    s4 = s3.to_scalar(0)
    assert s4 == 5.

    with pytest.raises(ValueError):
        s3.to_scalar(1)


def test_scalar_comprehensive_test_as_index_method() -> None:
    """Test as_index method."""

    np.random.seed(5678)

    s5 = Scalar([0, 1, 2, 3])
    idx = s5.as_index()
    assert np.allclose(idx, [0, 1, 2, 3])


def test_scalar_comprehensive_test_as_index_and_mask() -> None:
    """Test as_index_and_mask."""

    np.random.seed(5678)

    s6 = Scalar([0, 1, 2])
    idx2, mask2 = s6.as_index_and_mask()
    assert np.allclose(idx2, [0, 1, 2])
    assert not mask2


def test_scalar_comprehensive_test_int_method() -> None:
    """Test int() method."""

    np.random.seed(5678)

    s7 = Scalar(5.7)
    s8 = s7.int()
    assert s8 == 5
    assert s8.is_int()


def test_scalar_comprehensive_test_with_top_parameter_inclusive_true_by_default_so_the_top() -> None:
    """Test with top parameter; inclusive=True by default, so the top value itself is # in range (and gets shifted down by one), whereas anything above it is masked."""

    np.random.seed(5678)

    s9 = Scalar([1, 2, 3, 4, 5])
    s10 = s9.int(top=3, remask=True)
    assert np.all(s10.mask == [False, False, False, True, True])
    assert np.all(s10.values == [1, 2, 2, 4, 5])

    s10 = s9.int(top=3, remask=True, inclusive=False)
    assert np.all(s10.mask == [False, False, True, True, True])


def test_scalar_comprehensive_test_frac_method() -> None:
    """Test frac method."""

    np.random.seed(5678)

    s11 = Scalar(5.7)
    s12 = s11.frac()
    assert s12 == 0.7 or abs(s12 - 0.7) <= 1e-10


def test_scalar_comprehensive_test_sin_method() -> None:
    """Test sin method."""

    np.random.seed(5678)

    s13 = Scalar(np.pi/2, unit=Unit.RAD)
    s14 = s13.sin()
    assert s14 == 1. or abs(s14 - 1.) <= 1e-10


def test_scalar_comprehensive_test_cos_method() -> None:
    """Test cos method."""

    np.random.seed(5678)

    s15 = Scalar(0., unit=Unit.RAD)
    s16 = s15.cos()
    assert s16 == 1. or abs(s16 - 1.) <= 1e-10


def test_scalar_comprehensive_test_tan_method() -> None:
    """Test tan method."""

    np.random.seed(5678)

    s17 = Scalar(np.pi/4, unit=Unit.RAD)
    s18 = s17.tan()
    assert s18 == 1. or abs(s18 - 1.) <= 1e-10


def test_scalar_comprehensive_test_arcsin_method() -> None:
    """Test arcsin method."""

    np.random.seed(5678)

    s19 = Scalar(1.)
    s20 = s19.arcsin()
    assert s20 == np.pi/2 or abs(s20 - np.pi/2) <= 1e-10


def test_scalar_comprehensive_test_arccos_method() -> None:
    """Test arccos method."""

    np.random.seed(5678)

    s21 = Scalar(0.)
    s22 = s21.arccos()
    assert s22 == np.pi/2 or abs(s22 - np.pi/2) <= 1e-10


def test_scalar_comprehensive_test_arctan_method() -> None:
    """Test arctan method."""

    np.random.seed(5678)

    s23 = Scalar(1.)
    s24 = s23.arctan()
    assert s24 == np.pi/4 or abs(s24 - np.pi/4) <= 1e-10


def test_scalar_comprehensive_test_arctan2_method() -> None:
    """Test arctan2 method."""

    np.random.seed(5678)

    s25 = Scalar(1.)
    s26 = Scalar(1.)
    s27 = s25.arctan2(s26)
    assert s27 == np.pi/4 or abs(s27 - np.pi/4) <= 1e-10


def test_scalar_comprehensive_test_sqrt_method() -> None:
    """Test sqrt method."""

    np.random.seed(5678)

    s28 = Scalar(4.)
    s29 = s28.sqrt()
    assert s29 == 2.


def test_scalar_comprehensive_test_log_method() -> None:
    """Test log method."""

    np.random.seed(5678)

    s30 = Scalar(np.e)
    s31 = s30.log()
    assert s31 == 1. or abs(s31 - 1.) <= 1e-10


def test_scalar_comprehensive_test_exp_method() -> None:
    """Test exp method."""

    np.random.seed(5678)

    s32 = Scalar(1.)
    s33 = s32.exp()
    assert s33 == np.e or abs(s33 - np.e) <= 1e-10


def test_scalar_comprehensive_test_sign_method() -> None:
    """Test sign method."""

    np.random.seed(5678)

    s34 = Scalar([-2., 0., 2.])
    s35 = s34.sign()
    assert np.allclose(s35.vals, [-1., 0., 1.])


def test_scalar_comprehensive_test_solve_quadratic_static_method() -> None:
    """Test solve_quadratic static method."""

    np.random.seed(5678)

    a = Scalar(1.)
    b = Scalar(0.)
    c = Scalar(-1.)
    x0, x1 = Scalar.solve_quadratic(a, b, c)
    assert x0 == -1. or abs(x0 - -1.) <= 1e-10
    assert x1 == 1. or abs(x1 - 1.) <= 1e-10


def test_scalar_comprehensive_test_eval_quadratic_method() -> None:
    """Test eval_quadratic method."""

    np.random.seed(5678)

    s36 = Scalar(2.)
    s37 = s36.eval_quadratic(1., 0., -4.)
    assert s37 == 0.  # 1*2^2 + 0*2 - 4 = 0


def test_scalar_comprehensive_test_max_method() -> None:
    """Test max method."""

    np.random.seed(5678)

    s38 = Scalar([1., 5., 3., 2., 4.])
    s39 = s38.max()
    assert s39 == 5.

    s40 = s38.min()
    assert s40 == 1.

    s41 = s38.argmax()
    assert s41 == 1  # Index of max value

    s42 = s38.argmin()
    assert s42 == 0  # Index of min value


def test_scalar_comprehensive_test_maximum_static_method() -> None:
    """Test maximum static method."""

    np.random.seed(5678)

    s43 = Scalar([1., 3., 2.])
    s44 = Scalar([2., 1., 4.])
    s45 = Scalar.maximum(s43, s44)
    assert np.allclose(s45.vals, [2., 3., 4.])

    s46 = Scalar.minimum(s43, s44)
    assert np.allclose(s46.vals, [1., 1., 2.])


def test_scalar_comprehensive_test_median_method() -> None:
    """Test median method."""

    np.random.seed(5678)

    s47 = Scalar([1., 3., 2., 5., 4.])
    s48 = s47.median()
    assert s48 == 3.


def test_scalar_comprehensive_test_sort_method() -> None:
    """Test sort method."""

    np.random.seed(5678)

    s49 = Scalar([3., 1., 4., 2.])
    s50 = s49.sort()
    assert np.allclose(s50.vals, [1., 2., 3., 4.])


def test_scalar_comprehensive_test_reciprocal_method() -> None:
    """Test reciprocal method."""

    np.random.seed(5678)

    s51 = Scalar(2.)
    s52 = s51.reciprocal()
    assert s52 == 0.5


def test_scalar_comprehensive_test_identity_method() -> None:
    """Test identity method."""

    np.random.seed(5678)

    s53 = Scalar(5.)
    s54 = s53.identity()
    assert s54 == 1.
    assert s54.readonly


def test_scalar_comprehensive_test_abs_method() -> None:
    """Test __abs__ method."""

    np.random.seed(5678)

    s55 = Scalar(-5.)
    s56 = abs(s55)
    assert s56 == 5.


def test_scalar_comprehensive_test_pow_method() -> None:
    """Test __pow__ method."""

    np.random.seed(5678)

    s57 = Scalar(2.)
    s58 = s57 ** 3
    assert s58 == 8.
    s59 = s57 ** 0.5
    assert s59 == np.sqrt(2.) or abs(s59 - np.sqrt(2.)) <= 1e-10


def test_scalar_comprehensive_test_le_method() -> None:
    """Test __le__ method."""

    np.random.seed(5678)

    s60 = Scalar(2.)
    result = s60 <= 3.
    assert result

    result = s60 < 3.
    assert result

    result = s60 >= 1.
    assert result

    result = s60 > 1.
    assert result


def test_scalar_comprehensive_n_d_test_cases_test_sin_with_n_d_array() -> None:
    """n-D test cases # Test sin with n-D array."""

    np.random.seed(5678)

    s61 = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]], unit=Unit.RAD)
    s62 = s61.sin()
    assert s62[0, 0] == 0. or abs(s62[0, 0] - 0.) <= 1e-10
    assert s62[0, 1] == 1. or abs(s62[0, 1] - 1.) <= 1e-10


def test_scalar_comprehensive_test_max_with_axis() -> None:
    """Test max with axis."""

    np.random.seed(5678)

    s63 = Scalar([[1., 5., 3.], [2., 4., 6.]])
    s64 = s63.max(axis=1)
    assert np.allclose(s64.vals, [5., 6.])

    s65 = s63.min(axis=0)
    assert np.allclose(s65.vals, [1., 4., 3.])

    s66 = s63.median(axis=1)
    assert np.allclose(s66.vals, [3., 4.])


def test_scalar_comprehensive_test_as_scalar_with_boolean() -> None:
    """Test as_scalar with Boolean."""

    np.random.seed(5678)

    from polymath import Boolean
    b1 = Boolean(True)
    s67 = Scalar.as_scalar(b1)
    assert type(s67) == Scalar
    assert s67 == 1


def test_scalar_comprehensive_test_as_scalar_with_unit_unit_is_already_imported_at_top() -> None:
    """Test as_scalar with Unit (Unit is already imported at top)."""

    np.random.seed(5678)

    s68 = Scalar.as_scalar(Unit.RAD)
    assert type(s68) == Scalar

    assert s68.units == Unit.RAD


def test_scalar_comprehensive_test_as_scalar_with_recursive_false() -> None:
    """Test as_scalar with recursive=False."""

    np.random.seed(5678)

    s69 = Scalar(5.)
    s69.insert_deriv('t', Scalar(2.))
    s70 = Scalar.as_scalar(s69, recursive=False)
    assert len(s70.derivs) == 0


def test_scalar_comprehensive_test_to_scalar_with_recursive_false() -> None:
    """Test to_scalar with recursive=False."""

    np.random.seed(5678)

    s71 = Scalar(5.)
    s71.insert_deriv('t', Scalar(2.))
    s72 = s71.to_scalar(0, recursive=False)
    assert len(s72.derivs) == 0


def test_scalar_comprehensive_test_as_index_with_masked_parameter() -> None:
    """Test as_index with masked parameter."""

    np.random.seed(5678)

    s73 = Scalar([0, 1, 2, 3])
    idx3 = s73.as_index(masked=99)
    assert np.allclose(idx3, [0, 1, 2, 3])


def test_scalar_comprehensive_test_as_index_and_mask_with_masked_parameter() -> None:
    """Test as_index_and_mask with masked parameter."""

    np.random.seed(5678)

    s74 = Scalar([0, 1, 2])
    idx4, _ = s74.as_index_and_mask(masked=99)
    assert np.allclose(idx4, [0, 1, 2])


def test_scalar_comprehensive_test_as_index_and_mask_with_purge_true() -> None:
    """Test as_index_and_mask with purge=True."""

    np.random.seed(5678)

    s75 = Scalar([0, 1, 2])
    s75 = s75.mask_where_le(1)
    idx5, _ = s75.as_index_and_mask(purge=True)
    assert type(idx5) == np.ndarray


def test_scalar_comprehensive_test_int_with_clip_parameter() -> None:
    """Test int() with clip parameter."""

    np.random.seed(5678)

    s76 = Scalar([-1, 5, 3])
    s77 = s76.int(top=3, clip=True)

    assert np.allclose(s77.vals, [0, 2, 2])


def test_scalar_comprehensive_test_int_with_inclusive_parameter() -> None:
    """Test int() with inclusive parameter."""

    np.random.seed(5678)

    s78 = Scalar([0, 1, 2, 3])
    s79 = s78.int(top=3, inclusive=False, remask=True)

    assert isinstance(s79, Scalar)
    assert s79.mask[3]


def test_scalar_comprehensive_test_int_with_shift_parameter() -> None:
    """Test int() with shift parameter."""

    np.random.seed(5678)

    s80 = Scalar([0, 1, 2, 3])
    s81 = s80.int(top=3, shift=True, remask=True)
    assert isinstance(s81, Scalar)


def test_scalar_comprehensive_test_frac_with_n_d() -> None:
    """Test frac with n-D."""

    np.random.seed(5678)

    s82 = Scalar([[1.5, 2.7], [3.9, 4.1]])
    s83 = s82.frac()
    assert s83[0, 0] == 0.5 or abs(s83[0, 0] - 0.5) <= 1e-10


def test_scalar_comprehensive_test_sin_with_n_d_and_recursive_false() -> None:
    """Test sin with n-D and recursive=False."""

    np.random.seed(5678)

    s84 = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]], unit=Unit.RAD)
    s85 = s84.sin(recursive=False)
    assert s85[0, 1] == 1. or abs(s85[0, 1] - 1.) <= 1e-10


def test_scalar_comprehensive_test_cos_with_recursive_false() -> None:
    """Test cos with recursive=False."""

    np.random.seed(5678)

    s86 = Scalar(0., unit=Unit.RAD)
    s87 = s86.cos(recursive=False)
    assert s87 == 1. or abs(s87 - 1.) <= 1e-10


def test_scalar_comprehensive_test_tan_with_recursive_false() -> None:
    """Test tan with recursive=False."""

    np.random.seed(5678)

    s88 = Scalar(np.pi/4, unit=Unit.RAD)
    s89 = s88.tan(recursive=False)
    assert s89 == 1. or abs(s89 - 1.) <= 1e-10


def test_scalar_comprehensive_test_arcsin_with_recursive_false() -> None:
    """Test arcsin with recursive=False."""

    np.random.seed(5678)

    s90 = Scalar(1.)
    s91 = s90.arcsin(recursive=False)
    assert s91 == np.pi/2 or abs(s91 - np.pi/2) <= 1e-10


def test_scalar_comprehensive_test_arccos_with_recursive_false() -> None:
    """Test arccos with recursive=False."""

    np.random.seed(5678)

    s92 = Scalar(0.)
    s93 = s92.arccos(recursive=False)
    assert s93 == np.pi/2 or abs(s93 - np.pi/2) <= 1e-10


def test_scalar_comprehensive_test_arctan_with_recursive_false() -> None:
    """Test arctan with recursive=False."""

    np.random.seed(5678)

    s94 = Scalar(1.)
    s95 = s94.arctan(recursive=False)
    assert s95 == np.pi/4 or abs(s95 - np.pi/4) <= 1e-10


def test_scalar_comprehensive_test_arctan2_with_recursive_false() -> None:
    """Test arctan2 with recursive=False."""

    np.random.seed(5678)

    s96 = Scalar(1.)
    s97 = Scalar(1.)
    s98 = s96.arctan2(s97, recursive=False)
    assert s98 == np.pi/4 or abs(s98 - np.pi/4) <= 1e-10


def test_scalar_comprehensive_test_sqrt_with_recursive_false() -> None:
    """Test sqrt with recursive=False."""

    np.random.seed(5678)

    s99 = Scalar(4.)
    s100 = s99.sqrt(recursive=False)
    assert s100 == 2.


def test_scalar_comprehensive_test_log_with_recursive_false() -> None:
    """Test log with recursive=False."""

    np.random.seed(5678)

    s101 = Scalar(np.e)
    s102 = s101.log(recursive=False)
    assert s102 == 1. or abs(s102 - 1.) <= 1e-10


def test_scalar_comprehensive_test_exp_with_recursive_false() -> None:
    """Test exp with recursive=False."""

    np.random.seed(5678)

    s103 = Scalar(1.)
    s104 = s103.exp(recursive=False)
    assert s104 == np.e or abs(s104 - np.e) <= 1e-10


def test_scalar_comprehensive_test_sign_no_recursive_parameter() -> None:
    """Test sign (no recursive parameter)."""

    np.random.seed(5678)

    s105 = Scalar([-2., 0., 2.])
    s106 = s105.sign()
    assert np.allclose(s106.vals, [-1., 0., 1.])


def test_scalar_comprehensive_test_solve_quadratic_with_n_d() -> None:
    """Test solve_quadratic with n-D."""

    np.random.seed(5678)

    a2 = Scalar([1., 1.])
    b2 = Scalar([0., 0.])
    c2 = Scalar([-1., -4.])
    x0_2, x1_2 = Scalar.solve_quadratic(a2, b2, c2)
    assert x0_2[0] == -1. or abs(x0_2[0] - -1.) <= 1e-10
    assert x1_2[0] == 1. or abs(x1_2[0] - 1.) <= 1e-10


def test_scalar_comprehensive_test_eval_quadratic_with_recursive_false() -> None:
    """Test eval_quadratic with recursive=False."""

    np.random.seed(5678)

    s107 = Scalar(2.)
    s108 = s107.eval_quadratic(1., 0., -4., recursive=False)
    assert s108 == 0.


def test_scalar_comprehensive_test_max_no_recursive_parameter() -> None:
    """Test max (no recursive parameter)."""

    np.random.seed(5678)

    s109 = Scalar([1., 5., 3., 2., 4.])
    s110 = s109.max()
    assert s110 == 5.

    s111 = s109.min()
    assert s111 == 1.

    s112 = s109.argmax()
    assert s112 == 1

    s113 = s109.argmin()
    assert s113 == 0


def test_scalar_comprehensive_test_maximum_no_recursive_parameter() -> None:
    """Test maximum (no recursive parameter)."""

    np.random.seed(5678)

    s114 = Scalar([1., 3., 2.])
    s115 = Scalar([2., 1., 4.])
    s116 = Scalar.maximum(s114, s115)
    assert np.allclose(s116.vals, [2., 3., 4.])

    s117 = Scalar.minimum(s114, s115)
    assert np.allclose(s117.vals, [1., 1., 2.])


def test_scalar_comprehensive_test_median_no_recursive_parameter() -> None:
    """Test median (no recursive parameter)."""

    np.random.seed(5678)

    s118 = Scalar([1., 3., 2., 5., 4.])
    s119 = s118.median()
    assert s119 == 3.


def test_scalar_comprehensive_test_sort_no_recursive_parameter() -> None:
    """Test sort (no recursive parameter)."""

    np.random.seed(5678)

    s120 = Scalar([3., 1., 4., 2.])
    s121 = s120.sort()
    assert np.allclose(s121.vals, [1., 2., 3., 4.])


def test_scalar_comprehensive_test_reciprocal_with_recursive_false() -> None:
    """Test reciprocal with recursive=False."""

    np.random.seed(5678)

    s122 = Scalar(2.)
    s123 = s122.reciprocal(recursive=False)
    assert s123 == 0.5


def test_scalar_comprehensive_test_identity_no_recursive_parameter() -> None:
    """Test identity (no recursive parameter)."""

    np.random.seed(5678)

    s124 = Scalar(5.)
    s125 = s124.identity()
    assert s125 == 1.


def test_scalar_comprehensive_test_abs_with_recursive_false() -> None:
    """Test __abs__ with recursive=False."""

    np.random.seed(5678)

    s126 = Scalar(-5.)
    s127 = abs(s126)
    assert s127 == 5.


def test_scalar_comprehensive_test_pow_with_recursive_false() -> None:
    """Test __pow__ with recursive=False."""

    np.random.seed(5678)

    s128 = Scalar(2.)
    s129 = s128.__pow__(3, recursive=False)
    assert s129 == 8.


def test_scalar_comprehensive_test_pow_with_fractional_exponent() -> None:
    """Test __pow__ with fractional exponent."""

    np.random.seed(5678)

    s130 = Scalar(4.)
    s131 = s130.__pow__(0.5, recursive=False)
    assert s131 == 2. or abs(s131 - 2.) <= 1e-10


def test_scalar_comprehensive_test_le_with_n_d() -> None:
    """Test __le__ with n-D."""

    np.random.seed(5678)

    s132 = Scalar([1., 2., 3.])
    result = s132 <= 2.
    assert result[0]
    assert result[1]
    assert not result[2]

    result = s132 < 2.
    assert result[0]
    assert not result[1]
    assert not result[2]

    result = s132 >= 2.
    assert not result[0]
    assert result[1]
    assert result[2]

    result = s132 > 2.
    assert not result[0]
    assert not result[1]
    assert result[2]

    result = s132 == 2.
    assert not result[0]
    assert result[1]
    assert not result[2]

    result = s132 != 2.
    assert result[0]
    assert not result[1]
    assert result[2]


def test_scalar_comprehensive_test_max_with_multiple_axes() -> None:
    """Test max with multiple axes."""

    np.random.seed(5678)

    s133 = Scalar([[[1., 5.], [3., 2.]], [[4., 1.], [6., 3.]]])
    s134 = s133.max(axis=(0, 1))

    assert np.allclose(s134.vals, [6., 5.])

    s135 = s133.min(axis=(0, 1))
    assert np.allclose(s135.vals, [1., 1.])


def test_scalar_comprehensive_test_median_with_multiple_axes() -> None:
    """Test median with multiple axes."""

    np.random.seed(5678)

    s136 = Scalar([[[1., 5.], [3., 2.]], [[4., 1.], [6., 3.]]])
    s137 = s136.median(axis=(0, 1))
    assert np.allclose(s137.vals, [3.5, 2.5])


def test_scalar_comprehensive_test_sort_with_axis() -> None:
    """Test sort with axis."""

    np.random.seed(5678)

    s138 = Scalar([[3., 1., 4.], [2., 5., 1.]])
    s139 = s138.sort(axis=1)
    assert np.allclose(s139[0].vals, [1., 3., 4.])


def test_scalar_comprehensive_test_solve_quadratic_with_complex_roots_should_mask() -> None:
    """Test solve_quadratic with complex roots (should mask)."""

    np.random.seed(5678)

    a3 = Scalar(1.)
    b3 = Scalar(1.)
    c3 = Scalar(1.)
    _x0_3, _x1_3 = Scalar.solve_quadratic(a3, b3, c3)

    assert _x0_3.mask
    assert _x1_3.mask


def test_scalar_comprehensive_test_eval_quadratic_with_n_d() -> None:
    """Test eval_quadratic with n-D."""

    np.random.seed(5678)

    s140 = Scalar([[1., 2.], [3., 4.]])
    s141 = s140.eval_quadratic(1., 0., -1.)
    assert s141[0, 0] == 0.
    assert s141[0, 1] == 3.


##########################################################################################
