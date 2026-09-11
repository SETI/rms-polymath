##########################################################################################
# tests/test_matrix3_precision.py
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix, Matrix3, Scalar


def test_matrix3_precision_of_a_rotation_is_zero() -> None:
    """A matrix built as a rotation departs from unitarity by ~0."""

    result = Matrix3.x_rotation(0.3).precision()
    assert result.vals == pytest.approx(0., abs=1.e-15)


def test_matrix3_precision_returns_a_scalar() -> None:
    """The returned object is a Scalar."""

    assert type(Matrix3.x_rotation(0.3).precision()) is Scalar


def test_matrix3_precision_of_a_scaled_identity() -> None:
    """For 2I, self * self.T - I is 3I, whose RMS over nine elements is sqrt(3)."""

    result = Matrix3(2. * np.identity(3)).precision()
    assert result.vals == pytest.approx(np.sqrt(3.))


def test_matrix3_precision_of_one_stretched_axis() -> None:
    """For diag(1, 1, 1 + e), the only nonzero residual is 2e + e**2."""

    eps = 1.e-3
    result = Matrix3(np.diag([1., 1., 1. + eps])).precision()
    assert result.vals == pytest.approx((2.*eps + eps**2) / 3.)


def test_matrix3_precision_shape_matches_self() -> None:
    """The result has the same shape as this object."""

    np.random.seed(9012)
    angles = np.random.rand(2, 3) * 2.*np.pi
    result = Matrix3.z_rotation(angles).precision()
    assert result.shape == (2, 3)


def test_matrix3_precision_of_an_array_of_rotations() -> None:
    """Every matrix of an array of rotations departs from unitarity by ~0."""

    np.random.seed(4127)
    euler = np.random.rand(3, 20) * 2.*np.pi
    result = Matrix3.from_euler(*euler).precision()
    assert np.max(result.vals) == pytest.approx(0., abs=1.e-15)


def test_matrix3_precision_masks_a_reflection() -> None:
    """A matrix with a negative determinant has no measurable precision."""

    result = Matrix3(np.diag([1., 1., -1.])).precision()
    assert result.mask


def test_matrix3_precision_masks_a_singular_matrix() -> None:
    """A matrix with a zero determinant has no measurable precision."""

    result = Matrix3(np.diag([1., 1., 0.])).precision()
    assert result.mask


def test_matrix3_precision_preserves_the_mask_of_this_object() -> None:
    """An element masked in this object is masked in the result."""

    values = np.stack([np.identity(3), np.identity(3)])
    result = Matrix3(values, [False, True]).precision()
    assert np.array_equal(result.mask, [False, True])


def test_matrix3_precision_keeps_unmasked_values() -> None:
    """An element that is a valid rotation is evaluated."""

    values = np.stack([np.identity(3), np.diag([1., 1., -1.])])
    result = Matrix3(values).precision()
    assert result.vals[0] == 0.


def test_matrix3_precision_omits_derivatives() -> None:
    """Derivatives of this object are not carried into the result."""

    np.random.seed(7744)
    matrix = Matrix3.x_rotation(0.3)
    matrix.insert_deriv('t', Matrix(np.random.randn(3, 3)))
    assert not matrix.precision().derivs


def test_matrix3_precision_rejects_denominators() -> None:
    """An object with a denominator cannot be evaluated."""

    np.random.seed(3355)
    matrix = Matrix3(np.random.randn(4, 3, 3, 2), drank=1)
    with pytest.raises(ValueError, match='does not support denominators'):
        matrix.precision()

##########################################################################################
