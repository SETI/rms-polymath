##########################################################################################
# tests/test_matrix3_to_unitary.py
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix, Matrix3


def _rotations(n: int) -> Matrix3:
    """An array of n random rotation matrices."""

    angles = np.random.rand(3, n) * 2.*np.pi
    return Matrix3.from_euler(angles[0], angles[1], angles[2])


def _antisymmetric(n: int) -> np.ndarray:
    """An array of n random antisymmetric 3x3 arrays."""

    values = np.random.randn(n, 3, 3)
    return values - np.swapaxes(values, -1, -2)


def test_matrix3_to_unitary_returns_a_matrix3() -> None:
    """The returned object is a Matrix3."""

    assert type(Matrix3(2. * np.identity(3)).to_unitary()) is Matrix3


def test_matrix3_to_unitary_of_a_positive_definite_matrix_is_the_identity() -> None:
    """The unitary factor of a symmetric, positive-definite matrix is the identity."""

    result = Matrix3(np.diag([2., 3., 4.])).to_unitary()
    assert np.max(np.abs(result.vals - np.identity(3))) == pytest.approx(0., abs=1.e-15)


def test_matrix3_to_unitary_recovers_the_factor_of_a_polar_decomposition() -> None:
    """For R * S with S symmetric and positive-definite, the unitary factor is R."""

    np.random.seed(6001)
    rot = _rotations(10)
    stretch = np.random.randn(10, 3, 3)
    stretch = np.matmul(np.swapaxes(stretch, -1, -2), stretch) + np.identity(3)

    result = Matrix3(np.matmul(rot.vals, stretch)).to_unitary()
    assert np.max(np.abs(result.vals - rot.vals)) == pytest.approx(0., abs=1.e-13)


def test_matrix3_to_unitary_of_a_badly_perturbed_matrix_is_unitary() -> None:
    """A matrix perturbed by 30% is returned unitary to machine precision."""

    np.random.seed(6002)
    perturbed = Matrix3(_rotations(50).vals + 0.3 * np.random.randn(50, 3, 3))
    assert np.max(perturbed.to_unitary().precision().vals) < 1.e-14


def test_matrix3_to_unitary_of_a_perturbed_matrix_is_not_masked() -> None:
    """A 10% perturbation of a rotation is repairable, so nothing is masked."""

    np.random.seed(6016)
    perturbed = Matrix3(_rotations(100).vals + 0.1 * np.random.randn(100, 3, 3))
    assert not np.any(perturbed.to_unitary().mask)


def test_matrix3_to_unitary_leaves_a_rotation_unchanged() -> None:
    """A matrix that is already unitary is returned essentially unchanged."""

    np.random.seed(6003)
    rot = _rotations(20)
    assert np.max(np.abs(rot.to_unitary().vals - rot.vals)) < 1.e-14


def test_matrix3_to_unitary_preserves_shape() -> None:
    """The result has the same shape as this object."""

    np.random.seed(6004)
    matrix = Matrix3(np.random.randn(2, 3, 3, 3))
    assert matrix.to_unitary().shape == (2, 3)


def test_matrix3_to_unitary_accepts_a_numpy_false_mask() -> None:
    """A mask of np.False_ is recognized as unmasked, as a Python False is."""

    matrix = Matrix3(np.identity(3))
    matrix._mask = np.False_
    assert not np.any(matrix.to_unitary().mask)


def test_matrix3_to_unitary_masks_a_reflection() -> None:
    """No rotation is close to a matrix with a negative determinant."""

    result = Matrix3(np.diag([1., 1., -2.])).to_unitary()
    assert result.mask


def test_matrix3_to_unitary_of_a_reflection_still_returns_a_rotation() -> None:
    """The value substituted for a masked reflection is itself a proper rotation."""

    result = Matrix3(np.diag([1., 1., -2.])).to_unitary()
    assert np.linalg.det(result.vals) == pytest.approx(1.)


def test_matrix3_to_unitary_masks_a_singular_matrix() -> None:
    """No rotation is close to a matrix with a zero determinant."""

    result = Matrix3(np.diag([1., 1., 0.])).to_unitary()
    assert result.mask


def test_matrix3_to_unitary_preserves_the_mask() -> None:
    """An element masked in this object is masked in the result."""

    np.random.seed(6005)
    values = _rotations(4).vals
    result = Matrix3(values, [False, True, False, True]).to_unitary()
    assert np.array_equal(result.mask, [False, True, False, True])


def test_matrix3_to_unitary_evaluates_masked_elements() -> None:
    """A masked element is still made unitary."""

    np.random.seed(6006)
    values = _rotations(4).vals + 0.1 * np.random.randn(4, 3, 3)
    result = Matrix3(values, [False, True, False, True]).to_unitary()
    product = np.matmul(result.vals, np.swapaxes(result.vals, -1, -2))
    assert np.max(np.abs(product - np.identity(3))) < 1.e-14


def test_matrix3_to_unitary_tol_masks_an_imprecise_matrix() -> None:
    """A matrix whose precision reaches tol is masked."""

    result = Matrix3(np.diag([1., 1., 1.001])).to_unitary(tol=1.e-6)
    assert result.mask


def test_matrix3_to_unitary_tol_keeps_a_precise_matrix() -> None:
    """A matrix whose precision is below tol is not masked."""

    np.random.seed(6017)
    result = _rotations(10).to_unitary(tol=1.e-6)
    assert not np.any(result.mask)


def test_matrix3_to_unitary_without_tol_repairs_an_imprecise_matrix() -> None:
    """With the default tol of None, an imprecise matrix is repaired, not masked."""

    result = Matrix3(np.diag([1., 1., 1.001])).to_unitary()
    assert not result.mask


def test_matrix3_to_unitary_validate_rejects_a_reflection() -> None:
    """With validate=True, a matrix that would be masked raises an error instead."""

    with pytest.raises(ValueError, match='invalid rotation matrix'):
        Matrix3(np.diag([1., 1., -2.])).to_unitary(validate=True)


def test_matrix3_to_unitary_validate_rejects_an_imprecise_matrix() -> None:
    """With validate=True and a tol, an imprecise matrix raises an error."""

    with pytest.raises(ValueError, match='invalid rotation matrix'):
        Matrix3(np.diag([1., 1., 1.001])).to_unitary(tol=1.e-6, validate=True)


def test_matrix3_to_unitary_validate_accepts_a_rotation() -> None:
    """With validate=True, an array of valid rotations is converted normally."""

    np.random.seed(6018)
    result = _rotations(10).to_unitary(validate=True)
    assert not np.any(result.mask)


def test_matrix3_to_unitary_validate_ignores_an_already_masked_matrix() -> None:
    """With validate=True, a matrix that is already masked does not raise."""

    values = np.stack([np.identity(3), np.diag([1., 1., -1.])])
    result = Matrix3(values, [False, True]).to_unitary(validate=True)
    assert np.array_equal(result.mask, [False, True])


def test_matrix3_to_unitary_derivative_is_tangent() -> None:
    """The product of a returned derivative and the transposed matrix is antisymmetric."""

    np.random.seed(6007)
    matrix = Matrix3(_rotations(10).vals + 0.05 * np.random.randn(10, 3, 3))
    matrix.insert_deriv('t', Matrix(np.random.randn(10, 3, 3)))

    result = matrix.to_unitary()
    product = np.matmul(result.d_dt.vals, np.swapaxes(result.vals, -1, -2))
    residual = np.abs(product + np.swapaxes(product, -1, -2)).max()
    assert residual < 1.e-14


def test_matrix3_to_unitary_leaves_a_tangent_derivative_unchanged() -> None:
    """A derivative that is already tangent to the rotation is unchanged."""

    np.random.seed(6008)
    rot = _rotations(10).wod
    tangent = np.matmul(_antisymmetric(10), rot.vals)
    rot.insert_deriv('t', Matrix(tangent))

    assert np.max(np.abs(rot.to_unitary().d_dt.vals - tangent)) < 1.e-14


def test_matrix3_to_unitary_derivative_class_is_matrix() -> None:
    """A returned derivative is a Matrix, not a Matrix3."""

    np.random.seed(6009)
    matrix = Matrix3(_rotations(5).vals + 0.05 * np.random.randn(5, 3, 3))
    matrix.insert_deriv('t', Matrix(np.random.randn(5, 3, 3)))

    assert type(matrix.to_unitary().d_dt) is Matrix


def test_matrix3_to_unitary_masks_the_derivative_of_a_masked_matrix() -> None:
    """A derivative is masked wherever its matrix is masked."""

    np.random.seed(6019)
    values = np.stack([np.identity(3), np.diag([1., 1., -1.])])
    matrix = Matrix3(values)
    matrix.insert_deriv('t', Matrix(np.random.randn(2, 3, 3)))

    assert np.array_equal(matrix.to_unitary().d_dt.mask, [False, True])


def test_matrix3_to_unitary_preserves_a_derivative_denominator() -> None:
    """A derivative with a denominator retains its denominator."""

    np.random.seed(6010)
    matrix = Matrix3(_rotations(5).vals + 0.05 * np.random.randn(5, 3, 3))
    matrix.insert_deriv('uv', Matrix(np.random.randn(5, 3, 3, 2), drank=1))

    assert matrix.to_unitary().d_duv.denom == (2,)


def test_matrix3_to_unitary_derivative_with_a_denominator_is_tangent() -> None:
    """Each component of a derivative with a denominator is tangent to the rotation."""

    np.random.seed(6011)
    matrix = Matrix3(_rotations(5).vals + 0.05 * np.random.randn(5, 3, 3))
    matrix.insert_deriv('uv', Matrix(np.random.randn(5, 3, 3, 2), drank=1))

    result = matrix.to_unitary()
    dvals = np.moveaxis(result.d_duv.vals, (-3, -2), (-2, -1))
    mvals = result.vals.reshape(5, 1, 3, 3)
    product = np.matmul(dvals, np.swapaxes(mvals, -1, -2))
    residual = np.abs(product + np.swapaxes(product, -1, -2)).max()
    assert residual < 1.e-14


def test_matrix3_to_unitary_omits_derivatives_when_not_recursive() -> None:
    """With recursive=False, the result has no derivatives."""

    np.random.seed(6012)
    matrix = Matrix3(_rotations(5).vals + 0.05 * np.random.randn(5, 3, 3))
    matrix.insert_deriv('t', Matrix(np.random.randn(5, 3, 3)))

    assert not matrix.to_unitary(recursive=False).derivs


def test_matrix3_to_unitary_rejects_denominators() -> None:
    """An object with a denominator cannot be evaluated."""

    np.random.seed(6013)
    matrix = Matrix3(np.random.randn(4, 3, 3, 2), drank=1)
    with pytest.raises(ValueError, match='does not support denominators'):
        matrix.to_unitary()

##########################################################################################
