##########################################################################################
# tests/test_matrix3_mask_non_unitary.py
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix, Matrix3


def test_matrix3_mask_non_unitary_returns_a_matrix3() -> None:
    """The returned object is a Matrix3."""

    assert type(Matrix3(np.identity(3)).mask_non_unitary()) is Matrix3


def test_matrix3_mask_non_unitary_keeps_a_rotation_unmasked() -> None:
    """A valid rotation is not masked."""

    assert not Matrix3.x_rotation(0.3).mask_non_unitary().mask


def test_matrix3_mask_non_unitary_masks_a_reflection() -> None:
    """A matrix with a negative determinant is masked."""

    assert Matrix3(np.diag([1., 1., -1.])).mask_non_unitary().mask


def test_matrix3_mask_non_unitary_masks_a_singular_matrix() -> None:
    """A matrix with a zero determinant is masked."""

    assert Matrix3(np.diag([1., 1., 0.])).mask_non_unitary().mask


def test_matrix3_mask_non_unitary_keeps_an_imprecise_matrix_without_tol() -> None:
    """Without a tol, a merely imprecise matrix is not masked."""

    assert not Matrix3(np.diag([1., 1., 1.001])).mask_non_unitary().mask


def test_matrix3_mask_non_unitary_masks_an_imprecise_matrix_with_tol() -> None:
    """With a tol, a matrix whose precision reaches it is masked."""

    assert Matrix3(np.diag([1., 1., 1.001])).mask_non_unitary(tol=1.e-6).mask


def test_matrix3_mask_non_unitary_tol_is_inclusive() -> None:
    """A matrix whose precision exactly equals tol is masked."""

    matrix = Matrix3(np.diag([1., 1., 1.001]))
    assert matrix.mask_non_unitary(tol=float(matrix.precision().vals)).mask


def test_matrix3_mask_non_unitary_keeps_a_rotation_with_tol() -> None:
    """A valid rotation stays unmasked even under a tight tol."""

    np.random.seed(7001)
    angles = np.random.rand(3, 20) * 2.*np.pi
    result = Matrix3.from_euler(*angles).mask_non_unitary(tol=1.e-12)
    assert not np.any(result.mask)


def test_matrix3_mask_non_unitary_preserves_the_existing_mask() -> None:
    """An element already masked stays masked."""

    values = np.stack([np.identity(3), np.identity(3)])
    result = Matrix3(values, [False, True]).mask_non_unitary()
    assert np.array_equal(result.mask, [False, True])


def test_matrix3_mask_non_unitary_masks_only_the_bad_elements() -> None:
    """Masking is applied element by element."""

    values = np.stack([np.identity(3), np.diag([1., 1., -1.]), np.identity(3)])
    result = Matrix3(values).mask_non_unitary()
    assert np.array_equal(result.mask, [False, True, False])


def test_matrix3_mask_non_unitary_preserves_the_values() -> None:
    """The values are unchanged; only the mask differs."""

    values = np.stack([np.identity(3), np.diag([1., 1., -1.])])
    result = Matrix3(values).mask_non_unitary()
    assert np.array_equal(result.vals, values)


def test_matrix3_mask_non_unitary_keeps_derivatives() -> None:
    """Derivatives survive the remasking."""

    np.random.seed(7002)
    matrix = Matrix3.x_rotation(0.3)
    matrix.insert_deriv('t', Matrix(np.random.randn(3, 3)))
    assert 't' in matrix.mask_non_unitary().derivs


def test_matrix3_mask_non_unitary_allpos_keeps_a_reflection() -> None:
    """With allpos=True, a reflection is not masked."""

    assert not Matrix3(np.diag([1., 1., -1.])).mask_non_unitary(allpos=True).mask


def test_matrix3_mask_non_unitary_allpos_still_applies_tol() -> None:
    """With allpos=True, tol still masks an imprecise matrix."""

    matrix = Matrix3(np.diag([1., 1., 1.001]))
    assert matrix.mask_non_unitary(tol=1.e-6, allpos=True).mask


def test_matrix3_mask_non_unitary_rejects_denominators() -> None:
    """An object with a denominator cannot be evaluated."""

    np.random.seed(7003)
    matrix = Matrix3(np.random.randn(4, 3, 3, 2), drank=1)
    with pytest.raises(ValueError, match='does not support denominators'):
        matrix.mask_non_unitary()


def test_matrix3_mask_non_unitary_does_not_shadow_qube_remask() -> None:
    """Qube.remask() remains available on a Matrix3 and takes a positional mask."""

    values = np.stack([np.identity(3)] * 3)
    result = Matrix3(values).remask([True, False, True])
    assert np.array_equal(result.mask, [True, False, True])

##########################################################################################
