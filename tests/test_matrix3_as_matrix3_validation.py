##########################################################################################
# tests/test_matrix3_as_matrix3_validation.py: the unitarize, validate and tol options
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix, Matrix3, Quaternion


def test_matrix3_as_matrix3_returns_the_converted_object() -> None:
    """A plain array is converted to a Matrix3."""

    result = Matrix3.as_matrix3([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert np.array_equal(result.vals, np.identity(3))


def test_matrix3_as_matrix3_converts_a_matrix() -> None:
    """A Matrix of the right shape is converted to a Matrix3."""

    assert type(Matrix3.as_matrix3(Matrix(np.identity(3)))) is Matrix3


def test_matrix3_as_matrix3_passes_a_matrix3_through() -> None:
    """A Matrix3 is returned unchanged."""

    matrix = Matrix3.x_rotation(0.3)
    assert Matrix3.as_matrix3(matrix) is matrix


def test_matrix3_as_matrix3_drops_derivatives_when_not_recursive() -> None:
    """With recursive=False, the derivatives are dropped."""

    np.random.seed(8001)
    matrix = Matrix3.x_rotation(0.3)
    matrix.insert_deriv('t', Matrix(np.random.randn(3, 3)))
    assert not Matrix3.as_matrix3(matrix, recursive=False).derivs


def test_matrix3_as_matrix3_keeps_derivatives_by_default() -> None:
    """With the default recursive=True, the derivatives are kept."""

    np.random.seed(8002)
    matrix = Matrix3.x_rotation(0.3)
    matrix.insert_deriv('t', Matrix(np.random.randn(3, 3)))
    assert 't' in Matrix3.as_matrix3(matrix).derivs


def test_matrix3_as_matrix3_unitarize_repairs_a_matrix() -> None:
    """With unitarize=True, an imprecise matrix is made exactly unitary."""

    result = Matrix3.as_matrix3([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.001]], unitarize=True)
    assert result.precision().vals == pytest.approx(0., abs=1.e-15)


def test_matrix3_as_matrix3_without_unitarize_leaves_the_values_alone() -> None:
    """By default the values are not checked or repaired."""

    values = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.001]]
    assert np.array_equal(Matrix3.as_matrix3(values).vals, values)


def test_matrix3_as_matrix3_unitarize_masks_a_reflection() -> None:
    """With unitarize=True, a matrix that is not a rotation is masked."""

    result = Matrix3.as_matrix3([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], unitarize=True)
    assert result.mask


def test_matrix3_as_matrix3_unitarize_with_tol_masks_an_imprecise_matrix() -> None:
    """With unitarize=True and a tol, an imprecise matrix is masked."""

    result = Matrix3.as_matrix3([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.001]], unitarize=True, tol=1.e-6)
    assert result.mask


def test_matrix3_as_matrix3_validate_rejects_a_reflection() -> None:
    """With validate=True, a matrix that is not a rotation raises an error."""

    with pytest.raises(ValueError, match='invalid rotation matrix'):
        Matrix3.as_matrix3([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], validate=True)


def test_matrix3_as_matrix3_validate_rejects_an_imprecise_matrix() -> None:
    """With validate=True and a tol, an imprecise matrix raises an error."""

    with pytest.raises(ValueError, match='invalid rotation matrix'):
        Matrix3.as_matrix3([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.001]], validate=True, tol=1.e-6)


def test_matrix3_as_matrix3_validate_accepts_a_rotation() -> None:
    """With validate=True, a valid rotation is converted and repaired."""

    result = Matrix3.as_matrix3(Matrix3.x_rotation(0.3), validate=True)
    assert result.precision().vals == pytest.approx(0., abs=1.e-15)


def test_matrix3_as_matrix3_validate_supersedes_unitarize() -> None:
    """validate=True raises even when unitarize is False."""

    with pytest.raises(ValueError, match='invalid rotation matrix'):
        Matrix3.as_matrix3([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], unitarize=False, validate=True)


def test_matrix3_as_matrix3_unitarize_drops_derivatives_when_not_recursive() -> None:
    """The recursive option still applies when the matrices are unitarized."""

    np.random.seed(8003)
    matrix = Matrix3.x_rotation(0.3)
    matrix.insert_deriv('t', Matrix(np.random.randn(3, 3)))
    assert not Matrix3.as_matrix3(matrix, recursive=False, unitarize=True).derivs


def test_matrix3_as_matrix3_of_a_quaternion_skips_the_checks() -> None:
    """A Quaternion always yields a rotation, so unitarize has nothing to do."""

    q = Quaternion(np.array([1., 0., 0., 0.]))
    result = Matrix3.as_matrix3(q, unitarize=True)
    assert np.max(np.abs(result.vals - np.identity(3))) == pytest.approx(0., abs=1.e-15)


def test_matrix3_as_matrix3_of_a_quaternion_is_not_masked_by_validate() -> None:
    """A Quaternion is accepted under validate=True."""

    np.random.seed(8004)
    q = Quaternion(np.random.randn(6, 4)).unit()
    assert not np.any(Matrix3.as_matrix3(q, validate=True).mask)

##########################################################################################
