##########################################################################################
# tests/test_matrix_solve.py
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix, Unit, Vector, Vector3


@pytest.mark.parametrize('size', [1, 2, 3, 4, 6])
def test_matrix_solve_satisfies_the_equation(size: int) -> None:
    """The returned X satisfies A X = B for a square matrix of any size."""

    rng = np.random.default_rng(101 + size)
    a = Matrix(rng.normal(size=(size, size)))
    b = Vector(rng.normal(size=(size,)))

    x = a.solve(b)
    assert np.allclose((a * x).values, b.values)


def test_matrix_solve_agrees_with_multiplying_by_the_inverse() -> None:
    """solve() gives the same answer as multiplying by the inverse matrix."""

    rng = np.random.default_rng(202)
    a = Matrix(rng.normal(size=(3, 5, 5)))
    b = Vector(rng.normal(size=(3, 5)))

    assert np.allclose(a.solve(b).values, (a.inverse() * b).values)


def test_matrix_solve_agrees_with_numpy() -> None:
    """solve() gives the same answer as numpy.linalg.solve."""

    rng = np.random.default_rng(303)
    a = Matrix(rng.normal(size=(3, 5, 5)))
    b = Vector(rng.normal(size=(3, 5)))

    expected = np.linalg.solve(a.values, b.values[..., np.newaxis])[..., 0]
    assert np.allclose(a.solve(b).values, expected)


def test_matrix_solve_broadcasts_the_leading_shape() -> None:
    """A single right-hand side is broadcast across an array of matrices."""

    rng = np.random.default_rng(404)
    a = Matrix(rng.normal(size=(5, 4, 4)))
    b = Vector(rng.normal(size=(4,)))

    x = a.solve(b)
    assert x.shape == (5,)
    assert np.allclose((a * x).values, b.values)


def test_matrix_solve_returns_the_subclass_of_the_right_hand_side() -> None:
    """The result takes the subclass of the operand where that subclass fits."""

    a = Matrix(np.eye(3) * 2.)
    assert type(a.solve(Vector3([2., 4., 6.]))) is Vector3
    assert type(a.solve(Vector([2., 4., 6.]))) is Vector


def test_matrix_solve_masks_a_singular_matrix() -> None:
    """A singular matrix yields a masked solution and is not itself modified."""

    a = Matrix([[[1., 0.], [0., 1.]], [[1., 1.], [2., 2.]]])
    saved = a.values.copy()

    x = a.solve(Vector([[1., 2.], [3., 4.]]))
    assert not x.mask[0]
    assert x.mask[1]
    assert np.all(a.values == saved)


def test_matrix_solve_propagates_the_mask_of_either_operand() -> None:
    """A masked matrix or a masked right-hand side gives a masked solution."""

    a = Matrix([np.eye(2), np.eye(2)], mask=[True, False])
    b = Vector([[1., 2.], [3., 4.]], mask=[False, True])

    assert list(a.solve(b).mask) == [True, True]


def test_matrix_solve_with_nozeros_raises_on_a_singular_matrix() -> None:
    """nozeros=True skips the determinant check and reports a singular matrix."""

    with pytest.raises(ValueError, match='matrix is singular'):
        Matrix([[1., 1.], [2., 2.]]).solve(Vector([1., 2.]), nozeros=True)


def test_matrix_solve_divides_the_units() -> None:
    """The unit of the solution is the unit of the operand over that of the matrix."""

    a = Matrix([[2., 0.], [0., 2.]], unit=Unit.S)
    x = a.solve(Vector([2., 4.], unit=Unit.KM))

    assert str(x.unit_) == 'km/s'
    assert np.allclose(x.values, [1., 2.])


def test_matrix_solve_derivative_matches_a_finite_difference() -> None:
    """The derivative of the solution matches a central finite difference."""

    rng = np.random.default_rng(505)
    a_vals = rng.normal(size=(4, 4))
    b_vals = rng.normal(size=(4,))
    da = rng.normal(size=(4, 4))
    db = rng.normal(size=(4,))

    a = Matrix(a_vals)
    a.insert_deriv('t', Matrix(da))
    b = Vector(b_vals)
    b.insert_deriv('t', Vector(db))

    h = 1.e-7
    plus = Matrix(a_vals + h * da).solve(Vector(b_vals + h * db))
    minus = Matrix(a_vals - h * da).solve(Vector(b_vals - h * db))
    expected = (plus.values - minus.values) / (2. * h)

    assert np.allclose(a.solve(b).derivs['t'].values, expected, atol=1.e-6)


def test_matrix_solve_derivative_of_the_matrix_alone() -> None:
    """A derivative on the matrix alone matches the inverse-multiply result."""

    rng = np.random.default_rng(606)
    a = Matrix(rng.normal(size=(4, 4)))
    a.insert_deriv('t', Matrix(rng.normal(size=(4, 4))))
    b = Vector(rng.normal(size=(4,)))

    assert np.allclose(a.solve(b).derivs['t'].values,
                       (a.inverse() * b).derivs['t'].values)


def test_matrix_solve_derivative_of_the_operand_alone() -> None:
    """A derivative on the right-hand side alone matches the inverse-multiply result."""

    rng = np.random.default_rng(707)
    a = Matrix(rng.normal(size=(4, 4)))
    b = Vector(rng.normal(size=(4,)))
    b.insert_deriv('t', Vector(rng.normal(size=(4,))))

    assert np.allclose(a.solve(b).derivs['t'].values,
                       (a.inverse() * b).derivs['t'].values)


def test_matrix_solve_derivative_with_a_denominator() -> None:
    """A Jacobian-style derivative keeps its denominator through the solution."""

    rng = np.random.default_rng(808)
    a = Matrix(rng.normal(size=(4, 4)))
    a.insert_deriv('xy', Matrix(rng.normal(size=(4, 4, 2)), drank=1))
    b = Vector(rng.normal(size=(4,)))
    b.insert_deriv('xy', Vector(rng.normal(size=(4, 2)), drank=1))

    x = a.solve(b)
    assert x.derivs['xy'].denom == (2,)
    assert np.allclose(x.derivs['xy'].values, (a.inverse() * b).derivs['xy'].values)


def test_matrix_solve_without_recursive_drops_the_derivatives() -> None:
    """recursive=False returns a solution without derivatives."""

    a = Matrix(np.eye(2) * 2.)
    a.insert_deriv('t', Matrix(np.eye(2)))
    b = Vector([2., 4.])
    b.insert_deriv('t', Vector([1., 1.]))

    assert list(a.solve(b, recursive=False).derivs.keys()) == []
    assert list(a.solve(b, recursive=True).derivs.keys()) == ['t']


def test_matrix_solve_requires_a_square_matrix() -> None:
    """solve() rejects a matrix that is not square."""

    with pytest.raises(ValueError, match='requires a square matrix'):
        Matrix([[1., 2., 3.], [4., 5., 6.]]).solve(Vector([1., 2.]))


def test_matrix_solve_requires_matching_item_shapes() -> None:
    """solve() rejects a right-hand side of the wrong length."""

    with pytest.raises(ValueError, match='item shapes are incompatible'):
        Matrix(np.eye(3)).solve(Vector([1., 2.]))


def test_matrix_solve_rejects_a_denominator_on_the_matrix() -> None:
    """solve() rejects a matrix carrying a denominator."""

    a = Matrix(np.ones((2, 2, 3)), drank=1)
    with pytest.raises(ValueError, match='does not support denominators'):
        a.solve(Vector([1., 2.]))


def test_matrix_solve_rejects_a_denominator_on_the_operand() -> None:
    """solve() rejects a right-hand side carrying a denominator."""

    b = Vector(np.ones((2, 3)), drank=1)
    with pytest.raises(ValueError, match='right operand does not support denominators'):
        Matrix(np.eye(2)).solve(b)


##########################################################################################
